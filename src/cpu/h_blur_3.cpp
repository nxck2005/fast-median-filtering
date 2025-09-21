#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <span> 

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// =========================================================================
// == MEMORY ARENA                                                        ==
// =========================================================================
template<typename T>
class MemoryArena {
private:
    std::vector<T> memory;
    size_t current_pos = 0;
public:
    MemoryArena(size_t element_count) { memory.resize(element_count); }
    T* allocate(size_t count) {
        if (current_pos + count > memory.size()) { throw std::bad_alloc(); }
        T* ptr = &memory[current_pos];
        current_pos += count;
        return ptr;
    }
    void reset() { current_pos = 0; }
    // Get current usage for debugging or optimization
    size_t get_usage() const { return current_pos; }
};

// =========================================================================
// == HISTOGRAM SORT FOR uchar (8-bit)                                    ==
// =========================================================================
void histogram_sort(uchar* arr, size_t size) {
    if (size == 0) return;
    unsigned int counts[256] = {0};
    for (size_t i = 0; i < size; ++i) { counts[arr[i]]++; }
    size_t current_pos = 0;
    for (int i = 0; i < 256; ++i) {
        if (counts[i] > 0) {
            std::fill(arr + current_pos, arr + current_pos + counts[i], (uchar)i);
            current_pos += counts[i];
        }
    }
}

template<typename T> struct SIMD {
    static void sort(T* arr, size_t size) { if (size > 1) std::sort(arr, arr + size); }
};

template<> struct SIMD<uchar> {
    static void sort(uchar* arr, size_t size) { histogram_sort(arr, size); }
};

// =========================================================================
// == DYNAMIC THREAD POOL                                                 ==
// =========================================================================
class ThreadPool {
public:
    ThreadPool(size_t num_threads) : stop(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty()) return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }
    template<class F> void enqueue(F&& f) {
        { std::unique_lock<std::mutex> lock(queue_mutex); tasks.emplace(std::forward<F>(f)); }
        condition.notify_one();
    }
    ~ThreadPool() {
        { std::unique_lock<std::mutex> lock(queue_mutex); stop = true; }
        condition.notify_all();
        for (std::thread &worker : workers) { worker.join(); }
    }
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

// =========================================================================
// == CORE ALGORITHM                                                      ==
// =========================================================================
template<typename T> struct PixelSpan { T* data; size_t size; };
template<typename T> struct TileState {
    int x, y, w, h;
    PixelSpan<T> sorted_core;
    std::span<PixelSpan<T>> extra_cols;
    std::span<PixelSpan<T>> extra_rows;
};

template<typename T>
PixelSpan<T> merge_spans(PixelSpan<T> a, PixelSpan<T> b, MemoryArena<T>& arena) {
    if (a.size == 0) return b;
    if (b.size == 0) return a;
    T* dest = arena.allocate(a.size + b.size);
    std::merge(a.data, a.data + a.size, b.data, b.data + b.size, dest);
    return {dest, a.size + b.size};
}

template<typename T>
void process_tile_recursive(const TileState<T>& state, cv::Mat& output_image, int k_size, MemoryArena<T>& arena) {
    if (state.w <= 0 || state.h <= 0) return;

    // This lambda contains the core logic for merging, pruning, and finding the median.
    // It is called on a child tile state *before* the decision to recurse is made.
    auto process_child = [&](TileState<T>& child, PixelSpan<T> merging_data) {
        child.sorted_core = merge_spans(state.sorted_core, merging_data, arena);
        
        size_t total_elements = (size_t)k_size * k_size;
        size_t seen_count = child.sorted_core.size;

        if (seen_count >= total_elements) {
            size_t median_offset = (seen_count - total_elements) / 2 + (total_elements / 2);
            if(child.y < output_image.rows && child.x < output_image.cols && median_offset < child.sorted_core.size) {
                output_image.at<T>(child.y, child.x) = child.sorted_core.data[median_offset];
            }
            child.sorted_core.size = 0; // Signal that the median is found and recursion should stop for this path.
            return;
        }
        
        // Prune the candidate list to keep only the necessary elements for finding the median later.
        size_t unseen_count = total_elements - seen_count;
        size_t candidates_to_keep = unseen_count + 1;
        if (candidates_to_keep < child.sorted_core.size) {
            size_t start_idx = (child.sorted_core.size - candidates_to_keep) / 2;
            child.sorted_core.data += start_idx;
            child.sorted_core.size = candidates_to_keep;
        }
    };

    // =================================================================================
    // == FIX: The primary bug was here. The split decision must be based on whether a ==
    // == split is *possible*, not just on the tile's dimensions. Accessing an empty  ==
    // == span (e.g., state.extra_cols[...]) causes a segfault.                     ==
    // =================================================================================
    
    // Determine if splits are possible based on remaining rows/columns to merge.
    bool can_split_horizontally = state.w > 1 && !state.extra_cols.empty();
    bool can_split_vertically = state.h > 1 && !state.extra_rows.empty();

    // Base case: If no further splits are possible, we must be at a 1x1 tile.
    // The median is the middle of whatever candidates we have collected.
    if (!can_split_horizontally && !can_split_vertically) {
        if (state.w == 1 && state.h == 1 && state.sorted_core.size > 0) {
            output_image.at<T>(state.y, state.x) = state.sorted_core.data[state.sorted_core.size / 2];
        }
        return;
    }

    // Decide split direction: prefer splitting the larger dimension, but only if possible.
    bool split_horizontally = (can_split_horizontally && state.w >= state.h) || (can_split_horizontally && !can_split_vertically);

    if (split_horizontally) {
        int mid_col_idx = state.extra_cols.size() / 2;
        int w_left = state.w / 2;

        TileState<T> left_child {state.x, state.y, w_left, state.h};
        TileState<T> right_child {state.x + w_left, state.y, state.w - w_left, state.h};
        
        process_child(left_child, state.extra_cols[mid_col_idx]);
        process_child(right_child, state.extra_cols[mid_col_idx]);
        
        // FIX: The original code only checked left_child, but recursed on both unconditionally.
        // We must check each child independently to see if its median has been found.
        if (left_child.sorted_core.size > 0) {
            left_child.extra_cols = state.extra_cols.subspan(0, mid_col_idx);
            left_child.extra_rows = state.extra_rows; // Pass through rows for the next level
            process_tile_recursive(left_child, output_image, k_size, arena);
        }
        if (right_child.sorted_core.size > 0) {
            right_child.extra_cols = state.extra_cols.subspan(mid_col_idx + 1);
            right_child.extra_rows = state.extra_rows;
            process_tile_recursive(right_child, output_image, k_size, arena);
        }

    } else { // Vertical split (can_split_vertically must be true)
        int mid_row_idx = state.extra_rows.size() / 2;
        int h_top = state.h / 2;
        TileState<T> top_child {state.x, state.y, state.w, h_top};
        TileState<T> bottom_child {state.x, state.y + h_top, state.w, state.h - h_top};

        process_child(top_child, state.extra_rows[mid_row_idx]);
        process_child(bottom_child, state.extra_rows[mid_row_idx]);
        
        // FIX: Apply the same independent check for vertical children.
        if (top_child.sorted_core.size > 0) {
            top_child.extra_rows = state.extra_rows.subspan(0, mid_row_idx);
            top_child.extra_cols = state.extra_cols; // Pass through cols
            process_tile_recursive(top_child, output_image, k_size, arena);
        }
        if (bottom_child.sorted_core.size > 0) {
            bottom_child.extra_rows = state.extra_rows.subspan(mid_row_idx + 1);
            bottom_child.extra_cols = state.extra_cols;
            process_tile_recursive(bottom_child, output_image, k_size, arena);
        }
    }
}

template<typename T>
void process_root_tile(const cv::Mat& padded_channel, const cv::Mat& transposed_channel, cv::Mat& output_channel, int ksize, int x, int y, int tile_w, int tile_h, MemoryArena<T>& arena) {
    arena.reset();
    int pad = ksize / 2;

    int current_tile_w = std::min(tile_w, output_channel.cols - x);
    int current_tile_h = std::min(tile_h, output_channel.rows - y);

    if (current_tile_w <= 0 || current_tile_h <= 0) return;
    
    // NOTE on std::span usage: The `extra_cols_vec` and `extra_rows_vec` are local vectors.
    // Creating spans from them is safe *only because* the entire recursive call stack for this
    // tile completes synchronously within this function's scope. If the recursion were
    // enqueued as separate tasks, this would create dangling pointers.
    std::vector<PixelSpan<T>> extra_cols_vec, extra_rows_vec;
    extra_cols_vec.reserve(current_tile_w > 0 ? current_tile_w - 1 : 0);
    extra_rows_vec.reserve(current_tile_h > 0 ? current_tile_h - 1 : 0);

    // Calculate core region and initial sorted list
    int core_w = ksize - current_tile_w + 1;
    int core_h = ksize - current_tile_h + 1;
    PixelSpan<T> sorted_core = {arena.allocate((size_t)core_w * core_h), (size_t)(core_w * core_h)};
    for (int i = 0; i < core_h; ++i) {
        const T* row_ptr = padded_channel.ptr<T>(y + current_tile_h - 1 + i);
        std::copy(row_ptr + x + current_tile_w - 1, row_ptr + x + current_tile_w - 1 + core_w, sorted_core.data + (size_t)i * core_w);
    }
    SIMD<T>::sort(sorted_core.data, sorted_core.size);
    
    // Pre-sort all extra columns and rows needed for this tile
    for(int i = 0; i < current_tile_w - 1; ++i) {
        PixelSpan<T> col = {arena.allocate((size_t)ksize), (size_t)ksize};
        const T* trans_row_ptr = transposed_channel.ptr<T>(x + i);
        std::copy(trans_row_ptr + y, trans_row_ptr + y + ksize, col.data);
        SIMD<T>::sort(col.data, col.size);
        extra_cols_vec.push_back(col);
    }
    for(int i = 0; i < current_tile_h - 1; ++i) {
        PixelSpan<T> row = {arena.allocate((size_t)ksize), (size_t)ksize};
        const T* row_ptr = padded_channel.ptr<T>(y + i);
        std::copy(row_ptr + x, row_ptr + x + ksize, row.data);
        SIMD<T>::sort(row.data, row.size);
        extra_rows_vec.push_back(row);
    }
    
    // Set up the initial state for the entire tile and start recursion
    TileState<T> root_state;
    root_state.x = x; root_state.y = y;
    root_state.w = current_tile_w;
    root_state.h = current_tile_h;
    root_state.sorted_core = sorted_core;
    root_state.extra_cols = std::span(extra_cols_vec);
    root_state.extra_rows = std::span(extra_rows_vec);
    
    process_tile_recursive(root_state, output_channel, ksize, arena);
}


template<typename T>
void hMFilter(const cv::Mat& inputarr, cv::Mat& outputarr, int ksize) {
    if (ksize % 2 == 0) { std::cerr << "Kernel size must be odd." << std::endl; return; }
    
    int pad = ksize / 2;
    cv::Mat padded_input;
    cv::copyMakeBorder(inputarr, padded_input, pad, pad, pad, pad, cv::BORDER_REPLICATE);

    std::vector<cv::Mat> channels;
    cv::split(padded_input, channels);
    std::vector<cv::Mat> filtered_channels(channels.size());
    std::vector<cv::Mat> transposed_channels(channels.size());
    
    // Create the thread pool once. It will be destroyed when it goes out of scope, joining all threads.
    ThreadPool pool(std::thread::hardware_concurrency());

    for (size_t i = 0; i < channels.size(); ++i) {
        cv::transpose(channels[i], transposed_channels[i]);
    }

    for (size_t i = 0; i < channels.size(); ++i) {
        filtered_channels[i] = cv::Mat::zeros(inputarr.size(), CV_8U);
        const int tile_size = 64; 
        for (int y = 0; y < inputarr.rows; y += tile_size) {
            for (int x = 0; x < inputarr.cols; x += tile_size) {
                // The lambda captures loop variables by value, and vectors/mats by reference.
                // This is safe because the main thread waits for the pool to finish before these go out of scope.
                pool.enqueue([&, i, x, y, tile_size, ksize]() {
                    // Each thread gets its own memory arena to prevent data races and false sharing.
                    static thread_local MemoryArena<T> arena(30 * 1024 * 1024); // Increased size slightly for safety
                    process_root_tile<T>(channels[i], transposed_channels[i], filtered_channels[i], ksize, x, y, tile_size, tile_size, arena);
                });
            }
        }
    }
    // The ThreadPool destructor is called here, which implicitly waits for all tasks to complete.
    
    cv::merge(filtered_channels, outputarr);
}

// COMPILE WITH C++20 for std::span:
// g++ h_blur_fixed.cpp -o h_blur $(pkg-config --cflags --libs opencv4) -pthread -mavx2 -O3 -std=c++20
int main(int argc, char** argv) {
    if (argc < 2) { std::cout << "ERR: Please provide an image path!" << std::endl; return 1; }
    cv::Mat original_image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (original_image.empty()) { std::cout << "ERR: Img data empty" << std::endl; return 2; }
    
    cv::Mat filtered_image;
    int kernelsize = 79; 
    
    std::cout << "--- Processing 8-bit Image (uchar) ---" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    hMFilter<uchar>(original_image, filtered_image, kernelsize);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Optimized Hierarchical version took: " << duration.count() << "ms" << std::endl;

    cv::Mat opencv_filtered;
    std::cout << "Starting OpenCV Median Filter..." << std::endl;
    auto start_cv = std::chrono::high_resolution_clock::now();
    cv::medianBlur(original_image, opencv_filtered, kernelsize);
    auto end_cv = std::chrono::high_resolution_clock::now();
    auto duration_cv = std::chrono::duration_cast<std::chrono::milliseconds>(end_cv - start_cv);
    std::cout << "OpenCV's version took: " << duration_cv.count() << "ms" << std::endl;

    cv::imshow("Original", original_image);
    cv::imshow("Median Filtered (Hierarchical)", filtered_image);
    cv::imshow("Median Filtered (OpenCV)", opencv_filtered);
    
    cv::waitKey(0);
    return 0;
}
