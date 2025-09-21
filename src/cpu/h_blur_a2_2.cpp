// Implementation for Median filtering using hierarchical tiling.
// Implemented for the CPU by @nxck2005 based on the paper by Louis Sugy.
// This version includes advanced optimizations: Memory Arenas, Cache-Efficient
// Transposition, SIMD Sorting, and a Dynamic Thread Pool.

// MIT License

// Copyright (c) 2025 Nishchal Ravi

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION, CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


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

// For AVX2 Intrinsics
#include <immintrin.h>

// Only using openCV for the simplicity of manipulating images as Mat arrays.
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// =========================================================================
// == OPTIMIZATION 1: MEMORY ARENA                                        ==
// =========================================================================
template<typename T>
class MemoryArena {
private:
    std::vector<T> memory;
    size_t current_pos = 0;

public:
    MemoryArena(size_t element_count) {
        memory.resize(element_count);
    }
    T* allocate(size_t count) {
        if (current_pos + count > memory.size()) { throw std::bad_alloc(); }
        T* ptr = &memory[current_pos];
        current_pos += count;
        return ptr;
    }
    void reset() { current_pos = 0; }
};

// =========================================================================
// == OPTIMIZATION 2: TYPE-SAFE AVX2 SIMD SORTING                         ==
// =========================================================================
// Templated struct to provide type-specific SIMD functions.
template<typename T>
struct SIMD {
    // Generic fallback for types without a SIMD implementation.
    static void sort(T* arr, size_t size) {
        std::sort(arr, arr + size);
    }
};

// Specialization for unsigned char (uchar)
template<>
struct SIMD<uchar> {
    static void sort(uchar* arr, size_t size) {
        if (size == 32) {
            __m256i data = _mm256_loadu_si256((__m256i*)arr);
            for (int k = 2; k <= 32; k *= 2) {
                for (int j = k / 2; j > 0; j /= 2) {
                    __m256i perm_mask = _mm256_set_epi8( 31-j, 30-j, 29-j, 28-j, 27-j, 26-j, 25-j, 24-j, 23-j, 22-j, 21-j, 20-j, 19-j, 18-j, 17-j, 16-j, 15-j, 14-j, 13-j, 12-j, 11-j, 10-j, 9-j, 8-j, 7-j, 6-j, 5-j, 4-j, 3-j, 2-j, 1-j, 0-j);
                    if ((k / j) % 2 == 0) {
                        __m256i shuffled = _mm256_shuffle_epi8(data, perm_mask);
                        __m256i temp_min = _mm256_min_epu8(data, shuffled);
                        __m256i temp_max = _mm256_max_epu8(data, shuffled);
                        data = _mm256_blendv_epi8(temp_max, temp_min, shuffled);
                    }
                }
            }
            _mm256_storeu_si256((__m256i*)arr, data);
        } else {
            std::sort(arr, arr + size);
        }
    }
};

// =========================================================================
// == OPTIMIZATION 3: DYNAMIC THREAD POOL                                 ==
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

    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers) {
            worker.join();
        }
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};


// =========================================================================
// == CORE ALGORITHM (NOW FULLY TEMPLATED)                                ==
// =========================================================================
template<typename T>
struct PixelSpan { T* data; size_t size; };

template<typename T>
struct TileState {
    int x, y, w, h;
    PixelSpan<T> sorted_core;
    std::vector<PixelSpan<T>> extra_cols;
    std::vector<PixelSpan<T>> extra_rows;
};

template<typename T>
PixelSpan<T> merge_spans(PixelSpan<T> a, PixelSpan<T> b, MemoryArena<T>& arena) {
    T* dest = arena.allocate(a.size + b.size);
    std::merge(a.data, a.data + a.size, b.data, b.data + b.size, dest);
    return {dest, a.size + b.size};
}

template<typename T>
void process_tile_recursive(const TileState<T>& state, cv::Mat& output_image, int k_size, MemoryArena<T>& arena) {
    if (state.w == 1 && state.h == 1) {
        if (state.sorted_core.size > 0) {
            output_image.at<T>(state.y, state.x) = state.sorted_core.data[state.sorted_core.size / 2];
        }
        return;
    }

    bool split_horizontally = state.w >= state.h;

    if (split_horizontally) {
        if (state.w == 1 || state.extra_cols.empty()) {
             if (state.sorted_core.size > 0) { output_image.at<T>(state.y, state.x) = state.sorted_core.data[state.sorted_core.size / 2]; }
            return;
        }
        int mid_col_idx = state.extra_cols.size() / 2;
        TileState<T> left_child, right_child;
        left_child.x = state.x; left_child.y = state.y; left_child.w = state.w/2; left_child.h = state.h;
        right_child.x = state.x + state.w/2; right_child.y = state.y; right_child.w = state.w - state.w/2; right_child.h = state.h;

        auto process_child = [&](TileState<T>& child, PixelSpan<T> merging_data) {
            child.sorted_core = merge_spans(state.sorted_core, merging_data, arena);
            int total_elements = k_size * k_size;
            size_t seen_count = child.sorted_core.size;
            if (seen_count >= total_elements) return;
            int unseen_count = total_elements - seen_count;
            size_t candidates_to_keep = unseen_count + 1;
            if (candidates_to_keep < child.sorted_core.size) {
                size_t start_idx = (child.sorted_core.size - candidates_to_keep) / 2;
                child.sorted_core.data += start_idx;
                child.sorted_core.size = candidates_to_keep;
            }
        };
        process_child(left_child, state.extra_cols[mid_col_idx]);
        process_child(right_child, state.extra_cols[mid_col_idx]);
        left_child.extra_cols = std::vector<PixelSpan<T>>(state.extra_cols.begin(), state.extra_cols.begin() + mid_col_idx);
        right_child.extra_cols = std::vector<PixelSpan<T>>(state.extra_cols.begin() + mid_col_idx + 1, state.extra_cols.end());
        left_child.extra_rows = state.extra_rows; right_child.extra_rows = state.extra_rows;
        process_tile_recursive(left_child, output_image, k_size, arena);
        process_tile_recursive(right_child, output_image, k_size, arena);
    } else { // Vertical split
        if (state.h == 1 || state.extra_rows.empty()) {
            if (state.sorted_core.size > 0) { output_image.at<T>(state.y, state.x) = state.sorted_core.data[state.sorted_core.size / 2]; }
            return;
        }
        int mid_row_idx = state.extra_rows.size() / 2;
        TileState<T> top_child, bottom_child;
        top_child.x = state.x; top_child.y = state.y; top_child.w = state.w; top_child.h = state.h/2;
        bottom_child.x = state.x; bottom_child.y = state.y + state.h/2; bottom_child.w = state.w; bottom_child.h = state.h - state.h/2;
        auto process_child = [&](TileState<T>& child, PixelSpan<T> merging_data) {
            child.sorted_core = merge_spans(state.sorted_core, merging_data, arena);
            int total_elements = k_size * k_size; size_t seen_count = child.sorted_core.size;
            if (seen_count >= total_elements) return;
            int unseen_count = total_elements - seen_count; size_t candidates_to_keep = unseen_count + 1;
            if (candidates_to_keep < child.sorted_core.size) {
                size_t start_idx = (child.sorted_core.size - candidates_to_keep) / 2;
                child.sorted_core.data += start_idx; child.sorted_core.size = candidates_to_keep;
            }
        };
        process_child(top_child, state.extra_rows[mid_row_idx]);
        process_child(bottom_child, state.extra_rows[mid_row_idx]);
        top_child.extra_rows = std::vector<PixelSpan<T>>(state.extra_rows.begin(), state.extra_rows.begin() + mid_row_idx);
        bottom_child.extra_rows = std::vector<PixelSpan<T>>(state.extra_rows.begin() + mid_row_idx + 1, state.extra_rows.end());
        top_child.extra_cols = state.extra_cols; bottom_child.extra_cols = state.extra_cols;
        process_tile_recursive(top_child, output_image, k_size, arena);
        process_tile_recursive(bottom_child, output_image, k_size, arena);
    }
}


template<typename T>
void process_root_tile(const cv::Mat& input_channel, const cv::Mat& transposed_channel, cv::Mat& output_channel, int ksize, int x, int y, int tile_w, int tile_h) {
    MemoryArena<T> arena(50 * 1024 * 1024);
    int pad = ksize / 2;

    TileState<T> root_state;
    root_state.x = x; root_state.y = y;
    root_state.w = std::min(tile_w, input_channel.cols - x);
    root_state.h = std::min(tile_h, input_channel.rows - y);

    int core_w = ksize - root_state.w + 1;
    int core_h = ksize - root_state.h + 1;
    if (core_w <= 0 || core_h <= 0) return;
    
    // Allocate and fill core
    root_state.sorted_core = {arena.allocate(core_w * core_h), (size_t)(core_w * core_h)};
    for (int i = 0; i < core_h; ++i) {
        const T* row_ptr = input_channel.ptr<T>(y + root_state.h - 1 + i + pad);
        std::copy(row_ptr + x + root_state.w - 1 + pad, row_ptr + x + root_state.w - 1 + pad + core_w, root_state.sorted_core.data + i * core_w);
    }
    SIMD<T>::sort(root_state.sorted_core.data, root_state.sorted_core.size);
    
    // Allocate and fill extra columns using transposed image for cache efficiency
    for(int i = 0; i < root_state.w - 1; ++i) {
        PixelSpan<T> col = {arena.allocate(core_h), (size_t)core_h};
        const T* trans_row_ptr = transposed_channel.ptr<T>(x + i + pad);
        std::copy(trans_row_ptr + y + root_state.h - 1 + pad, trans_row_ptr + y + root_state.h - 1 + pad + core_h, col.data);
        SIMD<T>::sort(col.data, col.size);
        root_state.extra_cols.push_back(col);
    }

    // Allocate and fill extra rows
    for(int i = 0; i < root_state.h - 1; ++i) {
        PixelSpan<T> row = {arena.allocate(core_w), (size_t)core_w};
        const T* row_ptr = input_channel.ptr<T>(y + i + pad);
        std::copy(row_ptr + x + root_state.w - 1 + pad, row_ptr + x + root_state.w - 1 + pad + core_w, row.data);
        SIMD<T>::sort(row.data, row.size);
        root_state.extra_rows.push_back(row);
    }

    process_tile_recursive(root_state, output_channel, ksize, arena);
}

template<typename T>
void hMFilter(const cv::Mat& inputarr, cv::Mat& outputarr, int ksize) {
    if (ksize % 2 == 0) { std::cerr << "Kernel size must be odd." << std::endl; return; }
    
    std::vector<cv::Mat> channels;
    cv::split(inputarr, channels);
    std::vector<cv::Mat> filtered_channels(channels.size());
    std::vector<cv::Mat> transposed_channels(channels.size());
    
    ThreadPool pool(std::thread::hardware_concurrency());

    // Prepare transposed channels
    for (size_t i = 0; i < channels.size(); ++i) {
        cv::transpose(channels[i], transposed_channels[i]);
    }

    for (size_t i = 0; i < channels.size(); ++i) {
        filtered_channels[i] = cv::Mat::zeros(channels[i].size(), channels[i].type());
        const int tile_w = 8, tile_h = 8;
        for (int y = 0; y < channels[i].rows; y += tile_h) {
            for (int x = 0; x < channels[i].cols; x += tile_w) {
                pool.enqueue([&, i, x, y, tile_w, tile_h]() {
                    process_root_tile<T>(channels[i], transposed_channels[i], filtered_channels[i], ksize, x, y, tile_w, tile_h);
                });
            }
        }
    }
    // The thread pool destructor will wait for all tasks to complete.
    
    cv::merge(filtered_channels, outputarr);
}

// COMPILE WITH:
// g++ h_blur.cpp -o h_blur $(pkg-config --cflags --libs opencv4) -pthread -mavx2 -O3
int main(int argc, char** argv) {
    if (argc < 2) { std::cout << "ERR: Please provide an image path!" << std::endl; return 1; }
    cv::Mat original_image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (original_image.empty()) { std::cout << "ERR: Img data empty" << std::endl; return 2; }
    
    cv::Mat filtered_image;
    int kernelsize = 79;
    
    // Example for uchar (standard image)
    std::cout << "--- Processing 8-bit Image (uchar) ---" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    hMFilter<uchar>(original_image, filtered_image, kernelsize);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Hierarchical version took: " << duration.count() << "ms" << std::endl;

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