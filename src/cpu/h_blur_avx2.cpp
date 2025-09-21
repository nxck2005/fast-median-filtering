// Implementation for Median filtering using hierarchical tiling.
// Implemented for the CPU by @nxck2005 based on the paper by Louis Sugy.

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
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE for any claim, damages, or other
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>

// For AVX2 Intrinsics
#include <immintrin.h>

// Only using openCV for the simplicity of manipulating images as Mat arrays.
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using PixelType = uchar;
// =========================================================================
// == OPTIMIZATION 1: MEMORY ARENA                                        ==
// =========================================================================
// Avoids frequent, slow heap allocations (new/delete) by pre-allocating
// a large chunk of memory for each thread. Allocating from the arena is
// just a fast pointer bump.
class MemoryArena {
private:
    std::vector<PixelType> memory;
    size_t current_pos = 0;

public:
    MemoryArena(size_t size_in_bytes) {
        memory.resize(size_in_bytes);
    }

    PixelType* allocate(size_t count) {
        if (current_pos + count > memory.size()) {
            // In a real-world scenario, you might handle this more gracefully.
            // For this project, we pre-allocate enough to avoid this.
            throw std::bad_alloc();
        }
        PixelType* ptr = &memory[current_pos];
        current_pos += count;
        return ptr;
    }

    void reset() {
        current_pos = 0;
    }
};

// =========================================================================
// == OPTIMIZATION 2: AVX2 SIMD SORTING NETWORK                           ==
// =========================================================================
// This function uses AVX2 intrinsics to sort 32 bytes (pixels) at once.
// A single AVX instruction can perform an operation on all 32 bytes in parallel.
// This is significantly faster than scalar (one-by-one) operations.

// Helper to perform a vectorized compare-and-swap on two 256-bit registers
inline void SIMD_COMPARE_SWAP(__m256i& a, __m256i& b) {
    __m256i temp_min = _mm256_min_epu8(a, b);
    __m256i temp_max = _mm256_max_epu8(a, b);
    a = temp_min;
    b = temp_max;
}

// A single step of a bitonic sorting network
void bitonic_merge_step(__m256i& data, int j, int k) {
    // Create a shuffle mask to swap elements based on the step parameters
    __m256i perm_mask = _mm256_set_epi8(
        31-j, 30-j, 29-j, 28-j, 27-j, 26-j, 25-j, 24-j,
        23-j, 22-j, 21-j, 20-j, 19-j, 18-j, 17-j, 16-j,
        15-j, 14-j, 13-j, 12-j, 11-j, 10-j, 9-j, 8-j,
        7-j, 6-j, 5-j, 4-j, 3-j, 2-j, 1-j, 0-j
    );

    // Conditionally swap elements based on their position in the sorting network
    if ((k / j) % 2 == 0) {
        __m256i shuffled = _mm256_shuffle_epi8(data, perm_mask);
        SIMD_COMPARE_SWAP(data, shuffled);
    }
}

void simd_sort_32_pixels_avx2(PixelType* arr) {
    __m256i data = _mm256_loadu_si256((__m256i*)arr);

    // This is a hardcoded bitonic sorting network for 32 elements.
    // The sequence of `j` and `k` values defines the network structure.
    for (int k = 2; k <= 32; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
             bitonic_merge_step(data, j, k);
        }
    }
    _mm256_storeu_si256((__m256i*)arr, data);
}


//using PixelType = float;
using PixelSpan = std::pair<PixelType*, size_t>; // Pointer and size

// Merges two sorted spans into a new location within the arena.
PixelSpan merge_spans(PixelSpan a, PixelSpan b, MemoryArena& arena) {
    PixelType* dest = arena.allocate(a.second + b.second);
    std::merge(a.first, a.first + a.second, b.first, b.first + b.second, dest);
    return {dest, a.second + b.second};
}

struct TileState {
    int x, y, w, h;
    PixelSpan sorted_core;
    std::vector<PixelSpan> extra_cols;
    std::vector<PixelSpan> extra_rows;
};

void process_tile_recursive(const TileState& state, cv::Mat& output_image, int k_size, MemoryArena& arena) {
    if (state.w == 1 && state.h == 1) {
        if (state.sorted_core.second > 0) {
            output_image.at<PixelType>(state.y, state.x) = state.sorted_core.first[state.sorted_core.second / 2];
        }
        return;
    }

    bool split_horizontally = state.w >= state.h;

    if (split_horizontally) {
        if (state.extra_cols.empty()) {
             if (state.sorted_core.second > 0) {
                 output_image.at<PixelType>(state.y, state.x) = state.sorted_core.first[state.sorted_core.second / 2];
            }
            return;
        }

        int half_w = state.w / 2;
        int mid_col_idx = state.extra_cols.size() / 2;

        TileState left_child, right_child;
        left_child.x = state.x; left_child.y = state.y;
        left_child.w = half_w; left_child.h = state.h;

        right_child.x = state.x + half_w; right_child.y = state.y;
        right_child.w = state.w - half_w; right_child.h = state.h;
        
        const auto& merging_col = state.extra_cols[mid_col_idx];
        
        auto process_child = [&](TileState& child, PixelSpan merging_data) {
            child.sorted_core = merge_spans(state.sorted_core, merging_data, arena);

            int total_elements = k_size * k_size;
            int seen_count = child.sorted_core.second;
            if (seen_count >= total_elements) return;
            
            int unseen_count = total_elements - seen_count;
            int candidates_to_keep = unseen_count + 1;

            if (candidates_to_keep < child.sorted_core.second) {
                int start_idx = (child.sorted_core.second - candidates_to_keep) / 2;
                child.sorted_core.first += start_idx;
                child.sorted_core.second = candidates_to_keep;
            }
        };

        process_child(left_child, merging_col);
        process_child(right_child, merging_col);

        left_child.extra_cols = std::vector<PixelSpan>(state.extra_cols.begin(), state.extra_cols.begin() + mid_col_idx);
        right_child.extra_cols = std::vector<PixelSpan>(state.extra_cols.begin() + mid_col_idx + 1, state.extra_cols.end());
        left_child.extra_rows = state.extra_rows;
        right_child.extra_rows = state.extra_rows;
        
        process_tile_recursive(left_child, output_image, k_size, arena);
        process_tile_recursive(right_child, output_image, k_size, arena);

    } else { // Vertical split logic
         if (state.extra_rows.empty()) {
            if (state.sorted_core.second > 0) {
                 output_image.at<PixelType>(state.y, state.x) = state.sorted_core.first[state.sorted_core.second / 2];
            }
            return;
        }

        int half_h = state.h / 2;
        int mid_row_idx = state.extra_rows.size() / 2;
        
        TileState top_child, bottom_child;
        top_child.x = state.x; top_child.y = state.y;
        top_child.w = state.w; top_child.h = half_h;

        bottom_child.x = state.x; bottom_child.y = state.y + half_h;
        bottom_child.w = state.w; bottom_child.h = state.h - half_h;

        const auto& merging_row = state.extra_rows[mid_row_idx];
        
        auto process_child = [&](TileState& child, PixelSpan merging_data) {
            child.sorted_core = merge_spans(state.sorted_core, merging_data, arena);
            int total_elements = k_size * k_size;
            int seen_count = child.sorted_core.second;
            if (seen_count >= total_elements) return;
            int unseen_count = total_elements - seen_count;
            int candidates_to_keep = unseen_count + 1;
            if (candidates_to_keep < child.sorted_core.second) {
                int start_idx = (child.sorted_core.second - candidates_to_keep) / 2;
                child.sorted_core.first += start_idx;
                child.sorted_core.second = candidates_to_keep;
            }
        };

        process_child(top_child, merging_row);
        process_child(bottom_child, merging_row);

        top_child.extra_rows = std::vector<PixelSpan>(state.extra_rows.begin(), state.extra_rows.begin() + mid_row_idx);
        bottom_child.extra_rows = std::vector<PixelSpan>(state.extra_rows.begin() + mid_row_idx + 1, state.extra_rows.end());
        top_child.extra_cols = state.extra_cols;
        bottom_child.extra_cols = state.extra_cols;

        process_tile_recursive(top_child, output_image, k_size, arena);
        process_tile_recursive(bottom_child, output_image, k_size, arena);
    }
}

void process_single_channel(const cv::Mat& input_channel, cv::Mat& output_channel, int ksize, int start_row, int end_row) {
    MemoryArena arena(50 * 1024 * 1024); // 50 MB per thread
    int pad = ksize / 2;
    cv::Mat padded_image;
    cv::copyMakeBorder(input_channel, padded_image, pad, pad, pad, pad, cv::BORDER_REPLICATE);

    const int tile_w = 8, tile_h = 8;

    for (int y = start_row; y < end_row; y += tile_h) {
        for (int x = 0; x < input_channel.cols; x += tile_w) {
            arena.reset();
            TileState root_state;
            root_state.x = x; root_state.y = y;
            root_state.w = std::min(tile_w, input_channel.cols - x);
            root_state.h = std::min(tile_h, input_channel.rows - y);

            int core_start_x = x + root_state.w - 1;
            int core_start_y = y + root_state.h - 1;
            int core_w = ksize - root_state.w + 1;
            int core_h = ksize - root_state.h + 1;

            if (core_w <= 0 || core_h <= 0) continue;

            root_state.sorted_core.first = arena.allocate(core_w * core_h);
            root_state.sorted_core.second = core_w * core_h;
            for (int i = 0; i < core_h; ++i) {
                for (int j = 0; j < core_w; ++j) {
                    root_state.sorted_core.first[i * core_w + j] = padded_image.at<PixelType>(core_start_y + i, core_start_x + j);
                }
            }
            std::sort(root_state.sorted_core.first, root_state.sorted_core.first + root_state.sorted_core.second);
            
            for(int i = 0; i < root_state.w - 1; ++i) {
                PixelSpan col = {arena.allocate(core_h), (size_t)core_h};
                for(int j=0; j < core_h; ++j) { col.first[j] = padded_image.at<PixelType>(core_start_y + j, x + i); }
                std::sort(col.first, col.first + col.second);
                root_state.extra_cols.push_back(col);
            }
             for(int i = 0; i < root_state.h - 1; ++i) {
                PixelSpan row = {arena.allocate(core_w), (size_t)core_w};
                for(int j=0; j < core_w; ++j) { row.first[j] = padded_image.at<PixelType>(y + i, core_start_x + j); }
                std::sort(row.first, row.first + row.second);
                root_state.extra_rows.push_back(row);
            }

            process_tile_recursive(root_state, output_channel, ksize, arena);
        }
    }
}


void hMFilter(const cv::Mat& inputarr, cv::Mat& outputarr, int ksize) {
    if (ksize % 2 == 0) { std::cerr << "Kernel size must be odd." << std::endl; return; }
    
    std::vector<cv::Mat> channels;
    cv::split(inputarr, channels);
    std::vector<cv::Mat> filtered_channels(channels.size());

    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    for (size_t i = 0; i < channels.size(); ++i) {
        filtered_channels[i] = cv::Mat::zeros(inputarr.size(), channels[i].type());
        int rows_per_thread = inputarr.rows / num_threads;
        
        for (unsigned int t = 0; t < num_threads; ++t) {
            int start_row = t * rows_per_thread;
            int end_row = (t == num_threads - 1) ? inputarr.rows : start_row + rows_per_thread;
            
            threads.emplace_back([&, i, start_row, end_row]() {
                process_single_channel(channels[i], filtered_channels[i], ksize, start_row, end_row);
            });
        }
    }

    for (auto& t : threads) { if (t.joinable()) t.join(); }
    cv::merge(filtered_channels, outputarr);
}

// COMPILE WITH:
// g++ h_blur.cpp -o h_blur $(pkg-config --cflags --libs opencv4) -pthread -mavx2 -O3
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "ERR: Please provide an image path!" << std::endl;
        return 1;
    }
    cv::Mat original_image = cv::imread(argv[1]);
    if (original_image.empty()) {
        std::cout << "ERR: Img data empty" << std::endl;
        return 2;
    }
    cv::Mat filtered_image;

    int kernelsize = 79;
    
    std::cout << "Starting Hierarchical Median Filter..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    hMFilter(original_image, filtered_image, kernelsize);
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

    bool saved = cv::imwrite("filtered_output_h.jpg", filtered_image);
    if (saved) {
        std::cout << "Saved!" << std::endl;
    } else {
        std::cout << "Error while saving." << std::endl;
    }
    cv::waitKey(0);
    return 0;
}

