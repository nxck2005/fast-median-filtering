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

// Only using openCV for the simplicity of manipulating images as Mat arrays.
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// =========================================================================
// == HELPER FUNCTIONS AND DATA STRUCTURES                                ==
// =========================================================================

using PixelType = uchar;

// A simple compare-and-swap operation. The fundamental building block.
inline void compare_swap(PixelType& a, PixelType& b) {
    if (a > b) {
        std::swap(a, b);
    }
}

// Replaced std::sort with a true, data-oblivious sorting network.
// This implements an odd-even merge sort. For small, fixed-size inputs, this
// avoids data-dependent branching, which is better for CPU pipelines and SIMD.
void odd_even_merge_sort(std::vector<PixelType>& arr) {
    if (arr.size() <= 1) return;
    for (size_t p = 1; p < arr.size(); p <<= 1) {
        for (size_t k = p; k > 0; k >>= 1) {
            for (size_t j = k % p; j < arr.size() - k; j += 2 * k) {
                for (size_t i = 0; i < k; ++i) {
                    if (j + i + k < arr.size()) {
                        if (( (j + i) / (p * 2) ) == ( (j + i + k) / (p * 2) )) {
                            compare_swap(arr[j + i], arr[j + i + k]);
                        }
                    }
                }
            }
        }
    }
}

// Merges two sorted vectors into one.
std::vector<PixelType> merge_vectors(const std::vector<PixelType>& a, const std::vector<PixelType>& b) {
    std::vector<PixelType> result;
    result.reserve(a.size() + b.size());
    std::merge(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));
    return result;
}

// Represents the state of a tile.
struct TileState {
    int x, y, w, h;
    std::vector<PixelType> sorted_core;
    std::vector<std::vector<PixelType>> extra_cols;
    std::vector<std::vector<PixelType>> extra_rows;
};


// The recursive function that performs the tile splitting.
void process_tile_recursive(
    const TileState& state,
    cv::Mat& output_image,
    int k_size
) {
    // Base Case: If we have a 1x1 tile, we've found the median.
    if (state.w == 1 && state.h == 1) {
        if (!state.sorted_core.empty()) {
            output_image.at<PixelType>(state.y, state.x) = state.sorted_core[state.sorted_core.size() / 2];
        }
        return;
    }

    // Recursive Step: Split the tile on its longer side.
    bool split_horizontally = state.w >= state.h;

    if (split_horizontally) {
        if (state.extra_cols.empty()) {
             if (!state.sorted_core.empty()) {
                 output_image.at<PixelType>(state.y, state.x) = state.sorted_core[state.sorted_core.size() / 2];
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
        
        auto process_child = [&](TileState& child, const std::vector<PixelType>& merging_data) {
            child.sorted_core = merge_vectors(state.sorted_core, merging_data);

            // FINAL OPTIMIZATION: "Forgetfulness" Principle
            // After merging, we trim the list of candidates. We only need to keep the
            // values that could possibly be the median, given the number of elements
            // we still haven't "seen". This dramatically reduces the data passed down.
            int total_elements = k_size * k_size;
            int seen_count = child.sorted_core.size();
            if (seen_count >= total_elements) return; // All elements seen
            
            int unseen_count = total_elements - seen_count;
            int candidates_to_keep = unseen_count + 1;

            if (candidates_to_keep < child.sorted_core.size()) {
                int start_idx = (child.sorted_core.size() - candidates_to_keep) / 2;
                child.sorted_core = std::vector<PixelType>(
                    child.sorted_core.begin() + start_idx,
                    child.sorted_core.begin() + start_idx + candidates_to_keep
                );
            }
        };

        process_child(left_child, merging_col);
        process_child(right_child, merging_col);

        left_child.extra_cols = std::vector<std::vector<PixelType>>(state.extra_cols.begin(), state.extra_cols.begin() + mid_col_idx);
        right_child.extra_cols = std::vector<std::vector<PixelType>>(state.extra_cols.begin() + mid_col_idx + 1, state.extra_cols.end());
        left_child.extra_rows = state.extra_rows;
        right_child.extra_rows = state.extra_rows;
        
        process_tile_recursive(left_child, output_image, k_size);
        process_tile_recursive(right_child, output_image, k_size);

    } else { // Vertical split logic
         if (state.extra_rows.empty()) {
            if (!state.sorted_core.empty()) {
                 output_image.at<PixelType>(state.y, state.x) = state.sorted_core[state.sorted_core.size() / 2];
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
        
        // Using the same lambda for processing
        auto process_child = [&](TileState& child, const std::vector<PixelType>& merging_data) {
            child.sorted_core = merge_vectors(state.sorted_core, merging_data);
            int total_elements = k_size * k_size;
            int seen_count = child.sorted_core.size();
            if (seen_count >= total_elements) return;
            int unseen_count = total_elements - seen_count;
            int candidates_to_keep = unseen_count + 1;
            if (candidates_to_keep < child.sorted_core.size()) {
                int start_idx = (child.sorted_core.size() - candidates_to_keep) / 2;
                child.sorted_core = std::vector<PixelType>(child.sorted_core.begin() + start_idx, child.sorted_core.begin() + start_idx + candidates_to_keep);
            }
        };

        process_child(top_child, merging_row);
        process_child(bottom_child, merging_row);

        top_child.extra_rows = std::vector<std::vector<PixelType>>(state.extra_rows.begin(), state.extra_rows.begin() + mid_row_idx);
        bottom_child.extra_rows = std::vector<std::vector<PixelType>>(state.extra_rows.begin() + mid_row_idx + 1, state.extra_rows.end());
        top_child.extra_cols = state.extra_cols;
        bottom_child.extra_cols = state.extra_cols;

        process_tile_recursive(top_child, output_image, k_size);
        process_tile_recursive(bottom_child, output_image, k_size);
    }
}

void process_single_channel(const cv::Mat& input_channel, cv::Mat& output_channel, int ksize, int start_row, int end_row) {
    int pad = ksize / 2;
    cv::Mat padded_image;
    cv::copyMakeBorder(input_channel, padded_image, pad, pad, pad, pad, cv::BORDER_REPLICATE);

    const int tile_w = 8, tile_h = 8;

    for (int y = start_row; y < end_row; y += tile_h) {
        for (int x = 0; x < input_channel.cols; x += tile_w) {
            TileState root_state;
            root_state.x = x;
            root_state.y = y;
            root_state.w = std::min(tile_w, input_channel.cols - x);
            root_state.h = std::min(tile_h, input_channel.rows - y);

            int core_start_x = x + root_state.w - 1;
            int core_start_y = y + root_state.h - 1;
            int core_w = ksize - root_state.w + 1;
            int core_h = ksize - root_state.h + 1;

            if (core_w <= 0 || core_h <= 0) continue;

            for (int i = 0; i < core_h; ++i) {
                for (int j = 0; j < core_w; ++j) {
                    root_state.sorted_core.push_back(padded_image.at<PixelType>(core_start_y + i, core_start_x + j));
                }
            }
            odd_even_merge_sort(root_state.sorted_core);
            
            for(int i = 0; i < root_state.w - 1; ++i) {
                std::vector<PixelType> col;
                for(int j=0; j < core_h; ++j) {
                    col.push_back(padded_image.at<PixelType>(core_start_y + j, x + i));
                }
                odd_even_merge_sort(col);
                root_state.extra_cols.push_back(col);
            }
            
             for(int i = 0; i < root_state.h - 1; ++i) {
                std::vector<PixelType> row;
                for(int j=0; j < core_w; ++j) {
                    row.push_back(padded_image.at<PixelType>(y + i, core_start_x + j));
                }
                odd_even_merge_sort(row);
                root_state.extra_rows.push_back(row);
            }

            process_tile_recursive(root_state, output_channel, ksize);
        }
    }
}


// The actual method
void hMFilter(const cv::Mat& inputarr, cv::Mat& outputarr, int ksize) {
    if (ksize % 2 == 0) {
        std::cerr << "Kernel size must be odd." << std::endl;
        return;
    }
    
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

    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }

    cv::merge(filtered_channels, outputarr);
}

// g++ h_blur.cpp -o h_blur $(pkg-config --cflags --libs opencv4) -pthread
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

    int kernelsize = 17;
    
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

