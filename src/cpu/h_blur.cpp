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

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
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

// A sorting network for a vector of pixels.
// This uses a simple bubble sort approach for clarity. A real high-performance
// implementation would use fixed Batcher's odd-even or bitonic networks.
void sort_network(std::vector<PixelType>& arr) {
    std::sort(arr.begin(), arr.end());
}

// Merges two sorted vectors into one sorted vector.
std::vector<PixelType> merge_network(const std::vector<PixelType>& a, const std::vector<PixelType>& b) {
    std::vector<PixelType> result;
    result.reserve(a.size() + b.size());
    std::merge(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));
    return result;
}

// Represents the state of a tile during the recursion.
struct TileState {
    int x, y; // Top-left corner of the tile in the output image
    int w, h; // Dimensions of the tile
    std::vector<PixelType> sorted_core;
    std::vector<std::vector<PixelType>> extra_cols_left;
    std::vector<std::vector<PixelType>> extra_cols_right;
    std::vector<std::vector<PixelType>> extra_rows_top;
    std::vector<std::vector<PixelType>> extra_rows_bottom;
};


// The recursive function that performs the tile splitting.
void process_tile_recursive(
    TileState state,
    const cv::Mat& padded_image,
    cv::Mat& output_image,
    int k_size
) {
    // Base Case: If we have a 1x1 tile, we've found the median.
    if (state.w == 1 && state.h == 1) {
        if (!state.sorted_core.empty()) {
            // The middle element of the sorted core is the median.
            output_image.at<PixelType>(state.y, state.x) = state.sorted_core[state.sorted_core.size() / 2];
        }
        return;
    }

    // Recursive Step: Split the tile. The paper splits on the longer side.
    bool split_horizontally = state.w >= state.h;

    if (split_horizontally) {
        int half_w = state.w / 2;

        // Create states for the left and right child tiles.
        TileState left_child_state = state;
        left_child_state.w = half_w;

        TileState right_child_state = state;
        right_child_state.x = state.x + half_w;
        right_child_state.w = state.w - half_w;

        // Process Left Child
        {
            // The "new" inputs are the right extra columns of the parent tile.
            std::vector<PixelType> new_elements_to_merge;
            for (const auto& col : state.extra_cols_right) {
                new_elements_to_merge.insert(new_elements_to_merge.end(), col.begin(), col.end());
            }
            sort_network(new_elements_to_merge);
            left_child_state.sorted_core = merge_network(state.sorted_core, new_elements_to_merge);

            // "Forgetfulness" Principle: Trim the sorted core.
            // For a pixel in the child tile, we calculate how many elements it sees.
            int core_size = (k_size - left_child_state.w + 1) * (k_size - left_child_state.h + 1);
            int seen_count = core_size + new_elements_to_merge.size();
            int total_elements = k_size * k_size;
            int unseen_count = total_elements - seen_count;
            int median_of_n = (total_elements + 1) / 2;
            int median_candidate_count = unseen_count + 1;
            
            int start_idx = (left_child_state.sorted_core.size() - median_candidate_count) / 2;
            if(start_idx < 0) start_idx = 0;
            int end_idx = start_idx + median_candidate_count;
            if(end_idx > left_child_state.sorted_core.size()) end_idx = left_child_state.sorted_core.size();

            left_child_state.sorted_core = std::vector<PixelType>(
                left_child_state.sorted_core.begin() + start_idx,
                left_child_state.sorted_core.begin() + end_idx
            );

            process_tile_recursive(left_child_state, padded_image, output_image, k_size);
        }
        
        // Process Right Child (similar logic)
        {
             std::vector<PixelType> new_elements_to_merge;
            for (const auto& col : state.extra_cols_left) {
                new_elements_to_merge.insert(new_elements_to_merge.end(), col.begin(), col.end());
            }
            sort_network(new_elements_to_merge);
            right_child_state.sorted_core = merge_network(state.sorted_core, new_elements_to_merge);
            
            // Apply forgetfulness... (same logic as above)
             int core_size = (k_size - right_child_state.w + 1) * (k_size - right_child_state.h + 1);
            int seen_count = core_size + new_elements_to_merge.size();
            int total_elements = k_size * k_size;
            int unseen_count = total_elements - seen_count;
            int median_candidate_count = unseen_count + 1;
             int start_idx = (right_child_state.sorted_core.size() - median_candidate_count) / 2;
             if(start_idx < 0) start_idx = 0;
            int end_idx = start_idx + median_candidate_count;
            if(end_idx > right_child_state.sorted_core.size()) end_idx = right_child_state.sorted_core.size();

             right_child_state.sorted_core = std::vector<PixelType>(
                right_child_state.sorted_core.begin() + start_idx,
                right_child_state.sorted_core.begin() + end_idx
            );

            process_tile_recursive(right_child_state, padded_image, output_image, k_size);
        }

    } else { // Split vertically (similar logic to horizontal)
        int half_h = state.h / 2;
        TileState top_child_state = state;
        top_child_state.h = half_h;
        // ... and so on for the vertical split
    }
}

// The actual method
void hMFilter(const cv::Mat& inputarr, cv::Mat& outputarr, int ksize) {
    if (ksize % 2 == 0) {
        std::cerr << "Kernel size must be odd." << std::endl;
        return;
    }
    
    // For color images, we process each channel separately.
    if (inputarr.channels() == 3) {
        std::vector<cv::Mat> channels(3);
        cv::split(inputarr, channels);
        
        std::vector<cv::Mat> filtered_channels(3);
        
        // Process each channel in parallel
        std::vector<std::thread> threads;
        for (int i = 0; i < 3; ++i) {
            threads.emplace_back([&, i]() {
                hMFilter(channels[i], filtered_channels[i], ksize);
            });
        }
        for (auto& t : threads) {
            t.join();
        }

        cv::merge(filtered_channels, outputarr);
        return;
    }


    // 1. PAD THE IMAGE
    // Pad the image to handle pixels at the border.
    int pad = ksize / 2;
    cv::Mat padded_image;
    cv::copyMakeBorder(inputarr, padded_image, pad, pad, pad, pad, cv::BORDER_REPLICATE);

    outputarr = cv::Mat::zeros(inputarr.size(), inputarr.type());

    // 2. DEFINE ROOT TILE SIZE (e.g., 8x8 as in the paper)
    const int tile_w = 8;
    const int tile_h = 8;

    // 3. ITERATE OVER IMAGE IN ROOT TILES
    for (int y = 0; y < inputarr.rows; y += tile_h) {
        for (int x = 0; x < inputarr.cols; x += tile_w) {

            // =============================================================
            // == INITIALIZATION STAGE FOR ONE ROOT TILE                  ==
            // =============================================================
            
            // Define the footprint, core, etc. for this tile
            int core_w = ksize - tile_w + 1;
            int core_h = ksize - tile_h + 1;
            
            // Load core elements
            std::vector<PixelType> core_elements;
            for (int i = 0; i < core_h; ++i) {
                for (int j = 0; j < core_w; ++j) {
                     // Coords relative to padded_image
                    core_elements.push_back(padded_image.at<PixelType>(y + tile_h - 1 + i, x + tile_w - 1 + j));
                }
            }
            sort_network(core_elements);

            // Create the initial state for the root tile
            TileState root_state;
            root_state.x = x;
            root_state.y = y;
            root_state.w = std::min(tile_w, inputarr.cols - x);
            root_state.h = std::min(tile_h, inputarr.rows - y);
            root_state.sorted_core = core_elements;
            
            // In a full implementation, you'd also load and sort the extra_cols and extra_rows here.
            // This is a simplified version where we start the recursion with just the core.
            
            // =============================================================
            // == RECURSION STAGE                                         ==
            // =============================================================
            process_tile_recursive(root_state, padded_image, outputarr, ksize);
        }
    }
}

// g++ h_blur.cpp -o h_blur $(pkg-config --cflags --libs opencv4)
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
    
    // the timing section
    std::cout << "Starting Hierarchical Median Filter..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // replace with new func
    hMFilter(original_image, filtered_image, kernelsize);

    auto end = std::chrono::high_resolution_clock::now();
    // end of func

    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << duration.count() << "ms was taken" << std::endl;

    // Compare with OpenCV's implementation
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
