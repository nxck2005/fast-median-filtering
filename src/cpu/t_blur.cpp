#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <omp.h> // OpenMP for parallelization

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// --- Optimized Tiled Median Filter with Sliding Histogram ---

/**
 * @brief Finds the median value from a 256-bin histogram.
 * @param histogram An array of 256 integers representing the pixel counts.
 * @param median_pos The position of the median element (e.g., (kernel*kernel)/2).
 * @return The pixel value (0-255) corresponding to the median.
 */
inline uchar findMedianFromHistogram(const int* histogram, int median_pos) {
    int count = 0;
    for (int i = 0; i < 256; ++i) {
        count += histogram[i];
        if (count > median_pos) {
            return static_cast<uchar>(i);
        }
    }
    return 255; // Should not happen in practice
}

/**
 * @brief Performs median filtering on a horizontal slice of an image using a sliding histogram.
 *
 * This is the core optimized function. It processes a contiguous block of rows assigned to a single thread.
 * A histogram is built only once per row and then efficiently updated as the kernel window slides.
 *
 * @param src Pointer to the start of the source image data (single channel).
 * @param dst Pointer to the start of the destination image data (single channel).
 * @param width The full width of the source image.
 * @param height The full height of the source image.
 * @param start_row The first row this thread will process.
 * @param end_row The row after the last one this thread will process.
 * @param kernelSize The odd-sized dimension of the median filter kernel.
 */
void medianFilterSlice_Histogram(const uchar* src, uchar* dst, int width, int height,
                                 int start_row, int end_row, int kernelSize) {
    const int radius = kernelSize / 2;
    const int median_pos = (kernelSize * kernelSize) / 2;
    int histogram[256];

    // Process each row in the assigned slice
    for (int y = start_row; y < end_row; ++y) {
        // 1. Initialize histogram for the first kernel window in the row
        std::fill(histogram, histogram + 256, 0);
        for (int ky = -radius; ky <= radius; ++ky) {
            int py = std::max(0, std::min(height - 1, y + ky)); // Clamp to edge
            for (int kx = -radius; kx <= radius; ++kx) {
                int px = std::max(0, std::min(width - 1, 0 + kx)); // Clamp to edge
                histogram[src[py * width + px]]++;
            }
        }
        dst[y * width] = findMedianFromHistogram(histogram, median_pos);

        // 2. Slide the window horizontally across the row
        for (int x = 1; x < width; ++x) {
            // A. Decrement histogram for the column that slid out of the window
            int old_col_x = std::max(0, std::min(width - 1, x - radius - 1));
            for (int ky = -radius; ky <= radius; ++ky) {
                int py = std::max(0, std::min(height - 1, y + ky));
                histogram[src[py * width + old_col_x]]--;
            }

            // B. Increment histogram for the new column that slid into the window
            int new_col_x = std::max(0, std::min(width - 1, x + radius));
            for (int ky = -radius; ky <= radius; ++ky) {
                int py = std::max(0, std::min(height - 1, y + ky));
                histogram[src[py * width + new_col_x]]++;
            }
            
            // C. Find the new median and write to destination
            dst[y * width + x] = findMedianFromHistogram(histogram, median_pos);
        }
    }
}

/**
 * @brief An improved parallel median filter using a sliding histogram algorithm.
 * @param src The input single-channel 8-bit image.
 * @param dst The output single-channel 8-bit image.
 * @param kernelSize The odd-sized dimension of the median filter kernel.
 * @param num_threads The number of threads to use for parallelization.
 */
void TiledMedianFilterImproved(const cv::Mat& src, cv::Mat& dst, int kernelSize, int num_threads) {
    if (src.empty() || src.channels() != 1 || kernelSize % 2 == 0) {
        std::cerr << "ERR: Invalid input for TiledMedianFilterImproved." << std::endl;
        return;
    }

    dst.create(src.size(), src.type());
    const int width = src.cols;
    const int height = src.rows;
    const uchar* src_data = src.ptr<uchar>(0);
    uchar* dst_data = dst.ptr<uchar>(0);

    // Use OpenMP to parallelize the processing of horizontal slices
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < num_threads; ++i) {
        int start_row = i * height / num_threads;
        int end_row = (i + 1) * height / num_threads;
        medianFilterSlice_Histogram(src_data, dst_data, width, height, start_row, end_row, kernelSize);
    }
}


// --- Main Benchmarking Function ---

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./tiled_median_filter_improved <image_path>" << std::endl;
        return 1;
    }

    cv::Mat original_image_color = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (original_image_color.empty()) {
        std::cout << "ERR: Image data is empty at path: " << argv[1] << std::endl;
        return 2;
    }

    cv::Mat original_image;
    cv::cvtColor(original_image_color, original_image, cv::COLOR_BGR2GRAY);

    int kernelSize;
    std::cout << "Enter Kernel Size (must be odd, e.g., 3, 5, 7): ";
    std::cin >> kernelSize;

    if (kernelSize <= 1 || kernelSize % 2 == 0) {
        std::cout << "ERR: Invalid kernel size." << std::endl;
        return 3;
    }
    
    int max_threads = omp_get_max_threads();
    std::cout << "\n--- Benchmarking Median Filters ---" << std::endl;
    std::cout << "Image Resolution: " << original_image.cols << "x" << original_image.rows << std::endl;
    std::cout << "Kernel Size: " << kernelSize << "x" << kernelSize << std::endl;
    std::cout << "OpenMP Max Threads: " << max_threads << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    cv::Mat tiled_filtered_image;
    cv::Mat cv_filtered_image;
    
    // --- Benchmark TiledMedianFilterImproved ---
    auto start_tiled = std::chrono::high_resolution_clock::now();
    TiledMedianFilterImproved(original_image, tiled_filtered_image, kernelSize, max_threads);
    auto end_tiled = std::chrono::high_resolution_clock::now();
    auto duration_tiled = std::chrono::duration_cast<std::chrono::microseconds>(end_tiled - start_tiled);
    
    // --- Benchmark cv::medianBlur ---
    auto start_cv = std::chrono::high_resolution_clock::now();
    cv::medianBlur(original_image, cv_filtered_image, kernelSize);
    auto end_cv = std::chrono::high_resolution_clock::now();
    auto duration_cv = std::chrono::duration_cast<std::chrono::microseconds>(end_cv - start_cv);

    double time_tiled_ms = duration_tiled.count() / 1000.0;
    double time_cv_ms = duration_cv.count() / 1000.0;
    double speedup = time_cv_ms / time_tiled_ms;

    std::cout << "Improved TiledFilter time: " << time_tiled_ms << " ms" << std::endl;
    std::cout << "cv::medianBlur time:       " << time_cv_ms << " ms" << std::endl;
    std::cout << "Speedup vs OpenCV:         " << speedup << "x" << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    // --- Save and Display Results ---
    std::string tiled_output_path = "filtered_output_improved.jpg";
    if (cv::imwrite(tiled_output_path, tiled_filtered_image)) {
        std::cout << "Saved Improved TiledFilter result to " << tiled_output_path << std::endl;
    } else {
        std::cout << "Error saving Improved TiledFilter result." << std::endl;
    }
    
    cv::imshow("Original Grayscale", original_image);
    cv::imshow("Improved Tiled Median Filter", tiled_filtered_image);
    cv::imshow("OpenCV Median Filter", cv_filtered_image);
    
    cv::waitKey(0);
    return 0;
}
