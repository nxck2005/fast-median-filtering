#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "median_filter.cuh"
#include "matplotlibcpp.h"
#include <algorithm>

namespace plt = matplotlibcpp;

bool get_cmd_option(char** begin, char** end, const std::string& option, std::string& value) {
    char** itr = std::find(begin, end, option);
    if (itr!= end && ++itr!= end) {
        value = *itr;
        return true;
    }
    return false;
}

template <typename T>
void run_benchmark(const std::string& filename, int max_ksize) {
    cv::Mat src_img = cv::imread(filename, cv::IMREAD_UNCHANGED);
    if (src_img.empty()) {
        std::cerr << "Error: Could not open or find the image: " << filename << std::endl;
        return;
    }
    cv::Mat src;
    if (src_img.channels() > 1) {
        cv::cvtColor(src_img, src, cv::COLOR_BGR2GRAY);
    } else {
        src = src_img;
    }
    src.convertTo(src, cv::DataType<T>::type);

    std::cout << "Image loaded: " << src.cols << "x" << src.rows
              << ", Channels: " << src.channels()
              << ", Type: " << typeid(T).name() << std::endl;

    std::vector<int> kernel_sizes;
    std::vector<double> opencv_times;
    std::vector<double> ht_times;

    std::cout << std::left << std::setw(15) << "Kernel Size"
              << std::setw(20) << "OpenCV Time (ms)"
              << std::setw(30) << "Hierarchical Tiling Time (ms)"
              << std::setw(15) << "Speedup" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (int k = 7; k <= max_ksize; k += 2) {
        if (k < 3) continue;
        cv::Mat dst_ocv, dst_ht;
        kernel_sizes.push_back(k);

        auto start_ocv = std::chrono::high_resolution_clock::now();
        cv::medianBlur(src, dst_ocv, k);
        auto stop_ocv = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> time_ocv = stop_ocv - start_ocv;
        opencv_times.push_back(time_ocv.count());
        

        cudaEvent_t start_ht, stop_ht;
        cudaEventCreate(&start_ht);
        cudaEventCreate(&stop_ht);

        HierarchicalMedianFilter::apply<T>(src, dst_ht, k);

        cudaEventRecord(start_ht);
        HierarchicalMedianFilter::apply<T>(src, dst_ht, k);
        cudaEventRecord(stop_ht);
        cudaEventSynchronize(stop_ht);
        float time_ht_ms;
        cudaEventElapsedTime(&time_ht_ms, start_ht, stop_ht);
        ht_times.push_back(time_ht_ms);
        cudaEventDestroy(start_ht);
        cudaEventDestroy(stop_ht);

        std::cout << std::left << std::setw(15) << (std::to_string(k) + "x" + std::to_string(k))
                  << std::setw(20) << std::fixed << std::setprecision(2) << time_ocv.count()
                  << std::setw(30) << time_ht_ms
                  << std::setw(15) << (time_ocv.count() / time_ht_ms) << "x" << std::endl;
    }

    plt::figure_size(1200, 780);
    plt::named_plot("OpenCV cv::medianBlur", kernel_sizes, opencv_times, "r-o");
    plt::named_plot("Hierarchical Tiling (CUDA)", kernel_sizes, ht_times, "b-s");
    plt::title("Median Filter Performance Comparison");
    plt::xlabel("Kernel Diameter");
    plt::ylabel("Execution Time (ms)");
    plt::legend();
    plt::grid(true);
    plt::save("performance_graph.png");
    std::cout << "\nPerformance graph saved to performance_graph.png" << std::endl;
}

int main(int argc, char** argv) {
    std::string filename, type_str;
    std::string max_ksize_str;
    int max_ksize;

    if (!get_cmd_option(argv, argv + argc, "--filename", filename) ||
     !get_cmd_option(argv, argv + argc, "--type", type_str) ||
     !get_cmd_option(argv, argv + argc, "--max_ksize", max_ksize_str)) {
        std::cerr << "Usage: " << argv[0] << " --filename <path> --type <uint8|uint16|float> --max_ksize <size>" << std::endl; // <--- FIX
        return -1;
    }

    try {
        max_ksize = std::stoi(max_ksize_str);
    } catch (const std::exception& e) {
        std::cerr << "Error: Invalid max_ksize value." << std::endl;
        return -1;
    }

    if (type_str == "uint8") {
        run_benchmark<unsigned char>(filename, max_ksize);
    } else if (type_str == "uint16") {
        run_benchmark<unsigned short>(filename, max_ksize);
    } else if (type_str == "float") {
        run_benchmark<float>(filename, max_ksize);
    } else {
        std::cerr << "Error: Unsupported pixel type. Please use 'uint8', 'uint16', or 'float'." << std::endl;
        return -1;
    }

    return 0;
}