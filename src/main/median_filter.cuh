#pragma once

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>

// change for more verbose logging (what method was chosen, da/do)
bool DEBUG = false;

// Error checking lol, ignore
#define CUDA_CHECK(call)                                                    \
do {                                                                        \
    cudaError_t err = call;                                                 \
    if (err!= cudaSuccess) {                                               \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n",                \
                __FILE__, __LINE__, cudaGetErrorString(err));               \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while (0)


namespace HierarchicalMedianFilter {
namespace detail {

// data oblivious

template <typename T>
__device__ __forceinline__ void compare_swap(T& a, T& b) {
    T min_val = min(a, b);
    T max_val = max(a, b);
    a = min_val;
    b = max_val;
}

// bitonic odd even sort (batchers paper)
template <typename T, int N>
__device__ void sort_network(T arr[N]) {
    for (int i = 0; i < N; ++i) {
        if (i % 2 == 0) { 
            for (int j = 0; j < N / 2; ++j) {
                if (2 * j + 1 < N) {
                    compare_swap(arr[2 * j], arr[2 * j + 1]);
                }
            }
        } else { 
            for (int j = 0; j < (N - 1) / 2; ++j) {
                compare_swap(arr[2 * j + 1], arr[2 * j + 2]);
            }
        }
    }
}

template <typename T, int KSIZE, int TILE_W, int TILE_H>
__global__ void data_oblivious_kernel(const T* src, T* dst, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    constexpr int WINDOW_SIZE = KSIZE * KSIZE;
    T window[WINDOW_SIZE];
    int radius = KSIZE / 2;
    int count = 0;

    for (int j = -radius; j <= radius; ++j) {
        for (int i = -radius; i <= radius; ++i) {
            int sx = x + i;
            int sy = y + j;
            int clamped_sx = max(0, min(width - 1, sx));
            int clamped_sy = max(0, min(height - 1, sy));
            window[count++] = src[clamped_sy * width + clamped_sx];
        }
    }

    sort_network<T, WINDOW_SIZE>(window);

    dst[y * width + x] = window[WINDOW_SIZE / 2];
}


// data aware

template <typename T>
__global__ void data_aware_placeholder_kernel(const T* src, T* dst, int width, int height, int ksize) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int radius = ksize / 2;
    int window_size = ksize * ksize;
    
    // PLACEHOLDER malloc! the real template based approach would take long
    T* window = (T*)malloc(window_size * sizeof(T));
    if (window == nullptr) return;

    int count = 0;
    for (int j = -radius; j <= radius; ++j) {
        for (int i = -radius; i <= radius; ++i) {
            int sx = x + i;
            int sy = y + j;
            int clamped_sx = max(0, min(width - 1, sx));
            int clamped_sy = max(0, min(height - 1, sy));
            window[count++] = src[clamped_sy * width + clamped_sx];
        }
    }

    // placeholder bubble sort
    for (int i = 0; i < window_size - 1; ++i) {
        for (int j = 0; j < window_size - i - 1; ++j) {
            if (window[j] > window[j+1]) {
                T temp = window[j];
                window[j] = window[j+1];
                window[j+1] = temp;
            }
        }
    }

    dst[y * width + x] = window[window_size / 2];
    free(window);
}

// host-side function to launch the appropriate data-oblivious kernel
template <typename T>
void run_data_oblivious_kernel(const cv::Mat& src, cv::Mat& dst, int ksize) {
    if (ksize > 19) { 
        throw std::runtime_error("Data-oblivious kernel demo only supports ksize <= 19");
    }

    const T* d_src;
    T* d_dst;
    // using cudaMalloc for simplicity in placeholder code
    CUDA_CHECK(cudaMalloc((void**)&d_src, src.cols * src.rows * sizeof(T)));
    CUDA_CHECK(cudaMalloc((void**)&d_dst, dst.cols * dst.rows * sizeof(T)));
    CUDA_CHECK(cudaMemcpy( (void*)d_src, src.ptr(), src.cols * src.rows * sizeof(T), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((src.cols + block.x - 1) / block.x, (src.rows + block.y - 1) / block.y);

    // dispatch to a specific template instantiation based on ksize
    if (ksize == 7) data_oblivious_kernel<T, 7, 8, 8><<<grid, block>>>(d_src, d_dst, src.cols, src.rows);
    else if (ksize == 9) data_oblivious_kernel<T, 9, 8, 8><<<grid, block>>>(d_src, d_dst, src.cols, src.rows);
    else if (ksize == 11) data_oblivious_kernel<T, 11, 8, 8><<<grid, block>>>(d_src, d_dst, src.cols, src.rows);
    else if (ksize == 13) data_oblivious_kernel<T, 13, 8, 8><<<grid, block>>>(d_src, d_dst, src.cols, src.rows);
    else if (ksize == 15) data_oblivious_kernel<T, 15, 8, 8><<<grid, block>>>(d_src, d_dst, src.cols, src.rows);
    else if (ksize == 17) data_oblivious_kernel<T, 17, 8, 8><<<grid, block>>>(d_src, d_dst, src.cols, src.rows);
    else if (ksize == 19) data_oblivious_kernel<T, 19, 8, 8><<<grid, block>>>(d_src, d_dst, src.cols, src.rows);
    else { // fallback for other small sizes like 3 or 5
        data_oblivious_kernel<T, 5, 8, 8><<<grid, block>>>(d_src, d_dst, src.cols, src.rows);
    }
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dst.ptr(), d_dst, dst.cols * dst.rows * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree((void*)d_src));
    CUDA_CHECK(cudaFree((void*)d_dst));
}

// host-side function to launch the data-aware pipeline
template <typename T>
void run_data_aware_kernel(const cv::Mat& src, cv::Mat& dst, int ksize) {
    const T* d_src;
    T* d_dst;
    CUDA_CHECK(cudaMalloc((void**)&d_src, src.cols * src.rows * sizeof(T)));
    CUDA_CHECK(cudaMalloc((void**)&d_dst, dst.cols * dst.rows * sizeof(T)));
    CUDA_CHECK(cudaMemcpy( (void*)d_src, src.ptr(), src.cols * src.rows * sizeof(T), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((src.cols + block.x - 1) / block.x, (src.rows + block.y - 1) / block.y);
    
    data_aware_placeholder_kernel<T><<<grid, block>>>(d_src, d_dst, src.cols, src.rows, ksize);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dst.ptr(), d_dst, dst.cols * dst.rows * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree((void*)d_src));
    CUDA_CHECK(cudaFree((void*)d_dst));
}

} // namespace detail

// actual public api

template <typename T>
void apply(const cv::Mat& src, cv::Mat& dst, int ksize) {
    if (src.empty()) {
        throw std::invalid_argument("Source image is empty.");
    }
    if (ksize % 2 == 0) {
        throw std::invalid_argument("Kernel size must be odd.");
    }
    dst.create(src.size(), src.type());
    
    int crossover_ksize = 21;

    if (ksize < crossover_ksize) {
        if (DEBUG) std::cout << "Using Data-Oblivious Kernel for ksize=" << ksize << std::endl;
        detail::run_data_oblivious_kernel<T>(src, dst, ksize);
    } else {
        if (DEBUG) std::cout << "Using Data-Aware Kernel for ksize=" << ksize << std::endl;
        detail::run_data_aware_kernel<T>(src, dst, ksize);
    }
}

} // namespace HierarchicalMedianFilter