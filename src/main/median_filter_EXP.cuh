// An experimental version, vibe coded to fit a more optimized method for extrema exclusion
// using theoretical ideas presented in the supplemented paper.
// Only compiles upto 7, any higher is of polynomial complexity and manually written

#pragma once

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>

// Error checking macro for CUDA calls
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

// ==========================================================================
// == CORE ALGORITHM: DATA-OBLIVIOUS HIERARCHICAL TILING ====================
// ==========================================================================

// --- Sorting Network Primitives ---

template <typename T>
__device__ __forceinline__ void compare_swap(T& a, T& b) {
    T min_val = min(a, b);
    T max_val = max(a, b);
    a = min_val;
    b = max_val;
}

// Generic odd-even sorting network for a C-style array of size N
// Note: This is a full N-phase sort. For small, fixed N, 
// an optimal sorting network (e.g., Batcher's) would be faster.
template <typename T, int N>
__device__ __forceinline__ void sort_network(T arr[N]) {
    for (int i = 0; i < N; ++i) {
        if (i % 2 == 0) { // Even phase
            #pragma unroll
            for (int j = 0; j < (N + 1) / 2; ++j) {
                if (2 * j + 1 < N) {
                    compare_swap(arr[2 * j], arr[2 * j + 1]);
                }
            }
        } else { // Odd phase
            #pragma unroll
            for (int j = 0; j < N / 2; ++j) {
                if (2 * j + 2 < N) {
                    compare_swap(arr[2 * j + 1], arr[2 * j + 2]);
                }
            }
        }
    }
}

// --- Data Structures (as per Algorithm 1) ---

template <typename T, int KernelW, int KernelH, int TileW, int TileH>
struct TileData {
    // --- Constants derived from template parameters ---
    static constexpr int CoreW = KernelW - (TileW - 1);
    static constexpr int CoreH = KernelH - (TileH - 1);
    static constexpr int NumExtraCols = 2 * (TileW - 1); // Note: Original paper uses K-1
    static constexpr int NumExtraRows = 2 * (TileH - 1); // Note: Original paper uses K-1
    static constexpr int NumCorners = NumExtraCols * NumExtraRows;

    // Calculate how many elements are NOT in the core for this tile size
    static constexpr int Remaining = (TileW - 1) * CoreH + (TileH - 1) * CoreW + (TileW - 1) * (TileH - 1);
    
    // The size of the sorted core is the smaller of the full core size or
    // the number of remaining elements + 1 (Principle of Forgetfulness)
    static constexpr int SortedCoreSize = (CoreW * CoreH < Remaining + 1) ? (CoreW * CoreH) : (Remaining + 1);

    // --- Data arrays, sized at compile time ---
    T sortedCore[SortedCoreSize];             
    T extraCols[NumExtraCols][CoreH];
    T extraRows[NumExtraRows][CoreW];       
    T corners[NumCorners];                    
    
    // Final output for this tile
    T medians[TileH][TileW];                  

    // State for extrema exclusion
    int ns = 0; // Number of elements smaller than median
    int ng = 0; // Number of elements greater than median
};

// --- Merge and Crop (Extrema Exclusion) ---

template <typename T, int K, int P_SIZE, int Q_SIZE, int R_SIZE>
__device__ __forceinline__ void merge_and_crop(
    T* result,      // Must be a pointer or T result[R_SIZE]
    const T* p,   // Must be a pointer or const T p[P_SIZE]
    const T* q,   // Must be a pointer or const T q[Q_SIZE]
    int ns, int ng
) {
    T merged[P_SIZE + Q_SIZE]; // This is valid even if Q_SIZE=0 (becomes T merged[P_SIZE])
    int i = 0, j = 0, k = 0;

    // Standard merge of two sorted arrays p and q
    // These loops are correct even if Q_SIZE = 0
    while (i < P_SIZE && j < Q_SIZE) {
        if (p[i] <= q[j]) {
            merged[k++] = p[i++];
        } else {
            merged[k++] = q[j++];
        }
    }
    while (i < P_SIZE) merged[k++] = p[i++];
    while (j < Q_SIZE) merged[k++] = q[j++];

    // --- Apply Forgetfulness Principle (Formulas from Supplemental Material) ---
    constexpr int r = (K * K - 1) / 2; // Median rank
    const int p_new = P_SIZE + Q_SIZE;

    const int alpha = max(0, p_new + ng - r - 1);
    const int beta = min(p_new - 1, r - ns);

    // Copy the cropped region to the result array
    #pragma unroll
    for (int i = 0; i < R_SIZE; ++i) {
        if (alpha + i <= p_new - 1 && alpha + i <= beta) {
            result[i] = merged[alpha + i];
        }
    }
}

// --- Forward declarations for recursive calls ---
template <typename T, int K, int TW, int TH>
__device__ void recursion(TileData<T, K, K, TW, TH>& tile);

// --- Horizontal Split (as per Algorithm 2) ---
template <typename T, int K, int TW, int TH>
__device__ void horizontal_split(TileData<T, K, K, TW, TH>& parent) {
    constexpr int ChildW = TW / 2;
    constexpr int ChildH = TH;

    TileData<T, K, K, ChildW, ChildH> children[2];

    #pragma unroll
    for (int iChild = 0; iChild < 2; ++iChild) {
        // Define template args for merge_and_crop
        constexpr int NewColsSize = (ChildW - 1) * children[0].CoreH;
        constexpr int P_SIZE = parent.SortedCoreSize;
        constexpr int Q_SIZE = NewColsSize;
        constexpr int R_SIZE = children[iChild].SortedCoreSize;

        // FIX: Create a non-zero-sized array for the compiler
        // If NewColsSize is 0, create an array of size 1 as a placeholder
        constexpr int NewColsArraySize = (NewColsSize > 0) ? NewColsSize : 1;
        T newCols[NewColsArraySize]; 

        // FIX: Only populate and sort if the array is not empty
        if constexpr (NewColsSize > 0) {
            int count = 0;
            #pragma unroll
            for (int i = 0; i < (ChildW - 1); ++i) {
                #pragma unroll
                for (int j = 0; j < children[iChild].CoreH; ++j) {
                    // Placeholder: A real implementation would map parent.extraCols
                    newCols[count++] = parent.extraCols[iChild * (ChildW - 1) + i][j];
                }
            }
            sort_network<T, NewColsSize>(newCols);
        }

        // Update state for extrema exclusion
        children[iChild].ns = parent.ns;
        children[iChild].ng = parent.ng;

        merge_and_crop<T, K, P_SIZE, Q_SIZE, R_SIZE>(
            children[iChild].sortedCore, 
            parent.sortedCore, 
            newCols, // Pass the array; merge_and_crop handles Q_SIZE=0
            children[iChild].ns, 
            children[iChild].ng
        );
        
        // Update number of excluded elements
        children[iChild].ns += parent.CoreW * parent.CoreH - parent.SortedCoreSize;
        children[iChild].ng += parent.CoreW * parent.CoreH - parent.SortedCoreSize;
    }

    recursion(children[0]);
    recursion(children[1]);

    // Copy medians back to parent (this happens as recursion unwinds)
    #pragma unroll
    for (int y = 0; y < TH; ++y) {
        #pragma unroll
        for (int x = 0; x < ChildW; ++x) {
            parent.medians[y][x] = children[0].medians[y][x];
            parent.medians[y][x + ChildW] = children[1].medians[y][x];
        }
    }
}

// --- Vertical Split (Analogous to Horizontal) ---
template <typename T, int K, int TW, int TH>
__device__ void vertical_split(TileData<T, K, K, TW, TH>& parent) {
    constexpr int ChildW = TW;
    constexpr int ChildH = TH / 2;

    TileData<T, K, K, ChildW, ChildH> children[2];

    #pragma unroll
    for (int iChild = 0; iChild < 2; ++iChild) {
        constexpr int NewRowsSize = (ChildH - 1) * children[0].CoreW;
        constexpr int P_SIZE = parent.SortedCoreSize;
        constexpr int Q_SIZE = NewRowsSize;
        constexpr int R_SIZE = children[iChild].SortedCoreSize;
        
        // FIX: Create a non-zero-sized array for the compiler
        constexpr int NewRowsArraySize = (NewRowsSize > 0) ? NewRowsSize : 1;
        T newRows[NewRowsArraySize];

        // FIX: Only populate and sort if the array is not empty
        if constexpr (NewRowsSize > 0) {
            int count = 0;
            #pragma unroll
            for (int i = 0; i < (ChildH - 1); ++i) {
                #pragma unroll
                for (int j = 0; j < children[iChild].CoreW; ++j) {
                    // Placeholder: A real implementation would map parent.extraRows
                    newRows[count++] = parent.extraRows[iChild * (ChildH - 1) + i][j];
                }
            }
            sort_network<T, NewRowsSize>(newRows);
        }

        children[iChild].ns = parent.ns;
        children[iChild].ng = parent.ng;

        merge_and_crop<T, K, P_SIZE, Q_SIZE, R_SIZE>(
            children[iChild].sortedCore, 
            parent.sortedCore, 
            newRows, 
            children[iChild].ns, 
            children[iChild].ng
        );
        
        children[iChild].ns += parent.CoreW * parent.CoreH - parent.SortedCoreSize;
        children[iChild].ng += parent.CoreW * parent.CoreH - parent.SortedCoreSize;
    }

    recursion(children[0]);
    recursion(children[1]);

    // Copy medians back to parent
    #pragma unroll
    for (int y = 0; y < ChildH; ++y) {
        #pragma unroll
        for (int x = 0; x < TW; ++x) {
            parent.medians[y][x] = children[0].medians[y][x];
            parent.medians[y + ChildH][x] = children[1].medians[y][x];
        }
    }
}

// --- Main Recursive Function (as per Algorithm 3) ---
template <typename T, int K, int TW, int TH>
__device__ void recursion(TileData<T, K, K, TW, TH>& tile) {
    if constexpr (TW == 1 && TH == 1) {
        // Base case: 1x1 tile. The median is the first element 
        // of the (potentially size > 1) sortedCore.
        tile.medians[0][0] = tile.sortedCore[0];
        return;
    } else if constexpr (TW >= TH) {
        horizontal_split(tile);
    } else {
        vertical_split(tile);
    }
}

// --- The Main Kernel (implements Algorithm 4) ---
template <typename T, int K, int ROOT_TILE_W, int ROOT_TILE_H, int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void hierarchical_median_kernel(const T* src, T* dst, int width, int height) {
    const int tileX = blockIdx.x;
    const int tileY = blockIdx.y;
    const int tidX = threadIdx.x;
    const int tidY = threadIdx.y;

    // Each thread block processes one root tile
    const int root_tile_start_x = tileX * ROOT_TILE_W;
    const int root_tile_start_y = tileY * ROOT_TILE_H;

    // --- Collaborative loading into shared memory ---
    constexpr int FOOTPRINT_W = K + ROOT_TILE_W - 1;
    constexpr int FOOTPRINT_H = K + ROOT_TILE_H - 1;
    
    __shared__ T smem[FOOTPRINT_H][FOOTPRINT_W];

    constexpr int k_radius = (K - 1) / 2;
    const int load_start_x = root_tile_start_x - k_radius;
    const int load_start_y = root_tile_start_y - k_radius;

    for (int y = tidY; y < FOOTPRINT_H; y += BLOCK_DIM_Y) {
        for (int x = tidX; x < FOOTPRINT_W; x += BLOCK_DIM_X) {
            int img_x = load_start_x + x;
            int img_y = load_start_y + y;
            // Clamp to edge
            int clamped_x = max(0, min(width - 1, img_x));
            int clamped_y = max(0, min(height - 1, img_y));
            smem[y][x] = src[clamped_y * width + clamped_x];
        }
    }
    __syncthreads();

    // Only one thread per block continues from here to process its tile
    if (tidX == 0 && tidY == 0) {
        if (root_tile_start_x >= width || root_tile_start_y >= height) return;

        TileData<T, K, K, ROOT_TILE_W, ROOT_TILE_H> root_tile;

        // --- Initialization (from shared memory to registers) ---
        T core_cols[root_tile.CoreW][root_tile.CoreH];
        
        // 1. Sort core columns
        #pragma unroll
        for (int i = 0; i < root_tile.CoreW; ++i) {
            #pragma unroll
            for (int j = 0; j < root_tile.CoreH; ++j) {
                // Load from smem with correct offset
                // The root tile's core starts at (k_radius, k_radius) in smem
                core_cols[i][j] = smem[j + k_radius][i + k_radius];
            }
            sort_network<T, root_tile.CoreH>(core_cols[i]);
        }

        // 2. Multi-way merge sorted core columns and crop
        T core_flat[root_tile.CoreW * root_tile.CoreH];
        
        // Simplified merge for demo; a real one would be a k-way merge network
        int count = 0;
        #pragma unroll
        for(int i=0; i<root_tile.CoreW; ++i) {
            #pragma unroll
            for(int j=0; j<root_tile.CoreH; ++j) {
                core_flat[count++] = core_cols[i][j];
            }
        }
        sort_network<T, root_tile.CoreW * root_tile.CoreH>(core_flat);

        constexpr int r_init = (K * K - 1) / 2;
        const int alpha_init = max(0, (root_tile.CoreW * root_tile.CoreH) - r_init - 1);
        const int beta_init = min((root_tile.CoreW * root_tile.CoreH) - 1, r_init);

        #pragma unroll
        for (int i = 0; i < root_tile.SortedCoreSize; ++i) {
            if (alpha_init + i <= beta_init) {
                root_tile.sortedCore[i] = core_flat[alpha_init + i];
            }
        }
        root_tile.ns = alpha_init;
        root_tile.ng = (root_tile.CoreW * root_tile.CoreH) - 1 - beta_init;

        // 3. Initialize extra columns/rows
        // (This is also simplified; a full implementation would sort and copy
        // all extraCols, extraRows, and corners from shared memory here)
        #pragma unroll
        for(int i=0; i < root_tile.NumExtraCols; ++i) {
            #pragma unroll
            for(int j=0; j < root_tile.CoreH; ++j) {
                // Simplified load
                root_tile.extraCols[i][j] = smem[j + k_radius][i]; // Load from left side
            }
        }
        #pragma unroll
        for(int i=0; i < root_tile.NumExtraRows; ++i) {
            #pragma unroll
            for(int j=0; j < root_tile.CoreW; ++j) {
                root_tile.extraRows[i][j] = smem[i][j + k_radius]; // Load from top side
            }
        }


        // --- Start Recursion ---
        recursion(root_tile);

        // --- Write final medians to global memory ---
        #pragma unroll
        for (int y = 0; y < ROOT_TILE_H; ++y) {
            #pragma unroll
            for (int x = 0; x < ROOT_TILE_W; ++x) {
                int out_x = root_tile_start_x + x;
                int out_y = root_tile_start_y + y;
                if (out_x < width && out_y < height) {
                    dst[out_y * width + out_x] = root_tile.medians[y][x];
                }
            }
        }
    }
}

// Host-side function to launch the appropriate data-oblivious kernel
template <typename T>
void run_data_oblivious_kernel(const cv::Mat& src, cv::Mat& dst, int ksize) {
    const T* d_src;
    T* d_dst;
    CUDA_CHECK(cudaMalloc((void**)&d_src, src.cols * src.rows * sizeof(T)));
    CUDA_CHECK(cudaMalloc((void**)&d_dst, dst.cols * dst.rows * sizeof(T)));
    CUDA_CHECK(cudaMemcpy( (void*)d_src, src.ptr(), src.cols * src.rows * sizeof(T), cudaMemcpyHostToDevice));

    // For this algorithm, we launch one thread block per root tile.
    // Each thread block then collaboratively loads data, but only one thread
    // does the actual recursive computation for that tile.
    
    // Threads per block for collaborative load
    constexpr int BLOCK_X = 16; 
    constexpr int BLOCK_Y = 16;
    dim3 block(BLOCK_X, BLOCK_Y);

    if (ksize == 7) {
        constexpr int K = 7;
        constexpr int ROOT_W = 4; // Tile size must be power of 2
        constexpr int ROOT_H = 4;
        dim3 grid((src.cols + ROOT_W - 1) / ROOT_W, (src.rows + ROOT_H - 1) / ROOT_H);
        hierarchical_median_kernel<T, K, ROOT_W, ROOT_H, BLOCK_X, BLOCK_Y><<<grid, block>>>(d_src, d_dst, src.cols, src.rows);
    } 
    else if (ksize == 9) {
        constexpr int K = 9;
        constexpr int ROOT_W = 4;
        constexpr int ROOT_H = 4;
        dim3 grid((src.cols + ROOT_W - 1) / ROOT_W, (src.rows + ROOT_H - 1) / ROOT_H);
        hierarchical_median_kernel<T, K, ROOT_W, ROOT_H, BLOCK_X, BLOCK_Y><<<grid, block>>>(d_src, d_dst, src.cols, src.rows);
    }
    else if (ksize == 11) {
        constexpr int K = 11;
        constexpr int ROOT_W = 4;
        constexpr int ROOT_H = 4;
        dim3 grid((src.cols + ROOT_W - 1) / ROOT_W, (src.rows + ROOT_H - 1) / ROOT_H);
        hierarchical_median_kernel<T, K, ROOT_W, ROOT_H, BLOCK_X, BLOCK_Y><<<grid, block>>>(d_src, d_dst, src.cols, src.rows);
    }
    // ... (add more else if for 13, 15, 17, 19) ...
    else {
        throw std::runtime_error("Data-oblivious kernel for this ksize is not implemented.");
    }
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dst.ptr(), d_dst, dst.cols * dst.rows * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree((void*)d_src));
    CUDA_CHECK(cudaFree((void*)d_dst));
}

// Placeholder for the missing kernel
template <typename T>
__global__ void data_aware_placeholder_kernel(const T* src, T* dst, int width, int height, int ksize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    // Placeholder: just copy src to dst
    dst[y * width + x] = src[y * width + x];
}

// Host-side function to launch the data-aware pipeline (unchanged placeholder)
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

// ==========================================================================
// == PUBLIC API ============================================================
// ==========================================================================

template <typename T>
void apply(const cv::Mat& src, cv::Mat& dst, int ksize) {
    if (src.empty()) {
        throw std::invalid_argument("Source image is empty.");
    }
    if (ksize % 2 == 0) {
        throw std::invalid_argument("Kernel size must be odd.");
    }
    dst.create(src.size(), src.type());
    
    // Crossover point from the paper
    int crossover_ksize = 21;

    if (ksize < crossover_ksize) {
        std::cout << "Using Data-Oblivious Hierarchical Kernel for ksize=" << ksize << std::endl;
        detail::run_data_oblivious_kernel<T>(src, dst, ksize);
    } else {
        std::cout << "Using Data-Aware Kernel for ksize=" << ksize << std::endl;
        detail::run_data_aware_kernel<T>(src, dst, ksize);
    }
}

} // namespace HierarchicalMedianFilter