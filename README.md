# A Parallel Implementation of Hierarchical Tiling for Median Filtering

## Overview

This repository provides an academic C++ and CUDA implementation of the median filtering algorithm. The core of this project is based on the paper ["A Fast Parallel Median Filtering Algorithm Using Hierarchical Tiling"](https://dl.acm.org/doi/pdf/10.1145/3721238.3730709), which proposes a method to efficiently compute median filters with large kernel sizes.

The primary implementation in `src/main` is designed to run on NVIDIA GPUs.
It also benchmarks the implementation against OpenCV's implementation as you run it.

While the paper uses a NVIDIA L40S, The testing was done on a NVIDIA GeForce RTX 4060 Max-Q Mobile.

## Build from Source

The main benchmark executable is located in `src/main` and can be built using CMake.

### Dependencies

  * **CMake** (\>= 3.18)
  * **NVIDIA CUDA Toolkit** (The project is configured for compute capability `8.9`, e.g., Ada Lovelace architecture)
  * **OpenCV** (\>= 4.x)
  * **Python 3** (with development libraries)
  * **gcc14** (gcc14 is required for the needed nvcc compiler present in the toolkit version used)

### Compilation

1.  Navigate to the main project directory:
    ```sh
    cd src/main
    ```
2.  Create and enter a build directory:
    ```sh
    mkdir build && cd build
    ```
3.  Run CMake to configure the project:
    ```sh
    cmake ..
    ```
4.  Compile the project:
    ```sh
    make
    ```
    This will create an executable named `median_benchmark` in the `build` directory.

### Execution

The `median_benchmark` executable runs a performance comparison for various kernel sizes, pitting the hierarchical tiling implementation against OpenCV's.

**Usage:**

```
./median_benchmark --filename <path_to_image> --type <pixel_type> --max_ksize <size>
```

**Arguments:**

  * `--filename`: Path to the input image (e.g., `../../samples/sample1.JPG`).
  * `--type`: The pixel type to use for the benchmark. Supported types are `uint8`, `uint16`, and `float`.
  * `--max_ksize`: The maximum odd kernel size to test up to (e.g., `79`).

**Example:**

```sh
./median_benchmark --filename ../../samples/sample1.JPG --type uint8 --max_ksize 79
```


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.