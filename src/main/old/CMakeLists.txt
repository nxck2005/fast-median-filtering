cmake_minimum_required(VERSION 3.18)

project(HierarchicalMedianFilter VERSION 1.0 LANGUAGES CXX CUDA)


find_package(CUDAToolkit REQUIRED)


find_package(OpenCV REQUIRED)


find_package(Python 3 COMPONENTS Interpreter Development NumPy REQUIRED)


add_executable(median_benchmark main.cpp)

set_target_properties(median_benchmark PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)


# SM 8.9 corresponds to the NVIDIA Ada Lovelace architecture (e.g., RTX 40 series).

set_target_properties(median_benchmark PROPERTIES
    CUDA_ARCHITECTURES "89"
)


target_link_libraries(median_benchmark PRIVATE
    ${OpenCV_LIBS}
    Python::Python
)

target_include_directories(median_benchmark PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    ${Python_NumPy_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

message(STATUS "Project configured. Build with 'make' or your chosen generator.")
message(STATUS "Run executable from the build directory.")