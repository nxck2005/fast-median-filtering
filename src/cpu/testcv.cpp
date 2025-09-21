#include <iostream>
#include "opencv2/core.hpp"

// g++ cv_blur.cpp -o o_blur $(pkg-config --cflags --libs opencv4)
int main() {
    std::cout << "OpenCV has been successfully installed!" << std::endl;
    std::cout << "Version: " << CV_VERSION << std::endl;
    std::cout << "OpenCV is using " << cv::getNumThreads() << " threads." << std::endl;
    return 0;
}
