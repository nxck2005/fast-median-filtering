#include <iostream>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


// g++ cv_blur.cpp -o o_blur $(pkg-config --cflags --libs opencv4)
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
	int kernelsize = 19;
	
	auto start = std::chrono::high_resolution_clock::now();

	cv::medianBlur(original_image, filtered_image, kernelsize);

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	std::cout << duration.count() << "ms was taken" << std::endl;

	cv::imshow("Original", original_image);
	cv::imshow("Median Filtered", filtered_image);

	bool saved = cv::imwrite("../results/filtered_output_cv.jpg", filtered_image);
	if (saved) {
		std::cout << "Saved!" << std::endl;
	} else {
		std::cout << "Error while saving." << std::endl;
	}
	cv::waitKey(0);
	return 0;
}
