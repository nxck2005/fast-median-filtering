#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main() {
    // Create a blank black image
    cv::Mat image = cv::Mat::zeros(400, 600, CV_8UC3);

    // Put some text on it
    cv::putText(image,
                "Hello, OpenCV on Fedora!",
                cv::Point(50, 200),
                cv::FONT_HERSHEY_DUPLEX,
                1.0,
                cv::Scalar(118, 185, 0), // A nice green color
                2);

    // Display the image
    cv::imshow("Display window", image);

    // Wait for a key press indefinitely
    cv::waitKey(0); 

    std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    return 0;
}
