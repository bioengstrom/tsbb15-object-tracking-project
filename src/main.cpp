//This line includes all the opencv header files
#include <opencv2/opencv.hpp>

int main() {
	//Make a blank image
	cv::Mat picture = cv::Mat::zeros(100, 100, CV_32FC1);
	//define a rectangle
	cv::Rect box(30, 30, 25, 25);

	//draw a white rectangle in the image
	cv::rectangle(picture, box, cv::Scalar(1.0, 1.0, 1.0));

	//create the window that can be resized
	cv::namedWindow("window");

	//show the image in the window
	cv::imshow("window", picture);

	//wait until keypress
	cv::waitKey(0);

	return 0;
}
