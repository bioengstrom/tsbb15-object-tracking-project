//Opencv header files
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>  // Video write
#include <vector>
#include <algorithm>
#include <iterator>
#include <opencv2/core/mat.hpp>
#include <stdlib.h>     /* srand, rand */

// Header files
#include "BackgroundModelling.hpp"


int main() {

    //Name of video
    std::string source{ "Walk1.mpg" };
    cv::VideoCapture inputVideo(source); // Open input
    if (!inputVideo.isOpened())
    {
        std::cout << "Could not open the input video: " << source << std::endl;
        return -1;
    }

    cv::Mat frame;
    inputVideo >> frame;

    cv::Mat background_model = cv::Mat(frame.rows, frame.cols, CV_8U, cv::Scalar(0));
    
    std::vector<cv::Mat> variableMatrices;
  
    double var = 30.0; // 30
    double w = 0.002; // 0.002
    double alpha = 0.002;
    double lambda = 4.0; // golden number : 4.5

    int K = 5;
    for(int k = 0; k < K; k++) {
        //variableMatrices.push_back(cv::Mat(frame.rows, frame.cols, CV_64FC3, cv::Scalar(rand() % 255 + 1, var, w)));
        variableMatrices.push_back(cv::Mat(frame.rows, frame.cols, CV_64FC3, cv::Scalar(rand() % 255 + 1, var, w)));
        
    }
    
    variableMatrices[0].forEach<cv::Vec3d>([&](cv::Vec3d& pixel, const int position[]) -> void {
        pixel[0] = frame.at<double>(position[0], position[1]);
    });
    
    while (1) {

        cv::imshow("Original video", frame);
    
        //(cv::Mat &frame, double w_init, double var_init, int K = 5, double alpha = 0.002, double T = 0.7 AMAZIN)
        mixtureBackgroundModelling(frame, variableMatrices, background_model, w, var, K, alpha, 0.7, lambda);
        cv::imshow("Mixture model", frame);
        
        //Break if press ESC
        char c = (char)cv::waitKey(25);
        if (c == 27)
            break;

        inputVideo >> frame;
        if (frame.empty()) {
            break;
        }
    }

    return 0;
}
