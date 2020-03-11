#ifndef ExternalCode_hpp
#define ExternalCode_hpp

#include <string>
#include <opencv2/opencv.hpp>           //Base
#include <opencv2/core/core.hpp>        //Core

void ShowFourImages(std::string title, const cv::Mat& img0, const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &img3);

void cornerHarris_demo(cv::Mat &src, cv::Mat &src_gray, cv::Mat &result);

#endif /* ExternalCode_hpp */
