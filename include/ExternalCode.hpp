#ifndef ExternalCode_hpp
#define ExternalCode_hpp

#include <string>
//OPEN CV
#include <opencv2/opencv.hpp>           //Base
#include <opencv2/core/core.hpp>        //Core
#include <opencv2/highgui/highgui.hpp>  //Gui & video
#include <opencv2/imgproc.hpp>          //Image processing
#include <opencv2/tracking.hpp>    //Tracking
#include <opencv2/core/ocl.hpp>
#include <vector>
#include <iterator>
#include "Utilities.hpp"

void ShowFourImages(std::string title, const cv::Mat& img0, const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &img3);

void cornerHarris_demo(cv::Mat &src, cv::Mat &src_gray, cv::Mat &result);

void findBoundingBoxes(cv::Mat& bg_mask, std::vector<cv::Rect>& boundRect, int minRectArea) ;

#endif /* ExternalCode_hpp */
