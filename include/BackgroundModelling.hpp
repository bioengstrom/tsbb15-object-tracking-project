#ifndef BackgroundModelling_hpp
#define BackgroundModelling_hpp

//OPEN CV
#include <opencv2/opencv.hpp>           //Base
#include <opencv2/core/core.hpp>        //Core

cv::Mat medianBackgroundModelling(cv::Mat frame, cv::Mat background, int ksize = 3, double thresh = 90, int erosion_size = 1, int dilation_size = 2);

#endif /* BackgroundModelling_hpp */
