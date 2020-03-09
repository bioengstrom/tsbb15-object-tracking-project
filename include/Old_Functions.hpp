//
//  Old_Functions2.hpp
//  OpenCV_project
//
//  Created by Ylva Selling on 2020-03-09.
//

#ifndef Old_Functions_hpp
#define Old_Functions_hpp

#include <stdio.h>
//OPEN CV
#include <opencv2/opencv.hpp>           //Base
#include <opencv2/core/core.hpp>        //Core
#include <opencv2/highgui/highgui.hpp>  //Gui & video
#include <opencv2/imgproc.hpp>          //Image processing
#include <opencv2/tracking.hpp>    //Tracking
#include <opencv2/core/ocl.hpp>

cv::Mat medianBackgroundModelling(cv::Mat frame, cv::Mat background, int ksize = 3, double thresh = 90, int erosion_size = 1, int dilation_size = 2);

void cornerHarris_demo(cv::Mat &src, cv::Mat &src_gray, cv::Mat &result);


#endif /* Old_Functions_hpp */
