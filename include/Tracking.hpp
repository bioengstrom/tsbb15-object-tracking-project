#ifndef Tracking_hpp
#define Tracking_hpp
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

void matchUniqueObjToDetections(int invisibleFrameThreshold, std::vector<cv::Rect>& boundRect, std::vector<unique_object>& unique_objects);

#endif /* Tracking_hpp */
