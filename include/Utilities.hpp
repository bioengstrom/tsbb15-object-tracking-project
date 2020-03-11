#ifndef Utilities_hpp
#define Utilities_hpp

#include <opencv2/opencv.hpp>           //Base
#include <opencv2/core/core.hpp>        //Core

cv::Mat display(cv::Mat img);

void printEvalutationToCSV(std::ostream& os, int framenumber, int objectID, int ul_x, int ul_y, int width, int height);

#endif /* Utilities_hpp */
