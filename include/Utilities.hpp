#ifndef Utilities_hpp
#define Utilities_hpp

//OPEN CV
#include <opencv2/opencv.hpp>           //Base
#include <opencv2/core/core.hpp>        //Core
#include <opencv2/highgui/highgui.hpp>  //Gui & video
#include <opencv2/imgproc.hpp>          //Image processing
#include <opencv2/tracking.hpp>    //Tracking
#include <opencv2/core/ocl.hpp>

class unique_object  {
public:
    static cv::RNG rng;
    unique_object(cv::Rect rect_, int frames_unvisible_ = 0,bool overlap_found_ = false)
    : rect(rect_), frames_unvisible(frames_unvisible_), overlap_found(overlap_found_) {
        color = getRandomColor();
    }
    cv::Rect rect{};
    int frames_unvisible{};
    bool overlap_found{};
    cv::Scalar color{};
    cv::Scalar getRandomColor();
    
};


double jaccardIndex(cv::Rect& first, cv::Rect& second);

cv::Mat display(cv::Mat img);

void printEvalutationToCSV(std::ostream& os, int framenumber, int objectID, int ul_x, int ul_y, int width, int height);

#endif /* Utilities_hpp */
