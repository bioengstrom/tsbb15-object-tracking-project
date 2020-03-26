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
    static int counter;
    unique_object(cv::Rect rect_, int frames_invisible_ = 0,bool overlap_found_ = false)
    : rect(rect_), frames_invisible(frames_invisible_), overlap_found(overlap_found_) {
        color = getRandomColor();
        ID = counter;
        counter++;
    }
    cv::Rect rect{};
    int ID{};
    int frames_invisible{};
    bool overlap_found{};
    cv::Scalar color{};
    cv::Scalar getRandomColor();
    
};

double jaccardIndex(cv::Rect& first, cv::Rect& second);

cv::Mat display(cv::Mat img);

void printObjToCSV(std::ostream& os, int objectID, int ul_x, int ul_y, int width, int height);
void printFrameToCSV(std::ostream& os, int frameNumber, bool printInvisible, std::vector<unique_object>& unique_objects);

void drawRectangles(cv::Mat& img, cv::Scalar color, std::vector<cv::Rect>& rects);


#endif /* Utilities_hpp */
