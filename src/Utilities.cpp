#include "Utilities.hpp"

cv::Mat display(cv::Mat img) {
    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    img.convertTo(img, CV_8UC3, 255);
    return img;
}

void printEvalutationToCSV(std::ostream& os, int framenumber, int objectID, int ul_x, int ul_y, int width, int height) {
    
    os << framenumber << "," << objectID << "," << ul_x << "," << ul_y << "," << width << "," << height << "\n";
}
