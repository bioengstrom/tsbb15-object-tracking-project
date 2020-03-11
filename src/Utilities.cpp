#include "Utilities.hpp"

cv::Mat display(cv::Mat img) {
    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    img.convertTo(img, CV_8UC3, 255);
    return img;
}
