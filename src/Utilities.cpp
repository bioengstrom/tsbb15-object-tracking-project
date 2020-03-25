#include "Utilities.hpp"

cv::Mat display(cv::Mat img) {
    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    img.convertTo(img, CV_8UC3, 255);
    return img;
}

//Intersection over union
double jaccardIndex(cv::Rect& first, cv::Rect& second) {
    double the_intersection = (first & second).area();
    double the_union = first.area() + second.area() - the_intersection;
    return the_intersection / the_union;
}

void printEvalutationToCSV(std::ostream& os, int framenumber, int objectID, int ul_x, int ul_y, int width, int height) {
    
    os << framenumber << "," << objectID << "," << ul_x << "," << ul_y << "," << width << "," << height << "\n";
}


cv::RNG unique_object::rng = cv::RNG(0);

cv::Scalar unique_object::getRandomColor() {
    return cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255));
}
