#include "Utilities.hpp"

cv::Mat display(cv::Mat img) {
    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    img.convertTo(img, CV_8UC3, 255);
    return img;
}

//Intersection over union
double jaccardIndex(const cv::Rect& first, const cv::Rect& second) {
    double the_intersection = (first & second).area();
    double the_union = first.area() + second.area() - the_intersection;
    return the_intersection / the_union;
}

void printObjToCSV(std::ostream& os, int objectID, int ul_x, int ul_y, int width, int height) {
    
    os << "," << objectID << "," << ul_x << "," << ul_y << "," << width << "," << height;
}

void printFrameToCSV(std::ostream& os, int frameNumber, bool printInvisible, std::vector<unique_object>& unique_objects) {

    os << frameNumber;
    
    //Write each unique object to csv file
    for( int i = 0; i< unique_objects.size(); i++ )
    {
        if(!printInvisible && unique_objects[i].frames_invisible > 0) {
            continue;
        }
         printObjToCSV(os, unique_objects[i].ID, unique_objects[i].rect.x , unique_objects[i].rect.y, unique_objects[i].rect.width, unique_objects[i].rect.height);
        
    }
    
    os << std::endl;
}

//Initialize static members of unique object
cv::RNG unique_object::rng = cv::RNG(0);
int unique_object::counter = 0;

cv::Scalar unique_object::getRandomColor() {
    return cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255));
}

cv::Rect unique_object::predRect() {
    cv::Point predTopLeft = rect.tl() + static_cast<cv::Point>(delta);
    return cv::Rect(predTopLeft, predSize);
}

void drawRectangles(cv::Mat& img, cv::Scalar color, std::vector<cv::Rect>& rects) {
    
    for (cv::Rect rect : rects) {
        rectangle( img, rect.tl(), rect.br(), color, 2, 8, 0 );
    }
}

void drawUniqueObjects(cv::Mat& img, std::vector<unique_object>& unique_objects) {
    
    // Draw unique object bonding rects
    for (unique_object obj : unique_objects)
    {
        if(obj.frames_invisible == 0) {
            rectangle( img, obj.rect.tl(), obj.rect.br(), obj.color, 2, 8, 0 );
        }
    }
}
