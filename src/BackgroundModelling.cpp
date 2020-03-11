#include "BackgroundModelling.hpp"

cv::Mat medianBackgroundModelling(cv::Mat frame, cv::Mat background, int ksize, double thresh, int erosion_size, int dilation_size) {
    
    cv::Mat diff;
    cv::Mat binary;
    cv::medianBlur(background,background, ksize);

    absdiff(frame, background, diff);
    //Make grayscale
    //source, destination, color type
    cv::cvtColor( diff, diff, cv::COLOR_BGR2GRAY);
    //Threshold
    //source, destination, threshold, max value out, threshold type
    cv::threshold(diff, binary, thresh, 1.0, cv::ThresholdTypes::THRESH_BINARY);
    
    cv::Mat er_element = cv::getStructuringElement( cv::MORPH_RECT,
                         cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                         cv::Point( erosion_size, erosion_size ) );
    
    cv::erode(binary, binary, er_element);
    
    cv::Mat dil_element = getStructuringElement( cv::MORPH_RECT,
                         cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                         cv::Point( dilation_size, dilation_size ) );
    
    cv::dilate(binary, binary, dil_element);
    
    return binary;
}
