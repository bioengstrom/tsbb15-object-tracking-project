//
//  Old_Functions2.cpp
//  OpenCV_project
//
//  Created by Ylva Selling on 2020-03-09.
//

#include "Old_Functions.hpp"
//
//  Old_functions.cpp
//  OpenCV_project
//
//  Created by Ylva Selling on 2020-03-09.
//

#include <stdio.h>

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
/*
 Harris features
 Source code: https://docs.opencv.org/3.4/d4/d7d/tutorial_harris_detector.html
 */
void cornerHarris_demo(cv::Mat &src, cv::Mat &src_gray, cv::Mat &result)
{
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    int thresh = 200;
    int max_thresh = 255;
    
    cv::Mat dst = cv::Mat::zeros( src.size(), CV_32FC1 );
    cornerHarris( src_gray, dst, blockSize, apertureSize, k );
    
    cv::Mat dst_norm, dst_norm_scaled;
    cv::normalize( dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
    cv::convertScaleAbs( dst_norm, dst_norm_scaled );
    for( int i = 0; i < dst_norm.rows ; i++ )
    {
        for( int j = 0; j < dst_norm.cols; j++ )
        {
            if( (int) dst_norm.at<float>(i,j) > thresh )
            {
                circle( dst_norm_scaled, cv::Point(j,i), 5,  cv::Scalar(0), 2, 8, 0 );
            }
        }
    }
    result = dst_norm_scaled;
}
