//This line includes all the opencv header files
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>  // Video write

cv::Mat medianBackgroundModelling(cv::Mat frame, cv::Mat background, int ksize = 3, double thresh = 90, int erosion_size = 1, int dilation_size = 2) {
    
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
    
    //std::cout << binary << std::endl;
    binary.convertTo(binary, CV_32FC1);
    
    return binary;
}
/*
 Harris features
 Source code: https://docs.opencv.org/3.4/d4/d7d/tutorial_harris_detector.html
 */
void cornerHarris_demo(cv::Mat src, cv::Mat src_gray)
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
    cv::namedWindow( "Harris features" );
    cv::imshow("Harris features", dst_norm_scaled );
}

int main() {
	
    //Name of video
    std::string source{"Walk1.mpg"};
    
    cv::VideoCapture inputVideo(source); // Open input
    if (!inputVideo.isOpened())
    {
        std::cout  << "Could not open the input video: " << source << std::endl;
        return -1;
    }
    
    int threshold = 90;
    cv::namedWindow("Threshold", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar( "Threshold", "Threshold", &threshold, 255);
    
    while(1) {
        
        cv::Mat background = cv::imread ("Walk1000.jpg",cv::IMREAD_UNCHANGED);
        if(background.empty()) {
            return 1;
        }
        cv::Mat frame;
        cv::Mat bg_mask;
        
        inputVideo >> frame;
        if(frame.empty()) {
            break;
        }
        //frame, background, ksize for median filter, threshold, erosion size, dilation size
        bg_mask = medianBackgroundModelling(frame, background, 3, threshold, 1, 1);
        cv::imshow("Original video", frame);
        cv::imshow("Background model", bg_mask);
        //Make grayscale
        //source, destination, color type
        cv::Mat grayFrame;
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
        cornerHarris_demo(frame, grayFrame);
        
        /*
         TODO: use the cv::goodfunctionstotrack() function
         */

        //Break if press ESC
        char c = (char)cv::waitKey(25);
        if(c==27)
            break;
    }

	return 0;
}
