#include "ExternalCode.hpp"

/*
 Display four images at once.
 The code has been modified to only apply for 4 images.
 Source code: https://github.com/opencv/opencv/wiki/DisplayManyImages
 */

void ShowFourImages(std::string title, const cv::Mat& img0, const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &img3) {
int size;
int i;
int m, n;
int x, y;

// w - Maximum number of images in a row
// h - Maximum number of images in a column
int w = 2, h = 2;

// scale - How much we have to resize the image
float scale;
int max;


size = 300;

// Create a new 3 channel image
cv::Mat DispImage = cv::Mat::zeros(cv::Size(100 + size*w, 60 + size*h), CV_8UC3);

    // Loop for 4 number of arguments
    for (i = 0, m = 20, n = 20; i < 4; i++, m += (20 + size)) {
        // Get the Pointer to the IplImage
        cv::Mat img = img0;
        if(i == 1)
            img = img1;
        else if(i == 2)
            img = img2;
        else if(i == 3)
            img = img3;

        // Check whether it is NULL or not
        // If it is NULL, release the image, and return
        if(img.empty()) {
            printf("Invalid arguments");
            return;
        }

        // Find the width and height of the image
        x = img.cols;
        y = img.rows;

        // Find whether height or width is greater in order to resize the image
        max = (x > y)? x: y;

        // Find the scaling factor to resize the image
        scale = (float) ( (float) max / size );

        // Used to Align the images
        if( i % w == 0 && m!= 20) {
            m = 20;
            n+= 20 + size;
        }

        // Set the image ROI to display the current image
        // Resize the input image and copy the it to the Single Big Image
        cv::Rect ROI(m, n, (int)( x/scale ), (int)( y/scale ));
        cv::Mat temp;
        resize(img,temp, cv::Size(ROI.width, ROI.height));
        temp.copyTo(DispImage(ROI));
    }

    // Create a new window, and show the Single Big Image
    cv::imshow(title, DispImage);
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
