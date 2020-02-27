#include <cassert>
#include <stdio.h>
#include <stdarg.h>
#include <vector>
#include <iostream>

//OPEN CV
#include <opencv2/opencv.hpp>           //Base
#include <opencv2/core/core.hpp>        //Core
#include <opencv2/highgui/highgui.hpp>  //Gui & video
#include <opencv2/imgproc.hpp>          //Image processing
#include <opencv2/tracking.hpp>    //Tracking
#include <opencv2/core/ocl.hpp>

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

cv::Mat display(cv::Mat img) {
    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    img.convertTo(img, CV_8UC3, 255);
    return img;
}

int main() {
	
    /**************************************************************************
                    OPEN VIDEO
    ***************************************************************************/
    //Name of video
    std::string source{"Walk1.mpg"};
    
    cv::VideoCapture inputVideo(source); // Open input
    if (!inputVideo.isOpened())
    {
        std::cout  << "Could not open the input video: " << source << std::endl;
        return -1;
    }
    cv::Mat frame;
    inputVideo >> frame;
    cv::Mat tracking_frame = frame.clone();
    
    /**********************************************************************
                    TRACKING
    
    ************************************************************************/
    //Create single tracker
    cv::Ptr<cv::Tracker> tracker = cv::TrackerBoosting::create();
    // Create multitracker
    cv::Ptr<cv::MultiTracker> multiTracker = cv::MultiTracker::create();
    std::vector<cv::Rect> bboxes;
    
    int enough_feature_points = 5;
    bool feature_points_found = false;
    bool tracking_started = false;
    float fps{};
    
    
    /**************************************************************************
            SLIDER FOR ADJUSTING THRESHOLD
    ***************************************************************************/
    int threshold = 90;
    int threshold_slider_max = 255;
    cv::namedWindow("Image", cv::WINDOW_NORMAL);
    cv::createTrackbar("Threshold", "Image", &threshold, threshold_slider_max);
  
    while(1) {
        
        /**************************************************************************
                        BACKGROUND FRAME
        ***************************************************************************/
        cv::Mat background = cv::imread ("Walk1000.jpg",cv::IMREAD_UNCHANGED);
        if(background.empty()) {
            return 1;
        }
        /**************************************************************************
                        LOAD FRAMES
        ***************************************************************************/
        
        inputVideo >> frame;
        if(frame.empty()) {
            break;
        }
        tracking_frame = frame.clone();
        /**************************************************************************
                   BACKGROUND MODELLING
        ***************************************************************************/
        //frame, background, ksize for median filter, threshold, erosion size, dilation size
        cv::Mat bg_mask = medianBackgroundModelling(frame, background, 3, threshold, 1, 1);
        
        /**************************************************************************
                    HARRIS FEATURE POINTS
         ***************************************************************************/
        //Find good feature points to track
        //Make grayscale
        //source, destination, color type
        cv::Mat grayFrame;
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
        cv::Mat harris;
        cornerHarris_demo(frame, grayFrame, harris);
        
        
        /**************************************************************************
                   FIND GOOD FEATURE POINTS
        ***************************************************************************/
        cv::Mat displayFeatures = cv::Mat::zeros( frame.size(), CV_8UC1 );
        
        //Keep looking for feature points if we dont have enough
        if(!feature_points_found) {
            bg_mask.convertTo(bg_mask, CV_8UC1);
            assert(bg_mask.size() == frame.size());
            assert(!bg_mask.empty());
            assert(bg_mask.type() == CV_8UC1);
            
            std::vector<cv::Point> corners[2];
            int maxCorners = 5;
            double qualityLevel = 0.01;
            double minDistance = 5.0;
            int blockSize=3;
            bool useHarrisDetector=true;
            //Free parameter of the harris detector
            double k=0.04;
            //(InputArray image, OutputArray corners, int maxCorners, double qualityLevel, double minDistance, InputArray mask=noArray(), int blockSize=3, bool useHarrisDetector=false, double k=0.04 )
            cv::goodFeaturesToTrack(grayFrame, corners[1], maxCorners, qualityLevel,  minDistance, bg_mask, blockSize, useHarrisDetector, k);
            
            for( int i = 0; i < corners[1].size() ; i++ )
            {
                circle( displayFeatures, corners[1][i], 5, cv::Scalar(255, 255, 255), 2, 8, 0);
                //Push rectangle surronding feature point to vector
                bboxes.push_back(cv::Rect(corners[1][i].x, corners[1][i].x, 20, 20));
                std::cout << corners[1][i].x << " " << corners[1][i].x << std::endl;
            }
            if(bboxes.size() >= enough_feature_points) {
                std::cout << "Found enough feature points: " << bboxes.size() << std::endl;
                feature_points_found = true;
            }
        }
        /**********************************************************************
                        TRACKING
        
        ************************************************************************/
        else if (feature_points_found && !tracking_started){ //Initialize tracking the feature points
            std::cout << "Initializing multitracker..." << std::endl;
            
            // Initialize multitracker
            for(int i=0; i < bboxes.size(); i++) {
                //Make sure the bounding box is comletely inside the frame
                if(0 <= bboxes[i].x && 0 <= bboxes[i].width && bboxes[i].x + bboxes[i].width <= tracking_frame.cols && 0 <=  bboxes[i].y && 0 <=  bboxes[i].height &&  bboxes[i].y +  bboxes[i].height <= tracking_frame.rows) {
                    std::cout << "Initialized tracking at point: " << bboxes[i] << std::endl;
                    multiTracker->add(cv::TrackerBoosting::create(), tracking_frame, cv::Rect2d(bboxes[i]));
                }
            }
            tracking_started = true;
            std::cout << "Tracking objects..." << std::endl;
        }
        else {
            
            //Update the tracking result with new frame
            double timer = (double)cv::getTickCount();
            multiTracker->update(frame);
            // Calculate Frames per second (FPS)
            fps = cv::getTickFrequency() / ((double)cv::getTickCount() - timer);
            // Draw tracked objects
            for(unsigned i=0; i<multiTracker->getObjects().size(); i++)
            {
                rectangle(tracking_frame, multiTracker->getObjects()[i], cv::Scalar(255, 0,0), 2, 1);
            }
            
        }
        
        // Display tracker type on frame
        cv::putText(tracking_frame, "Booster Tracker", cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,255,255),2);
        // Display FPS on frame
        cv::putText(tracking_frame, "FPS : " + std::to_string(int(fps)), cv::Point(10,60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,255,255), 2);
        
        /**************************************************************************
                   DISPLAY IMAGES
        ***************************************************************************/
        //Display the threshold as text on the mask
        //(Mat& img, const string& text, Point org, int fontFace, double fontScale, Scalar color, int thickness=1, int lineType=8, bool bottomLeftOrigin=false )
        cv::putText(frame, "Original Frame", cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255),2);
        cv::putText(bg_mask, "Background modelling", cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255),2);
        cv::putText(bg_mask, "Threshold: " + std::to_string(threshold), cv::Point(10,60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255),2);
        cv::putText(displayFeatures, "Harris feature points" , cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255),2);
        //cv::putText(harris, "Harris feature points" , cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0),2);
        ShowFourImages("Image", frame, display(bg_mask), display(displayFeatures), tracking_frame);
        //Break if press ESC
        char c = (char)cv::waitKey(25);
        if(c==27)
            break;
    }
    cv::destroyAllWindows();
	return 0;
}
