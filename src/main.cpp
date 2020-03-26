#include "BackgroundModelling.hpp"
#include "ExternalCode.hpp"
#include "Tracking.hpp"
#include "Utilities.hpp"

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

int main() {
	
    /**************************************************************************
                    OPEN VIDEO
    ***************************************************************************/
    //Name of video
    std::string source{"Walk1"};
    std::string mov_format{"mpg"};
    
    cv::VideoCapture inputVideo(source + "." + mov_format); // Open input
    if (!inputVideo.isOpened())
    {
        std::cout  << "Could not open the input video: " << source << std::endl;
        return -1;
    }
    cv::Mat frame;
    inputVideo >> frame;
    cv::Mat tracking_frame = frame.clone();
    
    /**************************************************************************
                    EVALUATION
    ***************************************************************************/
    
    std::ofstream myfile;
    myfile.open(source + ".csv");
    if (!myfile.is_open())
    {
      std::cout << "Could not write to file!" << std::endl;
      myfile.close();
    }
    int frameNumber{0};
    
    /**********************************************************************
                    TRACKING
    
    ************************************************************************/
    //Create single tracker
  /*  cv::Ptr<cv::Tracker> tracker = cv::TrackerBoosting::create();
    // Create multitracker
    cv::Ptr<cv::MultiTracker> multiTracker = cv::MultiTracker::create();
    std::vector<cv::Rect> bboxes;
    std::vector<cv::Scalar> colors;
    
    int enough_feature_points = 5;
    bool feature_points_found = false;
    bool tracking_started = false;
    float fps{}; */
    //Random generator for generating random colors
    cv::RNG rng(0);

    /**********************************************************************
              MAINTAIN OBJECT IDENTITY
    
    ************************************************************************/
    
    std::vector<unique_object> unique_objects;
    
    /**************************************************************************
            SLIDER FOR ADJUSTING THRESHOLD
    ***************************************************************************/
    int threshold{85};
    int threshold_slider_max{255};
    int erosion_size{1};
    int dilation_size{9};
    cv::namedWindow("Image", cv::WINDOW_NORMAL);
    cv::createTrackbar("Threshold", "Image", &threshold, threshold_slider_max);
    cv::createTrackbar("Erosion", "Image", &erosion_size, 10);
    cv::createTrackbar("Dilation", "Image", &dilation_size, 10);
    
  
    while(1) {
        
        /**************************************************************************
                        BACKGROUND FRAME
        ***************************************************************************/
        std::string background_img{"Walk1000.jpg"};
        cv::Mat background = cv::imread (background_img ,cv::IMREAD_UNCHANGED);
        if(background.empty()) {
            std::cout << "Error! Could not find background image " << background_img << std::endl;
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
        cv::Mat bg_mask = medianBackgroundModelling(frame, background, 3, threshold, erosion_size, dilation_size);
        // Draw bonding rects
        cv::Mat drawing = cv::Mat::zeros( bg_mask.size(), CV_8UC3 );
        std::vector<cv::Rect> boundRect;
        
        findBoundingBoxes(bg_mask, boundRect);
        drawRectangles(tracking_frame, cv::Scalar( 0, 0, 255), boundRect);
        
         
        /**************************************************************************
                   DETECT OVERLAP
        ***************************************************************************/
        matchUniqueObjToDetections(boundRect, unique_objects);
        printFrameToCSV(myfile, frameNumber, false, unique_objects);
        drawUniqueObjects(drawing, unique_objects);
        
        /**************************************************************************
                    OCCLUSION MANAGEMENT
        ***************************************************************************/
        
        
        
        
        
        /**************************************************************************
                    HARRIS FEATURE POINTS
         ***************************************************************************/
        //Find good feature points to track
        //Make grayscale
        //source, destination, color type
        /*
        cv::Mat grayFrame;
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY); */
        //cv::Mat harris;
        //cornerHarris_demo(frame, grayFrame, harris);
        
        
        /**************************************************************************
                   FIND GOOD FEATURE POINTS
        ***************************************************************************/
  /*      cv::Mat displayFeatures = cv::Mat::zeros( frame.size(), CV_8UC1 );
        
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
            }
            if(bboxes.size() >= enough_feature_points) {
                std::cout << "Found enough feature points: " << bboxes.size() << std::endl;
                feature_points_found = true;
            }
        } */
        /**********************************************************************
                        TRACKING
        
        ************************************************************************/
    /*    else if (feature_points_found && !tracking_started){ //Initialize tracking the feature points
            std::cout << "Initializing multitracker..." << std::endl;
            
            // Initialize multitracker
            for(int i=0; i < bboxes.size(); i++) {
                //Make sure the bounding box is comletely inside the frame
                if(0 <= bboxes[i].x && 0 <= bboxes[i].width && bboxes[i].x + bboxes[i].width <= tracking_frame.cols && 0 <=  bboxes[i].y && 0 <=  bboxes[i].height &&  bboxes[i].y +  bboxes[i].height <= tracking_frame.rows) {
                    colors.push_back(cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255)));
                    multiTracker->add(cv::TrackerBoosting::create(), tracking_frame, cv::Rect2d(bboxes[i]));
                }
            }
            tracking_started = true;
            std::cout << "Tracking objects..." << std::endl;
        }
        else {
            
            double timer = (double)cv::getTickCount();
            //Update the tracking result with new frame
            multiTracker->update(tracking_frame);
            // Calculate Frames per second (FPS)
            fps = cv::getTickFrequency() / ((double)cv::getTickCount() - timer);
            // Draw tracked objects, if the tracking was ok
            for(unsigned i=0; i<multiTracker->getObjects().size(); i++)
            {
                rectangle(tracking_frame, multiTracker->getObjects()[i], colors[i], 2, 1);
            }
        }
        
        // Display tracker type on frame
        cv::putText(tracking_frame, "Booster Tracker", cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,255,255),2);
        // Display FPS on frame
        cv::putText(tracking_frame, "FPS : " + std::to_string(int(fps)), cv::Point(10,60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,255,255), 2); */
        
        /**************************************************************************
                   DISPLAY IMAGES
        ***************************************************************************/
        //Display the threshold as text on the mask
        //(Mat& img, const string& text, Point org, int fontFace, double fontScale, Scalar color, int thickness=1, int lineType=8, bool bottomLeftOrigin=false )
        cv::Scalar black = cv::Scalar(0, 0, 0);
        cv::Scalar white = cv::Scalar(255, 255, 255);
        
        cv::putText(frame, "Original Frame", cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.7, black,2);
        cv::putText(tracking_frame, "Bounding boxes", cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.7, black, 2);
        cv::putText(bg_mask, "Background model", cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255),2);
        cv::putText(bg_mask, "Threshold: " + std::to_string(threshold), cv::Point(10,60), cv::FONT_HERSHEY_SIMPLEX, 0.7, white,2);
        //cv::putText(displayFeatures, "Harris feature points" , cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255),2);
        std::string uniqueObjCounter = std::to_string(unique_objects.size() > 0 ? unique_objects[0].counter : 0);
        cv::putText(drawing, "Total dectected objects: " + uniqueObjCounter, cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.7, white,2);
        cv::putText(drawing, "Visible objects: " + std::to_string(unique_objects.size()), cv::Point(10,60), cv::FONT_HERSHEY_SIMPLEX, 0.7, white,2);
        //cv::putText(harris, "Harris feature points" , cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0),2);
        ShowFourImages("Image", frame, display(bg_mask), tracking_frame, drawing);
        //Increment frame number
        frameNumber++;
        //Break if press ESC
        char c = (char)cv::waitKey(25);
        if(c==27)
            break;
    }
    cv::destroyAllWindows();
    myfile.close();
	return 0;
}
