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

    //std::string source{"Rest_InChair"};
    std::string source{"eval3"};
    //std::string source{"Meet_Crowd"};
    //std::string source{"LeftBox"};
    std::string mov_format{"mpg"};

    cv::VideoCapture inputVideo("img3/%06d.jpg"); //like a printf pattern, leading zeros are important !
        

    //cv::VideoCapture inputVideo(source + "." + mov_format); // Open input

    if (!inputVideo.isOpened())
    {
        std::cout << "Could not open the input video: " << source << std::endl;
        return -1;
    }

    cv::Mat frame;
    inputVideo >> frame;

    // Nån trackinggrej från Tracking
    cv::Mat tracking_frame = frame.clone();
    cv::Mat background = frame.clone();

    /**************************************************************************
                    GMM PARAMETER INITIALIZATION
    ***************************************************************************/
    double m = 0; // median filtering
    cv::Mat background_model = cv::Mat(frame.rows, frame.cols, CV_8U, cv::Scalar(0));

    std::vector<cv::Mat> variableMatrices;

    double var = 50.0; // 30
    double w = 0.002; // 0.002
    double alpha = 0.002;
    double T = 0.8;
    double lambda = 3.0; // golden number : 4.5

    int K = 5;
    for (int k = 0; k < K; k++) {
        //variableMatrices.push_back(cv::Mat(frame.rows, frame.cols, CV_64FC3, cv::Scalar(rand() % 255 + 1, var, w)));
        variableMatrices.push_back(cv::Mat(frame.rows, frame.cols, CV_64FC3, cv::Scalar(rand() % 255 + 1, var, w)));
    }

    variableMatrices[0].forEach<cv::Vec3d>([&](cv::Vec3d& pixel, const int position[]) -> void {
        pixel[0] = frame.at<uchar>(position[0], position[1]);
        });

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
    int frameNumber{1};


    //Random generator for generating random colors

    int minRectArea = 1920*1080 * 0.25 * 0.01;
    /**********************************************************************
              MAINTAIN OBJECT IDENTITY
    ************************************************************************/

    std::vector<unique_object> unique_objects;
    int invisible_frame_treshold{50};
    int start_tracking_frame{10};

    /**************************************************************************
            SLIDER FOR ADJUSTING THRESHOLD
    ***************************************************************************/
    int closing_size{2};
    int opening_size{2};
    cv::namedWindow("Image", cv::WINDOW_NORMAL);
    cv::createTrackbar("Closing size", "Image", &closing_size, 10);
    cv::createTrackbar("Opening size", "Image", &opening_size, 10);
    


    while(1) {

        /**************************************************************************
                        LOAD FRAMES
        ***************************************************************************/

        inputVideo >> frame;
        cv::Size origSize(frame.rows, frame.cols);
        if(frame.empty()) {
            break;
        }

        cv::Mat tracking_frame = frame.clone();
        cv::Mat origFrame = frame.clone();

        /**************************************************************************
                   BACKGROUND MODELLING
        ***************************************************************************/
        // GMM från master
        pyrDown(frame, frame);
        pyrDown(frame, frame);
        //pyrDown(frame, frame);
        cv::Mat bg_mask = mixtureBackgroundModelling(frame, variableMatrices, background_model,
            w, var, K, alpha, T, lambda, closing_size, opening_size);

        cv::Mat drawing = cv::Mat::zeros( tracking_frame.size(), CV_8UC3 );

        //Start Gaussian mixture model after a certain frame
        if ( frameNumber > start_tracking_frame) {
            cv::Mat closing_small = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 7));
            cv::Mat opening_small = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
            cv::Mat closing = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, closing_size));
            cv::Mat opening = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(opening_size,opening_size));

            cv::morphologyEx(bg_mask, bg_mask, cv::MORPH_OPEN, opening_small);
            cv::morphologyEx(bg_mask, bg_mask, cv::MORPH_CLOSE, closing_small);

            //cv::morphologyEx(bg_mask, bg_mask, cv::MORPH_CLOSE, closing);
            //cv::morphologyEx(bg_mask, bg_mask, cv::MORPH_OPEN, opening);
            
            pyrUp(bg_mask, bg_mask);
            pyrUp(bg_mask, bg_mask);

            std::vector<cv::Rect> boundRect;

            findBoundingBoxes(bg_mask, boundRect, minRectArea);
            drawRectangles(tracking_frame, cv::Scalar( 0, 0, 255), boundRect);


            /**************************************************************************
                       DETECT OVERLAP
            ***************************************************************************/
            matchUniqueObjToDetections(invisible_frame_treshold, boundRect, unique_objects);
            drawUniqueObjects(drawing, unique_objects);
        }
        printFrameToCSV(myfile, frameNumber, false, unique_objects);
        /**************************************************************************
                   DISPLAY IMAGES
        ***************************************************************************/
        //Display the threshold as text on the mask
        //(Mat& img, const string& text, Point org, int fontFace, double fontScale, Scalar color, int thickness=1, int lineType=8, bool bottomLeftOrigin=false )
        cv::Scalar black = cv::Scalar(0, 0, 0);
        cv::Scalar white = cv::Scalar(255, 255, 255);
        pyrDown(tracking_frame, tracking_frame);
        cv::putText(origFrame, "Original Frame", cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.7, black,2);
        cv::putText(tracking_frame, "Detections", cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.7, black, 2);
        cv::putText(bg_mask, "Background model", cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255),2);
        //cv::putText(bg_mask, "Threshold: " + std::to_string(threshold), cv::Point(10,60), cv::FONT_HERSHEY_SIMPLEX, 0.7, white,2);
        //cv::putText(displayFeatures, "Harris feature points" , cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255),2);
        std::string uniqueObjCounter = unique_objects.size() > 0 ? std::to_string(unique_objects[0].counter) : "-";
        cv::putText(drawing, "Total unique objects: " + uniqueObjCounter, cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.7, white,2);
        cv::putText(drawing, "Current unique objects: " + std::to_string(unique_objects.size()), cv::Point(10,60), cv::FONT_HERSHEY_SIMPLEX, 0.7, white,2);
        //cv::putText(harris, "Harris feature points" , cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0),2);
        ShowFourImages("Image", origFrame, display(bg_mask), tracking_frame, drawing);
        //Increment frame number
        frameNumber++;

        //Break if press ESC
        char c = (char)cv::waitKey(25);
        if (c == 27)
            break;

    }
    cv::destroyAllWindows();
    myfile.close();
	return 0;
}
