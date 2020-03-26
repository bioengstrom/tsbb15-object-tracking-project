#include "Tracking.hpp"

/**************************************************************************
            BOUNDING BOXES
     source: https://docs.opencv.org/2.4/doc/tutorials/imgproc/shapedescriptors/bounding_rects_circles/bounding_rects_circles.html
 ***************************************************************************/
 
void findBoundingBoxes(cv::Mat& bg_mask, std::vector<cv::Rect>& boundRect) {
     
     std::vector<std::vector<cv::Point> > contours;
     std::vector<cv::Vec4i> hierarchy;
     
     // Find contours
     cv::findContours( bg_mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
     
     // Approximate contours to polygons + get bounding rects and circles
     std::vector<std::vector<cv::Point> > contours_poly( contours.size() );

     for( int i = 0; i < contours.size(); i++ )
    {
      approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 3, true );
      boundRect.push_back( boundingRect( cv::Mat(contours_poly[i]) ));
      //minEnclosingCircle( (cv::Mat)contours_poly[i], center[i], radius[i] );
    }
}

void matchUniqueObjToDetections(std::vector<cv::Rect>& boundRect, std::vector<unique_object>& unique_objects) {

    std::vector<unique_object>::iterator largest_found_intsect;
    int i{};
    //Greedy algorithm, assign the rectangle with most overlap (Jaccard index) to the unique object
    while(i < boundRect.size()) {
        largest_found_intsect = std::max_element(unique_objects.begin(), unique_objects.end(), [&](unique_object& first, unique_object& second){
            
            return jaccardIndex(boundRect[i], first.rect) < jaccardIndex(boundRect[i], second.rect);
        });
        //There is an overlap!! update position and remove detection from vector
        if(largest_found_intsect != unique_objects.end() && jaccardIndex(boundRect[i], largest_found_intsect->rect) > 0.2 && !largest_found_intsect->overlap_found) {
            
            largest_found_intsect->rect = boundRect[i];
            largest_found_intsect->overlap_found = true;
            //Remove the found overlap from the detections
            boundRect.erase(std::next(boundRect.begin(), i));
        }
        else {
            //Move on to the next object
            i++;
        }
    }
    std::for_each(unique_objects.begin(), unique_objects.end(), [] (unique_object& obj) {
        if(!obj.overlap_found) {
            obj.frames_invisible++;
        }
        obj.overlap_found = false;
    });
    
    //Remove all unique objects that have been invisible for too long (frame threshold)
    int invisibleFrameThreshold{10};
    unique_objects.erase(std::remove_if(unique_objects.begin(),unique_objects.end(), [&invisibleFrameThreshold] (unique_object& obj) {
        return obj.frames_invisible > invisibleFrameThreshold;
        }), unique_objects.end());
    

    //Add all detections with no overlap to the unique objects vector
    std::transform(boundRect.begin(),boundRect.end(),std::back_inserter(unique_objects), [&] (cv::Rect detection) {
        return unique_object(detection);
    });
}
