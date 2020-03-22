//
//  main.cpp
//  GT_Evaluation
//
//  Created by Ylva Selling on 2020-03-12.
//  Copyright © 2020 Ylva Selling. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc_c.h"

/*
 This program takes two arguments to two .csv files, the first is ground truth and the second is the results from the tracking program. The csv files should be parsed with the xml_parser.py if they are xml from the start. The csv files should be written in the format:
 
        framenumber, objectID, ul_x, ul_y, width, height

 where multiple objects are written after each other on the same line. End of line signifies a new frame.
 
 */

struct Object {
    int objectID{};
    cv::Rect rect{};
    
    friend std::istream& operator>> (std::istream &in, Object &obj);
    friend std::ostream& operator<< (std::ostream &out, Object &obj);
};

struct Evaluation {
    int true_positives{};
    int false_positives{};
    int false_negatives{};
    double total_tp_overlap{};
};

std::istream& operator>> (std::istream &in, Object &obj)
{
    char comma{};
    in >> comma;
    in >> obj.objectID;
    in >> comma;
    in >> obj.rect.x;
    in >> comma;
    in >> obj.rect.y;
    in >> comma;
    in >> obj.rect.width;
    in >> comma;
    in >> obj.rect.height;
    
    return in;
}

std::ostream& operator<< (std::ostream &out, Object &obj)
{
    out << obj.objectID << ",";
    out << obj.rect;
    
    return out;
}
//Gets one line from a csv file and inserts the objects into the vector
std::istream& operator>> (std::istream &in, std::vector<Object> &objects)
{
    //Remove beginning of line
    int frame_nr{};
    in >> frame_nr;
    
    //Read objects into vector
    Object temp{};
    while(in >> temp) {
        objects.push_back(temp);
        //std::cout << temp << std::endl;
    }
    return in;
}

//Evaluate true positives, false positives & false negatives
void evaluate(Evaluation &ev, std::vector<Object> &gt, std::vector<Object> &found) {
    
    /*
     True positives
     A detection that has at least 20% overlap with the associated ground truth bounding box.
     Overlap aka Jaccard index: Intersection over union (IoU)
    */
    std::vector<Object>::iterator largest_intersection;
    if(found.size() > 0) {
        for (int i = 0; i < gt.size(); i++) {
            //Find the object that has the largest intersection with the ground truth
            largest_intersection = std::max_element(found.begin(), found.end(), [&](Object& first, Object& second){
                return (gt[i].rect & first.rect).area() < (gt[i].rect & second.rect).area();
            });
          
            //Calculate overlap - intersection / union
            double the_intersection = (gt[i].rect & largest_intersection->rect).area();
            double the_union = gt[i].rect.area() + largest_intersection->rect.area() - the_intersection;
            double overlap = the_intersection / the_union;
            
            //If overlap is larger than 20% we have a true positive! Remove the objects from
            //each vector
            if(overlap > 0.2) {
                ev.true_positives++;
                //found.erase(std::remove(found.begin(), found.end(), largest_intersection), found.end());
                //gt.erase(std::remove(gt.begin(), gt.end(), std::next(gt.begin(),i), gt.end());
            }
        }
    }
    
    
    //False positives
    
    
    //False negatives
    
    
}



int main(int argc, const char * argv[]) {
    
    /************************************************************************
                    Open csv files
     ***********************************************************************/
    
    if(argc != 3) {
        std::cerr << "Error! Please enter two csv filenames in the following order: \n 1) ground truth \n 2) results from the tracking program" << std::endl;
        return -1;
    }
    std::ifstream ground_truth{argv[1]};
    std::ifstream tracking_results{argv[2]};
    if(!ground_truth) {
        std::cerr << "Error opening ground truth file" << std::endl;
        return -1;
    }
    if(!tracking_results) {
        std::cerr << "Error opening results file" << std::endl;
        return -1;
    }
    
    /************************************************************************
                  Evaluate files, line by line
    ***********************************************************************/
    
    std::vector<Object> true_obj{};
    std::vector<Object> found_obj{};
    std::string true_obj_str{};
    std::string found_obj_str{};
    Evaluation evaluation{};
    
    //Extract one line from each csv file to compare results
    while(std::getline(ground_truth, true_obj_str) && std::getline(tracking_results, found_obj_str)) {
        // Store current line in a stream
        std::stringstream ss_true(true_obj_str);
        std::stringstream ss_found(found_obj_str);
        
        //Read the objects into vectors for evaluation
        ss_true >> true_obj;
        ss_found >> found_obj;
        
        //Compare the vectors
        evaluate(evaluation, true_obj, found_obj);
        
        
        //Print result to file
        
        //Clear vectors
        true_obj.clear();
        found_obj.clear();
    }
    
    

    ground_truth.close();
    tracking_results.close();
    
    return 0;
}
