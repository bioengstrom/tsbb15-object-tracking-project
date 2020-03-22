//
//  main.cpp
//  GT_Evaluation
//
//  Created by Ylva Selling on 2020-03-12.
//  Copyright © 2020 Ylva Selling. All rights reserved.
//

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <map>
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
    int id_switches{};
    double total_tp_overlap{};
    std::map<int,int> id_pairs;
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

std::ostream& operator<< (std::ostream &out, Evaluation &ev)
{
    out << std::left << std::setw(30) << "True positives:";
    out << ev.true_positives << std::endl;
    out << std::left << std::setw(30) << "False positives:";
    out << ev.false_positives << std::endl;
    out << std::left << std::setw(30) << "False negatives:";
    out << ev.false_negatives << std::endl << std::endl;;
    
    //Precision: Defined as sum(TP)/(sum(TP) + sum(FP)).
    out << std::left << std::setw(30) << "Precision:";
    out << static_cast<double>(ev.true_positives)/static_cast<double>(ev.true_positives + ev.false_positives) << std::endl;
    
    //Recall: Defined as sum(TP)/(sum(TP) + sum(FN)).
    out << std::left << std::setw(30) << "Recall:";
    out << static_cast<double>(ev.true_positives)/static_cast<double>(ev.true_positives + ev.false_negatives) << std::endl;
    
    //Average TP overlap: Computed only over the true positives (with correct ID).
    out << std::left << std::setw(30) << "Average TP overlap:";
    out << static_cast<double>(ev.total_tp_overlap)/static_cast<double>(ev.true_positives) << std::endl;
    
    //Average TP overlap: Computed only over the true positives (with correct ID).
    out << std::left << std::setw(30) << "Identity switches:";
    out << ev.id_switches << std::endl;
    
    
    return out;
}

//Evaluate true positives, false positives & false negatives
void evaluate(Evaluation &ev, std::vector<Object> &gt, std::vector<Object> &found) {
    
    /*
     True positives
     A detection that has at least 20% overlap with the associated ground truth bounding box.
     Overlap aka Jaccard index: Intersection over union (IoU)
    */
    std::vector<Object>::iterator largest_found_intsect;
    int i{};
    if(found.size() > 0) {
        while(gt.size() > i) {
            //Find the object that has the largest intersection with the ground truth
            largest_found_intsect = std::max_element(found.begin(), found.end(), [&](Object& first, Object& second){
                return (gt[i].rect & first.rect).area() < (gt[i].rect & second.rect).area();
            });
          
            //Calculate overlap - intersection / union
            double the_intersection = (gt[i].rect & largest_found_intsect->rect).area();
            double the_union = gt[i].rect.area() + largest_found_intsect->rect.area() - the_intersection;
            double overlap = the_intersection / the_union;
            
            //If overlap is larger than 20% we have a true positive! Remove the objects from
            //each vector
            if(overlap > 0.2) {
                //Check for id switch
                //Is the object id new?
                auto it = ev.id_pairs.find(gt[i].objectID);
                
                //Identity switch!
                if( it != ev.id_pairs.end() && it->second != largest_found_intsect->objectID) {
                    ev.id_switches++;
                    it->second = largest_found_intsect->objectID;
                }
                else {
                    // True positive!
                    
                    //A new object has entered the video, add it to the pairs
                    if (it == ev.id_pairs.end()) {
                        ev.id_pairs.insert(std::pair<int,int>(gt[i].objectID, largest_found_intsect->objectID));
                    }
                    ev.true_positives++;
                    ev.total_tp_overlap += overlap;
                    //Remove the found pair from the vectors
                    found.erase(largest_found_intsect);
                    gt.erase(std::next(gt.begin(), i));
                }
            }
            else {
                i++;
            }
            
        }
    }
    
    /*
     False positives
     The number of found objects that have no correspondance in the ground truth.
     */
    ev.false_positives += found.size();
    
    /*False negatives
     The number of ground truth objects that are not found.
     */
    ev.false_negatives += gt.size();
    
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
             Evaluate files, compare line by line
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
        
        //Clear vectors
        true_obj.clear();
        found_obj.clear();
    }
    
    //Print result to file
    std::cout << evaluation << std::endl;
    
    ground_truth.close();
    tracking_results.close();
    
    return 0;
}
