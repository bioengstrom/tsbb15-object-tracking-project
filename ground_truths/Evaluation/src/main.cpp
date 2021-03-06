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
    int frameNr{};
    in >> frameNr;
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

std::string validateInput(std::string& in) {
    
    std::stringstream temp(in);
    std::stringstream out("");
    std::istream_iterator<char> it_temp (temp);         // stdin iterator
    std::istream_iterator<char> eos;              // end-of-stream iterator
    std::ostream_iterator<char> it_out (out, "");         // stdin iterator
    
    std::copy_if(it_temp, eos, it_out, [](char letter){
        return letter == ',' || isdigit(letter);
    });
    
    return out.str();
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
    //Read objects into vector
    Object temp{};
    while(in >> temp) {
        objects.push_back(temp);
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
    out << std::setprecision(2) << static_cast<double>(ev.true_positives)/static_cast<double>(ev.true_positives + ev.false_positives) << std::endl;
    
    //Recall: Defined as sum(TP)/(sum(TP) + sum(FN)).
    out << std::left << std::setw(30) << "Recall:";
    out << std::setprecision(2) << static_cast<double>(ev.true_positives)/static_cast<double>(ev.true_positives + ev.false_negatives) << std::endl;
    
    //Average TP overlap: Computed only over the true positives (with correct ID).
    out << std::left << std::setw(30) << "Average TP overlap:";
    out << std::setprecision(2) << static_cast<double>(ev.total_tp_overlap)/static_cast<double>(ev.true_positives) << std::endl;
    
    //Average TP overlap: Computed only over the true positives (with correct ID).
    out << std::left << std::setw(30) << "Identity switches:";
    out << ev.id_switches << std::endl;
    
    
    return out;
}

//Intersection over union
double jaccardIndex(cv::Rect& first, cv::Rect& second) {
    double the_intersection = (first & second).area();
    double the_union = first.area() + second.area() - the_intersection;
    return the_intersection / the_union;
}

//Evaluate true positives, false positives & false negatives
void evaluate(Evaluation &ev, std::vector<Object> &gt, std::vector<Object> &found) {
    
    /*
     True positives
     A detection that has at least 20% overlap with the associated ground truth bounding box.
     Overlap aka Jaccard index: Intersection over union (IoU)
    */
    int i{};

    if(found.size() > 0) {
        while(gt.size() > i) {
            std::vector<Object>::iterator largest_found_intsect;
            //Find the object that has the largest intersection with the ground truth
            largest_found_intsect = std::max_element(found.begin(), found.end(), [&](Object& first, Object& second){
                //return (gt[i].rect & first.rect).area() < (gt[i].rect & second.rect).area();
                return jaccardIndex(gt[i].rect, first.rect) < jaccardIndex(gt[i].rect, second.rect);
            });
            
            if(largest_found_intsect == found.end()) {
                break;
            }
          
            //Calculate overlap - intersection / union
            /* double the_intersection = (gt[i].rect & largest_found_intsect->rect).area();
            double the_union = gt[i].rect.area() + largest_found_intsect->rect.area() - the_intersection;
            double overlap = the_intersection / the_union; */
            double overlap = jaccardIndex(gt[i].rect, largest_found_intsect->rect);
            
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
                    largest_found_intsect = found.erase(largest_found_intsect);
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
    
    if(argc != 4) {
        std::cerr << "Error! Please enter two csv filenames in the following order: \n 1) ground truth \n 2) results from the tracking program" << std::endl;
        return -1;
    }
    std::ifstream ground_truth{argv[1]};
    std::ifstream tracking_results{argv[2]};
    std::string outFile{argv[3]};
    
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
    std::string tmp{};
    Evaluation evaluation{};
    
    // Store current line in a stream
    std::stringstream ss_true{};
    std::stringstream ss_found{};
    
    int frameNr{1};
    int tempFrNr{};
    
    while(ground_truth || tracking_results) {
        while(1) {
            //Read one frame into detection obj vector
            std::getline(ground_truth, tmp);
            std::cout << tmp << std::endl;
            std::string validInput = validateInput(tmp);
            ss_true.str(validInput);
            ss_true >> tempFrNr;
            if(tempFrNr == frameNr) {
                true_obj_str.append(tmp);
            }
            else {
                break;
            }
        }
        
        do {
            //Read one frame into detection obj vector
            std::getline(tracking_results, tmp);
            std::cout << tmp << std::endl;
            std::string validInput = validateInput(tmp);
            ss_found.str(validInput);
            ss_found >> tempFrNr;
            found_obj_str.append(tmp);
            std::cout << "hj" << std::endl;
        } while(tempFrNr == frameNr);
        
        ss_true.str(true_obj_str);
        ss_found.str(found_obj_str);
        //Read the objects into vectors for evaluation
        ss_true >> true_obj;
        ss_found >> found_obj;
        
        if(true_obj.size() == 0 && found_obj.size() == 0) {
            break;
        }
        
        //Compare the vectors
        evaluate(evaluation, true_obj, found_obj);
        
        //Clear vectors
        true_obj.clear();
        found_obj.clear();
    
    }
            
    std::cout << evaluation;
    
    //Print result to file
    std::ofstream result;
    result.open (outFile + ".txt");
    result << evaluation;
    result.close();
    
    ground_truth.close();
    tracking_results.close();
    
    return 0;
}
