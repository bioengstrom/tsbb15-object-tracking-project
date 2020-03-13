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

/*
 This program takes two arguments to two .csv files, the first is ground truth and the second is the results from the tracking program. The csv files should be parsed with the xml_parser.py if they are xml from the start. The csv files should be written in the format:
 
        framenumber, objectID, ul_x, ul_y, width, height

 where multiple objects are written after each other on the same line. End of line signifies a new frame.
 
 */

struct object {
    int objectID{};
    int x_tl{};
    int y_tl{};
    int width{};
    int height{};
    
    friend std::istream& operator>> (std::istream &in, object &obj);
};

std::istream& operator>> (std::istream &in, object &obj)
{
    in >> obj.objectID;
    in >> obj.x_tl;
    in >> obj.y_tl;
    in >> obj.width;
    in >> obj.height;
    
    return in;
}

int main(int argc, const char * argv[]) {
    
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
    
    object tmp{};
    int frame_number{};
    std::istringstream frame{};
    std::string tmp_string{};
    
    while(getline(ground_truth, tmp_string)) {
        std::cout << tmp_string <<std::endl;
        frame.str(tmp_string);
        int counter{};
        while(frame >> tmp) {
            std::cout << std::endl << "Hej" << std::endl;
            counter++;
        }
    }
    
    
    
    
    
    ground_truth.close();
    tracking_results.close();
    
    return 0;
}
