//Opencv header files
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>  // Video write
#include <vector>
#include <algorithm>
#include <iterator>
#include <opencv2/core/mat.hpp>
#include <stdlib.h>     /* srand, rand */

enum variables {
    MY = 0,
    VAR = 1,
    WEIGHT = 2
};

cv::Mat medianBackgroundModelling(cv::Mat frame, cv::Mat background, int ksize = 3, double thresh = 90, int erosion_size = 1, int dilation_size = 2) {

    cv::Mat diff;
    cv::Mat binary;
    cv::medianBlur(background, background, ksize);

    absdiff(frame, background, diff);
    //Make grayscale
    //source, destination, color type
    cv::cvtColor(diff, diff, cv::COLOR_BGR2GRAY);
    //Threshold
    //source, destination, threshold, max value out, threshold type
    cv::threshold(diff, binary, thresh, 1.0, cv::ThresholdTypes::THRESH_BINARY);

    cv::Mat er_element = cv::getStructuringElement(cv::MORPH_RECT,
        cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
        cv::Point(erosion_size, erosion_size));

    cv::erode(binary, binary, er_element);

    cv::Mat dil_element = getStructuringElement(cv::MORPH_RECT,
        cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
        cv::Point(dilation_size, dilation_size));

    cv::dilate(binary, binary, dil_element);

    //std::cout << binary << std::endl;
    binary.convertTo(binary, CV_32FC1);
    std::cout << binary << std::endl;

    return binary;
}

struct SortStruct
{
    double quotient;
    double w;
    double my;
    double covar;
};


double isForeground(double x, std::vector<cv::Vec3d*>& mix_comps, double w_init, int K = 5, double alpha = 0.002, double T = 0.8, double var_init = 10.0)  {

    std::vector<SortStruct> variables{};
    const double lambda = 2.5;
    bool match{false};
    int m{};
    double totWeight = 0.0;
    
    std::vector<double> d(K, 0.0); //mahalanobis
    std::vector<double> rho(K, 0.0);
    std::vector<double> quotient_vec(K, 0.0);

    // For all k mixture models, check if the pixel is foreground or background
    // Match == true means the pixel value is matching the mixture model
    for (int k = 0; k < mix_comps.size(); k++) {

        d[k] = sqrt((x - (*mix_comps[k])[MY]) * (x - (*mix_comps[k])[MY]));
        
        if (d[k] < lambda * sqrt((*mix_comps[k])[VAR])) {

            if (match == false) {
                m = k;
            }
            else if (((*mix_comps[k])[WEIGHT] / sqrt((*mix_comps[k])[VAR])) > ((*mix_comps[m])[WEIGHT] / sqrt((*mix_comps[m])[VAR]))) {
                m = k;
            }
            match = true;
        }
    }
    if (match == false) {
        m = mix_comps.size()-1;
        (*mix_comps[m])[WEIGHT] = alpha;
        (*mix_comps[m])[MY] = x;
        (*mix_comps[m])[VAR] = var_init;
    }
    else {
        (*mix_comps[m])[WEIGHT] = ((1 - alpha) * (*mix_comps[m])[WEIGHT]) + alpha;
        rho[m] = alpha / (*mix_comps[m])[WEIGHT];
        (*mix_comps[m])[MY] = ((1 - rho[m]) * (*mix_comps[m])[MY]) + (rho[m] * x);
        (*mix_comps[m])[VAR] = ((1 - rho[m]) * (*mix_comps[m])[VAR]) + (rho[m] * (x - (*mix_comps[m])[MY]) * (x - (*mix_comps[m])[MY]));
    }
        
    // RENORMALIZE W
    totWeight = 0;
    for (int i = 0; i < mix_comps.size(); i++)
    {
        totWeight += (*mix_comps[i])[WEIGHT];
    }
        
    for (int i = 0; i < mix_comps.size(); i++)
    {
        (*mix_comps[i])[WEIGHT] = (*mix_comps[i])[WEIGHT] / totWeight;
    }

    if (match) {

        // Sort w, my, covar with respect to weight/covariance ratio
        for (int i = 0; i < mix_comps.size(); i++) {
            quotient_vec[i] = (*mix_comps[i])[WEIGHT] / sqrt((*mix_comps[i])[VAR]);
            variables.push_back({ quotient_vec[i], (*mix_comps[i])[WEIGHT], (*mix_comps[i])[MY], (*mix_comps[i])[VAR]});
        }
        std::sort(variables.begin(), variables.end(),
            [](const SortStruct i, const SortStruct j) {return i.quotient < j.quotient; });

        for (int i = 0; i < mix_comps.size(); i++) {
            (*mix_comps[i])[WEIGHT] = variables[i].w;
            (*mix_comps[i])[MY] = variables[i].my;
            (*mix_comps[i])[VAR] = variables[i].covar;
        }
    }

    double sum = 0;
    int B = 0;
               
    for (int k = 0; k < mix_comps.size(); k++) {
        sum += (*mix_comps[k])[WEIGHT];
        if (sum > T) {
            B = k;
            break;
        }
    }
    //Background segmentation for gaussian mixture model
    for (int k = 0; k < B; k++)
    {
        d[k] = sqrt((x - (*mix_comps[k])[MY]) * (x - (*mix_comps[k])[MY]));
        if (d[k] < lambda * sqrt((*mix_comps[k])[VAR])) {
            return 1.0;
        }
    }
    return 0.0;
}


void mixtureBackgroundModelling(cv::Mat &frame, std::vector<cv::Mat>& variableMatrices, cv::Mat &background_model, double w_init, double var_init, int K = 5, double alpha = 0.002, double T = 0.8) {

    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    frame.convertTo(frame, CV_64F);
     
     frame.forEach<double>([&](double& pixel, const int position[]) -> void {
     std::vector<cv::Vec3d*> variables;
                
                for(int i = 0; i < variableMatrices.size(); i++) {
                    variables.push_back(&variableMatrices[i].at<cv::Vec3d>(position[0], position[1]));
                }
                pixel = isForeground(pixel, variables, w_init, K, alpha, T, var_init);
      });
}


int main() {

    //Name of video
    std::string source{ "Walk1.mpg" };
    cv::VideoCapture inputVideo(source); // Open input
    if (!inputVideo.isOpened())
    {
        std::cout << "Could not open the input video: " << source << std::endl;
        return -1;
    }

    cv::Mat frame;
    inputVideo >> frame;

    cv::Mat background_model = cv::Mat(frame.rows, frame.cols, CV_64F, cv::Scalar(0.000));
    
    std::vector<cv::Mat> variableMatrices;
    int K = 5;
    for(int k = 0; k < K; k++) {
        variableMatrices.push_back(cv::Mat(frame.rows, frame.cols, CV_64FC3, cv::Scalar(5 + rand() % 10 + 1 , 10.0, 0.002)));
    }
    
    while (1) {

        cv::imshow("Original video", frame);
    
        //(cv::Mat &frame, double w_init, double var_init, int K = 5, double alpha = 0.002, double T = 0.8)
        mixtureBackgroundModelling(frame, variableMatrices, background_model, 0.002, 10.0, 5, 0.002, 0.8);
        cv::imshow("Mixture model", frame);
        
        //Break if press ESC
        char c = (char)cv::waitKey(25);
        if (c == 27)
            break;

        inputVideo >> frame;
        if (frame.empty()) {
            break;
        }
    }

    return 0;
}
