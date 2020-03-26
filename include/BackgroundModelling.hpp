#ifndef BackgroundModelling_hpp
#define BackgroundModelling_hpp

//OPEN CV
#include <opencv2/opencv.hpp>           //Base
#include <opencv2/core/core.hpp>        //Core

int isForeground(double x, std::vector<cv::Vec3d*>& mix_comps,
    double w_init, int K, double alpha, double T, double var_init, double lambda);

cv::Mat mixtureBackgroundModelling(cv::Mat frame, std::vector<cv::Mat>& variableMatrices, cv::Mat& background_model,
    double w_init, double var_init, int K, double alpha = 0.002, double T = 0.8, double lambda = 2.5, int erosion_size = 1, int dilation_size = 2);

cv::Mat medianFiltering(cv::Mat frame, double& m);

#endif /* BackgroundModelling_hpp */
