#include "BackgroundModelling.hpp"

enum variables {
    MY = 0,
    VAR = 1,
    WEIGHT = 2
};

struct SortStruct
{
    double quotient;
    double w;
    double my;
    double covar;
};

int isForeground(double x, std::vector<cv::Vec3d*>& mix_comps,
    double w_init, int K, double alpha = 0.002, double T = 0.8, double var_init = 10.0, double lambda = 2.5) {

    std::vector<SortStruct> variables{};
    bool match{ false };
    int m = 0;
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
        m = K - 1;
        (*mix_comps[m])[WEIGHT] = w_init;
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
            variables.push_back({ quotient_vec[i], (*mix_comps[i])[WEIGHT], (*mix_comps[i])[MY], (*mix_comps[i])[VAR] });
        }
        std::sort(variables.begin(), variables.end(),
            [](const SortStruct i, const SortStruct j)
            {
                return i.quotient > j.quotient;
            });

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
            //B = k;
            break;
        }
    }

    for (int k = 0; k < B; k++)
    {
        d[k] = sqrt((x - (*mix_comps[k])[MY]) * (x - (*mix_comps[k])[MY]));
        if (d[k] < lambda * sqrt((*mix_comps[k])[VAR])) {
            return 0;
        }
    }
    return 255;
}

cv::Mat mixtureBackgroundModelling(cv::Mat frame, std::vector<cv::Mat>& variableMatrices, cv::Mat& background_model,
    double w_init, double var_init, int K, double alpha, double T, double lambda, int erosion_size, int dilation_size) {

    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

    std::vector<double> d(K, 0.0); //mahalanobis

    frame.forEach<uchar>([&](uchar& pixel, const int position[]) -> void {
        std::vector<cv::Vec3d*> variables;
        for (int i = 0; i < variableMatrices.size(); i++) {
            variables.push_back(&variableMatrices[i].at<cv::Vec3d>(position[0], position[1]));
        }
        pixel = isForeground(pixel, variables, w_init, K, alpha, T, var_init, lambda);
    });

    return frame;
}

cv::Mat medianFiltering(cv::Mat frame, double& m) {
    cv::Mat background;
    frame.copyTo(background);
    double T = 60.0;
    double a = 0.1;

    int x;
    cv::cvtColor(background, background, cv::COLOR_BGR2GRAY);

    for (int i = 0; i < background.rows; i++)
    {

        for (int j = 0; j < background.cols; j++)
        {
            x = background.at<uchar>(i, j);
            if (x > m) {
                m = m + a;
            }
            else if (x < m) {
                m = m - a;

            }

            if (abs(x - m) <= T) {
                background.at<uchar>(i, j) = 0;
            }
            else {
                background.at<uchar>(i, j) = 255;
            }
        }

    }

    return background;
}