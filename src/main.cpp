//This line includes all the opencv header files
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>  // Video write
#include <vector>
#include <algorithm>


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

double isForeground(double x, double w_init, int K = 5, double alpha = 0.002, double T = 0.8, double var_init = 10.0)  {

    std::vector<SortStruct> variables{};
    const double lambda = 2.5;
    bool match{false};
    int m{};
    double totWeight = 0.0;


    std::vector<double> d(K, 0.0); //mahalanobis
    std::vector<double> covar_init(K, var_init);
    std::vector<double> covar(K, var_init);
    std::vector<double> my(K, 15.0);
    std::vector<double> w(K, w_init);
    std::vector<double> rho(K, 0.0);
    std::vector<double> quotient_vec(K, 0.0);


    // For all k mixture models, check if the pixel is foreground or background
    // Match == true means the pixel value is matching the mixture model
    for (int k = 0; k < K; k++) {

        d[k] = sqrt((x - my[k]) * (x - my[k]));

        if (d[k] < lambda * sqrt(covar[k])) {

            if (match == false) {
                m = k;
            }
            else if ((w[k] / sqrt(covar[k])) > (w[m] / sqrt(covar[m]))) {
                m = k;
            }
            match = true;
        }
    }

	if (match == false) {
		m = K;
		w[m] = alpha;
		my[m] = x;
		covar[m] = covar_init[m];
	}
	else {
		w[m] = ((1 - alpha) * w[m]) + alpha;
		rho[m] = alpha / w[m];
		my[m] = ((1 - rho[m]) * my[m]) + (rho[m] * x);
		covar[m] = ((1 - rho[m]) * covar[m]) + (rho[m] * (x - my[m]) * (x - my[m]));
	}
        
	// RENORMALIZE W
	totWeight = 0;
	for (auto& n : w)
		totWeight += n;

	for (int i = 0; i < K; i++)
	{
		w[i] = w[i] / totWeight;
	}

	if (match) {

		// Sort w, my, covar with respect to weight/covariance ratio
		for (int i = 0; i < K; i++) {
			quotient_vec[i] = w[i] / sqrt(covar[i]);
			variables.push_back({ quotient_vec[i], w[i], my[i], covar[i] });
		}
		std::sort(variables.begin(), variables.end(),
			[](const SortStruct i, const SortStruct j) {return i.quotient < j.quotient; });

		for (int i = 0; i < K; i++) {
			w[i] = variables[i].w;
			my[i] = variables[i].my;
			covar[i] = variables[i].covar;
		}
	}

	double sum = 0;
	int B = 0;
			   
	for (int k = 0; k < K; k++) {
		sum += w[k];
		if (sum > T) {
			B = k;
			break;
		}
	}
	//Background segmentation for gaussian mixture model
	for (int k = 0; k < B; k++)
	{
		d[k] = sqrt((x - my[k]) * (x - my[k]));
		if (d[k] < lambda * sqrt(covar[k])) {
			return 1.0;
		}
	}
    return 0.0;
}

void mixtureBackgroundModelling(cv::Mat &frame, double w_init, double var_init, int K = 5, double alpha = 0.002, double T = 0.8) {

    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    frame.convertTo(frame, CV_64F);
    frame.forEach<double>([&] (double &x, const int * position) -> void {
        x = isForeground(x, w_init, K, alpha, T, var_init);
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

    int threshold = 10;
    cv::Mat frame;

    while (1) {

        inputVideo >> frame;
        if (frame.empty()) {
            break;
        }
        //(cv::Mat &frame, double w_init, double var_init, int K = 5, double alpha = 0.002, double T = 0.8)
        mixtureBackgroundModelling(frame, 0.002, 10.0, 5, 0.002, 0.8);
        cv::imshow("Original video", frame);
        
        //Break if press ESC
        char c = (char)cv::waitKey(25);
        if (c == 27)
            break;
    }

    return 0;
}
