//Opencv header files
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

double isForeground(cv::Mat &frame, std::vector<cv::Mat>& myMat, std::vector<cv::Mat>& varMat,
    std::vector<cv::Mat>& wMat, double w_init, int K = 5, double alpha = 0.002, double T = 0.8, double var_init = 10.0)  {

    std::vector<SortStruct> variables{};
    const double lambda = 2.5;
    //bool match{false};
   // int m{};
    double totWeight = 0.0;

    std::vector<cv::Mat> d(K, cv::Mat::zeros(frame.rows, frame.cols, CV_64F)); //mahalanobis
    std::vector<cv::Mat> rho(K, cv::Mat::zeros(frame.rows, frame.cols, CV_64F));
    std::vector<cv::Mat> quotient_vec(K, cv::Mat::zeros(frame.rows, frame.cols, CV_64F));

    cv::Mat match;
    int m;
    cv::Mat mk;
    //cv::Mat frame_my = cv::Mat::zeros(cv::Size(frame.rows, frame.cols), CV_64FC1);
    cv::Mat frame_my;
    frame.copyTo(frame_my);
    // For all k mixture models, check if the pixel is foreground or background
    // Match == true means the pixel value is matching the mixture model

    

    for (int k = 0; k < K; k++) {
        cv::subtract(frame, myMat[k], frame_my);
        //std::cout << frame_my << std::endl;
        cv::multiply(frame_my,frame_my, frame_my);
        //std::cout << frame_my << std::endl;
        cv::pow(d[k], 0.5, frame_my);
     
        cv::pow(varMat[k], 0.5, cv::Mat(varMat[k]));

        match = d[k] < lambda * varMat[k];

        match.forEach<int>([&](int& pixelMatch, const int * position) -> void
            {
                double& w = wMat[m].at<double>(position);
                double& var = varMat[m].at<double>(position);
                double& my = myMat[m].at<double>(position);
                double& x = frame.at<double>(position);

                if (pixelMatch == 0) {
                    m = k;
                }
                else if (w / sqrt(var) > w / sqrt(var)) {
                    m = k;
                }

                if (pixelMatch == 0) {
                    m = K;
                    w = alpha;
                    my = x;
                    var = var_init;
                }
                else {
                    w = ((1 - alpha) * w) + alpha;
                    rho[m] = alpha / w;
                    m = ((1 - rho[m].at<double>(position)) * my) + (rho[m].at<double>(position) * x);
                    var = ((1 - rho[m].at<double>(position)) * var + (rho[m].at<double>(position) * (x - my) * (x - my)));
                }
            }
        );
    }
         /* if (match == false) {
                m = k;
            }
            else if ((wMat[k] / sqrt(varMat[k])) > (wMat[m] / sqrt(varMat[m]))) {
                m = k;
            }
            match = true;
        }
    

	if (match == false) {
		m = K;
		wMat[m] = alpha;
		myMat[m] = frame;
		varMat[m] = var_init;
	}
	else {
		wMat[m] = ((1 - alpha) * wMat[m]) + alpha;
		rho[m] = alpha / wMat[m];
		myMat[m] = ((1 - rho[m]) * myMat[m]) + (rho[m] * frame);
		varMat[m] = ((1 - rho[m]) * varMat[m]) + (rho[m] * (frame - myMat[m]) * (frame - myMat[m]));
	}
        
	// RENORMALIZE wMat
	totWeight = 0;
	for (auto& n : wMat)
		totWeight += n;

	for (int i = 0; i < K; i++)
	{
		wMat[i] = wMat[i] / totWeight;
	}

	if (match) {

		// Sort w, my, varMat with respect to weight/varMatiance ratio
		for (int i = 0; i < K; i++) {
			quotient_vec[i] = wMat[i] / sqrt(varMat[i]);
			variables.push_back({ quotient_vec[i], wMat[i], myMat[i], varMat[i] });
		}
		std::sort(variables.begin(), variables.end(),
			[](const SortStruct i, const SortStruct j) {return i.quotient < j.quotient; });

		for (int i = 0; i < K; i++) {
			wMat[i] = variables[i].wMat;
			myMat[i] = variables[i].myMat;
			varMat[i] = variables[i].varMat;
		}
	}

	double sum = 0;
	int B = 0;
			   
	for (int k = 0; k < K; k++) {
		sum += wMat[k];
		if (sum > T) {
			B = k;
			break;
		}
	}
	//Background segmentation for gaussian miframeture model
	for (int k = 0; k < B; k++)
	{
		d[k] = sqrt((frame - myMat[k]) * (frame - myMat[k]));
		if (d[k] < lambda * sqrt(varMat[k])) {
			return 1.0;
		}
	}
    return 0.0;*/
        return 0;
}

void mixtureBackgroundModelling(cv::Mat &frame, std::vector<cv::Mat> &myMat, std::vector<cv::Mat>& varMat,
    std::vector<cv::Mat>& wMat, double w_init, double var_init, int K = 5, double alpha = 0.002, double T = 0.8) {

    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    frame.convertTo(frame, CV_64F);
 
    frame = isForeground(frame, myMat, varMat, wMat,w_init, K, alpha, T, var_init);
   
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

    cv::Mat myMat = cv::Mat(frame.rows, frame.cols, CV_64F, cv::Scalar(10.0));
    std::vector<cv::Mat> my(5, myMat);

    cv::Mat varMat = cv::Mat(frame.rows, frame.cols, CV_64F, cv::Scalar(10.0));
    std::vector<cv::Mat> var(5, varMat);

    cv::Mat wMat = cv::Mat(frame.rows, frame.cols, CV_64F, cv::Scalar(0.002));
    std::vector<cv::Mat> w(5, wMat);

    while (1) {

    
        //(cv::Mat &frame, double w_init, double var_init, int K = 5, double alpha = 0.002, double T = 0.8)
        mixtureBackgroundModelling(frame, my, var, w, 0.002, 10.0, 5, 0.002, 0.8);
        cv::imshow("Original video", frame);
        
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
