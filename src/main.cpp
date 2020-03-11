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

struct mix_comp
{
    mix_comp() {
        covar = 10.0;
        my = 10.0;
        w = 0.002;
    };

    double covar;
    double my;
    double w;
   

};

double isForeground(double x, std::vector<mix_comp> &mix_comps,double w_init, int K = 5, double alpha = 0.002, double T = 0.8, double var_init = 10.0)  {

    std::vector<SortStruct> variables{};
    const double lambda = 2.5;
    bool match{false};
    int m{};
    double totWeight = 0.0;
    

    std::vector<double> d(K, 0.0); //mahalanobis
    std::vector<double> covar_init(K, var_init);
    std::vector<double> rho(K, 0.0);
    std::vector<double> quotient_vec(K, 0.0);

    // For all k mixture models, check if the pixel is foreground or background
    // Match == true means the pixel value is matching the mixture model
    for (int k = 0; k < K; k++) {

        d[k] = sqrt((x - mix_comps[k].my) * (x - mix_comps[k].my));
        

        if (d[k] < lambda * sqrt(mix_comps[k].covar)) {

            if (match == false) {
                m = k;
            }
            else if ((mix_comps[k].w / sqrt(mix_comps[k].covar)) > (mix_comps[m].w / sqrt(mix_comps[k].covar))) {
                m = k;
            }
            match = true;
        }
    }

	if (match == false) {
		m = K;
        mix_comps[m].w = alpha;
        mix_comps[m].my = x;
		mix_comps[m].covar = covar_init[m];
	}
	else {
        mix_comps[m].w = ((1 - alpha) * mix_comps[m].w) + alpha;
		rho[m] = alpha / mix_comps[m].w;
        mix_comps[m].my = ((1 - rho[m]) * mix_comps[m].my) + (rho[m] * x);
		mix_comps[m].covar = ((1 - rho[m]) * mix_comps[m].covar) + (rho[m] * (x - mix_comps[m].my) * (x - mix_comps[m].my));
	}
        
	// RENORMALIZE W
	totWeight = 0;
    for (int i = 0; i < K; i++)
    {
        totWeight += mix_comps[i].w;
    }
		

	for (int i = 0; i < K; i++)
	{
        mix_comps[i].w = mix_comps[i].w / totWeight;
	}

	if (match) {

		// Sort w, my, covar with respect to weight/covariance ratio
		for (int i = 0; i < K; i++) {
			quotient_vec[i] = mix_comps[i].w / sqrt(mix_comps[i].covar);
			variables.push_back({ quotient_vec[i], mix_comps[i].w, mix_comps[i].my, mix_comps[i].covar});
		}
		std::sort(variables.begin(), variables.end(),
			[](const SortStruct i, const SortStruct j) {return i.quotient < j.quotient; });

		for (int i = 0; i < K; i++) {
            mix_comps[i].w = variables[i].w;
            mix_comps[i].my = variables[i].my;
			mix_comps[i].covar = variables[i].covar;
		}
	}

	double sum = 0;
	int B = 0;
			   
	for (int k = 0; k < K; k++) {
		sum += mix_comps[k].w;
		if (sum > T) {
			B = k;
			break;
		}
	}
	//Background segmentation for gaussian mixture model
	for (int k = 0; k < B; k++)
	{
		d[k] = sqrt((x - mix_comps[k].my) * (x - mix_comps[k].my));
		if (d[k] < lambda * sqrt(mix_comps[k].covar)) {
			return 1.0;
		}
	}
    return 0.0;
}

void mixtureBackgroundModelling(cv::Mat &frame, cv::Mat &mix_comps,double w_init, double var_init, int K = 5, double alpha = 0.002, double T = 0.8) {

    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    frame.convertTo(frame, CV_64F);
    frame.forEach<double>([&] (double &x, const int *position) -> void {
        x = isForeground(x, mix_comps.at<std::vector<mix_comp>>(position),w_init, K, alpha, T, var_init);
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
  
    int threshold = 10;
    
    //std::vector<std::vector<mix_comp>> mix_comps(frame.rows, std::vector<mix_comp>(frame.cols));
    //mix_comp lol = mix_comps[0][0];
    std::vector<mix_comp> kok(5);
    cv::Mat mix_comps(kok,frame.size);
   

    //cv::Mat<std::vector<mix_comp>> kek;

    while (1) {

        
        //(cv::Mat &frame, double w_init, double var_init, int K = 5, double alpha = 0.002, double T = 0.8)
        mixtureBackgroundModelling(frame, mix_comps,0.002, 10.0, 5, 0.002, 0.8);
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
