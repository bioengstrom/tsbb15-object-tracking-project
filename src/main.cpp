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

//struct getB {

	double getB(double &x)  {

		std::vector<SortStruct> lmao{};
		const double lambda = 2.5;
		bool match = 0;
		int m;
		double totWeight = 0.0;
		int K = 5;
		double var_init = 10.0;
		double alpha = 0.002;
		double T = 0.8;

		std::vector<double> d(K, 0.0); //mahalanobis
		std::vector<double> covar_init(K, var_init);
		std::vector<double> covar(K, var_init);
		std::vector<double> my(K, 15.0);
		std::vector<double> w(K, alpha);
		std::vector<double> rho(K, 0.0);

		std::vector<double> quotient_vec(K, 0.0);


		// for all mixture models
		for (int k = 0; k < K; k++) {

			//d = cv::Mahalanobis(x, mean, var);
			d[k] = sqrt((x - my[k]) * (x - my[k]));

			if (d[k] < lambda * sqrt(covar[k])) {

				if (match == 0) {
					m = k;
				}
				else if ((w[k] / sqrt(covar[k])) > (w[m] / sqrt(covar[m]))) {
					m = k;
				}
				match = 1;
			}
		} //end

	if (match == 0) {
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



	if (match != 0) {

		// Sort w, my, covar with respect to some shit
		for (int i = 0; i < K; i++) {
			quotient_vec[i] = w[i] / sqrt(covar[i]);
			lmao.push_back({ quotient_vec[i], w[i], my[i], covar[i] });
		}
		std::sort(lmao.begin(), lmao.end(),
			[](const SortStruct i, const SortStruct j) {return i.quotient < j.quotient; });

		for (int i = 0; i < K; i++) {
			w[i] = lmao[i].w;
			my[i] = lmao[i].my;
			covar[i] = lmao[i].covar;
		}
	}

	double sum = 0;
	//int b = 0; // idx when sum of w > T
	int B = 0; // idx of argmin w
			   // std::vector<double> w_b;
			   // sum w[k] until > T, then take index of argmin(w[k])
	for (int k = 0; k < K; k++) {
		sum += w[k];
		//w_b.push_back(w[k]);
		if (sum > T) {
			B = k;
			break;
		}
	}
	//std::vector<double>::iterator result = std::min_element(w_b.begin(), w_b.end());
	//B = std::distance(w_b.begin(), result);
	//std::cout << B << std::endl;
	//Background segmentation for gaussian mixture model

	int B_hat = 0;
	for (int k = 0; k < B; k++)
	{
		d[k] = sqrt((x - my[k]) * (x - my[k]));

		if (d[k] < lambda * sqrt(covar[k])) {
			B_hat = 1;
			// std::cout << B_hat << std::endl;
		}
	}

	//bg_mask.at<float>(row, col) = B_hat;
	//cv::Mat bg_mask = cv::Mat::zeros(rows, cols, CV_64F);
	//cv::circle(bg_mask, cv::Point(row, col), 5, B_hat, 2, 8, 0);
	//cv::imshow("bgmask", bg_mask);
	//cv::waitKey(25);
	return B_hat;
	}


//};

std::vector<cv::Mat> mixtureBackgroundModelling(cv::VideoCapture inputVideo, int K, double alpha, double T, double w_init, double var_init) {
	std::vector<cv::Mat> background_vector{};
	std::vector<SortStruct> lmao{};
    double x = 0; //current pixel val
    const double lambda = 2.5;
    bool match = 0;
    int m;
    double totWeight = 0.0;
	
  

    std::vector<double> d(K, 0.0); //mahalanobis
    std::vector<double> covar_init(K, var_init);
    std::vector<double> covar(K, var_init);
    std::vector<double> my(K, 15.0);
    std::vector<double> w(K, alpha);
    std::vector<double> rho(K, 0.0);

    std::vector<double> quotient_vec(K, 0.0);

	


    while (1) {


        cv::Mat frame{};

        inputVideo >> frame;
        if (frame.empty()) {
            break;
        }
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
		frame.convertTo(frame, CV_64F);
		cv::imshow("Frame", frame);
		cv::waitKey(25);
       // int stride = frame.step;
        int cols = frame.cols;
        int rows = frame.rows;
        //bg_mask = cv::Mat::zeros(rows, cols, CV_64F);
		cv::Mat bg_mask = cv::Mat::zeros(rows, cols, CV_64F);
        //frame.copyTo(bg_mask);
        //std::cout << frame << std::endl;

		//x = (int)frame.at<uchar>(0, 0);
		//std::fill(my.begin(), my.end(), x);

        // for each pixel in frame[row,col]
		//x = (int)frame.at<uchar>(0, 0);
		std::cout << "x: " << x << std::endl;
		std::cout << "running getB" << std::endl;
	//	frame.forEach<double>([](double &x, const int* position) -> void {
	//		x = getB(x);
	//	}
	//	);
		cv::imshow("bgmask", frame);
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				x = frame.at<double>(row, col);
				bg_mask.at<double>(row, col) = getB(x);
			}
		}
                /*
                //std::cout << x << std::endl;


                // for all mixture models
                for (int k = 0; k < K; k++) {

                    //d = cv::Mahalanobis(x, mean, var);
                    d[k] = sqrt((x - my[k]) * (x - my[k]));

                    if (d[k] < lambda * sqrt(covar[k])) {

                        if (match == 0) {
                            m = k;
                        }
                        else if ((w[k] / sqrt(covar[k])) > (w[m] / sqrt(covar[m]))) {
                            m = k;
                        }
                        match = 1;
                    }
                } //end

                if (match == 0) {
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
	


                if (match != 0) {
                    
                    // Sort w, my, covar with respect to some shit
                    for (int i = 0; i < K; i++) {
                        quotient_vec[i] = w[i] / sqrt(covar[i]);
                        lmao.push_back({ quotient_vec[i], w[i], my[i], covar[i] });
                    }
                    std::sort(lmao.begin(), lmao.end(),
                        [](const SortStruct i, const SortStruct j) {return i.quotient < j.quotient; });

                    for (int i = 0; i < K; i++) {
                        w[i] = lmao[i].w;
                        my[i] = lmao[i].my;
                        covar[i] = lmao[i].covar;
                    }
                }

                double sum = 0;
                //int b = 0; // idx when sum of w > T
                int B = 0; // idx of argmin w
               // std::vector<double> w_b;
                // sum w[k] until > T, then take index of argmin(w[k])
                for (int k = 0; k < K; k++) {
                    sum += w[k];
                    //w_b.push_back(w[k]);
                    if (sum > T) {
                        B = k;
                        break;
                    }
                }
                //std::vector<double>::iterator result = std::min_element(w_b.begin(), w_b.end());
                //B = std::distance(w_b.begin(), result);
                //std::cout << B << std::endl;
                //Background segmentation for gaussian mixture model

                int B_hat = 0;
                for (int k = 0; k < B; k++)
                {
                    d[k] = sqrt((x - my[k]) * (x - my[k]));

                    if (d[k] < lambda * sqrt(covar[k])) {
                        B_hat = 1;
                       // std::cout << B_hat << std::endl;
                    }
                }

                bg_mask.at<float>(row, col) = B_hat;

				cv::circle(bg_mask, cv::Point(row, col), 5, B_hat, 2, 8, 0);
				cv::imshow("bgmask", bg_mask);
				cv::waitKey(25);
               // std::cout << bg_mask << std::endl;
            }
        }
		*/
        //bg_mask.convertTo(bg_mask, CV_32FC1);
        //std::cout << bg_mask << std::endl;
		//std::cout << "heheheh" << std::endl;
		cv::imshow("bgmask",bg_mask);
		cv::waitKey(0);
       // background_vector.push_back(bg_mask);

    } 

    return background_vector;
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
  //  cv::namedWindow("Threshold", cv::WINDOW_AUTOSIZE);
   // cv::createTrackbar("Threshold", "Threshold", &threshold, 255);
    //cv::Mat bg_mask;
    std::vector<cv::Mat> b_mask_vec = mixtureBackgroundModelling(inputVideo, 5, 0.002, 0.8, 0.1, 10.0);
	//std::cout << b_mask_vec[200] << std::endl;
    //std::cout << b_mask_vec[1] << std::endl;
    cv::VideoCapture inputVideo1(source); // Open input
    if (!inputVideo1.isOpened())
    {
        std::cout << "Could not open the input video: " << source << std::endl;
        return -1;
    }
    int counter = 0;
    while (1) {

        cv::Mat background = cv::imread("Walk1000.jpg", cv::IMREAD_UNCHANGED);
        if (background.empty()) {
            return 1;
        }
        cv::Mat frame;
        inputVideo1 >> frame;
        if (frame.empty()) {
            break;
        }
        //frame, background, ksize for median filter, threshold, erosion size, dilation size
        //bg_mask = medianBackgroundModelling(frame, background, 3, threshold, 1, 1);
        cv::imshow("Original video", frame);
        cv::imshow("GaussianMixMod", b_mask_vec[counter]);
        //cv::imshow("Background model", bg_mask);
        counter++;
        //Break if press ESC
        char c = (char)cv::waitKey(25);
        if (c == 27)
            break;
    }

    return 0;
}
