//This line includes all the opencv header files
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>  // Video write

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

    return binary;
}

cv::Mat mixtureBackgroundModelling(cv::Mat frame, int K, int D, double alpha, double T, double w_init, double covar_init) {
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    cv::Size s = frame.size();
    double d; //mahalanobis
    double x = 0; //current pixel val
    double mean;
    cv::Vec<double,4> covar = covar_init;
    // for each pixel in frame[row,col]
    for (int row = 0; row < s.height; row++) {
        for (int col = 0; col < s.width; col++) {
            
            x = frame.at<double>(row, col);
            mean = x;
            //var = (x - mean) * (x - mean);

            // for all mixture models
            for (int k = 0; k < K; k++) {
                
                //d = cv::Mahalanobis(x, mean, var);

            }
        }

        
    }
    
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

    int threshold = 90;
    cv::namedWindow("Threshold", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Threshold", "Threshold", &threshold, 255);

    while (1) {

        cv::Mat background = cv::imread("Walk1000.jpg", cv::IMREAD_UNCHANGED);
        if (background.empty()) {
            return 1;
        }
        cv::Mat frame;
        cv::Mat bg_mask;

        inputVideo >> frame;
        if (frame.empty()) {
            break;
        }
        //frame, background, ksize for median filter, threshold, erosion size, dilation size
        //bg_mask = medianBackgroundModelling(frame, background, 3, threshold, 1, 1);
        
        cv::imshow("Original video", frame);
        //cv::imshow("Background model", bg_mask);

        //Break if press ESC
        char c = (char)cv::waitKey(25);
        if (c == 27)
            break;
    }

    return 0;
}
