//This line includes all the opencv header files
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>  // Video write

int main() {
	
    std::string source{"Walk1.mpg"};
    
    cv::VideoCapture inputVideo(source);// Open input
    if (!inputVideo.isOpened())
    {
        std::cout  << "Could not open the input video: " << source << std::endl;
        return -1;
    }
    
    while(1) {
    
     /*   //Load image
        cv::Mat img0 = cv::imread ("Walk1000.jpg",cv::IMREAD_UNCHANGED);
        cv::Mat img1 = cv::imread ("Walk1001.jpg",cv::IMREAD_UNCHANGED);
        cv::Mat walking = cv::imread ("Walk1321.jpg",cv::IMREAD_UNCHANGED);
        cv::Mat img_medblur{img1}; */
        cv::Mat background = cv::imread ("Walk1321.jpg",cv::IMREAD_UNCHANGED);
        
        cv::Mat frame;
        inputVideo >> frame;
        if(frame.empty()) {
            break;
        }
        
  /*
        if (!img0.data || !img1.data)
        {
            std::cout << "Image not loaded";
            return -1;
        } */
        int ksize = 3;
        
        cv::medianBlur(background,background, ksize);
        
        //show the image in the window
       // cv::imshow("img", walking);
        //cv::imshow("img blur", img_medblur);
        
        cv::Mat diff;
        cv::Mat binary;
        
        double thresh = 90;
        absdiff(frame, background, diff);
        cv::imshow("diff", diff);
        //Make grayscale
        //source, destination, color type
        cv::cvtColor( diff, diff, cv::COLOR_BGR2GRAY);
        //Threshold
        //source, destination, threshold, max value out, threshold type
        cv::threshold(diff, binary, thresh, 255, cv::ThresholdTypes::THRESH_BINARY);
        
        //cv::imshow("grayscale binary", binary);
        int erosion_size = 1;
        int dilation_size = 2;
        
        cv::Mat er_element = cv::getStructuringElement( cv::MORPH_RECT,
                             cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                             cv::Point( erosion_size, erosion_size ) );
        
        cv::erode(binary, binary, er_element);
        //cv::imshow("eroded", binary);
        
        cv::Mat dil_element = getStructuringElement( cv::MORPH_RECT,
                             cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                             cv::Point( dilation_size, dilation_size ) );
        
        cv::dilate(binary, binary, dil_element);
        cv::imshow("dilated", binary);

        //wait until keypress
        char c = (char)cv::waitKey(25);
        if(c==27)
            break;
    }

	return 0;
}
