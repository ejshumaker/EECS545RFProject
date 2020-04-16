#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <chrono>
#include "FastMCD.h"


#define VIDEO_OUT 0


int main(int argc, char **argv){
	//Handle Import of fastMCD Config Class ---------------------------------
	std::string yaml_path = argv[1];
	
	fastMCD_Config cfg;
	cfg.importYAML(yaml_path);

	//Create fastMCD Object -------------------------------------------------
	FastMCD bgs(cfg);



	cv::Mat frame;
	//--- INITIALIZE VIDEOCAPTURE
	cv::VideoCapture cap;
	cap.open(cfg.video_path);
	// check if we succeeded
	if (!cap.isOpened()) {
		std::cerr << "ERROR! Unable to open camera\n";
		return -1;
	}

#if VIDEO_OUT
	//Create Debug Writers --------------------------------------------------
	cv::VideoWriter klt_output;
	cv::Size S = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
	int ex = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
	klt_output.open("KLT_Output.avi", ex = -1, cap.get(cv::CAP_PROP_FPS), S, true);
#endif



	//--- GRAB AND WRITE LOOP
	std::cout << "Start grabbing" << std::endl
		<< "Press any key to terminate" << std::endl;
	for (int i = 0;;i++)
	{
		// wait for a new frame from camera and store it into 'frame'
		cap.read(frame);
		// check if we succeeded
		if (frame.empty()) {
			std::cerr << "ERROR! blank frame grabbed\n";
			break;
		}
		// show live and wait for a key with timeout long enough to show images
		cv::imshow("Live", frame);
		if (cv::waitKey(5) >= 0)
			break;
	
		if(i == 0){
			bgs.init(frame);
		printf("INIT \n");
		}
		else{
			auto start = std::chrono::high_resolution_clock::now();
			cv::Mat grayImage;
			cv::cvtColor(frame, grayImage, CV_BGR2GRAY, 1);
			cv::Mat blurImage;
			cv::GaussianBlur(grayImage, blurImage, cv::Size(5, 5), 0, 0);
			auto end = std::chrono::high_resolution_clock::now();
			float time_pp = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

			start = std::chrono::high_resolution_clock::now();
			IplImage* ipl_img = cvCloneImage(&(IplImage)blurImage);
			//cv::imshow("Gray Blur", blurImage);

			//cv::Mat testImage = cv::Mat::zeros(blurImage.size(), CV_32FC1);
			//bgs.testAccess(testImage);
			//cv::imshow("Test Access", testImage);
			//cv::imwrite("test.png", testImage);
			
			
			bgs.m_LucasKanade.RunTrack(ipl_img, 0);
			bgs.m_LucasKanade.GetHomography(bgs.m_h);
			bgs.blockInterpolation(bgs.m_h);
			end = std::chrono::high_resolution_clock::now();
			float time_mocomp = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

			//bgs.model_means.copyTo(bgs.temp_means);
			//bgs.model_vars.copyTo(bgs.temp_vars);
			//bgs.model_age.copyTo(bgs.temp_age);

			start = std::chrono::high_resolution_clock::now();
			bgs.updateModels(blurImage);
			end = std::chrono::high_resolution_clock::now();
			float time_update = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			std::printf("PP: %f BI: %f UPT: %f \n", time_pp, time_mocomp, time_update);
#if VIDEO_OUT
			klt_output << bgs.displayKLT(frame);
#endif		
		}
		bgs.displayKLT(frame);
		bgs.displayMeans(frame);
		bgs.displayVars(frame);
		bgs.displayAge(frame);
		bgs.displayMask();

		cv::waitKey(1);
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	
	return 0;
}