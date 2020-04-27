#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <chrono>
#include "FastMCD.h"


#define VIDEO_OUT 0
#define MEDIAN 0

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
	
	//--- INITIALIZE CSV
	std::ofstream timeFile;
	timeFile.open("timeFile.csv");
	timeFile << "PreProcess, KLT, parallelBlock, parallelUpdate,PostProcess,Total" << std::endl;
	

	
	// check if we succeeded
	if (!cap.isOpened()) {
		std::cerr << "ERROR! Unable to open camera\n";
		return -1;
	}

#if VIDEO_OUT
	//Create Debug Writers --------------------------------------------------
	cv::Size S = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
	int ex = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
	int fcc = CV_FOURCC('X', 'V', 'I', 'D');
	cv::VideoWriter mask_output_median;
	mask_output_median.open("Median_Output.avi", ex=-1, cap.get(cv::CAP_PROP_FPS), S, true);


	cv::VideoWriter mask_output_morph;
	mask_output_morph.open("Morph_Output.avi", ex=-1, cap.get(cv::CAP_PROP_FPS), S, true);

#endif
	cv::Mat closed_mask;
	cv::Mat open_mask;
	cv::Mat median_mask;

	cv::Mat element = cv::getStructuringElement(0,
		cv::Size(3, 3));

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
			closed_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
			open_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
			median_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
		}
		else{
			auto start = std::chrono::high_resolution_clock::now();
			cv::Mat grayImage;
			cv::cvtColor(frame, grayImage, CV_BGR2GRAY, 1);
			cv::Mat blurImage;
			cv::medianBlur(grayImage, blurImage, 5);
			auto end = std::chrono::high_resolution_clock::now();
			float time_pp = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()/1000.0;

			start = std::chrono::high_resolution_clock::now();
			IplImage* ipl_img = cvCloneImage(&(IplImage)blurImage);
			bgs.m_LucasKanade.RunTrack(ipl_img, 0);
			bgs.m_LucasKanade.GetHomography(bgs.m_h);
			end = std::chrono::high_resolution_clock::now();
			float time_klt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
			
			start = std::chrono::high_resolution_clock::now();
			bgs.blockInterpolation(bgs.m_h);
			end = std::chrono::high_resolution_clock::now();
			float time_parallelBlock = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

			start = std::chrono::high_resolution_clock::now();
			bgs.updateModels(blurImage);
			end = std::chrono::high_resolution_clock::now();
			float time_update = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

#if MEDIAN			
			start = std::chrono::high_resolution_clock::now();
			bgs.output_mask.copyTo(median_mask);
			cv::medianBlur(bgs.output_mask, median_mask, 7);
			end = std::chrono::high_resolution_clock::now();
			float time_post = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
			std::printf("Frame: %d PP: %f KLT: %f ||Block %f ||UPT: %f POST: %f\n", i, time_pp, time_klt, time_parallelBlock, time_update, time_post);
			timeFile << time_pp << "," << time_klt << "," << time_parallelBlock << "," << time_update << "," << time_post << std::endl;
			cv::imshow("Median_Blur", median_mask);

#else
			start = std::chrono::high_resolution_clock::now();
			bgs.output_mask.copyTo(closed_mask);
			cv::morphologyEx(bgs.output_mask, closed_mask, cv::MORPH_CLOSE, element, cv::Point(-1, -1), bgs.m_cfg.block_size);
			cv::morphologyEx(closed_mask, open_mask, cv::MORPH_OPEN, element, cv::Point(-1, -1), bgs.m_cfg.block_size);
			end = std::chrono::high_resolution_clock::now();
			float time_post = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
			float total = time_pp + time_klt + time_parallelBlock + time_update + time_post;
			std::printf("Frame: %d PP: %f KLT: %f ||Block %f ||UPT: %f POST: %f Total: %f\n", i, time_pp, time_klt, time_parallelBlock, time_update, time_post,total);
			timeFile << time_pp << "," << time_klt << "," << time_parallelBlock << "," << time_update << "," << time_post << "," << total << std::endl;
			cv::imshow("Closed_Mask", closed_mask);
			cv::imshow("Open_Mask", open_mask);
#endif

		}
		
		
		if(bgs.m_cfg.debug == 1){
			bgs.displayKLT(frame);
			bgs.displayMeans(frame);
			bgs.displayVars(frame);
			bgs.displayAge(frame);
			bgs.displayMask();
			bgs.displayUpdates(frame);
			bgs.model_updates = cv::Mat::zeros(bgs.model_dims, CV_32FC2);
		}

		char buf[1000];
		sprintf(buf, "./results/frm%05d_orig.png", i);

#if MEDIAN
		cv::imwrite(buf, median_mask);
#else
		cv::imwrite(buf, open_mask);
#endif

#if VIDEO_OUT
		mask_output_morph << open_mask;
		mask_output_median << median_mask;
#endif			

		cv::waitKey(1);
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	timeFile.close();
	return 0;
}