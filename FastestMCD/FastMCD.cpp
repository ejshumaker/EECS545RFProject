#include "FastMCD.h"


bool FastMCD::init(cv::Mat firstImg) {
	//First we set all the variables that rely on config
	model_dims.width = firstImg.cols / float(m_cfg.block_size);
	model_dims.height = firstImg.rows / float(m_cfg.block_size);

	temp_means = cv::Mat::zeros(model_dims, CV_32FC2);
	temp_age = cv::Mat::zeros(model_dims, CV_32FC2);
	temp_vars = cv::Mat::zeros(model_dims, CV_32FC2);

	model_means = cv::Mat::zeros(model_dims, CV_32FC2);
	model_age = cv::Mat::zeros(model_dims, CV_32FC2);
	model_vars = cv::Mat::zeros(model_dims, CV_32FC2);

	//second, we initialize the Interpolation ParallelLoopBody
	m_h[0] = 1.0;
	m_h[1] = 0.0;
	m_h[2] = 0.0;
	m_h[3] = 0.0;
	m_h[4] = 1.0;
	m_h[5] = 0.0;
	m_h[6] = 0.0;
	m_h[7] = 0.0;
	m_h[8] = 1.0;

	
	parallelInterpolation = ParallelInterpolation(model_dims, m_cfg.block_size, &model_means, &model_vars, &model_age,
		&temp_means, &temp_vars, &temp_age,
		m_cfg.min_vars, m_cfg.theta_v, m_cfg.max_age, m_cfg.lambda, m_cfg.init_vars);
	blockInterpolation(m_h);

	//Third, we grayscale and mean the image, inputing it as the 
	//We feed this into the KLT after converting into an IplImage
	cv::Mat grayImage;
	cv::cvtColor(firstImg, grayImage, CV_BGR2GRAY, 1);
	cv::Mat blurImage;
	cv::GaussianBlur(grayImage, blurImage, cv::Size(9, 9), 0, 0);

	output_mask = cv::Mat::zeros(blurImage.size(), CV_8UC1);
	parallelUpdate = ParallelUpdate(model_dims, m_cfg.block_size, &model_means, &model_vars, &model_age,
		&temp_means, &temp_vars, &temp_age,
		m_cfg.min_vars, m_cfg.max_age, m_cfg.theta_s, m_cfg.theta_d, m_cfg.init_vars, &output_mask);
	updateModels(blurImage);
	

	


	//Convert To IplImage
	IplImage* ipl_img = cvCloneImage(&(IplImage)blurImage);
	m_LucasKanade.Init(ipl_img);
	
	return true;
}


void FastMCD::testAccess(cv::Mat& image){
	ParallelAccess parallelAccess(model_dims, m_cfg.block_size, &image);
	cv::parallel_for_(cv::Range(0, model_means.cols * model_means.rows), parallelAccess,1);
}

void FastMCD::blockInterpolation(double h[9]) {
	parallelInterpolation.setH(h);
	cv::parallel_for_(cv::Range(0, model_means.rows*model_means.cols), parallelInterpolation);
}

void FastMCD::updateModels(cv::Mat& image) {
	parallelUpdate.setCurrentFrame(&image);
	cv::parallel_for_(cv::Range(0, model_means.rows*model_means.cols), parallelUpdate,1);
}

cv::Mat FastMCD::displayKLT(cv::Mat& image) {
	//First we need to have the points of the KLT available
	cv::Mat display;
	image.copyTo(display);

	for (int i = 0; i < m_LucasKanade.pt1.size(); i++) {
		float x = m_LucasKanade.pt1[i].x;
		float y = m_LucasKanade.pt1[i].y;
		cv::Point2f point = cv::Point2f(x,y);
		//printf("Point: %f, %f, \n", x, y);
		cv::circle(display, point, 3, cv::Scalar(0, 0, 255), -1);
		
		
		x = m_LucasKanade.pt2[i].x;
		y = m_LucasKanade.pt2[i].y;
	    cv::Point2f point2 = cv::Point2f(x, y);
		cv::line(display, point, point2, cv::Scalar(255, 0, 0), 2);
		cv::circle(display, point2, 3, cv::Scalar(0, 255, 0), 2);

		

	}

	cv::imshow("KLT", display);
	cv::waitKey(1);

	return display;
}

cv::Mat FastMCD::displayMeans(cv::Mat& image) {
	cv::Mat means[2];
	cv::split(model_means, means);

	cv::resize(means[0], means[0], image.size());
	cv::resize(means[1], means[1], image.size());
	
	cv::imshow("Model Mean", means[0]/255.0);
	cv::imshow("Cand Mean", means[1]/255.0);
	cv::waitKey(1);
	return means[0];
}

cv::Mat FastMCD::displayVars(cv::Mat& image) {
	cv::Mat vars[2];
	cv::split(model_vars, vars);

	cv::resize(vars[0], vars[0], image.size());
	cv::resize(vars[1], vars[1], image.size());
	
	for (int i = 0; i < vars[0].rows; i++) {
		for (int j = 0; j < vars[0].cols; j++) {
			for (int c = 0; c < 2; c++) {
				int val = vars[c].at<float>(i, j);
				float out = 0;
				if (val < 100)
					out = 0;
				else if (val < 500)
					out = 50;
				else if (val < 1000)
					out = 100;
				else if (val < 2000)
					out = 150;
				else if (val < 5000)
					out = 200;
				else
					out = 255;
				
				vars[c].at<float>(i, j) = out;
			}
		}
	}


	cv::imshow("Model Vars", vars[0] / 255.0);
	cv::imshow("Cand Vars", vars[1] / 255.0);
	cv::waitKey(1);
	return vars[0];
}

cv::Mat FastMCD::displayAge(cv::Mat& image) {
	cv::Mat age[2];
	cv::split(model_age, age);

	cv::resize(age[0], age[0], image.size());
	cv::resize(age[1], age[1], image.size());

	cv::imshow("Model Age", (age[0]*(255/30.0)) / 255.0);
	cv::imshow("Cand Age", (age[1]*(255/30.0)) / 255.0);
	cv::waitKey(1);
	return age[0];

}
cv::Mat FastMCD::displayMask() {
	cv::imshow("Mask", output_mask);
	cv::waitKey(1);
	return output_mask;
}
