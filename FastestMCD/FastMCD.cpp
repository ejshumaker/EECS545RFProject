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
	model_updates = cv::Mat::zeros(model_dims, CV_32FC2);
	
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
		m_cfg.min_vars, m_cfg.max_age, m_cfg.theta_s, m_cfg.theta_d, m_cfg.init_vars, &output_mask, &model_updates);
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
	cv::Scalar ill_offsets;
	cv::Scalar frame_mean = cv::mean(image);

	// "channels" is a vector of 3 Mat arrays:
	std::vector<cv::Mat> channels(2);
	// split img:
	split(model_means, channels);

	cv::Scalar model_mean_offset = cv::mean(channels[0]);
	cv::Scalar cand_mean_offset = cv::mean(channels[1]);
	float model_offset = (frame_mean - cv::mean(channels[0]))[0];
	float cand_offset = (frame_mean - cv::mean(channels[1]))[0];
	cv::Scalar pass = cv::Scalar(model_offset, cand_offset, 0, 0);

	parallelUpdate.setCurrentFrame(&image);
	parallelUpdate.setCurrentIllumination(&pass);
	cv::parallel_for_(cv::Range(0, model_means.rows*model_means.cols), parallelUpdate);
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
	cv::Mat image_means[2];
	cv::Mat image_model_means = cv::Mat::zeros(image.size(), CV_8UC1);
	cv::Mat image_cand_means = cv::Mat::zeros(image.size(), CV_8UC1);
	image_means[0] = image_model_means;
	image_means[1] = image_cand_means;

	int block_size = m_cfg.block_size;
	int val = 0;
	for (int y = 0; y < model_dims.height; y++)
		for (int x = 0; x < model_dims.width; x++)
			for (int c = 0; c < 2; c++) {
				val = uchar(means[c].at<float>(y, x));

				for (int jj = 0; jj < block_size; jj++) {
					int idx_j = y*block_size + jj;
					for (int ii = 0; ii < block_size; ii++) {
						int idx_i = x*block_size + ii;
						if (idx_i < 0 || idx_i >= image.cols || idx_j < 0 || idx_j >= image.rows)
							continue;
						image_means[c].at<uchar>(idx_j, idx_i) = val;
					}
				}

			}
	
	cv::imshow("Model Mean", image_means[0]);
	cv::imshow("Cand Mean", image_means[1]);
	cv::waitKey(1);
	return image_means[0];
}

cv::Mat FastMCD::displayVars(cv::Mat& image) {
	cv::Mat vars[2];
	cv::split(model_vars, vars);
	
	cv::Mat image_vars[2];
	cv::Mat image_model_vars = cv::Mat::zeros(image.size(), CV_8UC1);
	cv::Mat image_cand_vars = cv::Mat::zeros(image.size(), CV_8UC1);
	image_vars[0] = image_model_vars;
	image_vars[1] = image_cand_vars;

	int block_size = m_cfg.block_size;
	int val = 0;
	for(int y = 0; y < model_dims.height; y++)
		for( int x = 0; x < model_dims.width; x++)
			for (int c = 0; c < 2; c++) {
				val = vars[c].at<float>(y, x);
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

				for (int jj = 0; jj < block_size; jj++) {
					int idx_j = y*block_size + jj;
					for (int ii = 0; ii < block_size; ii++) {
						int idx_i = x*block_size + ii;
						if (idx_i < 0 || idx_i >= image.cols || idx_j < 0 || idx_j >= image.rows)
							continue;
						image_vars[c].at<uchar>(idx_j, idx_i) = out;
					}
				}

			}

	cv::imshow("Model Vars", image_vars[0]);
	cv::imshow("Cand Vars", image_vars[1]);
	cv::waitKey(1);
	return image_vars[0];
}

cv::Mat FastMCD::displayAge(cv::Mat& image) {
	cv::Mat age[2];
	cv::split(model_age, age);
	cv::Mat image_age[2];
	cv::Mat image_model_age = cv::Mat::zeros(image.size(), CV_8UC1);
	cv::Mat image_cand_age = cv::Mat::zeros(image.size(), CV_8UC1);
	image_age[0] = image_model_age;
	image_age[1] = image_cand_age;

	int block_size = m_cfg.block_size;
	int val = 0;
	for (int y = 0; y < model_dims.height; y++)
		for (int x = 0; x < model_dims.width; x++)
			for (int c = 0; c < 2; c++) {
				val = uchar(age[c].at<float>(y, x));

				for (int jj = 0; jj < block_size; jj++) {
					int idx_j = y*block_size + jj;
					for (int ii = 0; ii < block_size; ii++) {
						int idx_i = x*block_size + ii;
						if (idx_i < 0 || idx_i >= image.cols || idx_j < 0 || idx_j >= image.rows)
							continue;
						image_age[c].at<uchar>(idx_j, idx_i) = int(val * (255/float(m_cfg.max_age)));
					}
				}

			}
	cv::imshow("Model Age", image_age[0]);
	cv::imshow("Cand Age", image_age[1]);
	cv::waitKey(1);
	return image_age[0];

}

void FastMCD::displayUpdates(cv::Mat& image) {
	cv::Mat updates[2];
	cv::split(model_updates, updates);
	cv::Mat image_updates[2];
	cv::Mat image_model_updates = cv::Mat::zeros(image.size(), CV_8UC1);
	cv::Mat image_cand_updates = cv::Mat::zeros(image.size(), CV_8UC1);
	image_updates[0] = image_model_updates;
	image_updates[1] = image_cand_updates;

	int block_size = m_cfg.block_size;
	float val = 0;
	for (int y = 0; y < model_dims.height; y++)
		for (int x = 0; x < model_dims.width; x++)
			for (int c = 0; c < 2; c++) {
				val = updates[c].at<float>(y, x);

				for (int jj = 0; jj < block_size; jj++) {
					int idx_j = y*block_size + jj;
					for (int ii = 0; ii < block_size; ii++) {
						int idx_i = x*block_size + ii;
						if (idx_i < 0 || idx_i >= image.cols || idx_j < 0 || idx_j >= image.rows)
							continue;
						image_updates[c].at<uchar>(idx_j, idx_i) = int(val);
					}
				}

			}
	cv::imshow("Updates", image_updates[0]);
	cv::waitKey(1);

}

cv::Mat FastMCD::displayMask() {
	cv::imshow("Mask", output_mask);
	cv::waitKey(1);
	return output_mask;
}
