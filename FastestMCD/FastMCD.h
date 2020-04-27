#pragma once
#include "KLTWrapper.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <fstream>

struct fastMCD_Config {
	std::string video_path;
	int block_size = 4;
	float theta_v = 50 * 50;
	float theta_s = 2;
	float theta_d = 4;
	float lambda = 0.001;
	float max_age = 30;
	float min_vars = 5.0;
	float init_vars = 20 * 20;
	int debug = 0;
	fastMCD_Config() {}
	bool importYAML(std::string yaml_path) {
		cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
		video_path = fs["video_path"];
		block_size = fs["block_size"];
		theta_v = fs["theta_v"];
		theta_s = fs["theta_s"];
		theta_d = fs["theta_d"];
		lambda = fs["lambda"];
		max_age = fs["max_age"];
		min_vars = fs["min_vars"];
		debug = fs["debug"];
		return true;
	}
};

class ParallelAccess : public cv::ParallelLoopBody {
public:
	ParallelAccess() {}

	ParallelAccess(cv::Size _model_dims, int _block_size, cv::Mat* _matrix) {
		model_dims = _model_dims;
		matrix = _matrix;
		block_size = _block_size;
	}

	virtual void operator()(const cv::Range& range) const
	{
		for (int r = range.start; r < range.end; r++) {

			//Extract fastMCD Parameters
			int model_width = model_dims.width;
			int model_height = model_dims.height;
			int total_blocks = model_width * model_height;
			//Get Block XY from Index
			int b_Y = r / (model_width); //Rows
			int b_X = r % (model_width); //Columns
			//float val = (r / float(total_blocks));
			float val = r;
			//printf("Block [%d] Val: %f \n", r, val);

			for (int jj = 0; jj < block_size; jj++) {
				int idx_j = b_Y*block_size + jj;
				for (int ii = 0; ii < block_size; ii++) {
					int idx_i = b_X* block_size + ii;
					if (idx_i < 0 || idx_i >= matrix->cols || idx_j < 0 || idx_j >= matrix->rows)
						continue;
					float* frame_row_ptr = matrix->ptr<float>(idx_j);
					frame_row_ptr[idx_i] = val;
					
				}
			}
		}
	}

	ParallelAccess& operator=(ParallelAccess & new_model) {
		model_dims = new_model.model_dims;
		block_size = new_model.block_size;
		matrix = new_model.matrix;
		return *this;
	}
private:
	cv::Size model_dims;
	int block_size;
	cv::Mat* matrix;
};


class ParallelInterpolation : public cv::ParallelLoopBody {
public:
	ParallelInterpolation() {}
	
	ParallelInterpolation(cv::Size _model_dims, int _block_size, cv::Mat* _model_means, cv::Mat* _model_vars, cv::Mat* _model_age,
		cv::Mat* _temp_means, cv::Mat* _temp_vars, cv::Mat* _temp_age,
		float _min_vars, float _theta_v, float _max_age, float _lambda, float _init_vars) {
		model_dims = _model_dims;
		block_size = _block_size;
		model_means = _model_means;
		model_vars = _model_vars;
		model_age = _model_age;
		temp_means = _temp_means;
		temp_vars = _temp_vars;
		temp_age = _temp_age;
		min_vars = _min_vars;
		theta_v = _theta_v;
		max_age = _max_age;
		lambda = _lambda;
		init_vars = _init_vars;
	}

	void setH(double h[9]) {
		m_h[0] = h[0];
		m_h[1] = h[1];
		m_h[2] = h[2];
		m_h[3] = h[3];
		m_h[4] = h[4];
		m_h[5] = h[5];
		m_h[6] = h[6];
		m_h[7] = h[7];
		m_h[8] = h[8];

	}

	virtual void operator()(const cv::Range& range) const
	{
		for (int r = range.start; r < range.end; r++) {
			
			//Extract fastMCD Parameters
			int model_width = model_dims.width;
			int model_height = model_dims.height;

			//Get Block XY from Index
			int b_Y = r / (model_width); //Rows
			int b_X = r % (model_width); //Columns

			float* model_mean_row_ptr = model_means->ptr<float>(b_Y);
			float* model_age_row_ptr = model_age->ptr<float>(b_Y);
			float* model_vars_row_ptr = model_vars->ptr<float>(b_Y);

			float* temp_mean_row_ptr = temp_means->ptr<float>(b_Y);
			float* temp_age_row_ptr = temp_age->ptr<float>(b_Y);
			float* temp_vars_row_ptr = temp_vars->ptr<float>(b_Y);

			//Getting Xfrm Blocks -------------------------------------------------------
			

			//Transform From Model to Pixel Space
			float p_Y = b_Y*block_size + (block_size / 2.0);
			float p_X = b_X*block_size + (block_size / 2.0);

			//Transform Coordinates with H Matrix
			float xfrm_W = m_h[6] * p_X + m_h[7] * p_Y + m_h[8];
			float xfrm_X = (m_h[0] * p_X + m_h[1] * p_Y + m_h[2]) / xfrm_W;
			float xfrm_Y = (m_h[3] * p_X + m_h[4] * p_Y + m_h[5]) / xfrm_W;

			//Transform from Pixel Space to Model 
			float xb_Y = xfrm_Y / float(block_size);
			float xb_X = xfrm_X / float(block_size);

			//At this point we want to compare the transformed model point to the center of the pre xfrm model
			int self_Y = std::floor(xb_Y);
			int self_X = std::floor(xb_X);

			float dY = xb_Y - ((float)(self_Y) + 0.5);
			float dX = xb_X - ((float)(self_X) + 0.5);

			
			//Handle Mixing Procedure -------------------------------------------
			float neigh_weights[4] = { 0 };
			float neigh_indices[4] = { 0 };
			float neigh_means[4][2] = { 0 };
			float neigh_ages[4][2] = { 0 };
			float neigh_vars[4][2] = { 0 };
			float weight_sum = 0;
			//We check to see how many neighbors actually exist

			//Width Neighbor (idx = 0)
			if (dX != 0) {
				int neigh_Y = self_Y;
				int neigh_X = self_X;
				neigh_X += dX > 0 ? 1 : -1;
				if (neigh_X >= 0 && neigh_X < model_width && neigh_Y >= 0 && neigh_Y < model_height) {
					neigh_weights[0] = std::fabs(dX) * (1.0 - std::fabs(dY));
					neigh_indices[0] = neigh_X + neigh_Y * model_width;

					float* neigh_mean_row_ptr = model_means->ptr<float>(neigh_Y);
					float* neigh_age_row_ptr = model_age->ptr<float>(neigh_Y);

					//Record the weighted mean and age for this neighbor
					for (int c = 0; c < model_means->channels(); c++) {
						neigh_means[0][c] = neigh_weights[0] * neigh_mean_row_ptr[2 * neigh_X + c];
						neigh_ages[0][c] = neigh_weights[0] * neigh_age_row_ptr[2 * neigh_X + c];
					}
				}
			}
			//Height Neighbor (idx = 1)
			if (dY != 0) {
				int neigh_Y = self_Y;
				int neigh_X = self_X;
				neigh_Y += dY > 0 ? 1 : -1;
				if (neigh_X >= 0 && neigh_X < model_width && neigh_Y >= 0 && neigh_Y < model_height) {
					neigh_weights[1] = std::fabs(dY) * (1.0 - std::fabs(dX));
					neigh_indices[1] = neigh_X + neigh_Y * model_width;

					float* neigh_mean_row_ptr = model_means->ptr<float>(neigh_Y);
					float* neigh_age_row_ptr = model_age->ptr<float>(neigh_Y);

					//Record the weighted mean and age for this neighbor
					for (int c = 0; c < model_means->channels(); c++) {
						neigh_means[1][c] = neigh_weights[1] * neigh_mean_row_ptr[2 * neigh_X + c];
						neigh_ages[1][c] = neigh_weights[1] * neigh_age_row_ptr[2 * neigh_X + c];
					}
				}
			}
			//Diag Neighbor (idx = 2)
			if (dY != 0 && dX != 0) {
				int neigh_Y = self_Y;
				int neigh_X = self_X;
				neigh_Y += dY > 0 ? 1 : -1;
				neigh_X += dX > 0 ? 1 : -1;
				if (neigh_X >= 0 && neigh_X < model_width && neigh_Y >= 0 && neigh_Y < model_height) {
					neigh_weights[2] = std::fabs(dY) * (std::fabs(dX));
					neigh_indices[2] = neigh_X + neigh_Y * model_width;

					float* neigh_mean_row_ptr = model_means->ptr<float>(neigh_Y);
					float* neigh_age_row_ptr = model_age->ptr<float>(neigh_Y);

					//Record the weighted mean and age for this neighbor
					for (int c = 0; c < model_means->channels(); c++) {
						neigh_means[2][c] = neigh_weights[2] * neigh_mean_row_ptr[2 * neigh_X + c];
						neigh_ages[2][c] = neigh_weights[2] * neigh_age_row_ptr[2 * neigh_X + c];
					}
				}
			}
			//Self (idx = 3)
			{
				int neigh_Y = self_Y;
				int neigh_X = self_X;
				if (neigh_X >= 0 && neigh_X < model_width && neigh_Y >= 0 && neigh_Y < model_height) {
					neigh_weights[3] = (1.0 - std::fabs(dY)) * (1.0 - std::fabs(dX));
					neigh_indices[3] = neigh_X + neigh_Y * model_width;

					float* neigh_mean_row_ptr = model_means->ptr<float>(neigh_Y);
					float* neigh_age_row_ptr = model_age->ptr<float>(neigh_Y);

					//Record the weighted mean and age for this neighbor
					for (int c = 0; c < model_means->channels(); c++) {
						neigh_means[3][c] = neigh_weights[3] * neigh_mean_row_ptr[2 * neigh_X + c];
						neigh_ages[3][c] = neigh_weights[3] * neigh_age_row_ptr[2 * neigh_X + c];
					}
				}
			}

			weight_sum = neigh_weights[0] + neigh_weights[1] + neigh_weights[2] + neigh_weights[3];
			

			//If the Xfrmed Model has no neighbors, we short cut and reset the models
			if (weight_sum <= 0) {
				//We set all model variance to the initial var and each model age to 0
				for (int c = 0; c < model_means->channels(); c++) {
					temp_age_row_ptr[2*b_X + c] = 0;
					temp_vars_row_ptr[2*b_X + c] = init_vars;
				}
				continue;
			}
			
			//Add the weighted values of all the neighbors and divide by sum of the weights
			//We do this for each model and for both mean and age
			
			for (int c = 0; c < model_means->channels(); c++) {
				//For both the model and the candidate, we create the temp mean and age
				temp_mean_row_ptr[2*b_X + c] = (neigh_means[0][c] + neigh_means[1][c] + neigh_means[2][c] + neigh_means[3][c]) / weight_sum;
				temp_age_row_ptr[2*b_X + c] = (neigh_ages[0][c] + neigh_ages[1][c] + neigh_ages[2][c] + neigh_ages[3][c]) / weight_sum;
			}
			
			//Since Variance relies on the temp_mean, we now handle the calculated variance mix
			//Width Neighbor (idx = 0)
			if (dX != 0) {
				int neigh_Y = self_Y;
				int neigh_X = self_X;
				neigh_X += dX > 0 ? 1 : -1;
				if (neigh_X >= 0 && neigh_X < model_width && neigh_Y >= 0 && neigh_Y < model_height) {
					float* neigh_vars_row_ptr = model_vars->ptr<float>(neigh_Y);
					float* neigh_means_row_ptr = model_means->ptr<float>(neigh_Y);

					//Record the weighted mean and age for this neighbor
					//neighbor variance + pow( block_mean - neigh_mean                         )2
					neigh_vars[0][0] = neigh_weights[0] * (neigh_vars_row_ptr[2 * neigh_X] + std::pow(temp_mean_row_ptr[2 * b_X] - neigh_means_row_ptr[2 * neigh_X], 2));
					neigh_vars[0][1] = neigh_weights[0] * (neigh_vars_row_ptr[2 * neigh_X + 1] + std::pow(temp_mean_row_ptr[2 * b_X + 1] - neigh_means_row_ptr[2 * neigh_X + 1], 2));
				}
			}
			//Height Neighbor (idx = 1)
			if (dY != 0) {
				int neigh_Y = self_Y;
				int neigh_X = self_X;
				neigh_Y += dY > 0 ? 1 : -1;
				if (neigh_X >= 0 && neigh_X < model_width && neigh_Y >= 0 && neigh_Y < model_height) {
					float* neigh_vars_row_ptr = model_vars->ptr<float>(neigh_Y);
					float* neigh_means_row_ptr = model_means->ptr<float>(neigh_Y);

					//Record the weighted mean and age for this neighbor
					neigh_vars[1][0] = neigh_weights[1] * (neigh_vars_row_ptr[2 * neigh_X] + std::pow(temp_mean_row_ptr[2 * b_X] - neigh_means_row_ptr[2 * neigh_X], 2));
					neigh_vars[1][1] = neigh_weights[1] * (neigh_vars_row_ptr[2 * neigh_X + 1] + std::pow(temp_mean_row_ptr[2 * b_X + 1] - neigh_means_row_ptr[2 * neigh_X + 1], 2));
				}
			}
			//Diag Neighbor (idx = 2)
			if (dY != 0 && dX != 0) {
				int neigh_Y = self_Y;
				int neigh_X = self_X;
				neigh_Y += dY > 0 ? 1 : -1;
				neigh_X += dX > 0 ? 1 : -1;
				if (neigh_X >= 0 && neigh_X < model_width && neigh_Y >= 0 && neigh_Y < model_height) {
					float* neigh_vars_row_ptr = model_vars->ptr<float>(neigh_Y);
					float* neigh_means_row_ptr = model_means->ptr<float>(neigh_Y);

					//Record the weighted mean and age for this neighbor
					neigh_vars[2][0] = neigh_weights[2] * (neigh_vars_row_ptr[2 * neigh_X] + std::pow(temp_mean_row_ptr[2 * b_X] - neigh_means_row_ptr[2 * neigh_X], 2));
					neigh_vars[2][1] = neigh_weights[2] * (neigh_vars_row_ptr[2 * neigh_X + 1] + std::pow(temp_mean_row_ptr[2 * b_X + 1] - neigh_means_row_ptr[2 * neigh_X + 1], 2));
				}
			}
			//Self (idx = 3)
			{
				int neigh_Y = self_Y;
				int neigh_X = self_X;
				if (neigh_X >= 0 && neigh_X < model_width && neigh_Y >= 0 && neigh_Y < model_height) {
					float* neigh_vars_row_ptr = model_vars->ptr<float>(neigh_Y);
					float* neigh_means_row_ptr = model_means->ptr<float>(neigh_Y);

					//Record the weighted mean and age for this neighbor
					neigh_vars[3][0] = neigh_weights[3] * (neigh_vars_row_ptr[2 * neigh_X] + std::pow(temp_mean_row_ptr[2 * b_X] - neigh_means_row_ptr[2 * neigh_X], 2));
					neigh_vars[3][1] = neigh_weights[3] * (neigh_vars_row_ptr[2 * neigh_X + 1] + std::pow(temp_mean_row_ptr[2 * b_X + 1] - neigh_means_row_ptr[2 * neigh_X + 1], 2));
				}
			}

			for (int c = 0; c < model_means->channels(); c++) {
				//For both the model and the candidate, we create the temp vars
				temp_vars_row_ptr[2*b_X + c] = (neigh_vars[0][c] + neigh_vars[1][c] + neigh_vars[2][c] + neigh_vars[3][c]) / weight_sum;
			}


			//Limitations and Exceptions
			//There are two things we care about: 

			//the model must have a minimum variance.  
			//Too small a variance / static a scene and we can get foreground triggers on the slightest intensity shifts
			for (int c = 0; c < model_means->channels(); c++) {
				//Too small a variance / static a scene and we can get foreground triggers on the slightest intensity shifts
				temp_vars_row_ptr[2*b_X + c] = std::max(temp_vars_row_ptr[2*b_X + c], min_vars);

				//Another thing we need to do is reduce the age of the model if the variance is high.  Basically we shouldn't have too much confidence in a model with high variance
				temp_age_row_ptr[2*b_X + c] = temp_age_row_ptr[2*b_X + c] * std::exp(-lambda * std::max(float(0), temp_vars_row_ptr[2*b_X + c] - theta_v));

				//They are peforming the max age gate here but I don't know whether I want to now or later
				temp_age_row_ptr[2*b_X + c] = std::min(float(max_age), temp_age_row_ptr[2*b_X + c]);
			}


			
		}
	}

	ParallelInterpolation& operator=(ParallelInterpolation & new_model) {
		model_dims = new_model.model_dims;
		block_size = new_model.block_size;
		model_means = new_model.model_means;
		model_vars = new_model.model_vars;
		model_age = new_model.model_age;
		temp_means = new_model.temp_means;
		temp_vars = new_model.temp_vars;
		temp_age = new_model.temp_age;
		min_vars = new_model.min_vars;
		theta_v = new_model.theta_v;
		max_age = new_model.max_age;
		lambda = new_model.lambda;
		init_vars = new_model.init_vars;
		return *this;
	}
private:
	cv::Size model_dims;
	int block_size;
	cv::Mat* model_means;
	cv::Mat* model_vars;
	cv::Mat* model_age;
	cv::Mat* temp_means;
	cv::Mat* temp_vars;
	cv::Mat* temp_age;
	double m_h[9];
	float min_vars;
	float max_age;
	float theta_v;
	float lambda;
	float init_vars;
};

class ParallelUpdate : public cv::ParallelLoopBody {
public:
	ParallelUpdate() {
	}

	ParallelUpdate(cv::Size _model_dims, int _block_size, cv::Mat* _model_means, cv::Mat* _model_vars, cv::Mat* _model_age,
		cv::Mat* _temp_means, cv::Mat* _temp_vars, cv::Mat* _temp_age,
		float _min_vars, float _max_age, float _theta_s, float _theta_d, float _init_vars, cv::Mat* _output, cv::Mat* _model_updates) {
		model_dims = _model_dims;
		block_size = _block_size;
		model_means = _model_means;
		model_vars = _model_vars;
		model_age = _model_age;
		temp_means = _temp_means;
		temp_vars = _temp_vars;
		temp_age = _temp_age;
		theta_s = _theta_s;
		max_age = _max_age;
		min_vars = _min_vars;
		max_age = _max_age;
		output = _output;
		theta_d = _theta_d;
		init_vars = _init_vars;
		model_updates = _model_updates;
	}

	void setCurrentFrame(cv::Mat* _frame) {
		frame = _frame;
	}

	void setCurrentIllumination(cv::Scalar* _ill) {
		ill = _ill;
	}

	virtual void operator()(const cv::Range& range) const
	{
		for (int r = range.start; r < range.end; r++) {
			//Extract fastMCD Parameters ------------------------------------------------
			int model_width = model_dims.width;
			int model_height = model_dims.height;

			//Get Block XY from Index
			int b_Y = r / (model_width); //Rows
			int b_X = r % (model_width); //Columns

			float* model_mean_row_ptr = model_means->ptr<float>(b_Y);
			float* model_age_row_ptr = model_age->ptr<float>(b_Y);
			float* model_vars_row_ptr = model_vars->ptr<float>(b_Y);

			float* temp_mean_row_ptr = temp_means->ptr<float>(b_Y);
			float* temp_age_row_ptr = temp_age->ptr<float>(b_Y);
			float* temp_vars_row_ptr = temp_vars->ptr<float>(b_Y);
			
			//Calculate Current Block Mean -------------------------------------------------------
			float M = 0;
			float M_size = 0;
			for (int jj = 0; jj < block_size; jj++) {
				int idx_j = b_Y*block_size + jj;
				for (int ii = 0; ii < block_size; ii++) {
					int idx_i = b_X* block_size + ii;
					if (idx_i < 0 || idx_i >= frame->cols || idx_j < 0 || idx_j >= frame->rows)
						continue;
					uchar* frame_row_ptr = frame->ptr<uchar>(idx_j);
					M += frame_row_ptr[idx_i];
					M_size += 1.0;
				}
			}
			M /= M_size;

			//Perform a Model Swap if Necessary ------------------------------------------------
			//I am not sure if this is the best place to perform a model swap
			//Is there a world where after model mixing we have a candidate with higher age?
			//Mayyyyyybe 
			if (temp_age_row_ptr[2 * b_X] <= temp_age_row_ptr[2 * b_X + 1]) {
				temp_mean_row_ptr[2 * b_X] = temp_mean_row_ptr[2 * b_X + 1];
				temp_vars_row_ptr[2 * b_X] = temp_vars_row_ptr[2 * b_X + 1];
				temp_age_row_ptr[2 * b_X] = temp_age_row_ptr[2 * b_X + 1];

				temp_mean_row_ptr[2 * b_X + 1] = M;
				temp_vars_row_ptr[2 * b_X + 1] = init_vars;
				temp_age_row_ptr[2 * b_X + 1] = 0;
			}



			//Check for a Match to the Model ---------------------------------------------------
			bool model_match = std::pow(M - temp_mean_row_ptr[2 * b_X], 2) < theta_s * temp_vars_row_ptr[2*b_X] ? true : false;
			bool cand_match = std::pow(M - temp_mean_row_ptr[2 * b_X + 1], 2) < theta_s * temp_vars_row_ptr[2*b_X + 1] ? true : false;

			if (model_match) {
			    //float* model_updates_ptr = model_updates->ptr<float>(b_Y);
				//model_updates_ptr[2*b_X] = 255;

				//Calculate Model Mean
				float age = temp_age_row_ptr[2 * b_X];
				float age_weight = age / (age + 1.0);
				if (age < 1.0) {
					model_mean_row_ptr[2 * b_X] = M;
				}
				else {
					model_mean_row_ptr[2 * b_X] = (temp_mean_row_ptr[2 * b_X] + ill->val[0])* age_weight + M*(1.0 - age_weight);
				}

				//Calculate Current Block Variance -----------------------------------------------
				float S = 0;
				for (int jj = 0; jj < block_size; jj++) {
					int idx_j = b_Y*block_size + jj;
					for (int ii = 0; ii < block_size; ii++) {
						int idx_i = b_X* block_size + ii;
						if (idx_i < 0 || idx_i >= frame->cols || idx_j < 0 || idx_j >= frame->rows)
							continue;
						uchar* frame_row_ptr = frame->ptr<uchar>(idx_j);
						S = std::max(float(std::pow(frame_row_ptr[idx_i] - model_mean_row_ptr[2 * b_X], 2.0)), S);
					}
				}
				
				//Update Model Variance
				if (age == 0) {
					model_vars_row_ptr[2 * b_X] = std::max(init_vars,S);
				}
				else {
					model_vars_row_ptr[2*b_X] = std::max(float(temp_vars_row_ptr[2*b_X] * age_weight + S*(1.0 - age_weight)), min_vars);
				}

				model_age_row_ptr[2 * b_X] = std::min(temp_age_row_ptr[2 * b_X] + 1, max_age);

				//Update Candidate Model with Image Interpolated Temps
				model_mean_row_ptr[2 * b_X + 1] = temp_mean_row_ptr[2 * b_X + 1] + ill->val[1];
				model_vars_row_ptr[2 * b_X + 1] = temp_vars_row_ptr[2 * b_X + 1];
				model_age_row_ptr[2 * b_X + 1] = temp_age_row_ptr[2 * b_X + 1];
			}
			else if (!model_match && cand_match) {
				//Calculate Candidate Mean
				float age = temp_age_row_ptr[2 * b_X + 1];
				float age_weight = age / (age + 1.0);
				if (age < 1.0) {
					model_mean_row_ptr[2 * b_X + 1] = M;
				}
				else {
					model_mean_row_ptr[2 * b_X + 1] = (temp_mean_row_ptr[2 * b_X + 1] + ill->val[1])* age_weight + M*(1.0 - age_weight);
				}

				//Calculate Current Block Variance -----------------------------------------------
				float S = 0;
				for (int jj = 0; jj < block_size; jj++) {
					int idx_j = b_Y*block_size + jj;
					for (int ii = 0; ii < block_size; ii++) {
						int idx_i = b_X* block_size + ii;
						if (idx_i < 0 || idx_i >= frame->cols || idx_j < 0 || idx_j >= frame->rows)
							continue;
						uchar* frame_row_ptr = frame->ptr<uchar>(idx_j);
						S = std::max(float(std::pow(frame_row_ptr[idx_i] - model_mean_row_ptr[2 * b_X + 1], 2.0)), S);
					}
				}

				//Update Candidate Variance
				if (age == 0) {
					model_vars_row_ptr[2 * b_X + 1] = std::max(init_vars, S);
				}
				else {
					model_vars_row_ptr[2 * b_X + 1] = std::max(float(temp_vars_row_ptr[2 * b_X + 1] * age_weight + S*(1.0 - age_weight)), min_vars);
				}

				model_age_row_ptr[2 * b_X + 1] = std::min(temp_age_row_ptr[2 * b_X + 1] + 1, max_age);

				//Update Model with Image Interpolated Temps
				model_mean_row_ptr[2 * b_X] = temp_mean_row_ptr[2 * b_X] + ill->val[0];
				model_vars_row_ptr[2 * b_X] = temp_vars_row_ptr[2 * b_X];
				model_age_row_ptr[2 * b_X] = temp_age_row_ptr[2 * b_X];
			}
			else {
				//Reset Candidate Model
				model_mean_row_ptr[2 * b_X + 1] = M;

				//Calculate Current Block Variance -----------------------------------------------
				float S = 0;
				for (int jj = 0; jj < block_size; jj++) {
					int idx_j = b_Y*block_size + jj;
					for (int ii = 0; ii < block_size; ii++) {
						int idx_i = b_X* block_size + ii;
						if (idx_i < 0 || idx_i >= frame->cols || idx_j < 0 || idx_j >= frame->rows)
							continue;
						uchar* frame_row_ptr = frame->ptr<uchar>(idx_j);
						S = std::max(float(std::pow(frame_row_ptr[idx_i] - model_mean_row_ptr[2 * b_X + 1], 2.0)), S);
					}
				}

				model_vars_row_ptr[2*b_X + 1] = std::max(init_vars, S);
				model_age_row_ptr[2*b_X + 1] = 1;

				//Update Model with Image Interpolated Temps
				model_mean_row_ptr[2 * b_X] = temp_mean_row_ptr[2 * b_X] + ill->val[0];
				model_vars_row_ptr[2 * b_X] = temp_vars_row_ptr[2 * b_X];
				model_age_row_ptr[2 * b_X] = temp_age_row_ptr[2 * b_X];
			}
		
			//Perform Detection ----------------------------------------------------------------
			//I am choosing to do detection here instead of another loop because I already have every variable
			//I need here
			for (int jj = 0; jj < block_size; jj++) {
				int idx_j = b_Y*block_size + jj;
				for (int ii = 0; ii < block_size; ii++) {
					int idx_i = b_X*block_size + ii;
					if (idx_i < 0 || idx_i >= frame->cols || idx_j < 0 || idx_j >= frame->rows)
						continue;
					uchar* frame_row_ptr = frame->ptr<uchar>(idx_j);
					uchar* output_row_ptr = output->ptr<uchar>(idx_j);
				
					if (model_age_row_ptr[2 * b_X] > 1){
						output_row_ptr[idx_i] = 255 * int(std::pow(float(frame_row_ptr[idx_i]) - model_mean_row_ptr[2 * b_X], 2) > theta_d * model_vars_row_ptr[2 * b_X]);
					}
					else
						output_row_ptr[idx_i] = 0;
				}
			}
		}
	}

	ParallelUpdate& operator=(ParallelUpdate & new_model) {
		model_dims = new_model.model_dims;
		block_size = new_model.block_size;
		model_means = new_model.model_means;
		model_vars = new_model.model_vars;
		model_age = new_model.model_age;
		temp_means = new_model.temp_means;
		temp_vars = new_model.temp_vars;
		temp_age = new_model.temp_age;
		theta_s = new_model.theta_s;
		theta_d = new_model.theta_d;
		max_age = new_model.max_age;
		min_vars = new_model.min_vars;
		max_age = new_model.max_age;
		output = new_model.output;
		init_vars = new_model.init_vars;
		model_updates = new_model.model_updates;
		return *this;
	}
private:
	cv::Size model_dims;
	int block_size;
	cv::Mat* model_means;
	cv::Mat* model_vars;
	cv::Mat* model_age;
	cv::Mat* temp_means;
	cv::Mat* temp_vars;
	cv::Mat* temp_age;
	float max_age;
	float theta_s;
	float theta_d;
	float min_vars;
	float init_vars;
	cv::Mat* frame;
	cv::Scalar* ill;
	cv::Mat* output;
	cv::Mat* model_updates;
};


//NOTE: In the detection step we will always have '0' for age less than one

class FastMCD {

public:
	KLTWrapper m_LucasKanade;
	ParallelInterpolation parallelInterpolation;
	ParallelUpdate parallelUpdate;
	//Model Variables
	//All model variables are model_width x model_height x 2 where [0] is the current model and [1] is the candidate
	cv::Mat model_means;
	cv::Mat model_vars;
	cv::Mat model_age;
	cv::Mat model_updates;
	cv::Mat temp_means;
	cv::Mat temp_vars;
	cv::Mat temp_age;

	cv::Mat output_mask;
	double m_h[9];

	//Static Parameters
	cv::Size model_dims = cv::Size(4, 4);
	fastMCD_Config m_cfg;

	FastMCD::FastMCD(fastMCD_Config cfg) {
		m_cfg = cfg;
	}

	void blockInterpolation(double h[9]);
	void updateModels(cv::Mat& image);
	void testAccess(cv::Mat& image);


	cv::Mat displayKLT(cv::Mat& image);
	cv::Mat displayMask();
	cv::Mat displayMeans(cv::Mat& image);
	cv::Mat displayVars(cv::Mat& image);
	cv::Mat displayAge(cv::Mat& image);
	void displayUpdates(cv::Mat& image);
	bool init(cv::Mat firstImage);


};
