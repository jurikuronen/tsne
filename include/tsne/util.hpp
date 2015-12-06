#pragma once

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define TIME_NOW (std::chrono::high_resolution_clock::now())
#define TIME_TAKEN(start, end) (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count())

struct parameters;

void printParameters(parameters& PAR);
void printTime(uint32_t T);
void normalize_data(std::vector<double>& data, uint32_t N, uint32_t dim);
void sample(std::vector<double>& data, std::vector<uint8_t>& labels, uint32_t& N, uint32_t dim, uint32_t samples);
double compute_dist(std::pair<double, double> y1, std::pair<double, double> y2);

void DEBUG_good_neighbors(std::vector<double>& data, std::vector<uint8_t>& labels, parameters& PAR);
void DEBUG_P_proper(std::unordered_map<uint32_t, double>& P, parameters& PAR);
void DEBUG_statistics(parameters& PAR);

// Default values plugged in
struct parameters {
	/*
		GENERAL PARAMETERS
	*/
	uint32_t n_threads = 1;											// Number of threads to run
	uint32_t samples;												// Number of samples
	std::string input_file_name = "train-images.idx3-ubyte"; 		// Input file name
	std::string	labels_file_name = "train-labels.idx1-ubyte";		// Labels file name
	std::string	results_file_name = "res.txt";						// File name to store the results in (low dimensional points)
	bool DEBUG = false;												// DEBUG-mode

	/*
		DATA PARAMETERS
	*/
	uint32_t N;
	uint32_t dim;
		
	/*
		PARAMETERS USED TO COMPUTE INPUT SIMILARITIES IN THE HIGH DIMENSION
	*/
	uint32_t max_tries = 50;				// Maximum binary searches for the value tau
	uint32_t perplexity = 15;				// Perplexity that P_i should have (with some value tau)
	uint32_t k = 20;						// Number of neighbors for the k-NN search
	double tolerance = 1e-4;				// Tolerance for the above binary search
	uint32_t vp_select = 0;					// VP-tree vantage point selection method (random, min_median)
	uint32_t distance_function = 2;			// Distance function for vp-trees (d^{2x}_ij)
	
	/*
		PARAMETERS USED BY T-SNE
	*/
	uint32_t max_iters = 1000;				// Number of iterations to run
	uint32_t momentum_switch_iter = 250;	// Iteration number when we switch to final momentum
	uint32_t stop_exaggerating_iter = 50;	// Iteration number when we stop exaggerating P_ij-values
	double theta = 0.5;						// Criterion for when to compute particle-cluster interactions in the BH-tree
	double learning_rate = 500;				// Initial learning rate
	double momentum = 0.5;					// Initial momentum
	double final_momentum = 0.8;			// Final momentum
	double exaggerate = 12;					// The extent to which we exaggerate the P_ij-values
	double kappa = 0.2;						// Used in learning rate adaption (increment)
	double gamma = 0.8;						// Used in learning rate adaption (scale)
	uint32_t bh_max_level = 64;				// Maximum quad level in the BH-tree.
	
	/*
		PARAMETERS STORED BY DEBUG CODE
	*/
	uint32_t DEBUG_tau_iters = 0.0;			// Average iterations for tau binary search
	double DEBUG_tau_max = 0;				// Maximum computed tau_i value
	double DEBUG_tau_min = 9999999.0;		// Minimum computed tau_i value
	bool DEBUG_H_divzero = false;			// True if tau_i caused numeric problems in compute_H_i()
	uint32_t DEBUG_H_divzero_count = 0;		// Number of times this happened.
	double DEBUG_H_divzero_min_dist = 0.0;	// Example value of min dist when this happened
	double DEBUG_H_divzero_max_dist = 0.0;	// Example value of max dist when this happened
	double DEBUG_H_divzero_tau = 0.0;		// Example avlue of tau_i when this happened
	uint32_t DEBUG_kNN_iters = 0.0;			// Average nodes searched to find k-NN
	double DEBUG_kNN_min_dist = 9999999.0;	// Minimum distance of found neighbors
	double DEBUG_kNN_max_dist = 0.0;		// Maximum distance of found neighbors
};