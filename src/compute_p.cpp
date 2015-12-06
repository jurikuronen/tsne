#include <tsne/compute_p.hpp>
#include <tsne/util.hpp>
#include <tsne/vptree.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <thread>
#include <unordered_map>
#include <vector>

compute_p::compute_p(const std::vector<double>& data, parameters& p) : data_(data), PAR(p) {}
	
/*
	Computes P for high-dimensional data:
	For each data point i, find its PAR.k nearest neighbors and compute
		P_ij = (p_j|i + p_i|j) / (2 * PAR.N),
	where 
		P_j|i = exp(-||x_i - x_j||^2 * tau_i) / sum{k!=l} exp(-||x_k - x_l||^2 * tau_i).
	Otherwise, p_ij = 0.
	The appropriate value of tau_i is found by compute_tau().
*/
std::unordered_map<uint32_t, double> compute_p::compute_P() {
	std::cout << "Computing P-values...\n";
	std::vector<std::unordered_map<uint32_t, double>> P_tmp(PAR.n_threads);

	auto start = TIME_NOW;
	vptree vt(data_, PAR);
	vt.build_vptree();
	auto end = TIME_NOW;
	std::cout << "Built VP-tree in "; printTime(TIME_TAKEN(start, end));
	
	auto lambda = [&](uint32_t i) {
		double tau_i = 1.0; // Initial value of tau -- will be then used as an estimate for next points
		for (uint32_t j = i; j < PAR.N; j += PAR.n_threads) {
			std::vector<std::pair<double, uint32_t>> neighbors = vt.find_nearest_neighbors(j);
			// Prevent numerical errors by limiting tau_i value to 700.0 / max_neighbor_distance
			tau_i = std::fmin(700.0 / neighbors[PAR.k - 1].first, compute_tau(neighbors, tau_i));
			compute_P_i(P_tmp[i], neighbors, j, tau_i);
			if (PAR.DEBUG) {PAR.DEBUG_kNN_max_dist = std::max(PAR.DEBUG_kNN_max_dist, neighbors[PAR.k - 1].first); PAR.DEBUG_kNN_min_dist = std::min(PAR.DEBUG_kNN_min_dist, neighbors[0].first);}
		}
	};
	std::vector<std::thread> threads(PAR.n_threads);
	start = TIME_NOW;
	for (uint32_t i = 0; i < PAR.n_threads; ++i) threads[i] = std::thread(lambda, i);
	for (std::thread& thr : threads) thr.join();
	
	// Combine results
	std::unordered_map<uint32_t, double> P;
	for (uint32_t i = 0; i < PAR.n_threads; ++i) {
		for (auto& p_tmp : P_tmp[i]) P[p_tmp.first] += p_tmp.second;
	}
	end = TIME_NOW;
	std::cout << "P-values computed in "; printTime(TIME_TAKEN(start, end));
	return P;
}

/*
	Binary searches for the value of tau_i (= 1 / sigma_i) that produces P_i with a fixed
	perplexity that is specified by the user.
	Perplexity is defined as
		Perp(P_i) = 2^H(P_i),
	where H(P_i) is the entropy of P_i, defined as
		H(P_i) = -sum{j} (p_j|i log(p_j|i)).
		
	Note: sometimes the neighbors vector has elements with the same distance...
	in that case, sigma_i = 0, so tau_i = 1 / sigma_i = infinite, and compute_H_i
	will eventually divide by zero because of the std::exp()-function. In this case,
	compute_P() will modify tau_i so that it doesn't cause further numerical errors.
*/
double compute_p::compute_tau(const std::vector<std::pair<double, uint32_t>>& neighbors, double tau_i) {
	double tau_min = 0.0, log_perp = std::log(PAR.perplexity), 
		tau_max = std::numeric_limits<double>::max(), diff = compute_H_i(neighbors, tau_i) - log_perp;
	uint32_t tries = 0;
	for (; std::fabs(diff) > PAR.tolerance && ++tries < PAR.max_tries; diff = compute_H_i(neighbors, tau_i) - log_perp) {
		if (diff > 0) {
			tau_min = tau_i;
			tau_i = (tau_max == std::numeric_limits<double>::max() ? tau_i * 2 : (tau_i + tau_max) / 2);
		} else {
			tau_max = tau_i;
			tau_i = (tau_i + tau_min) / 2;
		}
	}
	if (std::fabs(diff) > PAR.tolerance) {std::cerr << "<No convergence in compute_tau()>\n"; throw 1;}
	if (PAR.DEBUG) {PAR.DEBUG_tau_iters += tries; PAR.DEBUG_tau_min = std::min(PAR.DEBUG_tau_min, tau_i); PAR.DEBUG_tau_max = std::max(PAR.DEBUG_tau_max, tau_i);}
	return tau_i;
}

/*
	Let sum_p = sum_{k!=i}(exp{-d^2_ik * tau_i}).
	Computes entropy H_i = -sum{j}(p_j|i * ln(p_j|i)) = -sum{j}(p_j|i * tau_i * d^2_ij) + ln(sum_p).
*/
double compute_p::compute_H_i(const std::vector<std::pair<double, uint32_t>> neighbors, double tau_i) {
	double sum_p = 0.0, H_i = 0.0, p = 0.0;
	for (auto j : neighbors) {
		sum_p += p = std::exp(-j.first * tau_i);
		H_i += p * j.first;
	}
	if (PAR.DEBUG && (sum_p == 0.0 || sum_p != sum_p)) {PAR.DEBUG_H_divzero = true; ++PAR.DEBUG_H_divzero_count; PAR.DEBUG_H_divzero_min_dist = neighbors[0].first; PAR.DEBUG_H_divzero_max_dist = neighbors[PAR.k - 1].first; PAR.DEBUG_H_divzero_tau = tau_i;}
	return tau_i * H_i / sum_p + std::log(sum_p);
}

/*
	Computes p_j|i for all neighbors j and sets p_ij = (p_j|i + p_i|j) / (2 * PAR.N), 
	which guarantees that sum_{j}(p_ij) > 1 / (2 * PAR.N).
*/
void compute_p::compute_P_i(std::unordered_map<uint32_t, double>& P, const std::vector<std::pair<double, uint32_t>>& neighbors, 
	uint32_t i, double tau_i) {
	double sum_p = 0.0;
	for (auto j : neighbors) {
		sum_p += std::exp(-j.first * tau_i);
	}
	if (sum_p == 0.0 || sum_p != sum_p) {std::cerr << "<Divide by zero in compute_P_i()>\n"; throw 1;} // This shouldn't happen
	sum_p *= 2 * PAR.N; // Add normalizing constant for symmetrized conditional probabilities
	for (auto j : neighbors) {
		P[PAR.N * i + j.second] = P[PAR.N * j.second + i] += std::exp(-j.first * tau_i) / sum_p;
	}
}