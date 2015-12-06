#include <tsne/bhtree.hpp>
#include <tsne/util.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

inline std::pair<double, double> operator+(const std::pair<double, double>& y1, const std::pair<double, double>& y2) {
	return {y1.first + y2.first, y1.second + y2.second};
}

quad::~quad() {
	if (NW) delete NW;
	if (NE) delete NE;
	if (SW) delete SW;
	if (SE) delete SE;
}

bhtree::~bhtree() {
	if(root_) delete root_;
}

bhtree::bhtree(parameters& p) : PAR(p) {}

/*
	If ||y_i - y_hat||^2 / width is less than theta (default=0.5), compute
	the repulsive force as N_points * q^2_ix * Z^2 * (y_i - y_hat),
	where q_ix * Z = 1 + (1 + ||y_1 - y_hat||^2).
	Otherwise, explore deeper cells until we reach max level or a leaf.
	If we reached max level, compute forces individually.
*/
std::pair<double, double> quad::compute_rep_force(std::pair<double, double> y, double& Z, double theta, uint32_t max_level) {
	if (n_points == 0) return {0.0, 0.0};
	double dist = compute_dist(y, y_hat);
	if (n_points == 1 || width / std::sqrt(dist) < theta) {
		double q_hat = 1 / (1 + dist), q2_hat = n_points * q_hat * q_hat;
		Z += n_points * q_hat;
		return {q2_hat * (y.first - y_hat.first), q2_hat * (y.second - y_hat.second)};
	}
	if (level == max_level) {
		std::pair<double, double> forces;
		for (auto it : max_nodes) {
			double q_ij = 1 / (1 + compute_dist(y, it)), q2_ij = q_ij * q_ij;
			Z += q_ij;
			forces.first += q2_ij * (y.first - it.first);
			forces.second += q2_ij * (y.second - it.second);
		}
		return forces;
	}
	return NW->compute_rep_force(y, Z, theta, max_level) + NE->compute_rep_force(y, Z, theta, max_level)
		+ SW->compute_rep_force(y, Z, theta, max_level) + SE->compute_rep_force(y, Z, theta, max_level);
}

bool quad::contains(std::pair<double, double> y) {
	return corner.first <= y.first && y.first < corner.first + width 
		&& corner.second <= y.second && y.second < corner.second + width;
}

void quad::insert(std::pair<double, double> y, uint32_t max_level) {
	// CASE: quad is empty, put y here
	if (n_points == 0) {
		n_points = 1;
		y_hat = y;
		return;
	}
	// CASE: quad is at max level, stop here no matter what
	if (level == max_level) {
		max_nodes.push_back(y);
		++n_points;
		y_hat.first = y_hat.first * (n_points - 1) / n_points + y.first / n_points;
		y_hat.second = y_hat.second * (n_points - 1) / n_points + y.second / n_points;
		return;
	}
	// CASE: quad contains 1 point, subdivide
	if (n_points == 1) {
		std::pair<double, double> y_old = y_hat;
		n_points = 2;
		y_hat = {(y.first + y_old.first) / 2, (y.second + y_old.second) / 2};
		double new_width = width / 2;
		NW = new quad({corner.first, corner.second + new_width}, new_width, level + 1);
		NE = new quad({corner.first + new_width, corner.second + new_width}, new_width, level + 1);
		SW = new quad(corner, new_width, level + 1);
		SE = new quad({corner.first + new_width, corner.second}, new_width, level + 1);
		
		if (NW->contains(y)) NW->insert(y, max_level);
		else if (NE->contains(y)) NE->insert(y, max_level);
		else if (SW->contains(y)) SW->insert(y, max_level);
		else SE->insert(y, max_level);
		
		if (NW->contains(y_old)) NW->insert(y_old, max_level);
		else if (NE->contains(y_old)) NE->insert(y_old, max_level);
		else if (SW->contains(y_old)) SW->insert(y_old, max_level);
		else SE->insert(y_old, max_level);
		return;
	}
	// CASE: quad contains many points, update y_hat and n_points and continue inserting
	++n_points;
	y_hat.first = y_hat.first * (n_points - 1) / n_points + y.first / n_points;
	y_hat.second = y_hat.second * (n_points - 1) / n_points + y.second / n_points;
	if (NW->contains(y)) NW->insert(y, max_level);
	else if (NE->contains(y)) NE->insert(y, max_level);
	else if (SW->contains(y)) SW->insert(y, max_level);
	else SE->insert(y, max_level);
}

void bhtree::build_bhtree(const std::vector<std::pair<double, double>>& Y) {
	// Compute the dimensions of the root quad
	std::pair<double, double> corner{std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
	double width = -std::numeric_limits<double>::max();
	for (auto it : Y) {
		corner.first = std::min(corner.first, it.first);
		corner.second = std::min(corner.second, it.second);
		width = std::max(width, std::max(it.first, it.second));
	}
	width = std::max(width - corner.first, width - corner.second) + 1e-5;
	root_ = new quad(corner, width, 0);
	if (!root_->contains(Y[0])) throw 1; // Sanity check
	for (auto it : Y) root_->insert(it, PAR.bh_max_level);
}

/*
	Compute Z * Frep = sum{j!=i} q^2_ij * Z^2 * (y_i - y_j),
	where q_ij * Z = 1 / (1 + ||y_1 - y_j||^2).
*/
std::vector<std::pair<double, double>> bhtree::compute_rep_forces(const std::vector<std::pair<double, double>>& Y) {
	double Z = 0.0;
	std::vector<std::pair<double, double>> forces(PAR.N);
	std::vector<double> Zs(PAR.n_threads);
	auto lambda = [&](uint32_t i){
		for (uint32_t j = i; j < PAR.N; j += PAR.n_threads) {
			forces[j] = root_->compute_rep_force(Y[j], Zs[i], PAR.theta, PAR.bh_max_level);
		}
	};
	std::vector<std::thread> threads(PAR.n_threads);
	for (uint32_t i = 0; i < PAR.n_threads; ++i) threads[i] = std::thread(lambda, i);
	for (std::thread& thr : threads) thr.join();
	for (double d : Zs) Z += d;
	
	auto lambda2 = [&](uint32_t i) {
		for (; i < PAR.N; i += PAR.n_threads) {
			forces[i].first /= Z;
			forces[i].second /= Z;
		}
	};
	for (uint32_t i = 0; i < PAR.n_threads; ++i) threads[i] = std::thread(lambda2, i);
	for (std::thread& thr : threads) thr.join();
	return forces;
}

