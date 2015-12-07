#include <tsne/bhtree.hpp>
#include <tsne/tsne.hpp>
#include <tsne/util.hpp>

#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

tsne::tsne(const std::unordered_map<uint32_t, double>& P, parameters& p) : P_(P), PAR(p) {}

/*
	Center the current solution by subtracting column means
*/
void tsne::center(std::vector<std::pair<double, double>>& Y) {
	double mean1 = 0.0, mean2 = 0.0;
	std::vector<std::pair<double, double>> means(PAR.n_threads);
	auto lambda = [&](uint32_t i) {
		for (uint32_t j = i; j < PAR.N; j += PAR.n_threads) {
			means[i].first += Y[j].first; means[i].second += Y[j].second;
		}
	};
	std::vector<std::thread> threads(PAR.n_threads);
	for (uint32_t i = 0; i < PAR.n_threads; ++i) threads[i] = std::thread(lambda, i);
	for (std::thread& thr : threads) thr.join();
	for (auto it : means) mean1 += it.first, mean2 += it.second;
	mean1 /= PAR.N; mean2 /= PAR.N;

	auto lambda2 = [&](uint32_t i) {
		for (; i < PAR.N; i += PAR.n_threads) {
			Y[i].first -= mean1; Y[i].second -= mean2;
		}
	};
	for (uint32_t i = 0; i < PAR.n_threads; ++i) threads[i] = std::thread(lambda2, i);
	for (std::thread& thr : threads) thr.join();
}

/*
	Compute Fattr = sum(j!=i) P_ij * q_ij * Z (y_i - y_j),
	where q_ij * Z = 1 / (1 + ||y_1 - y_j||^2).
*/
std::vector<std::pair<double, double>> tsne::compute_attr_forces(std::vector<std::pair<double, double>>& Y) {
	std::vector<std::pair<double, double>> forces(PAR.N);
	auto lambda = [&](uint32_t i) {
		auto it = P_.begin();
		for (uint32_t j = 0; it != P_.end() && j < i; ++it, ++j);
		while (it != P_.end()) {
			uint32_t y1 = it->first / PAR.N, y2 = it->first % PAR.N;
			double x = PAR.exaggerate * it->second / (1 + compute_dist(Y[y1], Y[y2]));
			forces[y1].first += x * (Y[y1].first - Y[y2].first);
			forces[y1].second += x * (Y[y1].second - Y[y2].second);
			for (uint32_t j = 0; it != P_.end() && j < PAR.n_threads; ++it, ++j);
		}
	};
	std::vector<std::thread> threads(PAR.n_threads);
	for (uint32_t i = 0; i < PAR.n_threads; ++i) threads[i] = std::thread(lambda, i);
	for (std::thread& thr : threads) thr.join();
	return forces;
}

/*
	Computes dC/dy_i = 4 * (Fattr - Frep) 
	= 4 * (sum(j!=i) P_ij * q_ij * Z (y_i - y_j) - sum{j!=i} q^2_ij * Z^2 * (y_i - y_j)),
	where q_ij * Z = 1 / (1 + ||y_1 - y_j||^2).
*/
std::vector<std::pair<double, double>> tsne::compute_grad(std::vector<std::pair<double, double>>& Y) {
	bhtree bh(PAR); bh.build_bhtree(Y);
	std::vector<std::pair<double, double>> grad(PAR.N),
		attr_forces = compute_attr_forces(Y),
		rep_forces = bh.compute_rep_forces(Y);
	auto lambda = [&](uint32_t i) {
		for (; i < PAR.N; i += PAR.n_threads) {
			grad[i].first = 4 * (attr_forces[i].first - rep_forces[i].first);
			grad[i].second = 4 * (attr_forces[i].second - rep_forces[i].second);
		}
	};
	std::vector<std::thread> threads(PAR.n_threads);
	for (uint32_t i = 0; i < PAR.n_threads; ++i) threads[i] = std::thread(lambda, i);
	for (std::thread& thr : threads) thr.join();
	return grad;
}

/*
	Performs one iteration which updates Y_t according to:
		Y_t = Y_{t-1} + eta * grad + momentum * (Y_{t-1} - Y_{t-2})
*/
void tsne::iterate(std::vector<std::pair<double, double>>& Y, std::vector<std::pair<double, double>>& dY, 
		std::vector<std::pair<double, double>>& weights, uint32_t t) {
	std::vector<std::pair<double, double>> grad = compute_grad(Y);
	auto lambda = [&](uint32_t i) {
		for (; i < PAR.N; i += PAR.n_threads) {
			// Adaptive weights updating for the learning rate
			weights[i].first = dY[i].first * grad[i].first > 0 ? weights[i].first + PAR.kappa : std::max(0.01, weights[i].first * PAR.gamma);
			weights[i].second = dY[i].second * grad[i].second > 0 ? weights[i].second + PAR.kappa : std::max(0.01, weights[i].second * PAR.gamma);
			
			// https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum
			dY[i].first = PAR.momentum * dY[i].first + weights[i].first * PAR.learning_rate * grad[i].first;
			dY[i].second = PAR.momentum * dY[i].second + weights[i].second * PAR.learning_rate * grad[i].second;
			
			Y[i].first -= dY[i].first;
			Y[i].second -= dY[i].second;
		}
	};
	std::vector<std::thread> threads(PAR.n_threads);
	for (uint32_t i = 0; i < PAR.n_threads; ++i) threads[i] = std::thread(lambda, i);
	for (std::thread& thr : threads) thr.join();
	center(Y);
}

/*
	
*/
std::vector<std::pair<double, double>> tsne::run_tsne() {
	std::vector<std::pair<double, double>> Y = sample_initial(), dY(PAR.N), weights(PAR.N, {1.0, 1.0});
	auto start = TIME_NOW;
	std::cout << "Running t-SNE...\n";
	for (uint32_t t = 1; t <= PAR.max_iters; ++t) {
		if (t == PAR.stop_exaggerating_iter) PAR.exaggerate = 1.0;
		if (t == PAR.momentum_switch_iter) PAR.momentum = PAR.final_momentum;
		if (t % (PAR.max_iters / 10) == 0) {
			std::cout << "Running iteration " << t << '/' << PAR.max_iters << " -- "; printTime(TIME_TAKEN(start, TIME_NOW));
		}
		iterate(Y, dY, weights, t);
	}
	return Y;
}

/*
	Sample initial solution Y_0 = {{y_11, y_12}, ..., {y_n1, y_n2}} from N(0, 1e-4).
*/
std::vector<std::pair<double, double>> tsne::sample_initial() {
	std::vector<std::pair<double, double>> Y_initial(PAR.N);
	std::mt19937 generator((uint32_t) TIME_NOW.time_since_epoch().count());
	std::normal_distribution<double> distribution(0, 1e-4);
	auto lambda = [&](uint32_t i) {
		for (; i < PAR.N; i += PAR.n_threads) {
			Y_initial[i].first = distribution(generator);
			Y_initial[i].second = distribution(generator);
		}
	};
	std::vector<std::thread> threads(PAR.n_threads);
	for (uint32_t i = 0; i < PAR.n_threads; ++i) threads[i] = std::thread(lambda, i);
	for (std::thread& thr : threads) thr.join();
	center(Y_initial);
	return Y_initial;
}