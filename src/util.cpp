#include <tsne/util.hpp>
#include <tsne/vptree.hpp>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

void printTime(uint32_t T) {
	if (T > 1000 * 60 * 5) std::cout << (T / 1000 / 60) << "min\n";
	else if (T > 5000) std::cout << (T / 1000) << "s\n";
	else std::cout << T << "ms\n";
}

void printParameters(parameters& PAR) {
	if (PAR.DEBUG) std::cout << "-- DEBUG MODE ON --\n\n";
	std::cout << "VP-tree vantage_point_select = ";
	switch (PAR.vp_select) {
		case 0: std::cout << "random"; break;
		case 1: std::cout << "min_median"; break;
		default: std::cout << "small_median";
	}
	std::cout << "\nVP-tree distance_function = ";
	switch (PAR.distance_function) {
		case 0: std::cout << "Euclidean distance"; break;
		case 1: std::cout << "Squared Euclidean distance"; break;
		case 2: std::cout << "(Euclidean distance)^4"; break;
		case 3: std::cout << "(Euclidean distance)^8"; break;
		default: std::cout << "(Euclidean distance)^16";
	}
	std::cout << "\nperplexity: " << PAR.perplexity << '\n' 
				<< "k: " << PAR.k << '\n' 
				<< "theta: " << PAR.theta << '\n';
	std::cout << "learning_rate: " << PAR.learning_rate << '\n'
				<< "exaggerate: " << PAR.exaggerate << '\n';
	std::cout << "momentum: " << PAR.momentum << '\n' 
				<< "final_momentum: " << PAR.final_momentum << '\n';
	std::cout << "momentum_switch_iter: " << PAR.momentum_switch_iter << '\n' 
				<< "stop_exaggerating_iter: " << PAR.stop_exaggerating_iter << "\n\n";
}
/*
	Normalizes data so that all values x are 0 <= x <= 1:
	first subtract the column means and then min-max normalize.
*/
void normalize_data(std::vector<double>& data, uint32_t N, uint32_t dim) {
	std::vector<double> mean(dim);
	for (uint32_t i = 0; i < N; ++i) {
		for (uint32_t j = 0; j < dim; ++j) {
			mean[j] += data[i * dim + j];
		}
	}
	for (double& m : mean) m /= N;
	double max_value = std::numeric_limits<double>::min(), 
		min_value = std::numeric_limits<double>::max();
	for (uint32_t i = 0; i < N; ++i) {
		for (uint32_t j = 0; j < dim; ++j) {
			data[i * dim + j] -= mean[j];
			min_value = std::min(min_value, data[i * dim + j]);
			max_value = std::max(max_value, data[i * dim + j]);
		}
	}
	for (double& d : data) d = (d - min_value) / (max_value - min_value);
}

/*
	Samples the original data to obtain a sample of size 'samples'.
	More technically, shuffles the data and then resizes the original
	vectors to contain only indices 0, ..., samples - 1.
*/
void sample(std::vector<double>& data, std::vector<uint8_t>& labels, uint32_t& N, uint32_t dim, uint32_t samples) {
	std::mt19937 generator((uint32_t) TIME_NOW.time_since_epoch().count());
	for (uint32_t i = 0; i < samples; ++i) {
		std::uniform_int_distribution<int> distribution(i, N - 1);
		uint32_t rand = distribution(generator);
		std::swap_ranges(data.begin() + i * dim, data.begin() + i * dim + dim, data.begin() + rand * dim);
		std::swap(labels[i], labels[rand]);
	}
	data.resize(samples * dim);
	labels.resize(samples);
	N = samples;
}

/*
	Computes the squared Euclidean distance between 
	two low-dimensional points y1 and y2.
*/
double compute_dist(std::pair<double, double> y1, std::pair<double, double> y2) {
	double d1 = y1.first - y2.first, d2 = y1.second - y2.second;
	return d1 * d1 + d2 * d2; 
}

void DEBUG_P_proper(std::unordered_map<uint32_t, double>& P, parameters& PAR) {
	double sum_all = 0.0;
	for (auto it : P) {
		if (P[it.first] != P[(it.first % PAR.N) * PAR.N + it.first / PAR.N]) {
			std::cout << "(DEBUG) P is not symmetric!\n";
			std::cout << "(DEBUG) P[" << it.first << "] != P[(" << (it.first % PAR.N) * PAR.N << " + " << it.first / PAR.N << ")] <=> "
				<< P[it.first] << " != " << P[(it.first % PAR.N) * PAR.N + it.first / PAR.N] << "\n\n";
			throw 1;
		}
		sum_all += it.second;
	}
	if (sum_all > 1.0000001 || sum_all < 0.9999999) {
		std::cout << "(DEBUG) sum(P) = " << sum_all << "!\n\n";
		throw 1;
	}
	std::cout << "(DEBUG) P is symmetric, sum(P) = " << sum_all << " and size(P) = " << P.size() << '\n';
}

void DEBUG_good_neighbors(std::vector<double>& data, std::vector<uint8_t>& labels, parameters& PAR) {
	std::vector<uint32_t> count_alls(PAR.n_threads);
	std::vector<double> avgs(PAR.n_threads);
	vptree vt(data, PAR);
	vt.build_vptree();
	auto lambda = [&](uint32_t i){
		for (uint32_t j = i; j < PAR.N; j += PAR.n_threads) {
			std::vector<std::pair<double, uint32_t>> neighbors = vt.find_nearest_neighbors(j);
			uint32_t count = 0;
			for (auto p : neighbors) {
				if (labels[p.second] == labels[j]) ++count;
			}
			avgs[i] += count; count_alls[i] += count == PAR.k;
		}
	};
	std::vector<std::thread> threads(PAR.n_threads);
	for (uint32_t i = 0; i < PAR.n_threads; ++i) threads[i] = std::thread(lambda, i);
	for (std::thread& thr : threads) thr.join();
	double avg = 0.0;
	uint32_t count_all = 0;
	for (double d : avgs) avg += d;
	for (uint32_t i : count_alls) count_all += i;
	avg /= PAR.N; 
	std::cout << "(DEBUG TEST) avg_correct_neigh = " << avg << " (" << (uint32_t) (100.0 * avg / PAR.k) << "%)\n" 
		<< "(DEBUG TEST) count_100%_correct = " << count_all << " (" << (uint32_t) (100.0 * count_all / PAR.N) << "%)\n\n"; 
}

void DEBUG_statistics(parameters& PAR) {
	std::cout << "(DEBUG) Average iterations to compute tau_i = " 
		<< ((double) PAR.DEBUG_tau_iters / PAR.N) << " / " << PAR.max_tries << '\n';
	std::cout << "(DEBUG) Minimum & maximum computed tau_i values = " 
		<< PAR.DEBUG_tau_min << " & " << PAR.DEBUG_tau_max << '\n';
	std::cout << "(DEBUG) Div by zero in compute_H_i() = " << (PAR.DEBUG_H_divzero ? "YES" : "NO") << '\n';
	if (PAR.DEBUG_H_divzero) {
		std::cout << "-> (DEBUG) Number of div by zeros = " << PAR.DEBUG_H_divzero_count << '\n'
			<< "-> (DEBUG) Example tau & min/max dist = " << PAR.DEBUG_H_divzero_tau << " & " << PAR.DEBUG_H_divzero_max_dist << '/' << PAR.DEBUG_H_divzero_max_dist << '\n';
	}
	std::cout << "(DEBUG) Min/max distance of found neighbors = " << PAR.DEBUG_kNN_min_dist << " / " << PAR.DEBUG_kNN_max_dist << '\n';
	std::cout << "(DEBUG) Average number of nodes searched to find k-NN = " 
		<< ((double) PAR.DEBUG_kNN_iters / PAR.N) << " (" << (uint32_t) (100.0 * PAR.DEBUG_kNN_iters / PAR.N / PAR.N) << "%)\n";
}
