#include <tsne/file_writing.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

bool file_writing::writeDefaultConfig() {
	std::ofstream file;
	file.open("tsne.config", std::ios::binary | std::ios::out | std::ios::trunc);
	if (!file.is_open()) return false;
	file << "input_file_name = train-images.idx3-ubyte\n";
	file << "labels_file_name = train-labels.idx1-ubyte\n";
	file << "results_file_name = res.txt\n";
	file << "\n";
	file << "max_tries = 50\n";
	file << "perplexity = 15\n";
	file << "k = 20\n";
	file << "tolerance = 0.0001\n";
	file << "vp_select = 0\n";
	file << "distance_function = 2\n";
	file << "\n";
	file << "max_iters = 1000\n";
	file << "momentum_switch_iter = 250\n";
	file << "stop_exaggerating_iter = 50\n";
	file << "theta = 0.5\n";
	file << "learning_rate = 500\n";
	file << "momentum = 0.5\n";
	file << "final_momentum = 0.8\n";
	file << "exaggerate = 12\n";
	file << "kappa = 0.2\n";
	file << "gamma = 0.8\n";
	file << "bh_max_level = 64\n";
	file << "\n";
	file << "DEBUG = 0\n";
	file.close();
	return true;
}

bool file_writing::writeResults(const std::string& file_name, std::vector<std::pair<double, double>>& Y, 
		std::vector<uint8_t>& labels, uint32_t n_images_) {
	std::ofstream file;
	file.open(file_name, std::ios::binary | std::ios::out | std::ios::trunc);
	if (!file.is_open()) {std::cout << "Problem creating " << file_name << ".\n"; return false;}
	for (uint32_t i = 0; i < n_images_; ++i) {
		file << (uint32_t) labels[i] + 1 << ',' << Y[i].first << ',' << Y[i].second << '\n';
	}
	file.close();
	return true;
}
