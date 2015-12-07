#include <tsne/file_reading.hpp>
#include <tsne/file_writing.hpp>
#include <tsne/util.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

parameters file_reading::readConfig() {
	parameters params;
	std::ifstream file;
	file.open("tsne.config", std::ios::binary | std::ios::in);
	if (!file.is_open()) {
		file_writing fw;
		if (!fw.writeDefaultConfig()) {std::cerr << "Problem creating tsne.config.\n"; throw 1;}
		std::cout << "tsne.config not found -- default config was created -- continue (Y/n)?\n";
		std::string cont;
		while (std::cin >> cont) {
			if (cont == "Y") break;
			if (cont == "n") {params.n_threads = 0; return params;}
		}
		file.open("tsne.config", std::ios::binary | std::ios::in);
		if (!file.is_open()) {std::cerr << "Problem opening tsne.config.\n"; throw 1;}
		return params; // Already has default values
	} else {
		std::string read;
		file >> read >> read >> params.input_file_name;
		file >> read >> read >> params.labels_file_name;
		file >> read >> read >> params.results_file_name;
		file >> read >> read >> read;
		params.max_tries = (uint32_t) std::stoi(read);
		file >> read >> read >> read;
		params.perplexity = (uint32_t) std::stoi(read);
		file >> read >> read >> read;
		params.k = (uint32_t) std::stoi(read);
		file >> read >> read >> read;
		params.tolerance = (double) std::stod(read);
		file >> read >> read >> read;
		params.vp_select = (uint32_t) std::stoi(read);
		file >> read >> read >> read;
		params.distance_function = (uint32_t) std::stoi(read);
		file >> read >> read >> read;
		params.max_iters = (uint32_t) std::stoi(read);
		file >> read >> read >> read;
		params.momentum_switch_iter = (uint32_t) std::stoi(read);
		file >> read >> read >> read;
		params.stop_exaggerating_iter = (uint32_t) std::stoi(read);
		file >> read >> read >> read;
		params.theta = (double) std::stod(read);
		file >> read >> read >> read;
		params.learning_rate = (double) std::stod(read);
		file >> read >> read >> read;
		params.momentum = (double) std::stod(read);
		file >> read >> read >> read;
		params.final_momentum = (double) std::stod(read);
		file >> read >> read >> read;
		params.exaggerate = (double) std::stod(read);
		file >> read >> read >> read;
		params.kappa = (double) std::stod(read);
		file >> read >> read >> read;
		params.gamma = (double) std::stod(read);
		file >> read >> read >> read;
		params.bh_max_level = (uint32_t) std::stoi(read);
		file >> read >> read >> read;
		params.DEBUG = (bool) std::stoi(read);
	}
	file.close();
	return params;
}

bool file_reading::readData(const std::string& images_file_name, const std::string& labels_file_name,
	std::vector<uint8_t>& data, std::vector<uint8_t>& labels, uint32_t& n_images, uint32_t& dim) {
	std::ifstream file;
	file.open(images_file_name, std::ios::binary | std::ios::in);
	if (!file.is_open()) {std::cerr << "Problem opening " << images_file_name << ".\n"; return false;}
	try {
		data = readImages(file, n_images, dim);
	} catch (int error) {
		std::cerr << "Problem reading " << images_file_name << ": magic number mismatch.\n";
		file.close();
		return false;
	}
	file.close();
	file.open(labels_file_name, std::ios::binary | std::ios::in);
	if (!file.is_open()) {std::cerr << "Problem opening " << labels_file_name << ".\n"; return false;}
	try {
		labels = readLabels(file);
	} catch (int error) {
		std::cerr << "Problem reading " << labels_file_name << ": magic number mismatch.\n";
		file.close();
		return false;
	}
	file.close();
	return true;
}

std::vector<uint8_t> file_reading::readImages(std::ifstream& file, 
		uint32_t& n_images, uint32_t& dim) {
	uint32_t magic_number, rows, cols;
	file.read((char*) &magic_number, sizeof(magic_number));
	file.read((char*) &n_images, sizeof(n_images));
	file.read((char*) &rows, sizeof(rows));
	file.read((char*) &cols, sizeof(cols));
	if (magic_number != 2051) {
		magic_number = __builtin_bswap32(magic_number);
		if (magic_number != 2051) throw 1;
		n_images = __builtin_bswap32(n_images);
		rows = __builtin_bswap32(rows);
		cols = __builtin_bswap32(cols);
	}
	dim = rows * cols;
	std::vector<uint8_t> data(n_images * dim);
	file.read((char*) data.data(), n_images * dim);
	return data;
}

std::vector<uint8_t> file_reading::readLabels(std::ifstream& file) {
	uint32_t magic_number, items;
	file.read((char*) &magic_number, 4);
	file.read((char*) &items, 4);
	if (magic_number != 2049) {
		magic_number = __builtin_bswap32(magic_number);
		if (magic_number != 2049) throw 1;
		items = __builtin_bswap32(items);
	}
	std::vector<uint8_t> labels(items);
	file.read((char*) labels.data(), items);
	return labels;
}