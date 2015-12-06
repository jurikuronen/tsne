#pragma once

#include "util.hpp"

#include <fstream>
#include <string>
#include <vector>

class file_reading {
public:
	parameters readConfig();
	bool readData(const std::string& images_file_name, const std::string& labels_file_name,
		std::vector<uint8_t>& data, std::vector<uint8_t>& labels, uint32_t& n_images, uint32_t& dim);
private:
	std::vector<uint8_t> readImages(std::ifstream& file, uint32_t& n_images, uint32_t& dim);
	std::vector<uint8_t> readLabels(std::ifstream& file);
};