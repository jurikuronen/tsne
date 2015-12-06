#pragma once

#include <fstream>
#include <string>
#include <utility>
#include <vector>

class file_writing {
public:
	bool writeDefaultConfig();
	bool writeResults(const std::string& file_name, std::vector<std::pair<double, double>>& Y, 
		std::vector<uint8_t>& labels, uint32_t n_images_);
};