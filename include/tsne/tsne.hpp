#pragma once

#include "util.hpp"

#include <iostream>
#include <thread>
#include <unordered_map>
#include <vector>


class tsne {
public:
	tsne(const std::unordered_map<uint32_t, double>& P, parameters& p);
	std::vector<std::pair<double, double>> run_tsne();
private:
	const std::unordered_map<uint32_t, double>& P_;
	parameters& PAR;
	std::vector<std::pair<double, double>> compute_attr_forces(std::vector<std::pair<double, double>>& Y);
	std::vector<std::pair<double, double>> compute_grad(std::vector<std::pair<double, double>>& Y);
	std::vector<std::pair<double, double>> sample_initial();
	void center(std::vector<std::pair<double, double>>& Y);
	void iterate(std::vector<std::pair<double, double>>& Y, std::vector<std::pair<double, double>>& dY, 
		std::vector<std::pair<double, double>>& weights, uint32_t t);
};