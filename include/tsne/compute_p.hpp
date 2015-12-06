#pragma once

#include "util.hpp"

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

class compute_p {
public:
	compute_p(const std::vector<double>& data, parameters& p);
	std::unordered_map<uint32_t, double> compute_P();
private:
	const std::vector<double>& data_;
	parameters& PAR;
	double compute_tau(const std::vector<std::pair<double, uint32_t>>& neighbors, double tau_i);
	double compute_H_i(const std::vector<std::pair<double, uint32_t>> neighbors, double tau_i);
	void compute_P_i(std::unordered_map<uint32_t, double>& P, 
		const std::vector<std::pair<double, uint32_t>>& neighbors, uint32_t i, double tau_i);
};