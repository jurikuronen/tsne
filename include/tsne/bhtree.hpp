#pragma once

#include "util.hpp"

#include <cstdint>
#include <utility>
#include <vector>

struct quad {
	~quad();
	quad(std::pair<double, double> crnr, double wdth, uint32_t lvl) 
	: corner(crnr), width(wdth), level(lvl), NW(nullptr), NE(nullptr), SW(nullptr), SE(nullptr) {};
	const std::pair<double, double> corner; // LL corner
	const double width;
	const uint32_t level;
	quad* NW, *NE, *SW, *SE;
	std::pair<double, double> y_hat;
	uint32_t n_points = 0;
	std::vector<std::pair<double, double>> max_nodes;
	bool contains(std::pair<double, double> y);
	void insert(std::pair<double, double> y, uint32_t max_level);
	std::pair<double, double> compute_rep_force(std::pair<double, double> y, double& Z, double theta, uint32_t max_level);
};

class bhtree {
public:
	~bhtree();
	bhtree(parameters& p);
	void build_bhtree(const std::vector<std::pair<double, double>>& Y);
	std::vector<std::pair<double, double>> compute_rep_forces(const std::vector<std::pair<double, double>>& Y);
private:
	parameters& PAR;
	quad* root_;
};