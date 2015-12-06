#pragma once

#include <cstdint>
#include <random>
#include <utility>
#include <vector>

struct vpnode {
	~vpnode();
	vpnode() : vp(0), mu(0.0), left(nullptr), right(nullptr) {};
	uint32_t vp;			// Vantage point representing this node
	double mu;				// Median distance from vp to all other elements
	vpnode	*	left,		// Elements e such that dist(vp, e) < mu
			*	right;		// Elements e such that dist(vp, e) >= mu
};

class vptree {
public:
	~vptree();
	vptree(const std::vector<double>& data, parameters& p);
	void build_vptree();
	std::vector<std::pair<double, uint32_t>> find_nearest_neighbors(uint32_t q);
private:
	const std::vector<double>& data_;
	parameters& PAR;
	vpnode* root_;
	std::mt19937 generator_;
	vpnode* build_vptree(std::vector<uint32_t>& elements);
	vpnode* initialize_single_node(std::vector<uint32_t>& elements);
	void continue_single_node(vpnode* node, std::vector<uint32_t>& elements);
	void getInstances(vpnode* node, std::vector<uint32_t>& elements, 
		std::vector<std::pair<vpnode*, std::vector<uint32_t>>>& nodes, uint32_t threads_to_use, uint32_t level);
	void divide_elements(vpnode* node, const std::vector<uint32_t>& elements, 
		std::vector<uint32_t>& left_elems, std::vector<uint32_t>& right_elems);
	std::pair<double, double> compute_dist(uint32_t img_i, uint32_t img_j);
	double compute_mu(const std::vector<uint32_t>& elements, uint32_t vp);
	uint32_t select_vp_index(const std::vector<uint32_t>& elements);
	uint32_t runif(uint32_t a, uint32_t b);
	uint32_t min_median_node(const std::vector<uint32_t>& elements);
	uint32_t small_median_node(const std::vector<uint32_t>& elements);
};