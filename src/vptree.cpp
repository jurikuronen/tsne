#include <tsne/util.hpp>
#include <tsne/vptree.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <queue>
#include <stack>
#include <thread>
#include <utility>
#include <vector>

vptree::~vptree() {
	if(root_) delete root_;
}

vpnode::~vpnode() {
	if (left) delete left;
	if (right) delete right;
}

vptree::vptree(const std::vector<double>& data, parameters& p) 
	: data_(data), PAR(p), generator_(std::mt19937((uint32_t) std::chrono::high_resolution_clock::now().time_since_epoch().count())) {}

/*
	Build VP-tree by utilizing X = 1, 2, 4, 8, ..., 2^x threads.
	First builds the tree until we have X nodes. Then gives one node
	to each thread and recursively builds the rest of the tree.
*/
void vptree::build_vptree() {
	std::vector<uint32_t> elements(PAR.N);
	for (uint32_t i = 0; i < PAR.N; elements[i] = i, ++i);
	root_ = initialize_single_node(elements);
	
	uint32_t threads_to_use = 1 << (31 - __builtin_clz(PAR.n_threads)); // Use 2^x threads
	std::vector<std::thread> threads(threads_to_use);
	std::vector<std::pair<vpnode*, std::vector<uint32_t>>> nodes;
	getInstances(root_, elements, nodes, threads_to_use, 1);

	auto lambda = [=](vpnode** node, std::vector<uint32_t>* elems) {continue_single_node(*node, *elems);};
	for (uint32_t i = 0; i < threads_to_use; ++i) {
		threads[i] = std::thread(lambda, &nodes[i].first, &nodes[i].second);
	}
	for (std::thread& thr : threads) thr.join();
}

/*
	Find the k nearest neighbors of query point q.
	http://www.huyng.com/posts/similarity-search-101-with-vantage-point-trees/
*/
std::vector<std::pair<double, uint32_t>> vptree::find_nearest_neighbors(uint32_t q) {
	if (PAR.k >= PAR.N) {std::cerr << "<Too large k in find_nearest_neighbors()>"; throw 1;} // This shouldn't happen
	double alpha = std::numeric_limits<double>::max();
	std::priority_queue<std::pair<std::pair<double, double>, uint32_t>> neighbors;
	std::stack<vpnode*> nodes; nodes.push(root_);
	while (!nodes.empty()) {
		if (PAR.DEBUG) ++PAR.DEBUG_kNN_iters;
		vpnode* node = nodes.top(); nodes.pop();
		std::pair<double, double> dist = compute_dist(q, node->vp);
		if (node->vp != q && dist.first < alpha) {
			neighbors.emplace(dist, node->vp);
			if (neighbors.size() > PAR.k) {
				neighbors.pop();
				alpha = neighbors.top().first.first;
			}
		}
		if (dist.first < node->mu) {
			if (node->left && dist.first < node->mu + alpha) nodes.push(node->left);
			if (node->right && dist.first >= node->mu - alpha) nodes.push(node->right);
		} else {
			if (node->right && dist.first >= node->mu - alpha) nodes.push(node->right);
			if (node->left && dist.first < node->mu + alpha) nodes.push(node->left);
		}
	}
	
	// Read neighbors and distances from the queue to a vector
	std::vector<std::pair<double, uint32_t>> neighbors_vec;
	while (!neighbors.empty()) {
		neighbors_vec.emplace_back(neighbors.top().first.second, neighbors.top().second); // Take the {d^2_ij, id}
		neighbors.pop();
	}
	return neighbors_vec;
}

/*
	Recursively divide until all elements have been added to the tree.
*/
vpnode* vptree::build_vptree(std::vector<uint32_t>& elements) {
	if (elements.size() == 0) return nullptr;
	vpnode* node = new vpnode();
	if (elements.size() == 1) return node;
	
	uint32_t vp_index = select_vp_index(elements);
	node->vp = elements[vp_index];
	std::swap(elements[vp_index], elements[elements.size() - 1]);
	elements.pop_back();
	
	node->mu = compute_mu(elements, node->vp);
	std::vector<uint32_t> left_elements, right_elements;
	divide_elements(node, elements, left_elements, right_elements);

	node->left = build_vptree(left_elements);
	node->right = build_vptree(right_elements);
	return node;
}

/*
	Required for multi-threaded build_vptree.
	This initializes a node but doesn't start recursively creating child nodes.
*/
vpnode* vptree::initialize_single_node(std::vector<uint32_t>& elements) {
	if (elements.size() == 0) return nullptr;
	vpnode* node = new vpnode();
	if (elements.size() == 1) return node;
	
	uint32_t size = elements.size(), vp_index = runif(0, size);
	node->vp = elements[vp_index];
	std::swap(elements[vp_index], elements[size - 1]);
	elements.pop_back();
	
	node->mu = compute_mu(elements, node->vp);
	return node;
}

/*
	Required for multi-threaded build_vptree.
	Continue from initialize_single_node().
*/
void vptree::continue_single_node(vpnode* node, std::vector<uint32_t>& elements) {
	std::vector<uint32_t> left_elements, right_elements;
	divide_elements(node, elements, left_elements, right_elements);
	node->left = build_vptree(left_elements);
	node->right = build_vptree(right_elements);
}

/*
	Required for multi-threaded build_vptree.
	Get the nodes (the instances) that will be given to the threads.
*/
void vptree::getInstances(vpnode* node, std::vector<uint32_t>& elements, 
		std::vector<std::pair<vpnode*, std::vector<uint32_t>>>& nodes, uint32_t threads_to_use, uint32_t level) {
	if (level == threads_to_use) {
		nodes.push_back({node, elements});
		return;
	}
	std::vector<uint32_t> left_elements, right_elements;
	divide_elements(node, elements, left_elements, right_elements);
	vpnode* left_node = initialize_single_node(left_elements);
	vpnode* right_node = initialize_single_node(right_elements);
	node->left = left_node;
	node->right = right_node;
	
	getInstances(left_node, left_elements, nodes, threads_to_use, level * 2);
	getInstances(right_node, right_elements, nodes, threads_to_use, level * 2);
}

/*
	Divide elements of current node into two equal parts according to their
	distance to the vantage point.
*/
void vptree::divide_elements(vpnode* node, const std::vector<uint32_t>& elements,
	std::vector<uint32_t>& left_elems, std::vector<uint32_t>& right_elems) {
	for (auto elem : elements) {
		if (compute_dist(node->vp, elem).first < node->mu) {
			left_elems.push_back(elem);
		} else {
			right_elems.push_back(elem);
		}
	}
}

/*
	Compute d^2_ij = ||x_i - x_j||^2, which is used to compute P.
	Return this value together with the distance measure used by vptree:
		0: Euclidean distance
		1: Squared Euclidean distance
		2: d^4_ij
		3: d^8_ij
		4: d^16_ij
*/
std::pair<double, double> vptree::compute_dist(uint32_t x_i, uint32_t x_j) {
	double dist = 0.0;
	for (uint32_t i = 0; i < PAR.dim; ++i) {
		double x = data_[x_i * PAR.dim + i] - data_[x_j * PAR.dim + i];
		dist += x * x;
	}
	switch (PAR.distance_function) {
		case 0: return {std::sqrt(dist), dist};
		case 1: return {dist, dist};
		case 2: return {dist * dist, dist};
		case 3: return {dist * dist * dist, dist};
		default: return {dist * dist * dist * dist, dist};
	}
}

/*
	Compute the median distance(vp, e) for e in elements.
*/
double vptree::compute_mu(const std::vector<uint32_t>& elements, uint32_t vp) {
	std::vector<double> dists;
	for (auto elem : elements) {
		dists.push_back(compute_dist(vp, elem).first);
	}
	std::sort(dists.begin(), dists.end());
	uint32_t size = dists.size();
	return size % 2 ? dists[size / 2] : (dists[size / 2 - 1] + dists[size / 2]) / 2;
}

/*
	Selects the vantage point according to the chosen method:
		0: Random index among elements
		1: The element with smallest median distance w.r.t. all other nodes
*/
uint32_t vptree::select_vp_index(const std::vector<uint32_t>& elements) {
	switch (PAR.vp_select) {
		case 0:	return runif(0, elements.size());
		case 1: return min_median_node(elements);
		default: return small_median_node(elements);
	}
}

/*
	Returns a random integer in [a, b).
*/
uint32_t vptree::runif(uint32_t a, uint32_t b) {
	std::uniform_int_distribution<uint32_t> distribution(a, b - 1);
	return distribution(generator_);
}

/*
	Returns the index of the node that has the 
	smallest median distance w.r.t. all other nodes.
*/
uint32_t vptree::min_median_node(const std::vector<uint32_t>& elements) {
	double min_median_dist = std::numeric_limits<double>::max();
	uint32_t min_median_node = 0;
	for (uint32_t i = 0; i < (uint32_t) elements.size(); ++i) {
		double dist = compute_mu(elements, i);
		if (dist < min_median_dist) {
			min_median_dist = dist; min_median_node = i;
		}
	}
	return min_median_node;
}

/*
	Try 50 random nodes and return the index of that node
	which had the smallest median distance w.r.t. all other nodes.
*/
uint32_t vptree::small_median_node(const std::vector<uint32_t>& elements) {
	double small_median_dist = std::numeric_limits<double>::max();
	uint32_t small_median_node = 0;
	for (uint32_t i = 0; i < (uint32_t) std::min(50, (int) elements.size()); ++i) {
		uint32_t rand_index = runif(0, elements.size());
		double dist = compute_mu(elements, rand_index);
		if (dist < small_median_dist) {
			small_median_dist = dist; small_median_node = rand_index;
		}
	}
	return small_median_node;
}