#include <tsne/compute_p.hpp>
#include <tsne/file_reading.hpp>
#include <tsne/file_writing.hpp>
#include <tsne/tsne.hpp>
#include <tsne/util.hpp>

#include <chrono>
#include <iostream>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
	auto start_program = TIME_NOW;
	
	/*
		FILE READING ROUTINE
	*/
	file_reading fr;
	parameters PAR = fr.readConfig();
	if (PAR.N == 0) return 0; // Signal to main to terminate the program
	
	// if (!validConfig()) return 1;
	
	std::vector<uint8_t> data2, labels;
	if (!fr.readData(PAR.input_file_name, PAR.labels_file_name, data2, labels, PAR.N, PAR.dim)) return 1;
	std::cout << "Read " << PAR.N << " images (d=" << PAR.dim << ")\n";
	
	std::vector<double> data(data2.begin(), data2.end()); // Convert raw data to double
	data2.clear();
	normalize_data(data, PAR.N, PAR.dim); 
	
	/*
		READ SAMPLE SIZE AND THREADS AS COMMAND-LINE ARGUMENTS
	*/
	if (argc >= 2) {
		uint32_t temp = std::atoi(argv[1]);
		if (temp > PAR.k && temp < PAR.N) PAR.samples = temp;
		else PAR.samples = PAR.N;
		if (argc >= 3) {
			temp = std::atoi(argv[2]); 
			if (temp > 1) PAR.n_threads = temp;
		}
	}
	
	if (PAR.samples < PAR.N) {
		sample(data, labels, PAR.N, PAR.dim, PAR.samples);
		std::cout << "Using a sample of " << PAR.N << " images\n";
	}
	std::cout << "Using " << PAR.n_threads << " threads\n\n";
	
	printParameters(PAR);

	/*
		COMPUTE HIGH-DIMENSIONAL PROBABILITIES P
	*/
	compute_p cp(data, PAR);
	std::unordered_map<uint32_t, double> P = cp.compute_P();
	
	if (PAR.DEBUG) {
		DEBUG_statistics(PAR);
		DEBUG_P_proper(P, PAR);
		DEBUG_good_neighbors(data, labels, PAR);
	}
	
	data.clear(); // No longer needed now that we have P
	
	/*
		FIND LOW-DIMENSIONAL REPRESENTATION OF DATA
	*/
	tsne tsne(P, PAR);
	std::vector<std::pair<double, double>> Y = tsne.run_tsne();
	
	file_writing fw;
	fw.writeResults(PAR.results_file_name, Y, labels, PAR.N);
	std::cout << "Wrote results to " << PAR.results_file_name << "\n\n";
	std::cout << "TOTAL TIME TAKEN "; printTime(TIME_TAKEN(start_program, TIME_NOW));
}