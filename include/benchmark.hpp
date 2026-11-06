#pragma once
#include "config.hpp"
#include "validation.hpp"
#include <vector>

namespace heat {

struct RunResult {
    double initialization_time;
    double computation_time;
    double validation_time;
    double total_time;
    ValidationResult validation;
};

struct ExperimentResult {
    int n;
    NumericalParameters parameters;
    double initialization_time;
    double computation_time;
    double computation_stddev;
    double validation_time;
    double total_time;
    double growth;
    ValidationResult validation;
};

// E2: one record for each MPI repetition.
struct MpiRunResult {
    int repetition;
    double total_runtime;
    double halo_runtime;
    ValidationResult validation;
};

// E2: aggregated result for one MPI process-count configuration.
struct MpiExperimentResult {
    int global_nx;
    int global_ny;
    int ranks;
    int process_rows;
    int process_cols;
    int steps;
    int repetitions;
    double runtime_median;
    double runtime_min;
    double runtime_max;
    double halo_runtime_median;
    double speedup_mpi;
    double speedup_serial;
    double parallel_efficiency;
    ValidationResult validation;
    std::vector<MpiRunResult> runs;
};

double mean(const std::vector<double>& values);
double median(std::vector<double> values);
double sample_stddev(const std::vector<double>& values, double average);
void calculate_growth(std::vector<ExperimentResult>& results);

} // namespace heat
