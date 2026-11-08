#pragma once
#include "config.hpp"
#include "validation.hpp"
#include <string>
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
    double initialization_time;       // Median
    double initialization_mean;
    double computation_time;          // Median; primary runtime metric
    double computation_mean;
    double computation_stddev;        // Sample standard deviation around the mean
    double validation_time;            // Median
    double validation_mean;
    double total_time;                 // Median
    double total_mean;
    double growth;                     // Ratio of computation medians
    ValidationResult validation;
};

// One repetition: per-phase max-rank timings; halo_ratio is paired with
// the rank that had the longest solver time (not max halo / max solver).
struct MpiRunResult {
    int repetition;
    double initialization;
    double solver;
    double halo;
    double ftcs;
    double validation;
    double total;
    double halo_ratio;
    bool passed;
};

// E2/E3: aggregated result for one MPI process-count configuration.
struct MpiExperimentResult {
    int global_nx;
    int global_ny;
    int ranks;
    int process_rows;
    int process_cols;
    int steps;
    int repetitions;
    double initialization_median = 0.0;
    double initialization_mean = 0.0;
    double solver_mean = 0.0;
    double solver_stddev = 0.0;
    double ftcs_median = 0.0;
    double ftcs_mean = 0.0;
    double halo_mean = 0.0;
    double validation_median = 0.0;
    double validation_mean = 0.0;
    double total_median = 0.0;
    double total_mean = 0.0;
    double halo_ratio_median = 0.0;
    std::vector<MpiRunResult> runs;
    double runtime_median;
    double runtime_min;
    double runtime_max;
    double halo_runtime_median;
    double speedup_mpi;
    double speedup_serial;
    double parallel_efficiency;
    double weak_scaling_efficiency = 0.0;
    ValidationResult validation;
};

// E4: one repetition and one MPI-rank/thread configuration.
struct HybridRunResult {
    int repetition;
    double initialization;
    double solver;
    double halo;
    double ftcs;
    double validation;
    double total;
    double halo_ratio;
    bool passed;
};

struct HybridExperimentResult {
    int total_cores;
    int ranks;
    int threads_per_rank;
    int process_rows;
    int process_cols;
    int steps;
    int repetitions;
    double initialization_median = 0.0;
    double initialization_mean = 0.0;
    double initialization_stddev = 0.0;
    double solver_median = 0.0;
    double solver_mean = 0.0;
    double solver_stddev = 0.0;
    double halo_median = 0.0;
    double halo_mean = 0.0;
    double halo_stddev = 0.0;
    double ftcs_median = 0.0;
    double ftcs_mean = 0.0;
    double ftcs_stddev = 0.0;
    double validation_median = 0.0;
    double validation_mean = 0.0;
    double total_median = 0.0;
    double total_mean = 0.0;
    double halo_ratio_median = 0.0;
    double configuration_speedup = 0.0;
    double hybrid_speedup = 0.0;
    double parallel_efficiency = 0.0;
    ValidationResult validation{};
    std::vector<HybridRunResult> runs;
};

struct TimingStatistics {
    double median = 0.0;
    double mean = 0.0;
    double stddev = 0.0;
};

struct GpuHybridRunResult {
    int repetition = 0;
    double solver_time = 0.0;
    double mpi_halo_time = 0.0;
    double gpu_compute_time = 0.0;
    double boundary_d2h_time = 0.0;
    double halo_h2d_time = 0.0;
    double gpu_setup_time = 0.0;
    double gpu_final_copy_time = 0.0;
    bool passed = false;
};

struct GpuHybridExperimentResult {
    std::string group;
    int total_cpu_cores = 0;
    int mpi_ranks = 0;
    int gpu_device_index = 0;
    int cpu_threads_per_rank = 0;
    int gpu_thread_limit = 0;
    int process_rows = 0;
    int process_cols = 0;
    int steps = 0;
    int repetitions = 0;
    double cpu_solver_median = 0.0;
    double cpu_gpu_speedup = 0.0;
    double within_group_configuration_speedup = 0.0;
    TimingStatistics solver_time;
    TimingStatistics mpi_halo_time;
    TimingStatistics gpu_compute_time;
    TimingStatistics boundary_d2h_time;
    TimingStatistics halo_h2d_time;
    TimingStatistics gpu_setup_time;
    TimingStatistics gpu_final_copy_time;
    ValidationResult validation{};
    std::vector<GpuHybridRunResult> runs;
};

double mean(const std::vector<double>& values);
double median(std::vector<double> values);
double sample_stddev(const std::vector<double>& values, double average);
TimingStatistics summarize_timings(const std::vector<double>& values);
void calculate_growth(std::vector<ExperimentResult>& results);
void calculate_hybrid_metrics(std::vector<HybridExperimentResult>& results);
void calculate_e5_gpu_metrics(std::vector<GpuHybridExperimentResult>& results);

} // namespace heat
