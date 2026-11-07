#pragma once
#include "benchmark.hpp"
#include <string>
#include <vector>

namespace heat {

struct SummaryConfig {
    std::string experiment;
    std::string implementation;
    int iterations;
    int repetitions;
    std::string runtime_metric;
    std::string expected_growth;
};

void print_summary(const std::vector<ExperimentResult>& results,
                   const SummaryConfig& config);
void write_e1_csv(const std::vector<ExperimentResult>& results,
                  const std::string& path);

// E2 output: one summary row per rank configuration.
void print_strong_scaling_summary(const std::vector<MpiExperimentResult>& results,
                                  double serial_baseline_seconds);
std::vector<MpiExperimentResult> read_e2_csv(const std::string& path);
double read_e1_serial_baseline(const std::string& path, int grid_size);
double read_e2_single_rank_baseline(const std::string& path);
void write_e2_runs_csv(const std::vector<MpiExperimentResult>& results,
                       const std::string& path);
void write_e2_csv(const std::vector<MpiExperimentResult>& results,
                  const std::string& path);

void print_weak_scaling_summary(
    const std::vector<MpiExperimentResult>& results);
std::vector<MpiExperimentResult> read_e3_csv(const std::string& path);
double read_e3_single_rank_baseline(const std::string& path);
void write_e3_csv(const std::vector<MpiExperimentResult>& results,
                  const std::string& path);

// E4 output: one summary row per configuration and one raw row per repetition.
void append_e4_csv(const HybridExperimentResult& result,
                   const std::string& path);
void append_e4_runs_csv(const HybridExperimentResult& result,
                        const std::string& path);
std::vector<HybridExperimentResult> read_e4_csv(const std::string& path);
void write_e4_csv(const std::vector<HybridExperimentResult>& results,
                  const std::string& path);
void print_hybrid_summary(const std::vector<HybridExperimentResult>& results);

void append_e5_gpu_csv(const GpuHybridExperimentResult& result,
                       const std::string& path);
void append_e5_gpu_runs_csv(const GpuHybridExperimentResult& result,
                            const std::string& path);
std::vector<GpuHybridExperimentResult> read_e5_gpu_csv(const std::string& path);
void print_e5_gpu_summary(
    const std::vector<GpuHybridExperimentResult>& results);

} // namespace heat
