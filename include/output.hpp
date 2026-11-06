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

// E2 output: one summary row per rank configuration and one raw row per run.
void print_strong_scaling_summary(const std::vector<MpiExperimentResult>& results,
                                  double serial_baseline_seconds);
void write_e2_csv(const std::vector<MpiExperimentResult>& results,
                  const std::string& path);
void write_e2_runs_csv(const std::vector<MpiExperimentResult>& results,
                       const std::string& path);

} // namespace heat
