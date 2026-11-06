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
void write_e1_csv(const std::vector<ExperimentResult>& results,const std::string& path);
}
