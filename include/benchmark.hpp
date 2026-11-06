#pragma once
#include "config.hpp"
#include "validation.hpp"
#include <vector>
namespace heat {
struct RunResult {
    double initialization_time,computation_time,validation_time,total_time;
    ValidationResult validation;
};
struct ExperimentResult {
    int n;
    NumericalParameters parameters;
    double initialization_time,computation_time,computation_stddev;
    double validation_time,total_time,growth;
    ValidationResult validation;
};
double mean(const std::vector<double>& values);
double sample_stddev(const std::vector<double>& values,double average);
void calculate_growth(std::vector<ExperimentResult>& results);
}
