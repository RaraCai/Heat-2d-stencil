#include "benchmark.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace heat {

double mean(const std::vector<double>& values)
{
    if (values.empty()) {
        throw std::invalid_argument("Cannot calculate mean of an empty sample.");
    }
    double sum = 0.0;
    for (double value : values) {
        sum += value;
    }
    return sum / static_cast<double>(values.size());
}

double median(std::vector<double> values)
{
    if (values.empty()) {
        throw std::invalid_argument("Cannot calculate median of an empty sample.");
    }
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 0) {
        return (values[middle - 1] + values[middle]) / 2.0;
    }
    return values[middle];
}

double sample_stddev(const std::vector<double>& values, double average)
{
    if (values.size() < 2) {
        return 0.0;
    }
    double sum = 0.0;
    for (double value : values) {
        const double difference = value - average;
        sum += difference * difference;
    }
    return std::sqrt(sum / static_cast<double>(values.size() - 1));
}

void calculate_growth(std::vector<ExperimentResult>& results)
{
    for (std::size_t i = 1; i < results.size(); ++i) {
        if (results[i - 1].computation_time > 0.0) {
            results[i].growth = results[i].computation_time /
                                results[i - 1].computation_time;
        }
    }
}


void calculate_hybrid_metrics(std::vector<HybridExperimentResult>& results)
{
    for (auto& result : results) {
        const auto config_baseline = std::find_if(
            results.begin(), results.end(), [&](const auto& candidate) {
                return candidate.total_cores == result.total_cores &&
                       candidate.ranks == result.total_cores &&
                       candidate.threads_per_rank == 1;
            });
        if (config_baseline == results.end() || result.solver_median <= 0.0) {
            throw std::runtime_error("Missing MPI-only baseline for E4 configuration.");
        }
        result.configuration_speedup =
            config_baseline->solver_median / result.solver_median;

        auto scaling_baseline = results.end();
        for (auto candidate = results.begin(); candidate != results.end(); ++candidate) {
            if (candidate->threads_per_rank == result.threads_per_rank &&
                (scaling_baseline == results.end() ||
                 candidate->ranks < scaling_baseline->ranks)) {
                scaling_baseline = candidate;
            }
        }
        if (scaling_baseline == results.end() ||
            scaling_baseline->solver_median <= 0.0) {
            throw std::runtime_error("Missing fixed-thread baseline for E4 scaling.");
        }
        result.hybrid_speedup =
            scaling_baseline->solver_median / result.solver_median;
        result.parallel_efficiency = result.hybrid_speedup /
            (static_cast<double>(result.ranks) / scaling_baseline->ranks);
    }
}

} // namespace heat
