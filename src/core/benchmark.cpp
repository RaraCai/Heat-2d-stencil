#include "benchmark.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

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

} // namespace heat
