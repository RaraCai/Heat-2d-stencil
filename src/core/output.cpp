#include "output.hpp"

#include "config.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace heat {
namespace {

std::vector<std::string> split_csv_row(const std::string& row)
{
    std::vector<std::string> fields;
    std::istringstream input(row);
    std::string field;
    while (std::getline(input, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

std::size_t column_index(const std::vector<std::string>& header,
                         const std::string& name,
                         const std::string& path)
{
    const auto column = std::find(header.begin(), header.end(), name);
    if (column == header.end()) {
        throw std::runtime_error("Missing " + name + " column in CSV: " + path);
    }
    return static_cast<std::size_t>(column - header.begin());
}

double read_csv_value(const std::string& path,
                      const std::string& key_column,
                      const std::string& key_value,
                      const std::string& value_column)
{
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open baseline CSV: " + path);
    }

    std::string line;
    if (!std::getline(file, line)) {
        throw std::runtime_error("Baseline CSV is empty: " + path);
    }
    const auto header = split_csv_row(line);
    const std::size_t key_index = column_index(header, key_column, path);
    const std::size_t value_index = column_index(header, value_column, path);

    while (std::getline(file, line)) {
        const auto fields = split_csv_row(line);
        if (fields.size() <= std::max(key_index, value_index)) {
            throw std::runtime_error("Malformed row in baseline CSV: " + path);
        }
        if (fields[key_index] == key_value) {
            const double value = std::stod(fields[value_index]);
            if (!std::isfinite(value) || value <= 0.0) {
                throw std::runtime_error("Baseline value must be positive in CSV: " + path);
            }
            return value;
        }
    }

    throw std::runtime_error("Baseline row not found in CSV: " + path);
}

} // namespace

void print_summary(const std::vector<ExperimentResult>& results,
                   const SummaryConfig& config)
{
    constexpr char separator[] =
        "=======================================================================";
    constexpr char table_separator[] =
        "-----------------------------------------------------------------------";

    std::ostringstream grid_sizes;
    for (std::size_t i = 0; i < results.size(); ++i) {
        if (i > 0) grid_sizes << ", ";
        grid_sizes << results[i].n;
    }

    bool all_passed = true;
    for (const auto& result : results) {
        all_passed = all_passed && result.validation.passed;
    }

    std::cout << separator << '\n'
              << "  " << config.experiment << " | " << config.implementation << '\n'
              << separator << '\n'
              << "  Configuration\n"
              << "  Grid Sizes       : " << grid_sizes.str() << '\n'
              << "  Iterations       : " << config.iterations << '\n'
              << "  Repetitions      : " << config.repetitions << '\n'
              << "  Runtime Metric   : " << config.runtime_metric << '\n'
              << '\n' << table_separator << '\n'
              << "  " << std::left << std::setw(14) << "Grid"
              << std::right << std::setw(17) << "Comp. Median (s)"
              << std::setw(15) << "Comp. Mean (s)"
              << std::setw(16) << "Std. Dev. (s)"
              << std::setw(12) << "Growth"
              << std::setw(10) << "Status" << '\n'
              << table_separator << '\n'
              << std::fixed << std::setprecision(6);

    for (const auto& result : results) {
        std::ostringstream grid_label;
        grid_label << result.n << " x " << result.n;
        std::cout << "  " << std::left << std::setw(14) << grid_label.str()
                  << std::right << std::setw(17) << result.computation_time
                  << std::setw(15) << result.computation_mean
                  << std::setw(16) << result.computation_stddev;

        if (result.growth > 0.0) {
            std::ostringstream growth;
            growth << std::fixed << std::setprecision(2) << result.growth << 'x';
            std::cout << std::setw(12) << growth.str();
        } else {
            std::cout << std::setw(12) << "-";
        }

        std::cout << std::setw(10)
                  << (result.validation.passed ? "PASS" : "FAIL") << '\n';
    }

    std::cout << table_separator << '\n'
              << "  Validation      : " << (all_passed ? "ALL PASSED" : "FAILED") << '\n'
              << "  Expected Growth : " << config.expected_growth << '\n'
              << separator << '\n';
}

void write_e1_csv(const std::vector<ExperimentResult>& results,
                  const std::string& path)
{
    const std::filesystem::path destination(path);
    if (destination.has_parent_path()) {
        std::filesystem::create_directories(destination.parent_path());
    }

    std::ofstream file(path, std::ios::trunc);
    if (!file) {
        throw std::runtime_error("Cannot open CSV output: " + path);
    }

    file << "grid_size,dx,dy,dt,simulated_time,num_steps,repetitions,"
            "initialization_median_s,initialization_mean_s,"
            "computation_median_s,computation_mean_s,computation_stddev_s,"
            "validation_median_s,validation_mean_s,"
            "total_median_s,total_mean_s,runtime_growth,min_temperature,"
            "max_temperature,max_boundary_error,validation_passed\n";
    file << std::setprecision(12);

    for (const auto& result : results) {
        file << result.n << ','
             << result.parameters.dx << ','
             << result.parameters.dy << ','
             << result.parameters.dt << ','
             << result.parameters.simulated_time << ','
             << kSteps << ',' << kRepeats << ','
             << result.initialization_time << ','
             << result.initialization_mean << ','
             << result.computation_time << ','
             << result.computation_mean << ','
             << result.computation_stddev << ','
             << result.validation_time << ','
             << result.validation_mean << ','
             << result.total_time << ','
             << result.total_mean << ','
             << result.growth << ','
             << result.validation.min_temperature << ','
             << result.validation.max_temperature << ','
             << result.validation.max_boundary_error << ','
             << (result.validation.passed ? 1 : 0) << '\n';
    }
}

void print_strong_scaling_summary(
    const std::vector<MpiExperimentResult>& results,
    double serial_baseline_seconds)
{
    constexpr char separator[] =
        "======================================================================================================================================";
    constexpr char table_separator[] =
        "--------------------------------------------------------------------------------------------------------------------------------------";

    if (results.empty()) {
        throw std::invalid_argument("Cannot print an empty E2 summary.");
    }

    const auto& configuration = results.front();
    bool all_passed = true;
    for (const auto& result : results) {
        all_passed = all_passed && result.validation.passed;
    }

    std::cout << '\n' << separator << '\n'
              << "  E2 | MPI STRONG SCALING\n"
              << separator << '\n'
              << "  Configuration\n"
              << "  Global Grid       : " << configuration.global_nx << " x "
              << configuration.global_ny << '\n'
              << "  Iterations        : " << configuration.steps << '\n'
              << "  Repetitions       : " << configuration.repetitions << '\n'
              << "  Runtime Metric    : Median of max-rank elapsed time (seconds)\n"
              << "  Serial Baseline   : " << std::fixed << std::setprecision(6)
              << serial_baseline_seconds << " s (E1, "
              << configuration.global_nx << " x " << configuration.global_ny
              << ")\n\n"
              << table_separator << '\n'
              << "  " << std::left << std::setw(9) << "Ranks"
              << std::setw(13) << "Process Grid"
              << std::right << std::setw(16) << "Median (s)"
              << std::setw(15) << "Halo (s)"
               << std::setw(15) << "FTCS (s)"
               << std::setw(14) << "Halo %"
              << std::setw(15) << "MPI Speedup"
              << std::setw(16) << "Serial Speedup"
              << std::setw(15) << "Efficiency"
              << std::setw(10) << "Status" << '\n'
              << table_separator << '\n'
              << std::fixed << std::setprecision(3);

    for (const auto& result : results) {
        std::ostringstream process_grid;
        process_grid << result.process_rows << 'x' << result.process_cols;
        std::ostringstream efficiency;
        efficiency << std::fixed << std::setprecision(2)
                   << result.parallel_efficiency * 100.0 << '%';
        std::cout << "  " << std::left << std::setw(9) << result.ranks
                  << std::setw(13) << process_grid.str()
                  << std::right << std::setw(16) << result.runtime_median
                  << std::setw(15) << result.halo_runtime_median
                   << std::setw(15) << result.ftcs_median
                   << std::setw(13) << result.halo_ratio_median * 100.0 << '%'
                  << std::setw(15) << result.speedup_mpi
                  << std::setw(16) << result.speedup_serial
                  << std::setw(15) << efficiency.str()
                  << std::setw(10)
                  << (result.validation.passed ? "PASS" : "FAIL") << '\n';
    }

    std::cout << table_separator << '\n'
              << "  Validation      : " << (all_passed ? "ALL PASSED" : "FAILED") << '\n'
              << "  Results CSV     : " << kE2OutputCsv << '\n'
              << separator << "\n\n";
}

void write_e2_csv(const std::vector<MpiExperimentResult>& results,
                  const std::string& path)
{
    const std::filesystem::path destination(path);
    if (destination.has_parent_path()) {
        std::filesystem::create_directories(destination.parent_path());
    }

    bool write_header = true;
    if (std::filesystem::exists(destination)) {
        std::ifstream existing(path);
        if (!existing) {
            throw std::runtime_error("Cannot read existing E2 CSV: " + path);
        }
        write_header = existing.peek() == std::ifstream::traits_type::eof();
        std::string line;
        if (!write_header) {
            std::getline(existing, line);
            const auto header = split_csv_row(line);
            if (std::find(header.begin(), header.end(), "initialization_median_s") == header.end())
                throw std::runtime_error("Existing E2 CSV has old schema; archive it before a new run: " + path);
            const std::size_t ranks_column = column_index(header, "ranks", path);
            while (std::getline(existing, line)) {
                const auto fields = split_csv_row(line);
                if (fields.size() <= ranks_column) {
                    throw std::runtime_error("Malformed row in existing E2 CSV: " + path);
                }
                for (const auto& result : results) {
                    if (fields[ranks_column] == std::to_string(result.ranks)) {
                        throw std::runtime_error(
                            "E2 CSV already contains results for " +
                            std::to_string(result.ranks) + " ranks: " + path);
                    }
                }
            }
        }
    }

    std::ofstream file(path, std::ios::app);
    if (!file) {
        throw std::runtime_error("Cannot open E2 CSV output: " + path);
    }

    if (write_header) {
        file << "global_nx,global_ny,ranks,process_rows,process_cols,steps,repetitions,"
                "runtime_median_s,runtime_min_s,runtime_max_s,halo_runtime_median_s,"
                 "initialization_median_s,initialization_mean_s,solver_mean_s,solver_stddev_s,"
                 "ftcs_median_s,ftcs_mean_s,halo_mean_s,validation_median_s,validation_mean_s,"
                 "total_median_s,total_mean_s,halo_ratio_median,"
                "speedup_mpi,speedup_serial,parallel_efficiency,"
                "min_temperature,max_temperature,max_boundary_error,validation_passed\n";
    }
    file << std::setprecision(12);

    for (const auto& result : results) {
        file << result.global_nx << ',' << result.global_ny << ','
             << result.ranks << ',' << result.process_rows << ','
             << result.process_cols << ',' << result.steps << ','
             << result.repetitions << ',' << result.runtime_median << ','
             << result.runtime_min << ',' << result.runtime_max << ','
             << result.halo_runtime_median << ','
             << result.initialization_median << ',' << result.initialization_mean << ','
             << result.solver_mean << ',' << result.solver_stddev << ','
             << result.ftcs_median << ',' << result.ftcs_mean << ','
             << result.halo_mean << ',' << result.validation_median << ','
             << result.validation_mean << ',' << result.total_median << ','
             << result.total_mean << ',' << result.halo_ratio_median << ','
             << result.speedup_mpi << ','
             << result.speedup_serial << ',' << result.parallel_efficiency << ','
             << result.validation.min_temperature << ','
             << result.validation.max_temperature << ','
             << result.validation.max_boundary_error << ','
             << (result.validation.passed ? 1 : 0) << '\n';
    }
}

double read_e1_serial_baseline(const std::string& path, int grid_size)
{
    return read_csv_value(path, "grid_size", std::to_string(grid_size),
                          "computation_median_s");
}

std::vector<MpiExperimentResult> read_e2_csv(const std::string& path)
{
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open E2 results CSV: " + path);
    }

    std::string line;
    if (!std::getline(file, line)) {
        throw std::runtime_error("E2 results CSV is empty: " + path);
    }
    const auto header = split_csv_row(line);
    const auto index = [&](const std::string& name) {
        return column_index(header, name, path);
    };
    const std::size_t global_nx = index("global_nx");
    const std::size_t global_ny = index("global_ny");
    const std::size_t ranks = index("ranks");
    const std::size_t process_rows = index("process_rows");
    const std::size_t process_cols = index("process_cols");
    const std::size_t steps = index("steps");
    const std::size_t repetitions = index("repetitions");
    const std::size_t runtime_median = index("runtime_median_s");
    const std::size_t runtime_min = index("runtime_min_s");
    const std::size_t runtime_max = index("runtime_max_s");
    const std::size_t halo_runtime_median = index("halo_runtime_median_s");
    const std::size_t speedup_mpi = index("speedup_mpi");
    const std::size_t speedup_serial = index("speedup_serial");
    const std::size_t parallel_efficiency = index("parallel_efficiency");
    const std::size_t min_temperature = index("min_temperature");
    const std::size_t max_temperature = index("max_temperature");
    const std::size_t max_boundary_error = index("max_boundary_error");
    const std::size_t validation_passed = index("validation_passed");
    const std::size_t required_fields = std::max({
        global_nx, global_ny, ranks, process_rows, process_cols, steps,
        repetitions, runtime_median, runtime_min, runtime_max,
        halo_runtime_median, speedup_mpi, speedup_serial,
        parallel_efficiency, min_temperature, max_temperature,
        max_boundary_error, validation_passed
    }) + 1;

    std::vector<MpiExperimentResult> results;
    while (std::getline(file, line)) {
        const auto fields = split_csv_row(line);
        if (fields.size() < required_fields) {
            throw std::runtime_error("Malformed row in E2 results CSV: " + path);
        }
        MpiExperimentResult result{};
        result.global_nx = std::stoi(fields[global_nx]);
        result.global_ny = std::stoi(fields[global_ny]);
        result.ranks = std::stoi(fields[ranks]);
        result.process_rows = std::stoi(fields[process_rows]);
        result.process_cols = std::stoi(fields[process_cols]);
        result.steps = std::stoi(fields[steps]);
        result.repetitions = std::stoi(fields[repetitions]);
        const auto optional_number = [&](const char* name) {
            const auto found = std::find(header.begin(), header.end(), name);
            return found == header.end() ? 0.0 :
                std::stod(fields[static_cast<std::size_t>(found - header.begin())]);
        };
        result.initialization_median = optional_number("initialization_median_s");
        result.initialization_mean = optional_number("initialization_mean_s");
        result.solver_mean = optional_number("solver_mean_s");
        result.solver_stddev = optional_number("solver_stddev_s");
        result.ftcs_median = optional_number("ftcs_median_s");
        result.ftcs_mean = optional_number("ftcs_mean_s");
        result.halo_mean = optional_number("halo_mean_s");
        result.validation_median = optional_number("validation_median_s");
        result.validation_mean = optional_number("validation_mean_s");
        result.total_median = optional_number("total_median_s");
        result.total_mean = optional_number("total_mean_s");
        result.halo_ratio_median = optional_number("halo_ratio_median");
        result.runtime_median = std::stod(fields[runtime_median]);
        result.runtime_min = std::stod(fields[runtime_min]);
        result.runtime_max = std::stod(fields[runtime_max]);
        result.halo_runtime_median = std::stod(fields[halo_runtime_median]);
        result.speedup_mpi = std::stod(fields[speedup_mpi]);
        result.speedup_serial = std::stod(fields[speedup_serial]);
        result.parallel_efficiency = std::stod(fields[parallel_efficiency]);
        result.validation.min_temperature = std::stod(fields[min_temperature]);
        result.validation.max_temperature = std::stod(fields[max_temperature]);
        result.validation.max_boundary_error = std::stod(fields[max_boundary_error]);
        result.validation.passed = fields[validation_passed] == "1";
        results.push_back(result);
    }
    if (results.empty()) {
        throw std::runtime_error("E2 results CSV has no data rows: " + path);
    }
    std::sort(results.begin(), results.end(),
              [](const auto& lhs, const auto& rhs) {
                  return lhs.ranks < rhs.ranks;
              });
    return results;
}

double read_e2_single_rank_baseline(const std::string& path)
{
    return read_csv_value(path, "ranks", "1", "runtime_median_s");
}

void print_weak_scaling_summary(
    const std::vector<MpiExperimentResult>& results)
{
    constexpr char separator[] =
        "==============================================================================================================";
    constexpr char table_separator[] =
        "--------------------------------------------------------------------------------------------------------------";

    if (results.empty()) {
        throw std::invalid_argument("Cannot print an empty E3 summary.");
    }

    const auto& configuration = results.front();
    const auto largest_grid = std::max_element(
        results.begin(), results.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.global_nx < rhs.global_nx;
        });
    bool all_passed = true;
    for (const auto& result : results) {
        all_passed = all_passed && result.validation.passed;
    }

    std::cout << '\n' << separator << '\n'
              << "  E3 | MPI WEAK SCALING\n"
              << separator << '\n'
              << "  Configuration\n"
              << "  Local Grid / Rank : " << kE3LocalNx << " x "
              << kE3LocalNy << '\n'
              << "  Global Grid Range : " << kE3LocalNx << " x " << kE3LocalNy
              << " to " << largest_grid->global_nx << " x "
              << largest_grid->global_ny << '\n'
              << "  Spatial Steps     : dx = " << std::scientific
              << std::setprecision(6) << kDomainLength / (kE3LocalNx - 1)
              << ", dy = " << kDomainLength / (kE3LocalNy - 1)
              << ", dt = " << kTheta /
                    (2.0 * kThermalDiffusivity *
                     (std::pow((kE3LocalNx - 1) / kDomainLength, 2) +
                      std::pow((kE3LocalNy - 1) / kDomainLength, 2)))
              << '\n'
              << "  Iterations        : " << configuration.steps << '\n'
              << "  Repetitions       : " << configuration.repetitions << '\n'
              << "  Runtime Metric    : Median of max-rank elapsed time (seconds)\n\n"
              << table_separator << '\n'
              << "  " << std::left << std::setw(9) << "Ranks"
              << std::setw(13) << "Process Grid"
              << std::setw(18) << "Global Grid"
              << std::setw(16) << "Local / Rank"
              << std::right << std::setw(16) << "Median (s)"
              << std::setw(15) << "Halo (s)"
              << std::setw(20) << "Weak Efficiency"
              << std::setw(10) << "Status" << '\n'
              << table_separator << '\n'
              << std::fixed << std::setprecision(3);

    for (const auto& result : results) {
        std::ostringstream process_grid;
        process_grid << result.process_rows << 'x' << result.process_cols;
        std::ostringstream global_grid;
        global_grid << result.global_nx << 'x' << result.global_ny;
        std::ostringstream local_grid;
        local_grid << result.global_nx / result.process_rows << 'x'
                   << result.global_ny / result.process_cols;
        std::ostringstream efficiency;
        efficiency << std::fixed << std::setprecision(2)
                   << result.weak_scaling_efficiency * 100.0 << '%';

        std::cout << "  " << std::left << std::setw(9) << result.ranks
                  << std::setw(13) << process_grid.str()
                  << std::setw(18) << global_grid.str()
                  << std::setw(16) << local_grid.str()
                  << std::right << std::setw(16) << result.runtime_median
                  << std::setw(15) << result.halo_runtime_median
                  << std::setw(20) << efficiency.str()
                  << std::setw(10)
                  << (result.validation.passed ? "PASS" : "FAIL") << '\n';
    }

    std::cout << table_separator << '\n'
              << "  Validation      : " << (all_passed ? "ALL PASSED" : "FAILED") << '\n'
              << "  Expected        : ~100% weak-scaling efficiency\n"
              << "  Results CSV     : " << kE3OutputCsv << '\n'
              << separator << "\n\n";
}

void write_e3_csv(const std::vector<MpiExperimentResult>& results,
                  const std::string& path)
{
    const std::filesystem::path destination(path);
    if (destination.has_parent_path()) {
        std::filesystem::create_directories(destination.parent_path());
    }

    bool write_header = true;
    if (std::filesystem::exists(destination)) {
        std::ifstream existing(path);
        if (!existing) {
            throw std::runtime_error("Cannot read existing E3 CSV: " + path);
        }
        write_header = existing.peek() == std::ifstream::traits_type::eof();
        std::string line;
        if (!write_header) {
            std::getline(existing, line);
            const auto header = split_csv_row(line);
            const std::size_t ranks_column = column_index(header, "ranks", path);
            column_index(header, "weak_scaling_efficiency", path);
            while (std::getline(existing, line)) {
                const auto fields = split_csv_row(line);
                if (fields.size() <= ranks_column) {
                    throw std::runtime_error("Malformed row in existing E3 CSV: " + path);
                }
                for (const auto& result : results) {
                    if (fields[ranks_column] == std::to_string(result.ranks)) {
                        throw std::runtime_error(
                            "E3 CSV already contains results for " +
                            std::to_string(result.ranks) + " ranks: " + path);
                    }
                }
            }
        }
    }

    std::ofstream file(path, std::ios::app);
    if (!file) {
        throw std::runtime_error("Cannot open E3 CSV output: " + path);
    }
    if (write_header) {
        file << "global_nx,global_ny,ranks,process_rows,process_cols,local_nx,local_ny,"
                "dx,dy,dt,steps,repetitions,runtime_median_s,runtime_min_s,"
                "runtime_max_s,halo_runtime_median_s,initialization_median_s,"
                "solver_mean_s,solver_stddev_s,ftcs_median_s,halo_mean_s,"
                "validation_median_s,total_median_s,weak_scaling_efficiency,"
                "min_temperature,max_temperature,max_boundary_error,validation_passed\n";
    }
    file << std::setprecision(12);
    for (const auto& result : results) {
        const double dx = kDomainLength / (kE3LocalNx - 1);
        const double dy = kDomainLength / (kE3LocalNy - 1);
        const double dt = kTheta /
            (2.0 * kThermalDiffusivity * (1.0 / (dx * dx) + 1.0 / (dy * dy)));
        file << result.global_nx << ',' << result.global_ny << ','
             << result.ranks << ',' << result.process_rows << ','
             << result.process_cols << ','
             << result.global_nx / result.process_rows << ','
             << result.global_ny / result.process_cols << ','
             << dx << ',' << dy << ',' << dt << ','
             << result.steps << ',' << result.repetitions << ','
             << result.runtime_median << ',' << result.runtime_min << ','
             << result.runtime_max << ',' << result.halo_runtime_median << ','
             << result.initialization_median << ',' << result.solver_mean << ','
             << result.solver_stddev << ',' << result.ftcs_median << ','
             << result.halo_mean << ',' << result.validation_median << ','
             << result.total_median << ',' << result.weak_scaling_efficiency << ','
             << result.validation.min_temperature << ','
             << result.validation.max_temperature << ','
             << result.validation.max_boundary_error << ','
             << (result.validation.passed ? 1 : 0) << '\n';
    }
}

std::vector<MpiExperimentResult> read_e3_csv(const std::string& path)
{
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open E3 results CSV: " + path);
    }
    std::string line;
    if (!std::getline(file, line)) {
        throw std::runtime_error("E3 results CSV is empty: " + path);
    }
    const auto header = split_csv_row(line);
    const auto index = [&](const std::string& name) {
        return column_index(header, name, path);
    };
    const std::size_t global_nx = index("global_nx");
    const std::size_t global_ny = index("global_ny");
    const std::size_t ranks = index("ranks");
    const std::size_t process_rows = index("process_rows");
    const std::size_t process_cols = index("process_cols");
    const std::size_t steps = index("steps");
    const std::size_t repetitions = index("repetitions");
    const std::size_t runtime = index("runtime_median_s");
    const std::size_t runtime_min = index("runtime_min_s");
    const std::size_t runtime_max = index("runtime_max_s");
    const std::size_t halo_runtime = index("halo_runtime_median_s");
    const std::size_t efficiency = index("weak_scaling_efficiency");
    const std::size_t minimum = index("min_temperature");
    const std::size_t maximum = index("max_temperature");
    const std::size_t boundary_error = index("max_boundary_error");
    const std::size_t validation = index("validation_passed");
    const std::size_t required_fields = std::max({
        global_nx, global_ny, ranks, process_rows, process_cols, steps,
        repetitions, runtime, runtime_min, runtime_max, halo_runtime,
        efficiency, minimum, maximum, boundary_error, validation
    }) + 1;

    std::vector<MpiExperimentResult> results;
    while (std::getline(file, line)) {
        const auto fields = split_csv_row(line);
        if (fields.size() < required_fields) {
            throw std::runtime_error("Malformed row in E3 results CSV: " + path);
        }
        MpiExperimentResult result{};
        result.global_nx = std::stoi(fields[global_nx]);
        result.global_ny = std::stoi(fields[global_ny]);
        result.ranks = std::stoi(fields[ranks]);
        result.process_rows = std::stoi(fields[process_rows]);
        result.process_cols = std::stoi(fields[process_cols]);
        result.steps = std::stoi(fields[steps]);
        result.repetitions = std::stoi(fields[repetitions]);
        result.runtime_median = std::stod(fields[runtime]);
        result.runtime_min = std::stod(fields[runtime_min]);
        result.runtime_max = std::stod(fields[runtime_max]);
        result.halo_runtime_median = std::stod(fields[halo_runtime]);
        result.weak_scaling_efficiency = std::stod(fields[efficiency]);
        result.validation.min_temperature = std::stod(fields[minimum]);
        result.validation.max_temperature = std::stod(fields[maximum]);
        result.validation.max_boundary_error = std::stod(fields[boundary_error]);
        result.validation.passed = fields[validation] == "1";
        results.push_back(result);
    }
    if (results.empty()) {
        throw std::runtime_error("E3 results CSV has no data rows: " + path);
    }
    std::sort(results.begin(), results.end(),
              [](const auto& lhs, const auto& rhs) {
                  return lhs.ranks < rhs.ranks;
              });
    return results;
}

double read_e3_single_rank_baseline(const std::string& path)
{
    return read_csv_value(path, "ranks", "1", "runtime_median_s");
}

} // namespace heat

namespace heat {
namespace {

void write_timing_columns(std::ostream& file, const TimingStatistics& timing)
{
    file << timing.median << ',' << timing.mean << ',' << timing.stddev;
}

void write_e5_gpu_header(std::ostream& file)
{
    file << "group,total_cpu_cores,gpu_count,node_count,mpi_ranks,ranks_per_node,"
            "gpu_device_index,cpu_threads_per_rank,gpu_thread_limit,"
            "process_rows,process_cols,"
            "steps,repetitions,cpu_solver_median_s,cpu_gpu_speedup,"
            "within_group_configuration_speedup,cross_group_gpu_scaling_speedup,"
            "solver_median_s,solver_mean_s,solver_stddev_s,"
            "mpi_halo_median_s,mpi_halo_mean_s,mpi_halo_stddev_s,"
            "gpu_compute_median_s,gpu_compute_mean_s,gpu_compute_stddev_s,"
            "boundary_d2h_median_s,boundary_d2h_mean_s,boundary_d2h_stddev_s,"
            "halo_h2d_median_s,halo_h2d_mean_s,halo_h2d_stddev_s,"
            "gpu_setup_median_s,gpu_setup_mean_s,gpu_setup_stddev_s,"
            "gpu_final_copy_median_s,gpu_final_copy_mean_s,gpu_final_copy_stddev_s,"
            "max_absolute_error,relative_l2_error,min_temperature,max_temperature,"
            "max_boundary_error,validation_passed\n";
}

void write_e5_gpu_row(std::ostream& file,
                      const GpuHybridExperimentResult& result)
{
    file << result.group << ',' << result.total_cpu_cores << ','
         << result.gpu_count << ',' << result.node_count << ','
         << result.mpi_ranks << ',' << result.ranks_per_node << ','
         << result.gpu_device_index << ',' << result.cpu_threads_per_rank << ','
         << result.gpu_thread_limit << ','
         << result.process_rows << ',' << result.process_cols << ','
         << result.steps << ',' << result.repetitions << ','
         << result.cpu_solver_median << ',' << result.cpu_gpu_speedup << ','
         << result.within_group_configuration_speedup << ','
         << result.cross_group_gpu_scaling_speedup << ',';
    write_timing_columns(file, result.solver_time);
    file << ',';
    write_timing_columns(file, result.mpi_halo_time);
    file << ',';
    write_timing_columns(file, result.gpu_compute_time);
    file << ',';
    write_timing_columns(file, result.boundary_d2h_time);
    file << ',';
    write_timing_columns(file, result.halo_h2d_time);
    file << ',';
    write_timing_columns(file, result.gpu_setup_time);
    file << ',';
    write_timing_columns(file, result.gpu_final_copy_time);
    file << ',' << result.max_absolute_error << ',' << result.relative_l2_error
         << ',' << result.validation.min_temperature << ','
         << result.validation.max_temperature << ','
         << result.validation.max_boundary_error << ','
         << (result.validation.passed ? 1 : 0) << '\n';
}

TimingStatistics read_timing(const std::vector<std::string>& fields,
                             const std::vector<std::string>& header,
                             const std::string& prefix,
                             const std::string& path)
{
    return {
        std::stod(fields[column_index(header, prefix + "_median_s", path)]),
        std::stod(fields[column_index(header, prefix + "_mean_s", path)]),
        std::stod(fields[column_index(header, prefix + "_stddev_s", path)])
    };
}

} // namespace

void append_e5_gpu_csv(const GpuHybridExperimentResult& result,
                       const std::string& path)
{
    const std::filesystem::path destination(path);
    if (destination.has_parent_path()) {
        std::filesystem::create_directories(destination.parent_path());
    }
    const bool write_header = !std::filesystem::exists(destination) ||
                              std::filesystem::file_size(destination) == 0;
    if (!write_header) {
        std::ifstream existing(path);
        std::string line;
        if (!std::getline(existing, line) ||
            line.find("within_group_configuration_speedup") == std::string::npos ||
            line.find("cross_group_gpu_scaling_speedup") == std::string::npos) {
            throw std::runtime_error("Existing E5 GPU CSV has an incompatible schema: " + path);
        }
        const auto header = split_csv_row(line);
        const std::size_t group = column_index(header, "group", path);
        const std::size_t ranks = column_index(header, "mpi_ranks", path);
        const std::size_t threads = column_index(header, "cpu_threads_per_rank", path);
        while (std::getline(existing, line)) {
            const auto fields = split_csv_row(line);
            if (fields.size() <= std::max({group, ranks, threads})) {
                throw std::runtime_error("Malformed row in existing E5 GPU CSV: " + path);
            }
            if (fields[group] == result.group &&
                fields[ranks] == std::to_string(result.mpi_ranks) &&
                fields[threads] == std::to_string(result.cpu_threads_per_rank)) {
                throw std::runtime_error("E5 GPU configuration already exists in CSV: " + path);
            }
        }
    }
    std::ofstream file(path, std::ios::app);
    if (!file) {
        throw std::runtime_error("Cannot open E5 GPU CSV: " + path);
    }
    file << std::setprecision(12);
    if (write_header) write_e5_gpu_header(file);
    write_e5_gpu_row(file, result);
    if (!file) throw std::runtime_error("Failed while writing E5 GPU CSV: " + path);
}

void append_e5_gpu_runs_csv(const GpuHybridExperimentResult& result,
                            const std::string& path)
{
    const std::filesystem::path destination(path);
    if (destination.has_parent_path()) {
        std::filesystem::create_directories(destination.parent_path());
    }
    const bool write_header = !std::filesystem::exists(destination) ||
                              std::filesystem::file_size(destination) == 0;
    std::ofstream file(path, std::ios::app);
    if (!file) {
        throw std::runtime_error("Cannot open E5 GPU runs CSV: " + path);
    }
    if (write_header) {
        file << "group,total_cpu_cores,gpu_count,node_count,mpi_ranks,ranks_per_node,"
                "gpu_device_index,cpu_threads_per_rank,gpu_thread_limit,"
                "repetition,solver_time_s,"
                "mpi_halo_time_s,gpu_compute_time_s,boundary_d2h_time_s,"
                "halo_h2d_time_s,gpu_setup_time_s,gpu_final_copy_time_s,"
                "max_absolute_error,relative_l2_error,validation_passed\n";
    }
    file << std::setprecision(12);
    for (const auto& run : result.runs) {
        file << result.group << ',' << result.total_cpu_cores << ','
             << result.gpu_count << ',' << result.node_count << ','
             << result.mpi_ranks << ',' << result.ranks_per_node << ','
             << result.gpu_device_index << ',' << result.cpu_threads_per_rank << ','
             << result.gpu_thread_limit << ','
             << run.repetition << ',' << run.solver_time << ','
             << run.mpi_halo_time << ',' << run.gpu_compute_time << ','
             << run.boundary_d2h_time << ',' << run.halo_h2d_time << ','
             << run.gpu_setup_time << ',' << run.gpu_final_copy_time << ','
             << run.max_absolute_error << ',' << run.relative_l2_error << ','
             << (run.passed ? 1 : 0) << '\n';
    }
    if (!file) throw std::runtime_error("Failed while writing E5 GPU runs CSV: " + path);
}

std::vector<GpuHybridExperimentResult> read_e5_gpu_csv(const std::string& path)
{
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open E5 GPU results CSV: " + path);
    std::string line;
    if (!std::getline(file, line)) {
        throw std::runtime_error("E5 GPU results CSV is empty: " + path);
    }
    const auto header = split_csv_row(line);
    const auto index = [&](const std::string& name) {
        return column_index(header, name, path);
    };
    std::vector<GpuHybridExperimentResult> results;
    while (std::getline(file, line)) {
        const auto fields = split_csv_row(line);
        if (fields.size() != header.size()) {
            throw std::runtime_error("Malformed row in E5 GPU results CSV: " + path);
        }
        GpuHybridExperimentResult result{};
        result.group = fields[index("group")];
        result.total_cpu_cores = std::stoi(fields[index("total_cpu_cores")]);
        result.gpu_count = std::stoi(fields[index("gpu_count")]);
        result.node_count = std::stoi(fields[index("node_count")]);
        result.mpi_ranks = std::stoi(fields[index("mpi_ranks")]);
        result.ranks_per_node = std::stoi(fields[index("ranks_per_node")]);
        result.gpu_device_index = std::stoi(fields[index("gpu_device_index")]);
        result.cpu_threads_per_rank = std::stoi(fields[index("cpu_threads_per_rank")]);
        result.gpu_thread_limit = std::stoi(fields[index("gpu_thread_limit")]);
        result.process_rows = std::stoi(fields[index("process_rows")]);
        result.process_cols = std::stoi(fields[index("process_cols")]);
        result.steps = std::stoi(fields[index("steps")]);
        result.repetitions = std::stoi(fields[index("repetitions")]);
        result.cpu_solver_median = std::stod(fields[index("cpu_solver_median_s")]);
        result.cpu_gpu_speedup = std::stod(fields[index("cpu_gpu_speedup")]);
        result.within_group_configuration_speedup =
            std::stod(fields[index("within_group_configuration_speedup")]);
        result.cross_group_gpu_scaling_speedup =
            std::stod(fields[index("cross_group_gpu_scaling_speedup")]);
        result.solver_time = read_timing(fields, header, "solver", path);
        result.mpi_halo_time = read_timing(fields, header, "mpi_halo", path);
        result.gpu_compute_time = read_timing(fields, header, "gpu_compute", path);
        result.boundary_d2h_time = read_timing(fields, header, "boundary_d2h", path);
        result.halo_h2d_time = read_timing(fields, header, "halo_h2d", path);
        result.gpu_setup_time = read_timing(fields, header, "gpu_setup", path);
        result.gpu_final_copy_time = read_timing(fields, header, "gpu_final_copy", path);
        result.max_absolute_error = std::stod(fields[index("max_absolute_error")]);
        result.relative_l2_error = std::stod(fields[index("relative_l2_error")]);
        result.validation.min_temperature = std::stod(fields[index("min_temperature")]);
        result.validation.max_temperature = std::stod(fields[index("max_temperature")]);
        result.validation.max_boundary_error =
            std::stod(fields[index("max_boundary_error")]);
        result.validation.passed = fields[index("validation_passed")] == "1";
        results.push_back(std::move(result));
    }
    std::sort(results.begin(), results.end(), [](const auto& lhs, const auto& rhs) {
        const auto lhs_config = std::find_if(
            kE5GpuConfigurations.begin(), kE5GpuConfigurations.end(),
            [&](const auto& config) {
                return config.mpi_ranks == lhs.mpi_ranks &&
                       config.cpu_threads_per_rank == lhs.cpu_threads_per_rank;
            });
        const auto rhs_config = std::find_if(
            kE5GpuConfigurations.begin(), kE5GpuConfigurations.end(),
            [&](const auto& config) {
                return config.mpi_ranks == rhs.mpi_ranks &&
                       config.cpu_threads_per_rank == rhs.cpu_threads_per_rank;
            });
        return lhs_config < rhs_config;
    });
    return results;
}

void print_e5_gpu_summary(
    const std::vector<GpuHybridExperimentResult>& results)
{
    if (results.size() != kE5GpuConfigurations.size()) {
        throw std::runtime_error("E5 GPU summary requires all 15 configurations.");
    }
    bool all_passed = true;
    std::cout << "\n========================================================================================\n"
              << "  E5 | MPI + OPENMP GPU HYBRID CONFIGURATION AND SCALING\n"
              << "========================================================================================\n"
              << "  Global Grid : " << kE5GlobalNx << " x " << kE5GlobalNy << '\n'
              << "  Iterations  : " << kE5Steps << '\n'
              << "  Repetitions : " << kE5Repeats << '\n'
              << "  Solver      : Max-rank wall time incl. GPU setup, transfers and MPI\n"
              << "  Speedups    : GPU/CPU = matching E4; In-Group = first config in group; Cross-Group = Group A at matching ranks\n\n"
              << std::left << std::setw(7) << "Group"
              << std::right << std::setw(7) << "Cores"
              << std::setw(6) << "GPUs"
              << std::setw(7) << "Nodes"
              << std::setw(7) << "Ranks"
              << std::setw(8) << "R/Node"
              << std::setw(7) << "Device"
              << std::setw(8) << "CPU T/R"
              << std::setw(9) << "GPU Lim"
              << std::setw(14) << "CPU Med (s)"
              << std::setw(14) << "GPU Med (s)"
              << std::setw(12) << "GPU/CPU E4"
              << std::setw(14) << "In-Group"
              << std::setw(14) << "Cross-Group"
              << std::setw(10) << "Status" << '\n'
              << "--------------------------------------------------------------------------------------------------\n"
              << std::fixed << std::setprecision(4);
    for (const auto& config : kE5GpuConfigurations) {
        const auto found = std::find_if(
            results.begin(), results.end(), [&](const auto& result) {
                return result.mpi_ranks == config.mpi_ranks &&
                       result.cpu_threads_per_rank == config.cpu_threads_per_rank &&
                       result.group == config.group;
            });
        if (found == results.end()) {
            throw std::runtime_error("E5 GPU summary is missing a configuration.");
        }
        const auto& result = *found;
        if (result.gpu_count != config.gpu_count ||
            result.node_count != config.node_count ||
            result.ranks_per_node != config.ranks_per_node ||
            result.gpu_device_index != config.gpu_device_index ||
            result.cpu_threads_per_rank != config.cpu_threads_per_rank ||
            result.gpu_thread_limit != config.thread_limit ||
            result.steps != kE5Steps || result.repetitions != kE5Repeats ||
            !result.validation.passed) {
            throw std::runtime_error("E5 GPU result failed configuration or validation checks.");
        }
        all_passed = all_passed && result.validation.passed;
        std::cout << std::left << std::setw(7) << result.group
                  << std::right << std::setw(7) << result.total_cpu_cores
                  << std::setw(6) << result.gpu_count
                  << std::setw(7) << result.node_count
                  << std::setw(7) << result.mpi_ranks
                  << std::setw(8) << result.ranks_per_node
                  << std::setw(7) << result.gpu_device_index
                  << std::setw(8) << result.cpu_threads_per_rank
                  << std::setw(9) << result.gpu_thread_limit
                  << std::setw(14) << result.cpu_solver_median
                  << std::setw(14) << result.solver_time.median
                  << std::setw(12) << result.cpu_gpu_speedup
                  << std::setw(14) << result.within_group_configuration_speedup;
        if (result.cross_group_gpu_scaling_speedup > 0.0) {
            std::cout << std::setw(14)
                      << result.cross_group_gpu_scaling_speedup;
        } else {
            std::cout << std::setw(14) << "n/a";
        }
        std::cout
                  << std::setw(10) << (result.validation.passed ? "PASS" : "FAIL")
                  << '\n';
    }
    std::cout << "----------------------------------------------------------------------------------------\n"
              << "  Validation : " << (all_passed ? "ALL PASSED" : "FAILED") << '\n'
              << "  Results    : " << kE5GpuOutputCsv << '\n'
              << "  Raw Runs   : " << kE5GpuRunsCsv << '\n'
              << "========================================================================================\n"
              << "\nE5 GPU timing diagnostics (median seconds; not additive across ranks):\n"
              << std::left << std::setw(7) << "Group"
              << std::right << std::setw(7) << "Ranks"
              << std::setw(14) << "Solver"
              << std::setw(14) << "MPI Halo"
              << std::setw(14) << "GPU Compute"
              << std::setw(14) << "Boundary D2H"
              << std::setw(14) << "Halo H2D"
              << std::setw(14) << "GPU Setup"
              << std::setw(14) << "Final Copy" << '\n';
    for (const auto& result : results) {
        std::cout << std::left << std::setw(7) << result.group
                  << std::right << std::setw(7) << result.mpi_ranks
                  << std::setw(14) << result.solver_time.median
                  << std::setw(14) << result.mpi_halo_time.median
                  << std::setw(14) << result.gpu_compute_time.median
                  << std::setw(14) << result.boundary_d2h_time.median
                  << std::setw(14) << result.halo_h2d_time.median
                  << std::setw(14) << result.gpu_setup_time.median
                  << std::setw(14) << result.gpu_final_copy_time.median << '\n';
    }
}

} // namespace heat

namespace heat {
void write_e2_runs_csv(const std::vector<MpiExperimentResult>& results,
                       const std::string& path)
{
    const std::filesystem::path destination(path);
    if (destination.has_parent_path())
        std::filesystem::create_directories(destination.parent_path());
    std::ofstream file(path, std::ios::trunc);
    if (!file) throw std::runtime_error("Cannot open E2 runs CSV: " + path);
    file << "ranks,repetition,initialization_s,solver_s,halo_s,ftcs_s,"
            "validation_s,total_s,halo_ratio,validation_passed\n";
    file << std::setprecision(12);
    for (const auto& result : results)
        for (const auto& run : result.runs)
            file << result.ranks << ',' << run.repetition << ','
                 << run.initialization << ',' << run.solver << ','
                 << run.halo << ',' << run.ftcs << ','
                 << run.validation << ',' << run.total << ','
                 << run.halo_ratio << ',' << (run.passed ? 1 : 0) << '\n';
}
} // namespace heat

namespace heat {
namespace {

std::vector<std::string> split_e4_csv_row(const std::string& row)
{
    std::vector<std::string> fields;
    std::istringstream input(row);
    std::string field;
    while (std::getline(input, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

std::size_t e4_column_index(const std::vector<std::string>& header,
                            const std::string& name,
                            const std::string& path)
{
    const auto column = std::find(header.begin(), header.end(), name);
    if (column == header.end()) {
        throw std::runtime_error("Missing " + name + " column in E4 CSV: " + path);
    }
    return static_cast<std::size_t>(column - header.begin());
}

void write_e4_header(std::ostream& file)
{
    file << "total_cores,ranks,threads_per_rank,process_rows,process_cols,steps,repetitions,"
            "initialization_median_s,initialization_mean_s,initialization_stddev_s,"
            "solver_median_s,solver_mean_s,solver_stddev_s,"
            "halo_median_s,halo_mean_s,halo_stddev_s,"
            "ftcs_median_s,ftcs_mean_s,ftcs_stddev_s,"
            "validation_median_s,validation_mean_s,total_median_s,total_mean_s,"
            "halo_ratio_median,configuration_speedup,hybrid_speedup,parallel_efficiency,"
            "min_temperature,max_temperature,max_boundary_error,validation_passed\n";
}

void write_e4_row(std::ostream& file, const HybridExperimentResult& result)
{
    file << result.total_cores << ',' << result.ranks << ','
         << result.threads_per_rank << ',' << result.process_rows << ','
         << result.process_cols << ',' << result.steps << ','
         << result.repetitions << ','
         << result.initialization_median << ',' << result.initialization_mean << ','
         << result.initialization_stddev << ','
         << result.solver_median << ',' << result.solver_mean << ','
         << result.solver_stddev << ','
         << result.halo_median << ',' << result.halo_mean << ','
         << result.halo_stddev << ','
         << result.ftcs_median << ',' << result.ftcs_mean << ','
         << result.ftcs_stddev << ','
         << result.validation_median << ',' << result.validation_mean << ','
         << result.total_median << ',' << result.total_mean << ','
         << result.halo_ratio_median << ','
         << result.configuration_speedup << ',' << result.hybrid_speedup << ','
         << result.parallel_efficiency << ','
         << result.validation.min_temperature << ','
         << result.validation.max_temperature << ','
         << result.validation.max_boundary_error << ','
         << (result.validation.passed ? 1 : 0) << '\n';
}

} // namespace

void append_e4_csv(const HybridExperimentResult& result, const std::string& path)
{
    const std::filesystem::path destination(path);
    if (destination.has_parent_path()) {
        std::filesystem::create_directories(destination.parent_path());
    }
    const bool write_header = !std::filesystem::exists(destination) ||
                              std::filesystem::file_size(destination) == 0;
    if (!write_header) {
        std::ifstream existing(path);
        std::string header;
        std::getline(existing, header);
        if (header.find("configuration_speedup") == std::string::npos) {
            throw std::runtime_error("Existing E4 CSV has an incompatible schema: " + path);
        }
        std::string line;
        while (std::getline(existing, line)) {
            const auto fields = split_e4_csv_row(line);
            if (fields.size() >= 3 && fields[1] == std::to_string(result.ranks) &&
                fields[2] == std::to_string(result.threads_per_rank)) {
                throw std::runtime_error("E4 configuration already exists in CSV: " + path);
            }
        }
    }
    std::ofstream file(path, std::ios::app);
    if (!file) {
        throw std::runtime_error("Cannot open E4 CSV output: " + path);
    }
    file << std::setprecision(12);
    if (write_header) write_e4_header(file);
    write_e4_row(file, result);
}

void append_e4_runs_csv(const HybridExperimentResult& result, const std::string& path)
{
    const std::filesystem::path destination(path);
    if (destination.has_parent_path()) {
        std::filesystem::create_directories(destination.parent_path());
    }
    const bool write_header = !std::filesystem::exists(destination) ||
                              std::filesystem::file_size(destination) == 0;
    std::ofstream file(path, std::ios::app);
    if (!file) {
        throw std::runtime_error("Cannot open E4 runs CSV output: " + path);
    }
    if (write_header) {
        file << "total_cores,ranks,threads_per_rank,repetition,initialization_s,"
                "solver_s,halo_s,ftcs_s,validation_s,total_s,halo_ratio,validation_passed\n";
    }
    file << std::setprecision(12);
    for (const auto& run : result.runs) {
        file << result.total_cores << ',' << result.ranks << ','
             << result.threads_per_rank << ',' << run.repetition << ','
             << run.initialization << ',' << run.solver << ',' << run.halo << ','
             << run.ftcs << ',' << run.validation << ',' << run.total << ','
             << run.halo_ratio << ',' << (run.passed ? 1 : 0) << '\n';
    }
}

std::vector<HybridExperimentResult> read_e4_csv(const std::string& path)
{
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open E4 results CSV: " + path);
    }
    std::string line;
    if (!std::getline(file, line)) {
        throw std::runtime_error("E4 results CSV is empty: " + path);
    }
    const auto header = split_e4_csv_row(line);
    const auto index = [&](const std::string& name) {
        return e4_column_index(header, name, path);
    };
    const std::size_t total_cores = index("total_cores");
    const std::size_t ranks = index("ranks");
    const std::size_t threads = index("threads_per_rank");
    const std::size_t rows = index("process_rows");
    const std::size_t cols = index("process_cols");
    const std::size_t steps = index("steps");
    const std::size_t repetitions = index("repetitions");
    const std::size_t init_median = index("initialization_median_s");
    const std::size_t init_mean = index("initialization_mean_s");
    const std::size_t init_stddev = index("initialization_stddev_s");
    const std::size_t solver_median = index("solver_median_s");
    const std::size_t solver_mean = index("solver_mean_s");
    const std::size_t solver_stddev = index("solver_stddev_s");
    const std::size_t halo_median = index("halo_median_s");
    const std::size_t halo_mean = index("halo_mean_s");
    const std::size_t halo_stddev = index("halo_stddev_s");
    const std::size_t ftcs_median = index("ftcs_median_s");
    const std::size_t ftcs_mean = index("ftcs_mean_s");
    const std::size_t ftcs_stddev = index("ftcs_stddev_s");
    const std::size_t validation_median = index("validation_median_s");
    const std::size_t validation_mean = index("validation_mean_s");
    const std::size_t total_median = index("total_median_s");
    const std::size_t total_mean = index("total_mean_s");
    const std::size_t halo_ratio = index("halo_ratio_median");
    const std::size_t config_speedup = index("configuration_speedup");
    const std::size_t hybrid_speedup = index("hybrid_speedup");
    const std::size_t efficiency = index("parallel_efficiency");
    const std::size_t minimum = index("min_temperature");
    const std::size_t maximum = index("max_temperature");
    const std::size_t boundary_error = index("max_boundary_error");
    const std::size_t passed = index("validation_passed");
    std::vector<HybridExperimentResult> results;
    while (std::getline(file, line)) {
        const auto fields = split_e4_csv_row(line);
        if (fields.size() < header.size()) {
            throw std::runtime_error("Malformed row in E4 results CSV: " + path);
        }
        HybridExperimentResult result{};
        result.total_cores = std::stoi(fields[total_cores]);
        result.ranks = std::stoi(fields[ranks]);
        result.threads_per_rank = std::stoi(fields[threads]);
        result.process_rows = std::stoi(fields[rows]);
        result.process_cols = std::stoi(fields[cols]);
        result.steps = std::stoi(fields[steps]);
        result.repetitions = std::stoi(fields[repetitions]);
        result.initialization_median = std::stod(fields[init_median]);
        result.initialization_mean = std::stod(fields[init_mean]);
        result.initialization_stddev = std::stod(fields[init_stddev]);
        result.solver_median = std::stod(fields[solver_median]);
        result.solver_mean = std::stod(fields[solver_mean]);
        result.solver_stddev = std::stod(fields[solver_stddev]);
        result.halo_median = std::stod(fields[halo_median]);
        result.halo_mean = std::stod(fields[halo_mean]);
        result.halo_stddev = std::stod(fields[halo_stddev]);
        result.ftcs_median = std::stod(fields[ftcs_median]);
        result.ftcs_mean = std::stod(fields[ftcs_mean]);
        result.ftcs_stddev = std::stod(fields[ftcs_stddev]);
        result.validation_median = std::stod(fields[validation_median]);
        result.validation_mean = std::stod(fields[validation_mean]);
        result.total_median = std::stod(fields[total_median]);
        result.total_mean = std::stod(fields[total_mean]);
        result.halo_ratio_median = std::stod(fields[halo_ratio]);
        result.configuration_speedup = std::stod(fields[config_speedup]);
        result.hybrid_speedup = std::stod(fields[hybrid_speedup]);
        result.parallel_efficiency = std::stod(fields[efficiency]);
        result.validation.min_temperature = std::stod(fields[minimum]);
        result.validation.max_temperature = std::stod(fields[maximum]);
        result.validation.max_boundary_error = std::stod(fields[boundary_error]);
        result.validation.passed = fields[passed] == "1";
        results.push_back(result);
    }
    std::sort(results.begin(), results.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.total_cores != rhs.total_cores) return lhs.total_cores < rhs.total_cores;
        return lhs.ranks < rhs.ranks;
    });
    return results;
}

void write_e4_csv(const std::vector<HybridExperimentResult>& results,
                  const std::string& path)
{
    const std::filesystem::path destination(path);
    if (destination.has_parent_path()) {
        std::filesystem::create_directories(destination.parent_path());
    }
    std::ofstream file(path, std::ios::trunc);
    if (!file) {
        throw std::runtime_error("Cannot rewrite E4 CSV: " + path);
    }
    file << std::setprecision(12);
    write_e4_header(file);
    for (const auto& result : results) write_e4_row(file, result);
}

void print_hybrid_summary(const std::vector<HybridExperimentResult>& results)
{
    if (results.size() != kE4Configurations.size()) {
        throw std::runtime_error("E4 summary requires all 15 configurations.");
    }
    std::cout << "\n==========================================================================\n"
              << "  E4 | MPI + OPENMP HYBRID CONFIGURATION AND STRONG SCALING\n"
              << "==========================================================================\n"
              << "  Global Grid : " << kE4GlobalNx << " x " << kE4GlobalNy << '\n'
              << "  Iterations  : " << kSteps << '\n'
              << "  Repetitions : " << kE4Repeats << '\n'
              << "  Runtime     : Median of max-rank elapsed time (seconds)\n\n"
              << std::left << std::setw(7) << "Group"
              << std::right << std::setw(8) << "Cores"
              << std::setw(8) << "Ranks"
              << std::setw(9) << "Threads"
              << std::setw(16) << "Solver Med (s)"
              << std::setw(15) << "Config Speedup"
              << std::setw(15) << "Hybrid Speedup"
              << std::setw(13) << "Efficiency"
              << std::setw(10) << "Status" << '\n'
              << "--------------------------------------------------------------------------\n"
              << std::fixed << std::setprecision(4);
    for (const auto& result : results) {
        const char* group = result.total_cores == 32 ? "A" :
                            result.total_cores == 64 ? "B" : "C";
        std::ostringstream efficiency;
        efficiency << std::setprecision(2) << result.parallel_efficiency * 100.0 << '%';
        std::cout << std::left << std::setw(7) << group
                  << std::right << std::setw(8) << result.total_cores
                  << std::setw(8) << result.ranks
                  << std::setw(9) << result.threads_per_rank
                  << std::setw(16) << result.solver_median
                  << std::setw(15) << result.configuration_speedup
                  << std::setw(15) << result.hybrid_speedup
                  << std::setw(13) << efficiency.str()
                  << std::setw(10) << (result.validation.passed ? "PASS" : "FAIL")
                  << '\n';
    }
    std::cout << "--------------------------------------------------------------------------\n"
              << "  Validation : "
              << (std::all_of(results.begin(), results.end(), [](const auto& r) {
                      return r.validation.passed;
                  }) ? "ALL PASSED" : "FAILED") << '\n'
              << "  Results    : " << kE4OutputCsv << '\n'
              << "  Raw Runs   : " << kE4RunsCsv << '\n'
              << "==========================================================================\n";
}

} // namespace heat
