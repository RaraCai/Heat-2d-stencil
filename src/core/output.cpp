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
