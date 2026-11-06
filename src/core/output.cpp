#include "output.hpp"

#include "config.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace heat {

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
              << std::right << std::setw(16) << "Comp. Mean (s)"
              << std::setw(20) << "Std. Dev. (s)"
              << std::setw(12) << "Growth"
              << std::setw(10) << "Status" << '\n'
              << table_separator << '\n'
              << std::fixed << std::setprecision(6);

    for (const auto& result : results) {
        std::ostringstream grid_label;
        grid_label << result.n << " x " << result.n;
        std::cout << "  " << std::left << std::setw(14) << grid_label.str()
                  << std::right << std::setw(16) << result.computation_time
                  << std::setw(20) << result.computation_stddev;

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
            "initialization_mean_s,computation_mean_s,computation_stddev_s,"
            "validation_mean_s,total_mean_s,runtime_growth,min_temperature,"
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
             << result.computation_time << ','
             << result.computation_stddev << ','
             << result.validation_time << ','
             << result.total_time << ','
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
        "==============================================================================================";
    constexpr char divider[] =
        "----------------------------------------------------------------------------------------------";

    std::cout << '\n' << separator << '\n'
              << "  E2 | MPI STRONG SCALING\n"
              << separator << '\n'
              << "  Global Grid       : 4096 x 4096\n"
              << "  Iterations        : " << kSteps << '\n'
              << "  Repetitions       : " << kE2Repeats << '\n'
              << "  Runtime           : Median of max-rank elapsed time (includes halo exchange)\n"
              << "  Serial Baseline   : " << std::fixed << std::setprecision(6)
              << serial_baseline_seconds << " s (E1 result)\n\n"
              << divider << '\n'
              << "  " << std::left << std::setw(8) << "Ranks"
              << std::setw(12) << "Process Grid"
              << std::right << std::setw(15) << "Runtime (s)"
              << std::setw(15) << "Halo (s)"
              << std::setw(14) << "MPI Speedup"
              << std::setw(16) << "Serial Speedup"
              << std::setw(12) << "Efficiency"
              << "  Status\n"
              << divider << '\n';

    for (const auto& result : results) {
        std::ostringstream process_grid;
        process_grid << result.process_rows << 'x' << result.process_cols;
        std::cout << "  " << std::left << std::setw(8) << result.ranks
                  << std::setw(12) << process_grid.str()
                  << std::right << std::setw(15) << result.runtime_median
                  << std::setw(15) << result.halo_runtime_median
                  << std::setw(14) << result.speedup_mpi
                  << std::setw(16) << result.speedup_serial
                  << std::setw(11) << result.parallel_efficiency * 100.0 << '%'
                  << "  " << (result.validation.passed ? "PASS" : "FAIL") << '\n';
    }

    std::cout << divider << '\n' << separator << "\n\n";
}

void write_e2_csv(const std::vector<MpiExperimentResult>& results,
                  const std::string& path)
{
    const std::filesystem::path destination(path);
    if (destination.has_parent_path()) {
        std::filesystem::create_directories(destination.parent_path());
    }

    std::ofstream file(path, std::ios::trunc);
    if (!file) {
        throw std::runtime_error("Cannot open E2 CSV output: " + path);
    }

    file << "global_nx,global_ny,ranks,process_rows,process_cols,steps,repetitions,"
            "runtime_median_s,runtime_min_s,runtime_max_s,halo_runtime_median_s,"
            "speedup_mpi,speedup_serial,parallel_efficiency,"
            "min_temperature,max_temperature,max_boundary_error,validation_passed\n";
    file << std::setprecision(12);

    for (const auto& result : results) {
        file << result.global_nx << ',' << result.global_ny << ','
             << result.ranks << ',' << result.process_rows << ','
             << result.process_cols << ',' << result.steps << ','
             << result.repetitions << ',' << result.runtime_median << ','
             << result.runtime_min << ',' << result.runtime_max << ','
             << result.halo_runtime_median << ',' << result.speedup_mpi << ','
             << result.speedup_serial << ',' << result.parallel_efficiency << ','
             << result.validation.min_temperature << ','
             << result.validation.max_temperature << ','
             << result.validation.max_boundary_error << ','
             << (result.validation.passed ? 1 : 0) << '\n';
    }
}

void write_e2_runs_csv(const std::vector<MpiExperimentResult>& results,
                       const std::string& path)
{
    const std::filesystem::path destination(path);
    if (destination.has_parent_path()) {
        std::filesystem::create_directories(destination.parent_path());
    }

    std::ofstream file(path, std::ios::trunc);
    if (!file) {
        throw std::runtime_error("Cannot open E2 raw-runs CSV output: " + path);
    }

    file << "global_nx,global_ny,ranks,process_rows,process_cols,repetition,"
            "runtime_s,halo_runtime_s,validation_passed\n";
    file << std::setprecision(12);

    for (const auto& result : results) {
        for (const auto& run : result.runs) {
            file << result.global_nx << ',' << result.global_ny << ','
                 << result.ranks << ',' << result.process_rows << ','
                 << result.process_cols << ',' << run.repetition << ','
                 << run.total_runtime << ',' << run.halo_runtime << ','
                 << (run.validation.passed ? 1 : 0) << '\n';
        }
    }
}

} // namespace heat
