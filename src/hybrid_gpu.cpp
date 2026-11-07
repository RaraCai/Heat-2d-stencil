#include "benchmark.hpp"
#include "config.hpp"
#include "mpi.hpp"
#include "output.hpp"
#include "stencil.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct GpuRun {
    int repetition;
    double initialization;
    double solver;
    double halo;
    double ftcs;
    double validation;
    double total;
    bool passed;
};

struct GpuResult {
    int total_cores;
    int ranks;
    int threads_per_rank;
    int gpus_used;
    int process_rows;
    int process_cols;
    double cpu_solver;
    double initialization;
    double solver;
    double solver_mean;
    double solver_stddev;
    double halo;
    double ftcs;
    double validation_time;
    double total;
    heat::ValidationResult validation;
    std::vector<GpuRun> runs;
};

const heat::HybridConfiguration* find_configuration(int ranks, int threads)
{
    const auto found = std::find_if(
        heat::kE4Configurations.begin(), heat::kE4Configurations.end(),
        [&](const auto& config) {
            return config.ranks == ranks && config.threads_per_rank == threads;
        });
    return found == heat::kE4Configurations.end() ? nullptr : &*found;
}

double read_cpu_solver_baseline(const heat::HybridConfiguration& configuration)
{
    const auto results = heat::read_e4_csv(heat::kE4OutputCsv);
    const auto found = std::find_if(
        results.begin(), results.end(), [&](const auto& result) {
            return result.total_cores == configuration.total_cores &&
                   result.ranks == configuration.ranks &&
                   result.threads_per_rank == configuration.threads_per_rank;
        });
    if (found == results.end()) {
        throw std::runtime_error(
            "Matching E4 CPU result is missing; run the complete CPU E4 first.");
    }
    if (found->steps != heat::kSteps ||
        found->repetitions != heat::kE4Repeats ||
        !found->validation.passed ||
        !std::isfinite(found->solver_median) || found->solver_median <= 0.0) {
        throw std::runtime_error(
            "Matching E4 CPU result has invalid steps, repetitions, timing, or validation.");
    }
    return found->solver_median;
}

std::vector<std::string> split_csv_row(const std::string& line)
{
    std::vector<std::string> fields;
    std::istringstream input(line);
    std::string field;
    while (std::getline(input, field, ',')) fields.push_back(field);
    return fields;
}

void append_result(const GpuResult& result)
{
    const std::string path = heat::kE4GpuOutputCsv;
    const std::filesystem::path destination(path);
    if (destination.has_parent_path()) {
        std::filesystem::create_directories(destination.parent_path());
    }
    const bool first_configuration =
        result.ranks == heat::kE4Configurations.front().ranks &&
        result.threads_per_rank == heat::kE4Configurations.front().threads_per_rank;
    const bool write_header = first_configuration ||
                              !std::filesystem::exists(destination) ||
                              std::filesystem::file_size(destination) == 0;
    if (!write_header) {
        std::ifstream existing(path);
        std::string header;
        if (!std::getline(existing, header) ||
            header.find("speedup_cpu_over_gpu") == std::string::npos) {
            throw std::runtime_error("Existing E4 GPU CSV has an incompatible schema.");
        }
        std::string line;
        while (std::getline(existing, line)) {
            const auto fields = split_csv_row(line);
            if (fields.size() >= 3 &&
                fields[1] == std::to_string(result.ranks) &&
                fields[2] == std::to_string(result.threads_per_rank)) {
                throw std::runtime_error(
                    "E4 GPU CSV already contains this ranks/threads configuration.");
            }
        }
    }

    std::ofstream output(path, first_configuration ? std::ios::trunc : std::ios::app);
    if (!output) {
        throw std::runtime_error("Cannot open E4 GPU CSV: " + path);
    }
    if (write_header) {
        output << "total_cores,ranks,threads_per_rank,gpus_used,process_rows,"
                  "process_cols,steps,repetitions,cpu_solver_median_s,"
                  "gpu_solver_median_s,speedup_cpu_over_gpu,"
                  "initialization_median_s,solver_mean_s,solver_stddev_s,"
                  "halo_median_s,ftcs_median_s,validation_median_s,total_median_s,"
                  "min_temperature,max_temperature,max_boundary_error,validation_passed\n";
    }
    output << std::setprecision(12)
           << result.total_cores << ',' << result.ranks << ','
           << result.threads_per_rank << ',' << result.gpus_used << ','
           << result.process_rows << ',' << result.process_cols << ','
           << heat::kSteps << ',' << heat::kE4Repeats << ','
           << result.cpu_solver << ',' << result.solver << ','
           << result.cpu_solver / result.solver << ','
           << result.initialization << ',' << result.solver_mean << ','
           << result.solver_stddev << ',' << result.halo << ','
           << result.ftcs << ',' << result.validation_time << ','
           << result.total << ','
           << result.validation.min_temperature << ','
           << result.validation.max_temperature << ','
           << result.validation.max_boundary_error << ','
           << (result.validation.passed ? 1 : 0) << '\n';
    if (!output) {
        throw std::runtime_error("Failed while writing E4 GPU CSV: " + path);
    }

    const std::string runs_path = heat::kE4GpuRunsCsv;
    const std::filesystem::path runs_destination(runs_path);
    if (runs_destination.has_parent_path()) {
        std::filesystem::create_directories(runs_destination.parent_path());
    }
    const bool write_runs_header = first_configuration ||
        !std::filesystem::exists(runs_destination) ||
        std::filesystem::file_size(runs_destination) == 0;
    std::ofstream runs_output(
        runs_path, first_configuration ? std::ios::trunc : std::ios::app);
    if (!runs_output) {
        throw std::runtime_error("Cannot open E4 GPU runs CSV: " + runs_path);
    }
    if (write_runs_header) {
        runs_output << "total_cores,ranks,threads_per_rank,gpus_used,repetition,"
                       "initialization_s,solver_s,halo_s,ftcs_s,validation_s,"
                       "total_s,validation_passed\n";
    }
    runs_output << std::setprecision(12);
    for (const auto& run : result.runs) {
        runs_output << result.total_cores << ',' << result.ranks << ','
                    << result.threads_per_rank << ',' << result.gpus_used << ','
                    << run.repetition << ',' << run.initialization << ','
                    << run.solver << ',' << run.halo << ',' << run.ftcs << ','
                    << run.validation << ',' << run.total << ','
                    << (run.passed ? 1 : 0) << '\n';
    }
    if (!runs_output) {
        throw std::runtime_error("Failed while writing E4 GPU runs CSV: " + runs_path);
    }
}

void print_summary()
{
    std::ifstream input(heat::kE4GpuOutputCsv);
    if (!input) {
        throw std::runtime_error("Cannot open E4 GPU results CSV.");
    }
    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("E4 GPU results CSV is empty.");
    }
    std::vector<std::vector<std::string>> rows;
    while (std::getline(input, line)) {
        auto fields = split_csv_row(line);
        if (fields.size() != 22) {
            throw std::runtime_error("Malformed row in E4 GPU results CSV.");
        }
        rows.push_back(std::move(fields));
    }
    if (rows.size() != heat::kE4Configurations.size()) {
        throw std::runtime_error("E4 GPU summary requires all 15 configurations.");
    }

    bool all_passed = true;
    std::cout << "\n==========================================================================\n"
              << "  E4 | MPI + OPENMP GPU OFFLOAD CONFIGURATION COMPARISON\n"
              << "==========================================================================\n"
              << "  Global Grid : " << heat::kE4GlobalNx << " x "
              << heat::kE4GlobalNy << '\n'
              << "  Iterations  : " << heat::kSteps << '\n'
              << "  Repetitions : " << heat::kE4Repeats << '\n'
              << "  Runtime     : Median of max-rank elapsed time (seconds)\n"
              << "  Speedup     : Matching CPU E4 solver median / GPU solver median\n\n"
              << std::left << std::setw(7) << "Group"
              << std::right << std::setw(8) << "Cores"
              << std::setw(8) << "Ranks"
              << std::setw(9) << "Threads"
              << std::setw(8) << "GPUs"
              << std::setw(16) << "CPU Med (s)"
              << std::setw(16) << "GPU Med (s)"
              << std::setw(15) << "Speedup"
              << std::setw(10) << "Status" << '\n'
              << "--------------------------------------------------------------------------\n"
              << std::fixed << std::setprecision(4);
    for (const auto& configuration : heat::kE4Configurations) {
        const auto found = std::find_if(
            rows.begin(), rows.end(), [&](const auto& row) {
                return std::stoi(row[0]) == configuration.total_cores &&
                       std::stoi(row[1]) == configuration.ranks &&
                       std::stoi(row[2]) == configuration.threads_per_rank;
            });
        if (found == rows.end()) {
            throw std::runtime_error("E4 GPU summary is missing a required configuration.");
        }
        const auto& row = *found;
        const int expected_gpus = configuration.ranks == 2 ? 1 : 4;
        const double cpu_solver = std::stod(row[8]);
        const double gpu_solver = std::stod(row[9]);
        const double speedup = std::stod(row[10]);
        const bool passed = row[21] == "1";
        const double expected_speedup = cpu_solver / gpu_solver;
        if (std::stoi(row[3]) != expected_gpus ||
            std::stoi(row[6]) != heat::kSteps ||
            std::stoi(row[7]) != heat::kE4Repeats ||
            !std::isfinite(cpu_solver) || cpu_solver <= 0.0 ||
            !std::isfinite(gpu_solver) || gpu_solver <= 0.0 ||
            !std::isfinite(speedup) ||
            std::abs(speedup - expected_speedup) >
                1e-10 * std::max(1.0, std::abs(expected_speedup)) ||
            !passed) {
            throw std::runtime_error(
                "E4 GPU result has an invalid configuration, speedup, or validation.");
        }
        all_passed = all_passed && passed;
        std::cout << std::left << std::setw(7) << configuration.group
                  << std::right << std::setw(8) << configuration.total_cores
                  << std::setw(8) << configuration.ranks
                  << std::setw(9) << configuration.threads_per_rank
                  << std::setw(8) << expected_gpus
                  << std::setw(16) << cpu_solver
                  << std::setw(16) << gpu_solver
                  << std::setw(15) << speedup
                  << std::setw(10) << (passed ? "PASS" : "FAIL") << '\n';
    }
    if (!all_passed) {
        throw std::runtime_error("E4 GPU validation failed.");
    }
    std::cout << "--------------------------------------------------------------------------\n"
              << "  Validation : "
              << (all_passed ? "ALL PASSED" : "FAILED") << '\n'
              << "  Results    : " << heat::kE4GpuOutputCsv << '\n'
              << "==========================================================================\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc == 2 && std::string(argv[1]) == "--summary") {
        try {
            print_summary();
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "E4 GPU summary error: " << error.what() << '\n';
            return 1;
        }
    }
    if (argc != 4 || std::string(argv[1]) != "--run") {
        std::cerr << "Usage: " << argv[0]
                  << " --run <mpi-ranks> <threads-per-rank> | --summary\n";
        return 2;
    }

    const int requested_ranks = std::stoi(argv[2]);
    const int threads = std::stoi(argv[3]);
    const auto* configuration = find_configuration(requested_ranks, threads);
    if (configuration == nullptr) {
        std::cerr << "Unsupported E4 GPU configuration: ranks=" << requested_ranks
                  << ", threads/rank=" << threads << '\n';
        return 2;
    }

    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        std::cerr << "MPI_Init failed.\n";
        return 1;
    }
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    heat::MpiDomain domain;
    MPI_Comm node_comm = MPI_COMM_NULL;
    try {
        if (world_size != configuration->ranks) {
            throw std::runtime_error("MPI world size does not match E4 GPU configuration.");
        }
        const double cpu_solver = read_cpu_solver_baseline(*configuration);
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                            MPI_INFO_NULL, &node_comm);
        int node_rank = 0;
        MPI_Comm_rank(node_comm, &node_rank);
        if (omp_get_num_devices() < 1) {
            throw std::runtime_error("No OpenMP GPU device is visible on this node.");
        }
        // All MPI ranks on a node intentionally share its first allocated GPU.
        omp_set_default_device(0);
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
        int initial_device = 1;
        #pragma omp target map(from: initial_device)
        {
            initial_device = omp_is_initial_device();
        }
        if (initial_device != 0) {
            throw std::runtime_error(
                "OpenMP target region ran on the host; check GCC nvptx offload support.");
        }

        int local_node_leader = node_rank == 0 ? 1 : 0;
        int gpus_used = 0;
        MPI_Allreduce(&local_node_leader, &gpus_used, 1, MPI_INT, MPI_SUM,
                      MPI_COMM_WORLD);
        const int expected_gpus = configuration->ranks == 2 ? 1 : 4;
        if (gpus_used != expected_gpus) {
            throw std::runtime_error(
                "MPI rank placement does not match the E4 four-node GPU mapping.");
        }

        heat::create_mpi_domain(domain, heat::kE4GlobalNx, heat::kE4GlobalNy);
        const auto params = heat::make_parameters(
            heat::kE4GlobalNx, heat::kE4GlobalNy, heat::kSteps);
        heat::Grid current = heat::make_local_grid(domain);
        heat::Grid next = heat::make_local_grid(domain);
        std::vector<GpuRun> runs;
        if (domain.rank == 0) runs.reserve(heat::kE4Repeats);
        bool all_passed = true;
        double min_temperature = std::numeric_limits<double>::max();
        double max_temperature = std::numeric_limits<double>::lowest();
        double max_boundary_error = 0.0;

        for (int repeat = 0; repeat < heat::kE4Repeats; ++repeat) {
            MPI_Barrier(domain.cart_comm);
            const double total_start = MPI_Wtime();
            const double init_start = MPI_Wtime();
            heat::initialize(current, domain.global_nx, domain.global_ny);
            next.data() = current.data();
            const double initialization = MPI_Wtime() - init_start;

            MPI_Barrier(domain.cart_comm);
            const double solver_start = MPI_Wtime();
            double halo_time = 0.0;
            double ftcs_time = 0.0;
            for (int step = 0; step < heat::kSteps; ++step) {
                heat::exchange_halos(current, domain, &halo_time);
                const double ftcs_start = MPI_Wtime();
                heat::ftcs_step_openmp_target(current, next, params,
                                               domain.global_nx, domain.global_ny);
                ftcs_time += MPI_Wtime() - ftcs_start;
                current.swap_data(next);
            }
            const double solver_time = MPI_Wtime() - solver_start;
            const double validation_start = MPI_Wtime();
            const auto validation = heat::validate_global(current, domain);
            const double validation_time = MPI_Wtime() - validation_start;
            const double total_time = MPI_Wtime() - total_start;
            const double local_times[] = {initialization, solver_time, halo_time,
                ftcs_time, validation_time, total_time};
            double max_times[6] = {};
            MPI_Reduce(local_times, max_times, 6, MPI_DOUBLE, MPI_MAX,
                       0, domain.cart_comm);

            all_passed = all_passed && validation.passed;
            min_temperature = std::min(min_temperature, validation.min_temperature);
            max_temperature = std::max(max_temperature, validation.max_temperature);
            max_boundary_error = std::max(
                max_boundary_error, validation.max_boundary_error);
            if (domain.rank == 0) {
                runs.push_back({repeat + 1, max_times[0], max_times[1],
                    max_times[2], max_times[3], max_times[4], max_times[5],
                    validation.passed});
                std::cerr << "\r[E4 GPU] Cores " << configuration->total_cores
                          << " | Ranks " << configuration->ranks
                          << " | Threads/Rank " << configuration->threads_per_rank
                          << " | Repeat " << repeat + 1 << '/'
                          << heat::kE4Repeats << std::flush;
            }
        }

        int exit_code = 0;
        if (domain.rank == 0) {
            std::vector<double> initialization, solver, halo, ftcs, validation, total;
            for (const auto& run : runs) {
                initialization.push_back(run.initialization);
                solver.push_back(run.solver);
                halo.push_back(run.halo);
                ftcs.push_back(run.ftcs);
                validation.push_back(run.validation);
                total.push_back(run.total);
            }
            GpuResult result{configuration->total_cores, configuration->ranks,
                configuration->threads_per_rank, gpus_used, domain.dims[0],
                domain.dims[1], cpu_solver, heat::median(initialization),
                heat::median(solver), heat::mean(solver),
                heat::sample_stddev(solver, heat::mean(solver)),
                heat::median(halo), heat::median(ftcs),
                heat::median(validation), heat::median(total),
                {all_passed, min_temperature, max_temperature, max_boundary_error},
                std::move(runs)};
            append_result(result);
            std::cerr << "\r[E4 GPU] Cores " << configuration->total_cores
                      << " | Ranks " << configuration->ranks
                      << " | Threads/Rank " << configuration->threads_per_rank
                      << " | GPUs " << gpus_used << " | GPU median "
                      << std::fixed << std::setprecision(3) << result.solver
                      << " s | CPU/GPU speedup " << std::setprecision(3)
                      << cpu_solver / result.solver << "             \n";
            if (!all_passed) exit_code = 2;
        }
        MPI_Bcast(&exit_code, 1, MPI_INT, 0, domain.cart_comm);
        heat::destroy_mpi_domain(domain);
        MPI_Comm_free(&node_comm);
        MPI_Finalize();
        return exit_code;
    } catch (const std::exception& error) {
        std::cerr << "Rank " << rank << " E4 GPU error: " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
