#include "benchmark.hpp"
#include "config.hpp"
#include "gpu_solver.hpp"
#include "mpi.hpp"
#include "output.hpp"
#include "stencil.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Configuration = heat::E5GpuConfiguration;

void log_e5_stage(int rank, int world_size, const std::string& stage)
{
    std::fprintf(stderr, "[E5][t=%.3fs][rank %d/%d] %s\n",
                 MPI_Wtime(), rank, world_size, stage.c_str());
    std::fflush(stderr);
}

const Configuration& configuration_at(int index)
{
    if (index < 0 ||
        index >= static_cast<int>(heat::kE5GpuConfigurations.size())) {
        throw std::invalid_argument("E5 GPU configuration index is out of range.");
    }
    return heat::kE5GpuConfigurations[static_cast<std::size_t>(index)];
}

void validate_gpu_mapping(const Configuration& config, int& device,
                          MPI_Comm& node_comm, int world_rank, int world_size)
{
    log_e5_stage(world_rank, world_size, "MPI_Comm_split_type begin");
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank,
                        MPI_INFO_NULL, &node_comm);
    log_e5_stage(world_rank, world_size, "MPI_Comm_split_type complete");
    int local_rank = 0;
    int local_size = 0;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_size(node_comm, &local_size);

    log_e5_stage(world_rank, world_size, "OpenMP device enumeration begin");
    const int device_count = omp_get_num_devices();
    log_e5_stage(world_rank, world_size,
                 "OpenMP device enumeration complete: " +
                     std::to_string(device_count) + " device(s)");
    int local_ok = local_size == config.mpi_ranks && device_count == 1;
    device = config.gpu_device_index;
    if (local_ok) omp_set_default_device(device);

    int uses_gpu = 0;
    if (local_ok) {
        int initial_device = 1;
        log_e5_stage(world_rank, world_size, "OpenMP target probe begin");
        #pragma omp target map(from: initial_device)
        {
            initial_device = omp_is_initial_device();
        }
        log_e5_stage(world_rank, world_size,
                     std::string("OpenMP target probe complete: ") +
                         (initial_device == 0 ? "GPU" : "initial device"));
        uses_gpu = initial_device == 0 ? 1 : 0;
        local_ok = uses_gpu;
    }

    const int node_leader = local_rank == 0 ? 1 : 0;
    int actual_nodes = 0;
    int all_ranks_ok = 0;
    log_e5_stage(world_rank, world_size, "GPU mapping MPI reductions begin");
    MPI_Allreduce(&node_leader, &actual_nodes, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_ok, &all_ranks_ok, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    log_e5_stage(world_rank, world_size,
                 "GPU mapping MPI reductions complete: nodes=" +
                     std::to_string(actual_nodes) + ", all_ranks_ok=" +
                     std::to_string(all_ranks_ok));
    if (actual_nodes != 1 || !all_ranks_ok) {
        throw std::runtime_error(
            "E5 requires one node and one GPU shared by all MPI ranks.");
    }

}

std::vector<double> reduce_max_timings(const heat::GpuSolverTiming& timing,
                                       MPI_Comm comm, int root)
{
    const std::array<double, 7> local{{
        timing.solver_time,
        timing.mpi_halo_time,
        timing.gpu_compute_time,
        timing.boundary_d2h_time,
        timing.halo_h2d_time,
        timing.gpu_setup_time,
        timing.gpu_final_copy_time
    }};
    std::array<double, 7> maximum{};
    MPI_Reduce(local.data(), maximum.data(), static_cast<int>(local.size()),
               MPI_DOUBLE, MPI_MAX, root, comm);
    return std::vector<double>(maximum.begin(), maximum.end());
}

std::vector<double> timing_samples(
    const std::vector<heat::GpuHybridRunResult>& runs,
    double heat::GpuHybridRunResult::*member)
{
    std::vector<double> samples;
    samples.reserve(runs.size());
    for (const auto& run : runs) samples.push_back(run.*member);
    return samples;
}

void print_configurations()
{
    for (std::size_t i = 0; i < heat::kE5GpuConfigurations.size(); ++i) {
        const auto& config = heat::kE5GpuConfigurations[i];
        std::cout << i << ' ' << config.group << ' '
                  << config.total_cpu_cores << ' ' << config.mpi_ranks << ' '
                  << config.cpu_threads_per_rank << ' '
                  << config.thread_limit << '\n';
    }
}

void run_configuration(int index, int& argc, char**& argv)
{
    const auto& config = configuration_at(index);
    int provided = MPI_THREAD_SINGLE;
    std::cerr << "[E5] configuration " << index << " MPI_Init_thread begin\n"
              << std::flush;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) !=
        MPI_SUCCESS) {
        throw std::runtime_error("MPI_Init_thread failed.");
    }
    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    log_e5_stage(world_rank, world_size,
                 "configuration " + std::to_string(index) +
                     " start: Group " + config.group + ", " +
                     std::to_string(config.mpi_ranks) + " ranks x " +
                     std::to_string(config.cpu_threads_per_rank) +
                     " CPU threads/rank");
    if (provided < MPI_THREAD_FUNNELED ||
        world_size != config.mpi_ranks) {
        if (world_rank == 0) {
            std::cerr << "MPI thread support or rank count does not match E5 config.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }

    MPI_Comm node_comm = MPI_COMM_NULL;
    heat::MpiDomain domain;
    try {
        log_e5_stage(world_rank, world_size, "matching E4 baseline lookup begin");
        const double cpu_solver = [&] {
            const auto cpu_results = heat::read_e4_csv(heat::kE4OutputCsv);
            const auto found = std::find_if(
                cpu_results.begin(), cpu_results.end(), [&](const auto& result) {
                    return result.total_cores == config.total_cpu_cores &&
                           result.ranks == config.mpi_ranks &&
                           result.threads_per_rank == config.cpu_threads_per_rank;
                });
            if (found == cpu_results.end() || found->steps != heat::kE5Steps ||
                found->repetitions != heat::kE5Repeats ||
                !found->validation.passed ||
                !std::isfinite(found->solver_median) ||
                found->solver_median <= 0.0) {
                throw std::runtime_error(
                    "A valid matching CPU E4 baseline is required for every E5 config.");
            }
            return found->solver_median;
        }();
        log_e5_stage(world_rank, world_size,
                     "matching E4 baseline lookup complete");

        int device = -1;
        validate_gpu_mapping(config, device, node_comm, world_rank, world_size);
        omp_set_dynamic(0);
        omp_set_num_threads(config.cpu_threads_per_rank);

        log_e5_stage(world_rank, world_size, "full experiment MPI domain creation begin");
        heat::create_mpi_domain(domain, heat::kE5GlobalNx, heat::kE5GlobalNy);
        log_e5_stage(world_rank, world_size,
                     "full experiment MPI domain creation complete");
        const auto params = heat::make_parameters(
            heat::kE5GlobalNx, heat::kE5GlobalNy, heat::kE5Steps);
        heat::Grid current = heat::make_local_grid(domain);
        heat::Grid next = heat::make_local_grid(domain);
        std::vector<heat::GpuHybridRunResult> runs;
        if (domain.rank == 0) runs.reserve(heat::kE5Repeats);
        bool all_passed = true;
        double global_min = std::numeric_limits<double>::max();
        double global_max = std::numeric_limits<double>::lowest();
        double global_boundary_error = 0.0;
        for (int repeat = 0; repeat < heat::kE5Repeats; ++repeat) {
            log_e5_stage(domain.rank, domain.size,
                         "repeat " + std::to_string(repeat + 1) +
                             " barrier begin");
            MPI_Barrier(domain.cart_comm);
            log_e5_stage(domain.rank, domain.size,
                         "repeat " + std::to_string(repeat + 1) +
                             " barrier complete; grid initialization begin");
            heat::initialize(current, domain.global_nx, domain.global_ny);
            next.data() = current.data();
            log_e5_stage(domain.rank, domain.size,
                         "repeat " + std::to_string(repeat + 1) +
                             " grid initialization complete; GPU solver begin");

            heat::GpuSolverTiming local_timing{};
            heat::solve_mpi_gpu(current, next, params, domain, heat::kE5Steps,
                                device, config.thread_limit, &local_timing);
            log_e5_stage(domain.rank, domain.size,
                         "repeat " + std::to_string(repeat + 1) +
                             " GPU solver complete; validation begin");
            const auto validation = heat::validate_global(current, domain);
            log_e5_stage(domain.rank, domain.size,
                         "repeat " + std::to_string(repeat + 1) +
                             " validation complete: " +
                             (validation.passed ? "PASS" : "FAIL"));
            log_e5_stage(domain.rank, domain.size,
                         "repeat " + std::to_string(repeat + 1) +
                             " timing reduction begin");
            const auto max_times = reduce_max_timings(
                local_timing, domain.cart_comm, 0);
            log_e5_stage(domain.rank, domain.size,
                         "repeat " + std::to_string(repeat + 1) +
                             " timing reduction complete");
            all_passed = all_passed && validation.passed;
            global_min = std::min(global_min, validation.min_temperature);
            global_max = std::max(global_max, validation.max_temperature);
            global_boundary_error = std::max(
                global_boundary_error, validation.max_boundary_error);
            if (domain.rank == 0) {
                runs.push_back({
                    repeat + 1,
                    max_times[0], max_times[1], max_times[2], max_times[3],
                    max_times[4], max_times[5], max_times[6],
                    validation.passed
                });
                std::cerr << "[E5 GPU] Group " << config.group
                          << " | " << config.mpi_ranks << " ranks x "
                          << config.cpu_threads_per_rank << " CPU threads/rank"
                          << " | Repeat " << repeat + 1 << '/'
                          << heat::kE5Repeats
                          << " complete | solver " << std::fixed
                          << std::setprecision(3) << max_times[0] << " s\n"
                          << std::flush;
            }
        }

        int exit_code = 0;
        if (domain.rank == 0) {
            if (!all_passed) {
                throw std::runtime_error(
                    "E5 GPU full-run validation failed; result was not written.");
            }
            heat::GpuHybridExperimentResult result{};
            result.group = config.group;
            result.total_cpu_cores = config.total_cpu_cores;
            result.mpi_ranks = config.mpi_ranks;
            result.gpu_device_index = config.gpu_device_index;
            result.cpu_threads_per_rank = config.cpu_threads_per_rank;
            result.gpu_thread_limit = config.thread_limit;
            result.process_rows = domain.dims[0];
            result.process_cols = domain.dims[1];
            result.steps = heat::kE5Steps;
            result.repetitions = heat::kE5Repeats;
            result.cpu_solver_median = cpu_solver;
            result.solver_time = heat::summarize_timings(
                timing_samples(runs, &heat::GpuHybridRunResult::solver_time));
            result.mpi_halo_time = heat::summarize_timings(
                timing_samples(runs, &heat::GpuHybridRunResult::mpi_halo_time));
            result.gpu_compute_time = heat::summarize_timings(
                timing_samples(runs, &heat::GpuHybridRunResult::gpu_compute_time));
            result.boundary_d2h_time = heat::summarize_timings(
                timing_samples(runs, &heat::GpuHybridRunResult::boundary_d2h_time));
            result.halo_h2d_time = heat::summarize_timings(
                timing_samples(runs, &heat::GpuHybridRunResult::halo_h2d_time));
            result.gpu_setup_time = heat::summarize_timings(
                timing_samples(runs, &heat::GpuHybridRunResult::gpu_setup_time));
            result.gpu_final_copy_time = heat::summarize_timings(
                timing_samples(runs, &heat::GpuHybridRunResult::gpu_final_copy_time));
            result.validation = {all_passed, global_min, global_max,
                                 global_boundary_error};
            result.runs = std::move(runs);

            const auto previous = std::filesystem::exists(heat::kE5GpuOutputCsv)
                ? heat::read_e5_gpu_csv(heat::kE5GpuOutputCsv)
                : std::vector<heat::GpuHybridExperimentResult>{};
            auto metrics = previous;
            metrics.push_back(result);
            heat::calculate_e5_gpu_metrics(metrics);
            const auto calculated = std::find_if(
                metrics.begin(), metrics.end(), [&](const auto& candidate) {
                    return candidate.group == result.group &&
                           candidate.mpi_ranks == result.mpi_ranks &&
                           candidate.cpu_threads_per_rank ==
                               result.cpu_threads_per_rank;
                });
            if (calculated == metrics.end()) {
                throw std::runtime_error("Could not calculate E5 speedup metrics.");
            }
            result.cpu_gpu_speedup = calculated->cpu_gpu_speedup;
            result.within_group_configuration_speedup =
                calculated->within_group_configuration_speedup;
            heat::append_e5_gpu_csv(result, heat::kE5GpuOutputCsv);
            heat::append_e5_gpu_runs_csv(result, heat::kE5GpuRunsCsv);
            std::cerr << "\r[E5 GPU] Group " << config.group
                      << " | " << config.mpi_ranks << " ranks x "
                      << config.cpu_threads_per_rank << " threads/rank"
                      << " | 1 GPU"
                      << " | Median solver " << std::fixed
                      << std::setprecision(3) << result.solver_time.median
                      << " s | CPU/GPU " << std::setprecision(3)
                      << result.cpu_gpu_speedup << "x                 \n";
        }
        log_e5_stage(domain.rank, domain.size,
                     "configuration result broadcast begin");
        MPI_Bcast(&exit_code, 1, MPI_INT, 0, domain.cart_comm);
        log_e5_stage(domain.rank, domain.size,
                     "configuration result broadcast complete; MPI cleanup begin");
        heat::destroy_mpi_domain(domain);
        MPI_Comm_free(&node_comm);
        log_e5_stage(domain.rank, domain.size,
                     "MPI communicators released; MPI_Finalize begin");
        MPI_Finalize();
        if (exit_code != 0) throw std::runtime_error("E5 GPU configuration failed.");
    } catch (const std::exception& error) {
        std::cerr << "Rank " << world_rank << " E5 GPU error: "
                  << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void print_summary()
{
    auto results = heat::read_e5_gpu_csv(heat::kE5GpuOutputCsv);
    if (results.size() != heat::kE5GpuConfigurations.size()) {
        throw std::runtime_error("E5 GPU summary requires all five Group A configurations.");
    }
    heat::calculate_e5_gpu_metrics(results);
    heat::print_e5_gpu_summary(results);
}

} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc == 2 && std::string(argv[1]) == "--list-configurations") {
            print_configurations();
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--summary") {
            print_summary();
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--run") {
            run_configuration(std::stoi(argv[2]), argc, argv);
            return 0;
        }
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " --list-configurations\n"
                  << "  " << argv[0] << " --run <configuration-index>\n"
                  << "  " << argv[0] << " --summary\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "E5 GPU error: " << error.what() << '\n';
        return 1;
    }
}
