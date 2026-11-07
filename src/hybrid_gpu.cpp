#include "benchmark.hpp"
#include "config.hpp"
#include "gpu_solver.hpp"
#include "mpi.hpp"
#include "output.hpp"
#include "stencil.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
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

const Configuration& configuration_at(int index)
{
    if (index < 0 ||
        index >= static_cast<int>(heat::kE5GpuConfigurations.size())) {
        throw std::invalid_argument("E5 GPU configuration index is out of range.");
    }
    return heat::kE5GpuConfigurations[static_cast<std::size_t>(index)];
}

int visible_cuda_device_count()
{
    const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
    if (visible == nullptr || *visible == '\0') return -1;
    int count = 1;
    for (const char* cursor = visible; *cursor != '\0'; ++cursor) {
        if (*cursor == ',') ++count;
    }
    return count;
}

void validate_gpu_mapping(const Configuration& config, int& device,
                          MPI_Comm& node_comm, int world_rank)
{
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank,
                        MPI_INFO_NULL, &node_comm);
    int local_rank = 0;
    int local_size = 0;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_size(node_comm, &local_size);

    const int devices_per_node = config.gpu_count / config.node_count;
    const int process_visible_devices = omp_get_num_devices();
    int local_ok = local_size == config.ranks_per_node &&
                   process_visible_devices > config.gpu_device_index;
    const int visible_count = visible_cuda_device_count();
    if (visible_count >= 0 &&
        visible_count != devices_per_node) {
        local_ok = 0;
    } else if (visible_count < 0 &&
               process_visible_devices != devices_per_node) {
        local_ok = 0;
    }
    device = config.gpu_device_index;
    if (local_ok) omp_set_default_device(device);

    int uses_gpu = 0;
    if (local_ok) {
        int initial_device = 1;
        #pragma omp target map(from: initial_device)
        {
            initial_device = omp_is_initial_device();
        }
        uses_gpu = initial_device == 0 ? 1 : 0;
        local_ok = uses_gpu;
    }

    const int node_leader = local_rank == 0 ? 1 : 0;
    int actual_nodes = 0;
    int all_ranks_ok = 0;
    MPI_Allreduce(&node_leader, &actual_nodes, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_ok, &all_ranks_ok, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    if (actual_nodes != config.node_count || !all_ranks_ok) {
        throw std::runtime_error(
            "E5 GPU allocation, rank placement, or OpenMP device mapping "
            "does not match this configuration.");
    }

    int local_cuda_visible = visible_count;
    int min_visible = 0;
    int max_visible = 0;
    MPI_Allreduce(&local_cuda_visible, &min_visible, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_cuda_visible, &max_visible, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    if (min_visible >= 0 && max_visible >= 0 && min_visible != max_visible) {
        throw std::runtime_error(
            "Ranks report different CUDA_VISIBLE_DEVICES allocations.");
    }
}

heat::ReferenceComparison run_reference_cases(const Configuration& config,
                                               int device,
                                               MPI_Comm comm)
{
    heat::MpiDomain test_domain;
    heat::create_mpi_domain(test_domain, 128, 130, comm);
    heat::ReferenceComparison worst{0.0, 0.0, true};
    for (int steps : {1, 2}) {
        const auto params = heat::make_parameters(128, 130, steps);
        heat::Grid cpu = heat::make_local_grid(test_domain);
        heat::Grid cpu_next = heat::make_local_grid(test_domain);
        heat::Grid gpu = heat::make_local_grid(test_domain);
        heat::Grid gpu_next = heat::make_local_grid(test_domain);
        heat::initialize(cpu, 128, 130);
        cpu_next.data() = cpu.data();
        gpu.data() = cpu.data();
        gpu_next.data() = gpu.data();

        heat::solve_mpi(cpu, cpu_next, params, test_domain, steps);
        heat::solve_mpi_gpu(gpu, gpu_next, params, test_domain, steps, device,
                            config.thread_limit);
        const auto comparison = heat::compare_global_reference(
            gpu, cpu, test_domain, heat::kE5MaxAbsoluteErrorTolerance,
            heat::kE5RelativeL2ErrorTolerance);
        const auto validation = heat::validate_global(gpu, test_domain);
        worst.max_absolute_error =
            std::max(worst.max_absolute_error, comparison.max_absolute_error);
        worst.relative_l2_error =
            std::max(worst.relative_l2_error, comparison.relative_l2_error);
        worst.passed = worst.passed && comparison.passed && validation.passed;
    }
    heat::destroy_mpi_domain(test_domain);
    if (!worst.passed) {
        throw std::runtime_error(
            "GPU result failed odd/even-step CPU-reference comparison.");
    }
    return worst;
}

heat::ReferenceComparison compare_full_cpu_reference(
    const heat::Grid& gpu_grid, const heat::NumericalParameters& params,
    const heat::MpiDomain& domain, int cpu_threads)
{
    heat::Grid cpu = heat::make_local_grid(domain);
    heat::Grid cpu_next = heat::make_local_grid(domain);
    heat::initialize(cpu, domain.global_nx, domain.global_ny);
    cpu_next.data() = cpu.data();
    for (int step = 0; step < heat::kE5Steps; ++step) {
        heat::exchange_halos(cpu, domain);
        heat::ftcs_step_openmp(cpu, cpu_next, params, domain.global_nx,
                               domain.global_ny, cpu_threads);
        cpu.swap_data(cpu_next);
    }
    const auto cpu_validation = heat::validate_global(cpu, domain);
    auto comparison = heat::compare_global_reference(
        gpu_grid, cpu, domain, heat::kE5MaxAbsoluteErrorTolerance,
        heat::kE5RelativeL2ErrorTolerance);
    comparison.passed = comparison.passed && cpu_validation.passed;
    return comparison;
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
                  << config.total_cpu_cores << ' ' << config.gpu_count << ' '
                  << config.node_count << ' ' << config.mpi_ranks << ' '
                  << config.ranks_per_node << ' '
                  << config.cpu_threads_per_rank << ' '
                  << config.thread_limit << '\n';
    }
}

void run_correctness_test(int& argc, char**& argv)
{
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) !=
        MPI_SUCCESS) {
        throw std::runtime_error("MPI_Init_thread failed in --test.");
    }
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int local_ok = provided >= MPI_THREAD_FUNNELED &&
                   omp_get_num_devices() > 0;
    int all_ok = 0;
    MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!all_ok) {
        if (rank == 0) std::cerr << "GPU correctness test has no device/runtime.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }
    omp_set_default_device(0);
    const auto& config = heat::kE5GpuConfigurations.front();
    try {
        const auto comparison = run_reference_cases(
            config, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "PASS E5 GPU odd/even steps, " << [&] {
                int size = 0;
                MPI_Comm_size(MPI_COMM_WORLD, &size);
                return size;
            }() << " MPI rank(s); max_abs="
                      << comparison.max_absolute_error
                      << ", relative_l2=" << comparison.relative_l2_error << '\n';
        }
        MPI_Finalize();
    } catch (const std::exception& error) {
        std::cerr << "Rank " << rank << " GPU correctness test error: "
                  << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void run_configuration(int index, int& argc, char**& argv)
{
    const auto& config = configuration_at(index);
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) !=
        MPI_SUCCESS) {
        throw std::runtime_error("MPI_Init_thread failed.");
    }
    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
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

        int device = -1;
        validate_gpu_mapping(config, device, node_comm, world_rank);
        omp_set_dynamic(0);
        omp_set_num_threads(config.cpu_threads_per_rank);
        const auto reference = run_reference_cases(
            config, device, MPI_COMM_WORLD);

        heat::create_mpi_domain(domain, heat::kE5GlobalNx, heat::kE5GlobalNy);
        const auto params = heat::make_parameters(
            heat::kE5GlobalNx, heat::kE5GlobalNy, heat::kE5Steps);
        heat::Grid current = heat::make_local_grid(domain);
        heat::Grid next = heat::make_local_grid(domain);
        std::vector<heat::GpuHybridRunResult> runs;
        if (domain.rank == 0) runs.reserve(heat::kE5Repeats);

        bool all_passed = true;
        heat::ReferenceComparison full_reference{0.0, 0.0, true};
        double global_min = std::numeric_limits<double>::max();
        double global_max = std::numeric_limits<double>::lowest();
        double global_boundary_error = 0.0;
        for (int repeat = 0; repeat < heat::kE5Repeats; ++repeat) {
            MPI_Barrier(domain.cart_comm);
            heat::initialize(current, domain.global_nx, domain.global_ny);
            next.data() = current.data();

            heat::GpuSolverTiming local_timing{};
            heat::solve_mpi_gpu(current, next, params, domain, heat::kE5Steps,
                                device, config.thread_limit, &local_timing);
            const auto validation = heat::validate_global(current, domain);
            if (repeat == 0) {
                full_reference = compare_full_cpu_reference(
                    current, params, domain, config.cpu_threads_per_rank);
                if (!full_reference.passed) {
                    throw std::runtime_error(
                        "Full 4096x4096 GPU result failed CPU-reference comparison.");
                }
            }
            const auto max_times = reduce_max_timings(
                local_timing, domain.cart_comm, 0);
            all_passed = all_passed && validation.passed &&
                         reference.passed && full_reference.passed;
            global_min = std::min(global_min, validation.min_temperature);
            global_max = std::max(global_max, validation.max_temperature);
            global_boundary_error = std::max(
                global_boundary_error, validation.max_boundary_error);
            if (domain.rank == 0) {
                runs.push_back({
                    repeat + 1,
                    max_times[0], max_times[1], max_times[2], max_times[3],
                    max_times[4], max_times[5], max_times[6],
                    full_reference.max_absolute_error,
                    full_reference.relative_l2_error,
                    validation.passed && reference.passed && full_reference.passed
                });
                std::cerr << "\r[E5 GPU] Group " << config.group
                          << " | " << config.mpi_ranks << " ranks x "
                          << config.cpu_threads_per_rank << " CPU threads/rank"
                          << " | Repeat " << repeat + 1 << '/'
                          << heat::kE5Repeats << std::flush;
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
            result.gpu_count = config.gpu_count;
            result.node_count = config.node_count;
            result.mpi_ranks = config.mpi_ranks;
            result.ranks_per_node = config.ranks_per_node;
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
            result.max_absolute_error = full_reference.max_absolute_error;
            result.relative_l2_error = full_reference.relative_l2_error;
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
            result.cross_group_gpu_scaling_speedup =
                calculated->cross_group_gpu_scaling_speedup;
            heat::append_e5_gpu_csv(result, heat::kE5GpuOutputCsv);
            heat::append_e5_gpu_runs_csv(result, heat::kE5GpuRunsCsv);
            std::cerr << "\r[E5 GPU] Group " << config.group
                      << " | " << config.mpi_ranks << " ranks x "
                      << config.cpu_threads_per_rank << " threads/rank"
                      << " | " << config.gpu_count << " GPU(s)"
                      << " | Median solver " << std::fixed
                      << std::setprecision(3) << result.solver_time.median
                      << " s | CPU/GPU " << std::setprecision(3)
                      << result.cpu_gpu_speedup << "x                 \n";
        }
        MPI_Bcast(&exit_code, 1, MPI_INT, 0, domain.cart_comm);
        heat::destroy_mpi_domain(domain);
        MPI_Comm_free(&node_comm);
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
        throw std::runtime_error("E5 GPU summary requires all 15 configurations.");
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
        if (argc == 2 && std::string(argv[1]) == "--test") {
            run_correctness_test(argc, argv);
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--run") {
            run_configuration(std::stoi(argv[2]), argc, argv);
            return 0;
        }
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " --list-configurations\n"
                  << "  " << argv[0] << " --test\n"
                  << "  " << argv[0] << " --run <configuration-index>\n"
                  << "  " << argv[0] << " --summary\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "E5 GPU error: " << error.what() << '\n';
        return 1;
    }
}
