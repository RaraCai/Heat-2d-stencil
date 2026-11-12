#include "config.hpp"
#include "gpu_solver.hpp"
#include "mpi.hpp"
#include "stencil.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
constexpr int kCpuThreads = 32;
constexpr int kWarmupSteps = 5;
constexpr int kRepetitions = 10;
constexpr int kGpuThreadLimit = heat::kE5GpuThreadLimit;
constexpr const char* kSummaryPath =
    "results/raw/e5_single_rank_gpu_optimized.csv";
constexpr const char* kRunsPath =
    "results/raw/e5_single_rank_gpu_optimized_runs.csv";

struct RunResult {
    int repetition = 0;
    double cpu_loop = 0.0;
    double gpu_iteration = 0.0;
    double gpu_end_to_end = 0.0;
    double gpu_compute = 0.0;
    double gpu_setup = 0.0;
    double gpu_final_copy = 0.0;
    double max_abs_error = 0.0;
    double relative_l2_error = 0.0;
    bool passed = false;
};

double median(std::vector<double> values)
{
    if (values.empty()) throw std::invalid_argument("median requires samples");
    std::sort(values.begin(), values.end());
    const std::size_t m = values.size() / 2;
    return values.size() % 2 ? values[m] : 0.5 * (values[m - 1] + values[m]);
}

double mean(const std::vector<double>& values)
{
    double sum = 0.0;
    for (double value : values) sum += value;
    return sum / static_cast<double>(values.size());
}

double sample_stddev(const std::vector<double>& values)
{
    if (values.size() < 2) return 0.0;
    const double average = mean(values);
    double squared_deviations = 0.0;
    for (double value : values) {
        const double difference = value - average;
        squared_deviations += difference * difference;
    }
    return std::sqrt(squared_deviations /
                     static_cast<double>(values.size() - 1));
}

void initialize_pair(heat::Grid& current, heat::Grid& next,
                     const heat::MpiDomain& domain)
{
    heat::initialize(current, domain.global_nx, domain.global_ny);
    next.data() = current.data();
}

void run_cpu_loop(heat::Grid& current, heat::Grid& next,
                  const heat::NumericalParameters& params,
                  const heat::MpiDomain& domain, int steps)
{
    // One MPI rank has no remote halo neighbors.
    for (int step = 0; step < steps; ++step) {
        heat::ftcs_step_openmp(current, next, params,
                               domain.global_nx, domain.global_ny, kCpuThreads);
        current.swap_data(next);
    }
}

void write_results(const std::vector<RunResult>& runs, bool all_passed)
{
    std::filesystem::create_directories("results/raw");
    if (std::filesystem::exists(kSummaryPath) || std::filesystem::exists(kRunsPath))
        throw std::runtime_error("Single-rank result files already exist; archive them first.");

    std::vector<double> cpu, gpu_iter, gpu_e2e, gpu_compute;
    double max_abs = 0.0, max_rel = 0.0;
    for (const auto& r : runs) {
        cpu.push_back(r.cpu_loop);
        gpu_iter.push_back(r.gpu_iteration);
        gpu_e2e.push_back(r.gpu_end_to_end);
        gpu_compute.push_back(r.gpu_compute);
        max_abs = std::max(max_abs, r.max_abs_error);
        max_rel = std::max(max_rel, r.relative_l2_error);
    }

    {
        std::ofstream out(kRunsPath);
        if (!out) throw std::runtime_error("Cannot create single-rank runs CSV.");
        out << "repetition,cpu_loop_s,gpu_iteration_s,gpu_end_to_end_s,gpu_compute_s,"
               "gpu_setup_s,gpu_final_copy_s,max_absolute_error,relative_l2_error,validation_passed\n";
        out << std::setprecision(17);
        for (const auto& r : runs)
            out << r.repetition << ',' << r.cpu_loop << ',' << r.gpu_iteration << ','
                << r.gpu_end_to_end << ',' << r.gpu_compute << ',' << r.gpu_setup << ','
                << r.gpu_final_copy << ',' << r.max_abs_error << ','
                << r.relative_l2_error << ',' << (r.passed ? 1 : 0) << '\n';
    }
    {
        std::ofstream out(kSummaryPath);
        if (!out) throw std::runtime_error("Cannot create single-rank summary CSV.");
        out << "mpi_ranks,cpu_threads,gpu_devices,global_nx,global_ny,steps,repetitions,"
               "cpu_loop_median_s,cpu_loop_mean_s,cpu_loop_stddev_s,"
               "gpu_iteration_median_s,gpu_iteration_mean_s,gpu_iteration_stddev_s,"
               "gpu_end_to_end_median_s,gpu_end_to_end_mean_s,gpu_end_to_end_stddev_s,"
               "gpu_kernel_median_s,gpu_kernel_mean_s,gpu_kernel_stddev_s,"
               "cpu_over_gpu_iteration_speedup,max_absolute_error,relative_l2_error,validation_passed\n";
        out << std::setprecision(17)
            << 1 << ',' << kCpuThreads << ',' << 1 << ','
            << heat::kE5GlobalNx << ',' << heat::kE5GlobalNy << ','
            << heat::kE5Steps << ',' << kRepetitions << ','
            << median(cpu) << ',' << mean(cpu) << ',' << sample_stddev(cpu) << ','
            << median(gpu_iter) << ',' << mean(gpu_iter) << ','
            << sample_stddev(gpu_iter) << ','
            << median(gpu_e2e) << ',' << mean(gpu_e2e) << ','
            << sample_stddev(gpu_e2e) << ','
            << median(gpu_compute) << ',' << mean(gpu_compute) << ','
            << sample_stddev(gpu_compute) << ','
            << median(cpu) / median(gpu_iter) << ',' << max_abs << ',' << max_rel << ','
            << (all_passed ? 1 : 0) << '\n';
    }
}
} // namespace

int main(int argc, char** argv)
{
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        std::cerr << "MPI_Init_thread failed.\n";
        return 1;
    }
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
        if (provided < MPI_THREAD_FUNNELED || size != 1)
            throw std::runtime_error("Requires exactly one MPI rank and MPI_THREAD_FUNNELED.");
        if (omp_get_num_devices() != 1)
            throw std::runtime_error("Requires exactly one visible OpenMP GPU device.");
        if (std::filesystem::exists(kSummaryPath) || std::filesystem::exists(kRunsPath))
            throw std::runtime_error("Single-rank CSVs already exist; archive them before rerunning.");

        omp_set_dynamic(0);
        omp_set_num_threads(kCpuThreads);
        omp_set_default_device(0);
        int initial_device = 1;
        #pragma omp target map(from: initial_device)
        { initial_device = omp_is_initial_device(); }
        if (initial_device != 0)
            throw std::runtime_error("OpenMP target did not execute on the GPU.");

        heat::MpiDomain domain;
        heat::GpuSolverWorkspace gpu_workspace;
        heat::create_mpi_domain(domain, heat::kE5GlobalNx, heat::kE5GlobalNy);
        const auto params = heat::make_parameters(
            heat::kE5GlobalNx, heat::kE5GlobalNy, heat::kE5Steps);

        std::cout << "E5 single-rank CPU/GPU diagnostic\n"
                  << "MPI ranks: 1 | CPU threads: " << kCpuThreads << " | GPUs: 1\n"
                  << "Grid: " << heat::kE5GlobalNx << 'x' << heat::kE5GlobalNy
                  << " | Steps: " << heat::kE5Steps
                  << " | Repetitions: " << kRepetitions << '\n';

        // Untimed warm-up for both paths.
        {
            auto cpu = heat::make_local_grid(domain);
            auto cpu_next = heat::make_local_grid(domain);
            auto gpu = heat::make_local_grid(domain);
            auto gpu_next = heat::make_local_grid(domain);
            initialize_pair(cpu, cpu_next, domain);
            initialize_pair(gpu, gpu_next, domain);
            run_cpu_loop(cpu, cpu_next, params, domain, kWarmupSteps);
            heat::solve_mpi_gpu(gpu, gpu_next, params, domain,
                                kWarmupSteps, 0, kGpuThreadLimit, nullptr,
                                &gpu_workspace);
        }

        std::vector<RunResult> runs;
        runs.reserve(kRepetitions);
        for (int rep = 1; rep <= kRepetitions; ++rep) {
            auto cpu = heat::make_local_grid(domain);
            auto cpu_next = heat::make_local_grid(domain);
            auto gpu = heat::make_local_grid(domain);
            auto gpu_next = heat::make_local_grid(domain);
            initialize_pair(cpu, cpu_next, domain);
            initialize_pair(gpu, gpu_next, domain);

            double cpu_elapsed = 0.0;
            heat::GpuSolverTiming gpu_timing{};
            auto measure_cpu = [&]() {
                MPI_Barrier(domain.cart_comm);
                const double start = MPI_Wtime();
                run_cpu_loop(cpu, cpu_next, params, domain, heat::kE5Steps);
                cpu_elapsed = MPI_Wtime() - start;
            };
            auto measure_gpu = [&]() {
                MPI_Barrier(domain.cart_comm);
                heat::solve_mpi_gpu(gpu, gpu_next, params, domain,
                                    heat::kE5Steps, 0, kGpuThreadLimit,
                                    &gpu_timing, &gpu_workspace);
            };
            if (rep % 2) { measure_cpu(); measure_gpu(); }
            else { measure_gpu(); measure_cpu(); }

            const auto errors = heat::compare_global_reference(
                gpu, cpu, domain, heat::kE5MaxAbsoluteErrorTolerance,
                heat::kE5RelativeL2ErrorTolerance);
            const auto physical = heat::validate_global(gpu, domain);
            const bool passed = errors.passed && physical.passed;
            runs.push_back({rep, cpu_elapsed, gpu_timing.iteration_time,
                            gpu_timing.solver_time, gpu_timing.gpu_compute_time,
                            gpu_timing.gpu_setup_time, gpu_timing.gpu_final_copy_time,
                            errors.max_absolute_error, errors.relative_l2_error, passed});
            std::cout << "Repetition " << rep << '/' << kRepetitions
                      << " | CPU loop " << std::fixed << std::setprecision(4)
                      << cpu_elapsed << " s | GPU iteration " << gpu_timing.iteration_time
                      << " s | GPU end-to-end " << gpu_timing.solver_time << " s | "
                      << (passed ? "PASS" : "FAIL") << '\n';
        }

        const bool all_passed = std::all_of(runs.begin(), runs.end(),
            [](const RunResult& r) { return r.passed; });
        write_results(runs, all_passed);
        std::vector<double> cpu, gpu_iter, gpu_e2e;
        for (const auto& r : runs) {
            cpu.push_back(r.cpu_loop);
            gpu_iter.push_back(r.gpu_iteration);
            gpu_e2e.push_back(r.gpu_end_to_end);
        }
        std::cout << "\nCPU loop median: " << median(cpu) << " s\n"
                  << "GPU iteration median: " << median(gpu_iter) << " s\n"
                  << "GPU end-to-end median: " << median(gpu_e2e) << " s\n"
                  << "CPU/GPU iteration speedup: " << median(cpu) / median(gpu_iter) << "x\n"
                  << "Numerical validation: " << (all_passed ? "PASS" : "FAIL") << '\n'
                  << "Summary: " << kSummaryPath << '\n'
                  << "Runs: " << kRunsPath << '\n';
        heat::destroy_mpi_domain(domain);
        MPI_Finalize();
        return all_passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Single-rank GPU diagnostic error: " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
