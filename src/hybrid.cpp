#include "benchmark.hpp"
#include "config.hpp"
#include "mpi.hpp"
#include "output.hpp"
#include "stencil.hpp"
#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

namespace {

const heat::HybridConfiguration* find_configuration(int ranks, int threads)
{
    const auto found = std::find_if(
        heat::kE4Configurations.begin(), heat::kE4Configurations.end(),
        [&](const auto& config) {
            return config.ranks == ranks && config.threads_per_rank == threads;
        });
    return found == heat::kE4Configurations.end() ? nullptr : &*found;
}

void print_usage(const char* program)
{
    std::cerr << "Usage:\n"
              << "  " << program << " --run <mpi-ranks> <threads-per-rank>\n"
              << "  " << program << " --summary\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc == 2 && std::string(argv[1]) == "--summary") {
        try {
            auto results = heat::read_e4_csv(heat::kE4OutputCsv);
            if (results.size() != heat::kE4Configurations.size()) {
                throw std::runtime_error("E4 requires all 15 configurations before summary.");
            }
            for (const auto& expected : heat::kE4Configurations) {
                const auto found = std::find_if(
                    results.begin(), results.end(), [&](const auto& result) {
                        return result.ranks == expected.ranks &&
                               result.threads_per_rank == expected.threads_per_rank &&
                               result.total_cores == expected.total_cores;
                    });
                if (found == results.end()) {
                    throw std::runtime_error("E4 summary is missing a required configuration.");
                }
                if (found->repetitions != heat::kE4Repeats) {
                    throw std::runtime_error(
                        "E4 configuration does not contain the required number of repetitions.");
                }
                if (!std::isfinite(found->solver_median) || found->solver_median <= 0.0) {
                    throw std::runtime_error(
                        "E4 configuration contains an invalid solver median.");
                }
            }
            heat::calculate_hybrid_metrics(results);
            for (const auto& result : results) {
                if (!std::isfinite(result.configuration_speedup) ||
                    !std::isfinite(result.hybrid_speedup) ||
                    !std::isfinite(result.parallel_efficiency) ||
                    result.configuration_speedup <= 0.0 ||
                    result.hybrid_speedup <= 0.0 ||
                    result.parallel_efficiency <= 0.0) {
                    throw std::runtime_error(
                        "E4 metric calculation produced an invalid speedup or efficiency.");
                }
            }
            heat::write_e4_csv(results, heat::kE4OutputCsv);
            heat::print_hybrid_summary(results);
            return std::all_of(results.begin(), results.end(), [](const auto& result) {
                return result.validation.passed;
            }) ? 0 : 2;
        }
        catch (const std::exception& error) {
            std::cerr << "E4 summary error: " << error.what() << '\n';
            return 1;
        }
    }

    if (argc != 4 || std::string(argv[1]) != "--run") {
        print_usage(argv[0]);
        return 2;
    }

    const int requested_ranks = std::stoi(argv[2]);
    const int threads = std::stoi(argv[3]);
    const heat::HybridConfiguration* configuration =
        find_configuration(requested_ranks, threads);
    if (configuration == nullptr) {
        std::cerr << "Unsupported E4 configuration: ranks=" << requested_ranks
                  << ", threads/rank=" << threads << '\n';
        return 2;
    }

    int provided = MPI_THREAD_SINGLE;
    const int init_status = MPI_Init_thread(&argc, &argv,
                                            MPI_THREAD_FUNNELED, &provided);
    if (init_status != MPI_SUCCESS) {
        std::cerr << "MPI_Init_thread failed.\n";
        return 1;
    }

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int exit_code = 0;
    heat::MpiDomain domain;

    try {
        if (provided < MPI_THREAD_FUNNELED) {
            throw std::runtime_error("MPI does not provide MPI_THREAD_FUNNELED.");
        }
        if (world_size != configuration->ranks) {
            throw std::runtime_error("MPI world size does not match requested E4 ranks.");
        }

        omp_set_dynamic(0);
        omp_set_num_threads(threads);

        heat::create_mpi_domain(domain, heat::kE4GlobalNx, heat::kE4GlobalNy);
        const heat::NumericalParameters params =
            heat::make_parameters(heat::kE4GlobalNx, heat::kE4GlobalNy,
                                  heat::kSteps);
        heat::Grid current = heat::make_local_grid(domain);
        heat::Grid next = heat::make_local_grid(domain);

        std::vector<heat::HybridRunResult> runs;
        if (domain.rank == 0) runs.reserve(heat::kE4Repeats);

        bool all_passed = true;
        double global_min = std::numeric_limits<double>::max();
        double global_max = std::numeric_limits<double>::lowest();
        double global_boundary_error = 0.0;
        const double experiment_start = MPI_Wtime();

        for (int repeat = 0; repeat < heat::kE4Repeats; ++repeat) {
            MPI_Barrier(domain.cart_comm);
            const double total_start = MPI_Wtime();
            const double init_start = MPI_Wtime();
            heat::initialize(current, domain.global_nx, domain.global_ny);
            next.data() = current.data();
            const double local_initialization = MPI_Wtime() - init_start;

            MPI_Barrier(domain.cart_comm);
            const double solver_start = MPI_Wtime();
            double local_halo = 0.0;
            double local_ftcs = 0.0;
            for (int step = 0; step < heat::kSteps; ++step) {
                heat::exchange_halos(current, domain, &local_halo);
                const double ftcs_start = MPI_Wtime();
                heat::ftcs_step_openmp(current, next, params,
                                       domain.global_nx, domain.global_ny, threads);
                local_ftcs += MPI_Wtime() - ftcs_start;
                current.swap_data(next);
            }
            const double local_solver = MPI_Wtime() - solver_start;

            const double validation_start = MPI_Wtime();
            const heat::ValidationResult validation =
                heat::validate_global(current, domain);
            const double local_validation = MPI_Wtime() - validation_start;
            const double local_total = MPI_Wtime() - total_start;

            const double local_times[6] = {
                local_initialization, local_solver, local_halo,
                local_ftcs, local_validation, local_total
            };
            double max_times[6] = {};
            MPI_Reduce(local_times, max_times, 6, MPI_DOUBLE, MPI_MAX,
                       0, domain.cart_comm);

            struct {
                double value;
                int rank;
            } candidate{local_solver, domain.rank}, critical{};
            MPI_Allreduce(&candidate, &critical, 1, MPI_DOUBLE_INT,
                          MPI_MAXLOC, domain.cart_comm);
            double critical_halo = 0.0;
            if (domain.rank == critical.rank) critical_halo = local_halo;
            MPI_Bcast(&critical_halo, 1, MPI_DOUBLE, critical.rank, domain.cart_comm);

            all_passed = all_passed && validation.passed;
            global_min = std::min(global_min, validation.min_temperature);
            global_max = std::max(global_max, validation.max_temperature);
            global_boundary_error = std::max(
                global_boundary_error, validation.max_boundary_error);

            if (domain.rank == 0) {
                runs.push_back({repeat + 1, max_times[0], max_times[1],
                                max_times[2], max_times[3], max_times[4],
                                max_times[5],
                                max_times[1] > 0.0 ? critical_halo / max_times[1] : 0.0,
                                validation.passed});
                std::cerr << "\r[E4] Cores " << configuration->total_cores
                          << " | Ranks " << configuration->ranks
                          << " | Threads/Rank " << configuration->threads_per_rank
                          << " | Repeat " << repeat + 1 << '/' << heat::kE4Repeats
                          << std::flush;
            }
        }

        if (domain.rank == 0) {
            std::vector<double> initialization, solver, halo, ftcs;
            std::vector<double> validation, total, halo_ratio;
            for (const auto& run : runs) {
                initialization.push_back(run.initialization);
                solver.push_back(run.solver);
                halo.push_back(run.halo);
                ftcs.push_back(run.ftcs);
                validation.push_back(run.validation);
                total.push_back(run.total);
                halo_ratio.push_back(run.halo_ratio);
            }

            heat::HybridExperimentResult result{};
            result.total_cores = configuration->total_cores;
            result.ranks = configuration->ranks;
            result.threads_per_rank = configuration->threads_per_rank;
            result.process_rows = domain.dims[0];
            result.process_cols = domain.dims[1];
            result.steps = heat::kSteps;
            result.repetitions = heat::kE4Repeats;
            result.initialization_median = heat::median(initialization);
            result.initialization_mean = heat::mean(initialization);
            result.initialization_stddev =
                heat::sample_stddev(initialization, result.initialization_mean);
            result.solver_median = heat::median(solver);
            result.solver_mean = heat::mean(solver);
            result.solver_stddev = heat::sample_stddev(solver, result.solver_mean);
            result.halo_median = heat::median(halo);
            result.halo_mean = heat::mean(halo);
            result.halo_stddev = heat::sample_stddev(halo, result.halo_mean);
            result.ftcs_median = heat::median(ftcs);
            result.ftcs_mean = heat::mean(ftcs);
            result.ftcs_stddev = heat::sample_stddev(ftcs, result.ftcs_mean);
            result.validation_median = heat::median(validation);
            result.validation_mean = heat::mean(validation);
            result.total_median = heat::median(total);
            result.total_mean = heat::mean(total);
            result.halo_ratio_median = heat::median(halo_ratio);
            result.validation = {all_passed, global_min, global_max,
                                global_boundary_error};
            result.runs = std::move(runs);

            heat::append_e4_runs_csv(result, heat::kE4RunsCsv);
            heat::append_e4_csv(result, heat::kE4OutputCsv);

            // Report completion only after both result files were written successfully.
            std::cerr << "\r[E4] Cores " << configuration->total_cores
                      << " | Ranks " << configuration->ranks
                      << " | Threads/Rank " << configuration->threads_per_rank
                      << " | " << heat::kE4Repeats << '/' << heat::kE4Repeats
                      << " completed | Finished in " << std::fixed
                      << std::setprecision(3) << MPI_Wtime() - experiment_start
                      << " s          \n";

            if (!all_passed) exit_code = 2;
        }

        MPI_Bcast(&exit_code, 1, MPI_INT, 0, domain.cart_comm);
    }
    catch (const std::exception& error) {
        std::cerr << "Rank " << world_rank << " error: " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit_code = 1;
    }

    heat::destroy_mpi_domain(domain);
    MPI_Finalize();
    return exit_code;
}
