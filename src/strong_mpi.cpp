#include "benchmark.hpp"
#include "config.hpp"
#include "mpi.hpp"
#include "output.hpp"
#include "stencil.hpp"
#include <algorithm>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool supported_rank_count(int ranks)
{
    return std::find(heat::kE2RankCounts.begin(),
                     heat::kE2RankCounts.end(), ranks) !=
           heat::kE2RankCounts.end();
}

} // namespace

int main(int argc, char** argv)
{
    if (argc == 2 && std::string(argv[1]) == "--summary") {
        try {
            const double serial_baseline = heat::read_e1_serial_baseline(
                heat::kE1BaselineCsv, heat::kE2GlobalNx);
            const auto results = heat::read_e2_csv(heat::kE2OutputCsv);
            heat::print_strong_scaling_summary(results, serial_baseline);
            return std::all_of(
                results.begin(), results.end(),
                [](const auto& result) { return result.validation.passed; })
                ? 0 : 2;
        } catch (const std::exception& error) {
            std::cerr << "E2 summary error: " << error.what() << '\n';
            return 1;
        }
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
        if (!supported_rank_count(world_size)) {
            throw std::runtime_error(
                "Unsupported E2 rank count. Use 1, 4, 16, 64, or 256.");
        }

        heat::create_mpi_domain(domain,
                                heat::kE2GlobalNx,
                                heat::kE2GlobalNy);

        if (domain.dims[0] != domain.dims[1]) {
            throw std::runtime_error("E2 requires a square Cartesian process grid.");
        }

        const heat::NumericalParameters params =
            heat::make_parameters(heat::kE2GlobalNx,
                                  heat::kE2GlobalNy,
                                  heat::kSteps);

        heat::Grid current = heat::make_local_grid(domain);
        heat::Grid next = heat::make_local_grid(domain);

        std::vector<double> total_samples;
        std::vector<double> halo_samples;
        if (world_rank == 0) {
            total_samples.reserve(heat::kE2Repeats);
            halo_samples.reserve(heat::kE2Repeats);
        }

        bool all_passed = true;
        double global_min = std::numeric_limits<double>::max();
        double global_max = std::numeric_limits<double>::lowest();
        double global_boundary_error = 0.0;
        const double experiment_start = MPI_Wtime();

        for (int repeat = 0; repeat < heat::kE2Repeats; ++repeat) {
            heat::initialize(current, domain.global_nx, domain.global_ny);
            next.data() = current.data();

            heat::MpiTiming timing;
            MPI_Barrier(domain.cart_comm);
            const double start = MPI_Wtime();

            heat::solve_mpi(current, next, params, domain,
                            heat::kSteps, &timing);

            const double local_total = MPI_Wtime() - start;
            double global_total = 0.0;
            double global_halo = 0.0;

            MPI_Reduce(&local_total, &global_total, 1, MPI_DOUBLE,
                       MPI_MAX, 0, domain.cart_comm);
            MPI_Reduce(&timing.halo_exchange_seconds, &global_halo, 1,
                       MPI_DOUBLE, MPI_MAX, 0, domain.cart_comm);

            // Validation is deliberately outside the timed solve interval.
            const heat::ValidationResult validation =
                heat::validate_global(current, domain);

            all_passed = all_passed && validation.passed;
            global_min = std::min(global_min, validation.min_temperature);
            global_max = std::max(global_max, validation.max_temperature);
            global_boundary_error = std::max(
                global_boundary_error, validation.max_boundary_error);

            if (world_rank == 0) {
                total_samples.push_back(global_total);
                halo_samples.push_back(global_halo);
                std::cerr << "\r[E2] Grid "
                          << std::left << std::setw(11)
                          << (std::to_string(heat::kE2GlobalNx) + " x " +
                              std::to_string(heat::kE2GlobalNy))
                          << " | Ranks " << std::right << std::setw(3)
                          << world_size << " | Repeat " << std::setw(2)
                          << repeat + 1 << '/' << heat::kE2Repeats
                          << std::flush;
            }
        }

        if (world_rank == 0) {
            std::cerr << "\r[E2] Grid "
                      << std::left << std::setw(11)
                      << (std::to_string(heat::kE2GlobalNx) + " x " +
                          std::to_string(heat::kE2GlobalNy))
                      << " | Ranks " << std::right << std::setw(3) << world_size
                      << " | " << heat::kE2Repeats << '/' << heat::kE2Repeats
                      << " completed | Finished in " << std::fixed
                      << std::setprecision(3) << MPI_Wtime() - experiment_start
                      << " s          \n";

            const auto minmax = std::minmax_element(
                total_samples.begin(), total_samples.end());
            heat::MpiExperimentResult result{};
            result.global_nx = heat::kE2GlobalNx;
            result.global_ny = heat::kE2GlobalNy;
            result.ranks = world_size;
            result.process_rows = domain.dims[0];
            result.process_cols = domain.dims[1];
            result.steps = heat::kSteps;
            result.repetitions = heat::kE2Repeats;
            result.runtime_median = heat::median(total_samples);
            result.runtime_min = *minmax.first;
            result.runtime_max = *minmax.second;
            result.halo_runtime_median = heat::median(halo_samples);
            result.validation = {
                all_passed, global_min, global_max, global_boundary_error};
            const double serial_baseline = heat::read_e1_serial_baseline(
                heat::kE1BaselineCsv, heat::kE2GlobalNx);
            const double mpi_baseline = world_size == 1
                ? result.runtime_median
                : heat::read_e2_single_rank_baseline(heat::kE2OutputCsv);
            result.speedup_mpi = mpi_baseline / result.runtime_median;
            result.speedup_serial = serial_baseline / result.runtime_median;
            result.parallel_efficiency =
                result.speedup_mpi / static_cast<double>(world_size);
            heat::write_e2_csv({result}, heat::kE2OutputCsv);

            if (!all_passed) {
                exit_code = 2;
            }
        }

        MPI_Bcast(&exit_code, 1, MPI_INT, 0, domain.cart_comm);
    }
    catch (const std::exception& error) {
        std::cerr << "Rank " << world_rank << " error: " << error.what() << '\n';
        exit_code = 1;
        MPI_Abort(MPI_COMM_WORLD, exit_code);
    }

    heat::destroy_mpi_domain(domain);
    MPI_Finalize();
    return exit_code;
}
