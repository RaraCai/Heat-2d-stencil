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

namespace {

bool supported_rank_count(int ranks)
{
    return std::find(heat::kE3RankCounts.begin(),
                     heat::kE3RankCounts.end(), ranks) !=
           heat::kE3RankCounts.end();
}

} // namespace

int main(int argc, char** argv)
{
    if (argc == 2 && std::string(argv[1]) == "--summary") {
        try {
            const auto results = heat::read_e3_csv(heat::kE3OutputCsv);
            heat::print_weak_scaling_summary(results);
            return std::all_of(
                results.begin(), results.end(),
                [](const auto& result) { return result.validation.passed; })
                ? 0 : 2;
        } catch (const std::exception& error) {
            std::cerr << "E3 summary error: " << error.what() << '\n';
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
                "Unsupported E3 rank count. Use 1, 4, 16, 64, or 256.");
        }

        const int process_side = static_cast<int>(
            std::sqrt(static_cast<double>(world_size)));
        if (process_side * process_side != world_size) {
            throw std::runtime_error("E3 requires a square MPI process grid.");
        }
        const int global_nx = heat::kE3LocalNx * process_side;
        const int global_ny = heat::kE3LocalNy * process_side;
        heat::create_mpi_domain(domain, global_nx, global_ny);
        if (domain.dims[0] != process_side ||
            domain.dims[1] != process_side ||
            domain.local_nx != heat::kE3LocalNx ||
            domain.local_ny != heat::kE3LocalNy) {
            throw std::runtime_error(
                "E3 decomposition must give every rank a 256 x 256 grid.");
        }

        const double dx = heat::kDomainLength / (heat::kE3LocalNx - 1);
        const double dy = heat::kDomainLength / (heat::kE3LocalNy - 1);
        const heat::NumericalParameters params = heat::make_parameters(
            global_nx, global_ny, heat::kSteps, dx, dy);
        heat::Grid current = heat::make_local_grid(domain);
        heat::Grid next = heat::make_local_grid(domain);

        std::vector<heat::MpiRunResult> runs;
        if (domain.rank == 0) runs.reserve(heat::kE3Repeats);

        bool all_passed = true;
        double global_min = std::numeric_limits<double>::max();
        double global_max = std::numeric_limits<double>::lowest();
        double global_boundary_error = 0.0;
        const double experiment_start = MPI_Wtime();

        for (int repeat = 0; repeat < heat::kE3Repeats; ++repeat) {
            MPI_Barrier(domain.cart_comm);
            const double total_start = MPI_Wtime();
            const double init_start = MPI_Wtime();
            heat::initialize(current, global_nx, global_ny, dx, dy);
            next.data() = current.data();
            const double local_init = MPI_Wtime() - init_start;

            MPI_Barrier(domain.cart_comm);
            const double solver_start = MPI_Wtime();
            heat::MpiTiming timing;
            heat::solve_mpi(current, next, params, domain,
                            heat::kSteps, &timing);
            const double local_solver = MPI_Wtime() - solver_start;

            const double validation_start = MPI_Wtime();
            const heat::ValidationResult validation =
                heat::validate_global(current, domain);
            const double local_validation = MPI_Wtime() - validation_start;
            const double local_total = MPI_Wtime() - total_start;

            const double local_times[6] = {
                local_init, local_solver, timing.halo_exchange_seconds,
                timing.ftcs_seconds, local_validation, local_total
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
            if (domain.rank == critical.rank) {
                critical_halo = timing.halo_exchange_seconds;
            }
            MPI_Bcast(&critical_halo, 1, MPI_DOUBLE, critical.rank,
                      domain.cart_comm);

            all_passed = all_passed && validation.passed;
            global_min = std::min(global_min, validation.min_temperature);
            global_max = std::max(global_max, validation.max_temperature);
            global_boundary_error = std::max(
                global_boundary_error, validation.max_boundary_error);

            if (domain.rank == 0) {
                runs.push_back({
                    repeat + 1, max_times[0], max_times[1], max_times[2],
                    max_times[3], max_times[4], max_times[5],
                    max_times[1] > 0.0 ? critical_halo / max_times[1] : 0.0,
                    validation.passed
                });
                std::cerr << "\r[E3] Local grid " << heat::kE3LocalNx << "x"
                          << heat::kE3LocalNy << " | Ranks " << world_size
                          << " | Repeat " << repeat + 1 << '/'
                          << heat::kE3Repeats << std::flush;
            }
        }

        if (domain.rank == 0) {
            std::cerr << "\r[E3] Local grid " << heat::kE3LocalNx << "x"
                      << heat::kE3LocalNy << " | Ranks " << world_size
                      << " | " << heat::kE3Repeats << '/'
                      << heat::kE3Repeats << " completed | Finished in "
                      << std::fixed << std::setprecision(3)
                      << MPI_Wtime() - experiment_start << " s          \n";

            std::vector<double> init, solver, halo, ftcs, validation, total, ratios;
            for (const auto& run : runs) {
                init.push_back(run.initialization);
                solver.push_back(run.solver);
                halo.push_back(run.halo);
                ftcs.push_back(run.ftcs);
                validation.push_back(run.validation);
                total.push_back(run.total);
                ratios.push_back(run.halo_ratio);
            }

            const auto minmax = std::minmax_element(solver.begin(), solver.end());
            heat::MpiExperimentResult result{};
            result.global_nx = global_nx;
            result.global_ny = global_ny;
            result.ranks = world_size;
            result.process_rows = domain.dims[0];
            result.process_cols = domain.dims[1];
            result.steps = heat::kSteps;
            result.repetitions = heat::kE3Repeats;
            result.initialization_median = heat::median(init);
            result.initialization_mean = heat::mean(init);
            result.runtime_median = heat::median(solver);
            result.solver_mean = heat::mean(solver);
            result.solver_stddev =
                heat::sample_stddev(solver, result.solver_mean);
            result.runtime_min = *minmax.first;
            result.runtime_max = *minmax.second;
            result.halo_runtime_median = heat::median(halo);
            result.halo_mean = heat::mean(halo);
            result.ftcs_median = heat::median(ftcs);
            result.validation_median = heat::median(validation);
            result.total_median = heat::median(total);
            result.halo_ratio_median = heat::median(ratios);
            result.validation = {
                all_passed, global_min, global_max, global_boundary_error};

            const double baseline = world_size == 1
                ? result.runtime_median
                : heat::read_e3_single_rank_baseline(heat::kE3OutputCsv);
            result.weak_scaling_efficiency = baseline / result.runtime_median;
            heat::write_e3_csv({result}, heat::kE3OutputCsv);
            if (!all_passed) exit_code = 2;
        }

        MPI_Bcast(&exit_code, 1, MPI_INT, 0, domain.cart_comm);
    } catch (const std::exception& error) {
        std::cerr << "Rank " << world_rank << " error: "
                  << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    heat::destroy_mpi_domain(domain);
    MPI_Finalize();
    return exit_code;
}
