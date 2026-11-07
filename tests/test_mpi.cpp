#include "mpi.hpp"
#include "stencil.hpp"

#include <cmath>
#include <iostream>
#include <vector>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int local_ok = 1;
    heat::MpiDomain domain;

    try {
        heat::create_mpi_domain(domain, 8, 8);
        heat::Grid grid = heat::make_local_grid(domain);

        for (int i = 0; i < grid.nx(); ++i) {
            for (int j = 0; j < grid.ny(); ++j) {
                const int gi = grid.global_row() + i;
                const int gj = grid.global_col() + j;
                grid.at(i, j) = static_cast<double>(gi * 100 + gj);
            }
        }

        heat::exchange_halos(grid, domain);

        // The OpenMP FTCS kernel must match the existing serial local kernel.
        heat::Grid serial_next = heat::make_local_grid(domain);
        heat::Grid openmp_next = heat::make_local_grid(domain);
        const heat::NumericalParameters params = heat::make_parameters(8, 8, 1);
        heat::ftcs_step(grid, serial_next, params, 8, 8);
        heat::ftcs_step_openmp(grid, openmp_next, params, 8, 8, 2);
        if (heat::max_absolute_error(serial_next, openmp_next) > 1e-12) {
            local_ok = 0;
        }

        // Verify each face halo against the global-coordinate encoding.
        if (domain.north != MPI_PROC_NULL) {
            for (int j = 0; j < grid.ny(); ++j) {
                const int gi = grid.global_row() - 1;
                const int gj = grid.global_col() + j;
                if (std::abs(grid.at(-1, j) - (gi * 100 + gj)) > 1e-12) {
                    local_ok = 0;
                }
            }
        }
        if (domain.south != MPI_PROC_NULL) {
            for (int j = 0; j < grid.ny(); ++j) {
                const int gi = grid.global_row() + grid.nx();
                const int gj = grid.global_col() + j;
                if (std::abs(grid.at(grid.nx(), j) - (gi * 100 + gj)) > 1e-12) {
                    local_ok = 0;
                }
            }
        }
        if (domain.west != MPI_PROC_NULL) {
            for (int i = 0; i < grid.nx(); ++i) {
                const int gi = grid.global_row() + i;
                const int gj = grid.global_col() - 1;
                if (std::abs(grid.at(i, -1) - (gi * 100 + gj)) > 1e-12) {
                    local_ok = 0;
                }
            }
        }
        if (domain.east != MPI_PROC_NULL) {
            for (int i = 0; i < grid.nx(); ++i) {
                const int gi = grid.global_row() + i;
                const int gj = grid.global_col() + grid.ny();
                if (std::abs(grid.at(i, grid.ny()) - (gi * 100 + gj)) > 1e-12) {
                    local_ok = 0;
                }
            }
        }

        std::vector<double> global;
        heat::gather_global_grid(grid, domain, global);
        if (domain.rank == 0) {
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j) {
                    const double expected = static_cast<double>(i * 100 + j);
                    if (std::abs(global[static_cast<std::size_t>(i) * 8 + j] -
                                 expected) > 1e-12) {
                        local_ok = 0;
                    }
                }
            }
        }

        int global_ok = 0;
        MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN,
                      domain.cart_comm);
        if (domain.rank == 0) {
            std::cout << (global_ok ? "PASS" : "FAIL")
                      << " MPI 2D decomposition / halo / gather test\n";
        }
        local_ok = global_ok;
    }
    catch (const std::exception& error) {
        std::cerr << "Rank " << world_rank << ": " << error.what() << '\n';
        local_ok = 0;
    }

    heat::destroy_mpi_domain(domain);
    int all_ok = 0;
    MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Finalize();
    return all_ok ? 0 : 1;
}
