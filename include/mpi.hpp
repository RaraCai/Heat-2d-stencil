#pragma once
#include "config.hpp"
#include "grid.hpp"
#include "validation.hpp"
#include <mpi.h>
#include <vector>

namespace heat {

// Two-dimensional MPI Cartesian decomposition of the global grid.
// The local Grid stores owned cells plus one halo layer on every side.
struct MpiDomain {
    MPI_Comm cart_comm = MPI_COMM_NULL;
    MPI_Datatype column_type = MPI_DATATYPE_NULL;

    int rank = 0;
    int size = 1;

    int dims[2] = {1, 1};
    int coords[2] = {0, 0};

    int north = MPI_PROC_NULL;
    int south = MPI_PROC_NULL;
    int west = MPI_PROC_NULL;
    int east = MPI_PROC_NULL;

    int global_nx = 0;
    int global_ny = 0;

    int local_nx = 0;
    int local_ny = 0;
    int global_row = 0;
    int global_col = 0;
};

struct MpiTiming {
    double halo_exchange_seconds = 0.0;
    double ftcs_seconds = 0.0;
};

// Create a 2D Cartesian communicator and a balanced (possibly uneven)
// block decomposition. E2's selected process counts produce square grids.
void create_mpi_domain(MpiDomain& domain,
                       int global_nx,
                       int global_ny,
                       MPI_Comm parent_comm = MPI_COMM_WORLD);

// Release the derived datatype and Cartesian communicator before MPI_Finalize.
void destroy_mpi_domain(MpiDomain& domain);

// Construct a local Grid with halo width one and the correct global offset.
Grid make_local_grid(const MpiDomain& domain);

// Exchange the four face halos using blocking MPI_Sendrecv.
// Column communication uses a committed MPI_Type_vector datatype.
void exchange_halos(Grid& grid, const MpiDomain& domain,
                    double* elapsed_seconds = nullptr);

// Distributed time loop. The caller owns MPI initialization/finalization.
// Each iteration performs halo exchange, the shared FTCS kernel, and buffer swap.
void solve_mpi(Grid& current, Grid& next,
               const NumericalParameters& params,
               const MpiDomain& domain,
               int steps,
               MpiTiming* timing = nullptr);

// Aggregate local validation statistics across all ranks.
ValidationResult validate_global(const Grid& local_grid,
                                 const MpiDomain& domain);

// Development/debug utility: gather owned cells into a row-major global grid
// on root. On non-root ranks, global_grid is left empty.
void gather_global_grid(const Grid& local_grid,
                        const MpiDomain& domain,
                        std::vector<double>& global_grid,
                        int root = 0);

} // namespace heat
