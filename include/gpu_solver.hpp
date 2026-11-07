#pragma once

#include "grid.hpp"
#include "mpi.hpp"
#include "stencil.hpp"

namespace heat {

struct GpuSolverTiming {
    double solver_time = 0.0;
    double mpi_halo_time = 0.0;
    double gpu_compute_time = 0.0;
    double boundary_d2h_time = 0.0;
    double halo_h2d_time = 0.0;
    double gpu_setup_time = 0.0;
    double gpu_final_copy_time = 0.0;
};

struct ReferenceComparison {
    double max_absolute_error = 0.0;
    double relative_l2_error = 0.0;
    bool passed = false;
};

// Solves on the GPU with resident double buffers. MPI halo exchange remains
// on the host; only owned boundaries and received halos are transferred per step.
void solve_mpi_gpu(Grid& current, const Grid& next,
                   const NumericalParameters& params,
                   const MpiDomain& domain, int steps, int device,
                   int thread_limit, GpuSolverTiming* timing = nullptr);

ReferenceComparison compare_global_reference(
    const Grid& gpu_grid, const Grid& cpu_grid, const MpiDomain& domain,
    double max_absolute_tolerance, double relative_l2_tolerance);

} // namespace heat
