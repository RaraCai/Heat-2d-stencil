#pragma once

#include "grid.hpp"
#include "mpi.hpp"
#include "stencil.hpp"

#include <memory>

namespace heat {

struct GpuSolverTiming;

class GpuSolverWorkspace {
public:
    // Reuse one workspace across sequential solves on the same MPI rank.
    GpuSolverWorkspace();
    ~GpuSolverWorkspace();

    GpuSolverWorkspace(const GpuSolverWorkspace&) = delete;
    GpuSolverWorkspace& operator=(const GpuSolverWorkspace&) = delete;
    GpuSolverWorkspace(GpuSolverWorkspace&&) noexcept;
    GpuSolverWorkspace& operator=(GpuSolverWorkspace&&) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    friend void solve_mpi_gpu(Grid&, const Grid&,
                              const NumericalParameters&, const MpiDomain&,
                              int, int, int, GpuSolverTiming*,
                              GpuSolverWorkspace*);
};

struct GpuSolverTiming {
    double solver_time = 0.0;
    // Iterative-loop time, excluding GPU allocation and host/device setup/final copies.
    double iteration_time = 0.0;
    double mpi_halo_time = 0.0;
    double gpu_compute_time = 0.0;
    double gpu_pack_time = 0.0;
    double gpu_unpack_time = 0.0;
    double gpu_setup_time = 0.0;
    double gpu_final_copy_time = 0.0;
};

struct ReferenceComparison {
    double max_absolute_error = 0.0;
    double relative_l2_error = 0.0;
    bool passed = false;
};

// E5: resident grid buffers and direct MPI communication using
// OpenMP-allocated GPU send/receive buffers. MPI halo exchange remains
// blocking; no explicit per-step host staging copies are performed.
void solve_mpi_gpu(Grid& current, const Grid& next,
                   const NumericalParameters& params,
                   const MpiDomain& domain, int steps, int device,
                   int thread_limit, GpuSolverTiming* timing = nullptr,
                   GpuSolverWorkspace* workspace = nullptr);

ReferenceComparison compare_global_reference(
    const Grid& gpu_grid, const Grid& cpu_grid, const MpiDomain& domain,
    double max_absolute_tolerance, double relative_l2_tolerance);

} // namespace heat
