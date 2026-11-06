#pragma once
#include "config.hpp"
#include "grid.hpp"
namespace heat {
// Global offsets allow exactly the same initialization for MPI subdomains.
void initialize(Grid& grid,int global_nx,int global_ny);
void initialize(Grid& grid,int global_nx,int global_ny,double dx,double dy);
// Updates only owned physical points; global physical boundaries are excluded.
// Caller must update halo cells before this call when using MPI.
void ftcs_step(const Grid& current,Grid& next,
               const NumericalParameters& params,int global_nx,int global_ny);
// Caller manages communication and parallelization between steps.
void solve_serial(Grid& current,Grid& next,const NumericalParameters& params,
                  int global_nx,int global_ny,int steps);
}
