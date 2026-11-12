# Heat 2D Stencil — E1–E5 experiments

C++17, Linux/WSL, CMake + Ninja. `heat_core` contains shared grid, initialization,
FTCS step, validation, benchmark statistics and CSV output. E1, MPI E2 strong
scaling, MPI E3 weak scaling, E4 MPI + OpenMP CPU, and E5 MPI + OpenMP GPU
experiments are implemented.

## Build and run

```bash
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/release
./build/release/stencil_serial
```

Run from repository root; output is `results/raw/e1_serial.csv`.
The full E1 workload is five grids, 500 iterations and 10 repeats per grid;
4096² can take substantial time. The executable updates one compact repeat
counter per grid on stderr; the summary and CSV are written after all grids
finish.
`build/debug` may be used for debugging but must not be used for performance
measurements.

Run E2 strong scaling with `scripts/local/run_e2.sh`. It writes one row per
rank configuration to `results/raw/e2_strong.csv`; the 1-rank configuration
is run first to establish the MPI speedup baseline (and is automatically
included when a subset of rank counts is requested).
Individual repetition samples are used for statistics but are not written to
CSV. The script prints one E1-style summary table after all selected runs.

Run E3 weak scaling with `scripts/local/run_e3.sh [ranks...]`. Supported rank
counts are 1, 4, 16, 64, and 256; the 1-rank baseline is automatically run
first. Each rank owns a fixed 256 x 256 grid, while the global grid scales from
256 x 256 to 4096 x 4096. Spatial spacing, time step, and iteration count are
held fixed. Results are written to `results/raw/e3_weak.csv`.
On Vanda, submit the full E3 job with `qsub scripts/vanda/e3_weak.pbs`.

The E4 CPU experiment is unchanged and is submitted with
`qsub scripts/vanda/e4_hybrid_CPU.pbs`. E5's GCC 13.3/OpenMPI 5.0.3 build is
performed with `scripts/local/build_e5.sh`. The one-rank performance diagnostic
and five-configuration GPU experiment use separate PBS jobs:

```bash
git clone https://github.com/5555555559/Heat-2d-stencil.git
cd Heat-2d-stencil

# Run E4 first and wait for it to finish.
qsub scripts/vanda/e4_hybrid_CPU.pbs

# Once E4 has produced its full baseline, build E5 and run CTest.
./scripts/local/build_e5.sh

# Run the fixed 4096 x 4096, 500-step, 10-repetition CPU/GPU comparison.
qsub scripts/vanda/e5_single_rank_GPU.pbs

# Submit the five E5 Group A configurations on one GPU.
qsub scripts/vanda/e5_hybrid_GPU.pbs
```

`scripts/local/build_e5.sh` loads the GCC 13.3, OpenMPI 5.0.3 and CMake 3.29.3
modules, verifies the complete E4 baseline and builds the GCC NVPTX-offload
targets into `build/vanda_e5_gcc13/`.

The PBS job does not compile. It rechecks the E4 baseline and E5 executable,
then runs the five production configurations. Check the PBS job status with
`qstat -u "$USER"` and read its combined output log for progress and the final
summary.

E5 reports concise rank-0 repetition progress and a configuration summary to
the PBS log. Errors remain explicit; normal execution does not emit per-rank
stage diagnostics.

The job requests one node, 36 CPU cores and one GPU. E5 runs only the five
Group A MPI-rank and CPU threads-per-rank configurations from E4 (2, 4, 8, 16,
and 32 ranks); all ranks share GPU device 0. `gpu_thread_limit` (256) is a
separate device-kernel setting; CPU threads-per-rank continue to control host
rank placement. Static configuration checks and runtime rank/node/device
checks reject oversubscribed or mismatched layouts.

The GPU Solver keeps two full grids resident on the GPU for all 500 steps.
Callers performing repeated solves should retain one `GpuSolverWorkspace` per
MPI rank and pass it to each sequential solve; its grid and halo buffers are
reused. The single-rank path allocates no halo buffers and uses a stencil-only
kernel, since no MPI faces need packing or exchange.
Owned edges are packed into OpenMP device buffers and exchanged directly with
blocking MPI calls; the application does not stage halos through host memory.
After each exchange, one GPU kernel reads remote face values from the receive
buffer directly, updates the owned grid, and packs the updated edges for the
next step. This avoids separate per-step pack and halo-unpack kernel launches.
Only setup and the final result copy transfer complete grids. Solver time
includes setup, iterations, halo synchronization and the final copy. The output
separates MPI halo, GPU compute, setup, and final-copy timing; component maxima
are diagnostics and must not be added together as a total runtime. Initial edge
packing is reported as GPU pack time; pack work fused into each stencil launch
is included in GPU compute time, and no separate halo-unpack launch remains.
The solver wall time also includes releasing its resident device buffers.

Each configuration runs only the recorded GPU repetitions; there are no
additional CPU/GPU reference solves. Every GPU repetition undergoes global
physical-boundary/temperature validation. E5 reads the matching Group A solver
median from `results/raw/e4_hybrid.csv` as its CPU baseline and uses it to
calculate GPU/CPU speedup. Failed configurations are not written as valid
timing results.
The summary and CSV report two speedups:

- **Within-group configuration speedup:** the 2-rank Group A GPU median
  divided by this configuration's GPU solver median.
- **GPU vs CPU E4 speedup:** the matching E4 rank/thread configuration median
  divided by this E5 GPU solver median.

E5 and E4 Group A use matching CPU rank/thread configurations. GPU solver time
includes GPU setup, boundary/halo transfers, MPI exchange,
kernel launch/synchronization and final copy; E4 solver time does not include
GPU transfers, so this is an end-to-end runtime comparison, not an equal-resource
CPU/GPU efficiency comparison.

Results are written to `results/raw/e5_gpu.csv` and
`results/raw/e5_gpu_runs.csv` (5 summary rows and 50 repetition rows). The PBS
job records its ID, allocated nodes and visible GPU model in its standard job
log. Existing E5 CSV files cause the run to stop rather than be overwritten;
move or remove both result files explicitly before starting a fresh run.

The one-rank diagnostic writes ten per-run samples and summary median, mean and
sample standard deviation for CPU loop, GPU iteration, GPU end-to-end and GPU
kernel times. `scripts/vanda/e5_single_rank_nsys.pbs` profiles the same
diagnostic without overwriting benchmark CSVs. The Nsight trace from the
pre-optimization version recorded 10,056 `cuMemAlloc_v2`/`cuMemFree_v2` calls
for 5,005 FTCS launches; its API events had no call stacks, so this count alone
does not identify the exact libgomp allocation site or imply that the summed
API time can be subtracted from solver time.

## MPI extension contract

- `Grid(nx, ny, halo, global_row, global_col)` owns a local rectangular region.
- `initialize` uses **global** coordinates and sets only owned cells.
- Before `ftcs_step`, exchange one-layer halos with four Cartesian neighbors.
- `ftcs_step` computes only owned, nonphysical-boundary cells; MPI owns the
  communication and the `swap_data` call between steps.
- On global physical boundaries, initialize both buffers consistently; for
  MPI local regions touching a physical boundary, copy those owned boundary
  cells to both buffers before time stepping.
- `validate_local` scans owned cells; aggregate its metrics using MPI reductions.
- Serial and MPI reference comparisons require matching global indexing,
  iteration counts and boundary handling.

The `ftcs_step` kernel is serial by design; E4 CPU and E5 GPU use separate
OpenMP kernels without changing the physical model.
