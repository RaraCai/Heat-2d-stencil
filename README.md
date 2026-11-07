# Heat 2D Stencil — E1–E4-ready refactor

C++17, Linux/WSL, CMake + Ninja. `heat_core` contains shared grid, initialization,
FTCS step, validation, benchmark statistics and CSV output. E1, MPI E2 strong
scaling, MPI E3 weak scaling, and E4 MPI + OpenMP CPU/GPU experiments are
implemented.

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

The E4 CPU experiment remains unchanged and is submitted with
`qsub scripts/vanda/e4_hybrid_CPU.pbs`. For E4 GPU strong scaling, submit
`qsub scripts/vanda/e4_hybrid_GPU.pbs` after the CPU E4 job has completed.
The GPU script requires all 15 rows in `results/raw/e4_hybrid.csv`; it does not
modify or rerun the CPU experiment. It requests the same four nodes and CPU
slots as the CPU job, plus one GPU per node. It then runs the same 15 MPI-rank
and OpenMP-thread configurations, with the same grid, iteration count, and
repetitions, using the CPU E4 rank placement. MPI ranks on each node share that
node's single GPU; the 2-rank configuration follows the CPU script's placement
and uses one node/GPU, while the other configurations use all four nodes/GPUs.
Each rank selects device 0 on its node, exchanges MPI halos on the host, and
offloads the FTCS update to the GPU. The host/device mapping is performed per
step, leaving the CPU E4 design and MPI halo/grid-swap sequence unchanged.

`results/raw/e4_gpu.csv` stores each GPU configuration beside its matching CPU
solver median and the per-configuration speedup (`CPU solver median / GPU
solver median`). It contains one summary row for each of the 15 configurations.
`results/raw/e4_gpu_runs.csv` stores the 10 individual repetitions for each
configuration.
The GPU version retains the CPU `threads per rank` setting to match
configurations; GPU kernel parallelism itself is provided by OpenMP target
teams, not by that host-thread count. The first GPU configuration replaces
both GPU result files on a new run; the PBS script does not archive older GPU
CSV files.

The GPU job requires Vanda's GCC 13.3.0 compiler to have GCC `nvptx-none`
OpenMP offload support, and available NVIDIA GPUs on the requested nodes. If
the nvptx toolchain requires a site CUDA module, set `CUDA_MODULE` when
submitting with the module name provided by Vanda (for example,
`qsub -v CUDA_MODULE=CUDA/<version> scripts/vanda/e4_hybrid_GPU.pbs`).
`OMP_TARGET_OFFLOAD=MANDATORY` and an
in-program device check make the job fail rather than silently running the
kernel on the CPU. The results are written to `results/raw/e4_gpu.csv`.
The 10-per-configuration timing records are written to
`results/raw/e4_gpu_runs.csv`.

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

The `ftcs_step` kernel is serial by design; the CPU and GPU E4 variants use
separate OpenMP kernels without changing the physical model.
