# Heat 2D Stencil — E1–E4-ready refactor

C++17, Linux/WSL, CMake + Ninja. `heat_core` contains shared grid, initialization,
FTCS step, validation, benchmark statistics and CSV output. E1 is implemented.
E2/E3/E4 entry points are intentionally placeholders; no MPI/OpenMP functionality
is claimed until domain decomposition, halo exchange, reductions and threading
are implemented and tested.

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

The `ftcs_step` kernel is serial by design; E4 can add a separate OpenMP-enabled
kernel or parallel execution policy without changing the physical model.
