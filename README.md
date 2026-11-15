# Heat 2D Stencil

> DSA5208 Project 3: Parallelizing an Algorithm with MPI+X

A C++17 implementation of the two-dimensional heat-equation FTCS stencil and five experiments:

- **E1:** Serial baseline
- **E2:** MPI strong scaling
- **E3:** MPI weak scaling
- **E4:** MPI + OpenMP CPU hybrid
- **E5:** MPI + OpenMP GPU hybrid using GCC NVPTX offload

The primary workflow in this README reproduces E1–E5 on Vanda. Local Linux instructions are provided as a short alternative for E1–E4; a host-only local machine cannot produce the formal E5 GPU results.

## Repository layout

```text
.
├── CMakeLists.txt                  # Build targets, dependencies, and tests
├── include/                        # Public headers and experiment configuration
├── src/
│   ├── core/                       # Grid, FTCS, validation, statistics, CSV output
│   ├── gpu/                        # GPU solver and stencil
│   ├── serial.cpp                  # E1
│   ├── strong_mpi.cpp              # E2
│   ├── weak_mpi.cpp                # E3
│   ├── hybrid.cpp                  # E4
│   ├── hybrid_gpu.cpp              # E5
│   └── mpi.cpp                     # MPI communication
├── scripts/
│   ├── local/                      # Local build and run scripts
│   └── vanda/                      # Vanda PBS job scripts
├── tests/                          # Core, MPI, and GPU correctness tests
├── results/raw/                    # Experiment CSV output (created at runtime)
└── build/                          # CMake build trees (created at runtime)
```

`heat_core` contains the shared grid, initialization, FTCS update, validation, benchmark statistics, and CSV output. MPI domain decomposition and halo exchange are implemented in `heat_mpi`; E4 and E5 use separate OpenMP CPU and GPU kernels.

## Reproduce E1–E5 workflow on Vanda

Run the experiments from a shared project directory visible to Vanda compute nodes. The PBS scripts load GCC 13.3.0, OpenMPI 5.0.3, and CMake 3.29.3. E5 additionally requires a GPU allocation and GCC OpenMP `nvptx-none` offload support.

### 1. Clone the repository

```bash
git clone https://github.com/5555555559/Heat-2d-stencil.git
cd Heat-2d-stencil
```

Run all subsequent commands from this repository directory.

### 2. Set your Vanda PBS account

Every PBS script contains the neutral `YOUR_PBS_ACCOUNT` placeholder. Before submitting jobs, replace it with the Vanda project/account assigned to you. A personal allocation is commonly named `personal-<your-username>` (for example, `personal-jane.doe`); confirm the exact value for your account. Do not use another person's project.

Run this command and enter your own authorized account:

```bash
read -r -p "Authorized Vanda PBS account: " PBS_PROJECT
if [[ -z "$PBS_PROJECT" ]]; then
    echo "PBS account is required." >&2
    exit 1
fi

for script in scripts/vanda/*.pbs; do
    sed -i "s/^#PBS -P .*/#PBS -P ${PBS_PROJECT}/" "$script"
done

grep -n '^#PBS -P' scripts/vanda/*.pbs
```

Check that the output shows your account and no `YOUR_PBS_ACCOUNT` placeholders. PBS jobs use the directory from which `qsub` is run as their working directory.

### 3. Submit E1, the serial baseline

```bash
qsub scripts/vanda/e1_serial.pbs
qstat -u "$USER"
```

Wait for the job to finish and inspect its PBS output log. Continue only after it reports successful completion and the baseline exists:

```bash
wc -l results/raw/e1_serial.csv
```

E1 runs five grid sizes (256, 512, 1024, 2048, and 4096), with 500 iterations and 10 repetitions per grid. The CSV should contain five data rows plus its header (6 lines).

### 4. Submit E2, MPI strong scaling

E2 checks that the E1 baseline is present and includes the 4096 grid result.

```bash
qsub scripts/vanda/e2_strong.pbs
qstat -u "$USER"
```

Wait for successful completion before proceeding. E2 uses 1, 4, 16, 64, and 256 MPI ranks and writes `results/raw/e2_strong.csv` (5 data rows plus the header).

### 5. Submit E3, MPI weak scaling

```bash
qsub scripts/vanda/e3_weak.pbs
qstat -u "$USER"
```

Each rank owns a 256 × 256 grid. E3 uses 1, 4, 16, 64, and 256 ranks and writes `results/raw/e3_weak.csv` (5 data rows plus the header). Wait for successful completion before proceeding.

### 6. Submit E4, MPI + OpenMP CPU

```bash
qsub scripts/vanda/e4_hybrid_CPU.pbs
qstat -u "$USER"
```

E4 runs 15 MPI-rank/OpenMP-thread configurations on a 4096 × 4096 grid, with 500 iterations and 10 repetitions per configuration. Wait for successful completion, then verify both E4 outputs:

```bash
wc -l results/raw/e4_hybrid.csv results/raw/e4_hybrid_runs.csv
```

The expected line counts are 16 for the summary and 151 for the repetition records, including the CSV headers. E5 requires the complete 15-configuration summary in `results/raw/e4_hybrid.csv`.

### 7. Build and submit E5, MPI + OpenMP GPU

After E4 has completed successfully, build E5 from the Vanda login environment at the repository root:

```bash
bash scripts/local/build_e5.sh
```

The script loads the required modules, verifies the complete E4 baseline, builds the GCC NVPTX targets in `build/vanda_e5_gcc13/`, and runs CTest. Then submit the formal GPU experiment:

```bash
qsub scripts/vanda/e5_hybrid_GPU.pbs
qstat -u "$USER"
```

The job requests one node and one GPU, runs five Group A rank/thread configurations with 10 repetitions each, and compares them with the matching E4 CPU configurations. The GPU timings include setup, MPI halo exchange, kernel execution, and final copy; they are end-to-end timings, not equal-resource CPU/GPU efficiency measurements.

Expected output:

- `results/raw/e5_gpu.csv` — 5 configuration summaries
- `results/raw/e5_gpu_runs.csv` — 50 repetition records

### 8. Verify all experiment outputs

After all jobs have completed, confirm the expected CSV files and row counts:

```bash
wc -l \
    results/raw/e1_serial.csv \
    results/raw/e2_strong.csv \
    results/raw/e3_weak.csv \
    results/raw/e4_hybrid.csv \
    results/raw/e4_hybrid_runs.csv \
    results/raw/e5_gpu.csv \
    results/raw/e5_gpu_runs.csv
```

Expected line counts, including headers: E1 = 6, E2 = 6, E3 = 6, E4 summary = 16, E4 runs = 151, E5 summary = 6, and E5 runs = 51. A smaller count indicates an incomplete run; check the corresponding PBS log before using the data. Use `qstat -u "$USER"` to check job status and `ls -lt` in the submission directory to locate recent PBS logs.

## Local Linux quick reference (E1–E4 only)

Local runs require a C++17 compiler, CMake 3.16 or newer, MPI (`mpicxx` and `mpirun`), and OpenMP. E1 also requires Ninja; E2/E3 use GCC-specific compiler options. On Ubuntu or WSL:

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build openmpi-bin libopenmpi-dev
```

From the repository root, run the experiment script:

```bash
bash scripts/local/run_e1.sh  # E1
bash scripts/local/run_e2.sh  # E2
bash scripts/local/run_e3.sh  # E3
bash scripts/local/run_e4.sh  # E4
```

E2 and E3 default to all five MPI sizes (1, 4, 16, 64, and 256 ranks), which may exceed the resources available on a workstation. To run a subset for local validation, pass rank counts, for example:

```bash
bash scripts/local/run_e2.sh 1 4 16 256
bash scripts/local/run_e3.sh 1 4 16 256
```

The formal E5 experiment requires Vanda GPU resources and cannot be reproduced as a GPU benchmark on a host-only local build.

## Results and reruns

All experiment CSVs are written to `results/raw/`. Vanda scripts archive previous E2–E4 outputs where applicable. Local E2/E3 scripts delete their corresponding CSV before running. The E5 formal run stops if either E5 output CSV already exists; archive existing files before rerunning:

```bash
BACKUP_DIR="results/archive/e5_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
for file in results/raw/e5_gpu.csv results/raw/e5_gpu_runs.csv; do
    if [[ -e "$file" ]]; then
        mv "$file" "$BACKUP_DIR/"
    fi
done
```

Do not mix CSVs from different runs when comparing experiments.
