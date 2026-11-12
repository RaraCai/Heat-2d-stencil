#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

module purge
source /app1/ebenv
module load GCC/13.3.0
module load OpenMPI/5.0.3-GCC-13.3.0
module load CMake/3.29.3-GCCcore-13.3.0

export LD_LIBRARY_PATH="/opt/pbs/lib:${LD_LIBRARY_PATH:-}"
BUILD_DIR="build/vanda_e5_gcc13"

 echo "========================================"
echo " E5 GCC 13.3 + OpenMPI 5.0.3 Direct GPU MPI Build"
echo "========================================"
echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo unavailable)"
echo "Build directory: $BUILD_DIR"
module list
echo
g++ --version
mpicxx --showme:version
mpirun --version
cmake --version

if [[ ! -f results/raw/e4_hybrid.csv ]]; then
    echo "ERROR: Missing E4 CPU baseline: results/raw/e4_hybrid.csv" >&2
    exit 1
fi
baseline_rows=$(($(wc -l < results/raw/e4_hybrid.csv) - 1))
if (( baseline_rows != 15 )); then
    echo "ERROR: Expected 15 E4 baseline configurations; found $baseline_rows." >&2
    exit 1
fi

cmake --fresh -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON \
    -DCMAKE_CXX_COMPILER="$(command -v g++)" \
    -DMPI_CXX_COMPILER="$(command -v mpicxx)" \
    -DHEAT_ENABLE_GCC_NVPTX=ON

LINK_FILE="$BUILD_DIR/CMakeFiles/stencil_hybrid_gpu.dir/link.txt"
if [[ ! -f "$LINK_FILE" ]] || \
   ! grep -q -- '-fopenmp' "$LINK_FILE" || \
   ! grep -q -- '-foffload=nvptx-none' "$LINK_FILE"; then
    echo "ERROR: GPU final link is missing OpenMP/NVPTX flags." >&2
    [[ ! -f "$LINK_FILE" ]] || cat "$LINK_FILE"
    exit 1
fi

echo
echo "GPU final link command:"
cat "$LINK_FILE"

cmake --build "$BUILD_DIR" \
    --target stencil_hybrid_gpu test_gpu test_gpu_single_rank test_core test_mpi \
    --parallel 4

for executable in stencil_hybrid_gpu test_gpu test_gpu_single_rank test_core test_mpi; do
    [[ -x "$BUILD_DIR/$executable" ]] || {
        echo "ERROR: Missing executable $BUILD_DIR/$executable" >&2
        exit 1
    }
done

for executable in stencil_hybrid_gpu test_gpu test_gpu_single_rank; do
    if ldd "$BUILD_DIR/$executable" | grep -q 'not found'; then
        echo "ERROR: Missing dynamic library for $executable." >&2
        ldd "$BUILD_DIR/$executable"
        exit 1
    fi
done

ctest --test-dir "$BUILD_DIR" --output-on-failure

echo
echo "========================================"
echo " E5 DIRECT GPU MPI BUILD SUCCESSFUL"
echo "========================================"
echo "GPU executable: $BUILD_DIR/stencil_hybrid_gpu"
echo "GPU correctness executable: $BUILD_DIR/test_gpu"
echo "Single-rank diagnostic: $BUILD_DIR/test_gpu_single_rank"
echo "Next: qsub scripts/vanda/e5_single_rank_GPU.pbs"
