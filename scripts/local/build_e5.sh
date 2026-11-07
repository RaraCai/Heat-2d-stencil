#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source /app1/ebenv
module load GCC/13.2.0
module load NVHPC/24.7-CUDA-12.3.0
module load OpenMPI/4.1.6-GCC-13.2.0
module load CMake/3.27.6-GCCcore-13.2.0

export LD_LIBRARY_PATH="/opt/pbs/lib:${LD_LIBRARY_PATH:-}"
export OMPI_CXX=nvc++

echo "E5 GPU login-node build"
echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo unavailable)"
module list
nvc++ --version
mpicxx --showme:version
cmake --version
echo "OMPI_CXX=$OMPI_CXX"

if [[ ! -f results/raw/e4_hybrid.csv ]]; then
    echo "Missing E4 CPU baseline: results/raw/e4_hybrid.csv" >&2
    echo "Run E4 CPU to completion before building E5 GPU." >&2
    exit 1
fi
baseline_rows=$(($(wc -l < results/raw/e4_hybrid.csv) - 1))
if (( baseline_rows != 15 )); then
    echo "E4 CPU baseline must contain all 15 configurations; found $baseline_rows." >&2
    exit 1
fi

BUILD_DIR="build/vanda_gpu"
cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON \
    -DCMAKE_CXX_COMPILER="$(command -v nvc++)" \
    -DMPI_CXX_COMPILER="$(command -v mpicxx)" \
    -DHEAT_ENABLE_NVHPC_GPU=ON \
    -DHEAT_NVHPC_GPU_ARCH=cc86
cmake --build "$BUILD_DIR" \
    --target stencil_hybrid_gpu test_core test_mpi \
    --parallel 4
ctest --test-dir "$BUILD_DIR" --output-on-failure

test -x "$BUILD_DIR/stencil_hybrid_gpu"
echo "E5 GPU binaries built in $BUILD_DIR; CPU/MPI CTest passed."
echo "Submit with: qsub scripts/vanda/e5_hybrid_GPU.pbs"
