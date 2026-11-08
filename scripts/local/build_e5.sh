#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Match the verified E4 compiler/MPI/CMake environment exactly.
module purge
source /app1/ebenv
module load GCC/13.3.0
module load OpenMPI/5.0.3-GCC-13.3.0
module load CMake/3.29.3-GCCcore-13.3.0

export LD_LIBRARY_PATH="/opt/pbs/lib:${LD_LIBRARY_PATH:-}"

BUILD_DIR="${BUILD_DIR:-build/vanda_e5_gcc13}"

 echo "========================================"
echo " E5 GCC 13.3 + OpenMPI 5.0.3 GPU Build"
echo "========================================"
echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo unavailable)"
echo "Build directory: ${BUILD_DIR}"
echo
module list

echo
echo "GCC:"
g++ --version

echo "MPI wrapper:"
command -v mpicxx
mpicxx --showme:version
mpicxx --showme:command

echo "MPI runtime:"
mpirun --version

echo "CMake:"
cmake --version

# Confirm the same PBS/TM runtime dependency used by the successful E4 job.
PRTE_TM="$(dirname "$(dirname "$(command -v mpirun)")")/lib/prte/prte_mca_plm_tm.so"
if [[ -f "$PRTE_TM" ]]; then
    echo
    echo "PRRTE PBS/TM plugin: ${PRTE_TM}"
    ldd "$PRTE_TM" | grep -E 'libpbs|not found' || true
else
    echo "WARNING: PRRTE TM plugin not found at inferred path: ${PRTE_TM}" >&2
fi

# E5 compares each GPU configuration with its matching E4 CPU result.
if [[ ! -f results/raw/e4_hybrid.csv ]]; then
    echo "ERROR: Missing E4 CPU baseline: results/raw/e4_hybrid.csv" >&2
    exit 1
fi
baseline_rows=$(($(wc -l < results/raw/e4_hybrid.csv) - 1))
if [[ "$baseline_rows" -ne 15 ]]; then
    echo "ERROR: E4 CPU baseline must contain 15 configurations; found ${baseline_rows}." >&2
    exit 1
fi

# Use a new build directory so no NVHPC/OpenMPI 4.1.6 CMake cache can leak in.
if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    cached_compiler=$(sed -n 's/^CMAKE_CXX_COMPILER:FILEPATH=//p' "${BUILD_DIR}/CMakeCache.txt" | head -n 1)
    if [[ -n "$cached_compiler" && "$cached_compiler" != "$(command -v g++)" ]]; then
        echo "ERROR: Existing build cache uses a different compiler: ${cached_compiler}" >&2
        echo "Remove ${BUILD_DIR} explicitly, then rerun this script." >&2
        exit 1
    fi
fi

cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON \
    -DHEAT_ENABLE_GCC_NVPTX=ON

cmake --build "$BUILD_DIR" \
    --target stencil_hybrid_gpu test_core test_mpi test_gpu \
    --parallel 4

# Only the pure core test is run on the login node. MPI and real-GPU tests
# are run inside the PBS allocation by debug.pbs / e5_hybrid_GPU.pbs.
ctest --test-dir "$BUILD_DIR" -R '^core_smoke$' --output-on-failure

for executable in stencil_hybrid_gpu test_gpu; do
    path="${BUILD_DIR}/${executable}"
    test -x "$path"
    echo
    echo "Dynamic-link check: ${path}"
    ldd "$path" | grep -E 'libmpi|libgomp|not found' || true
    if ldd "$path" | grep -q 'not found'; then
        echo "ERROR: Missing dynamic library in ${path}." >&2
        exit 1
    fi
    if ! ldd "$path" | grep -q 'OpenMPI/5.0.3-GCC-13.3.0.*libmpi'; then
        echo "ERROR: ${path} is not linked to the E4 OpenMPI 5.0.3 runtime." >&2
        exit 1
    fi
    if ! ldd "$path" | grep -q 'GCCcore/13.3.0/lib64/libgomp.so.1'; then
        echo "ERROR: ${path} is not linked to GCC 13.3 libgomp." >&2
        exit 1
    fi
done

echo
echo "E5 GCC/NVPTX binaries built successfully in ${BUILD_DIR}."
echo "Next: qsub scripts/vanda/debug.pbs"
