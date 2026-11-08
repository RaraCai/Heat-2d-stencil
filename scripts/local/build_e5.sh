
#!/usr/bin/env bash
set -euo pipefail

# ==================================================
# E5 GPU Build
# GCC 13.3 + OpenMPI 5.0.3 + GCC NVPTX
# ==================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# --------------------------------------------------
# 1. Load E4-compatible environment
# --------------------------------------------------

module purge
source /app1/ebenv

module load GCC/13.3.0
module load OpenMPI/5.0.3-GCC-13.3.0
module load CMake/3.29.3-GCCcore-13.3.0

export LD_LIBRARY_PATH="/opt/pbs/lib:${LD_LIBRARY_PATH:-}"

BUILD_DIR="build/vanda_e5_gcc13"

echo "========================================"
echo " E5 GCC 13.3 + OpenMPI 5.0.3 GPU Build"
echo "========================================"

echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo unavailable)"
echo "Build directory: $BUILD_DIR"

echo
echo "Loaded modules:"
module list

echo
echo "GCC:"
g++ --version

echo
echo "MPI wrapper:"
which mpicxx
mpicxx --showme:version
mpicxx --showme:command

echo
echo "MPI runtime:"
mpirun --version

echo
echo "CMake:"
cmake --version

# --------------------------------------------------
# 2. Verify E4 CPU baseline
# --------------------------------------------------

if [[ ! -f results/raw/e4_hybrid.csv ]]; then
    echo "ERROR: Missing E4 CPU baseline." >&2
    exit 1
fi

baseline_rows=$(($(wc -l < results/raw/e4_hybrid.csv) - 1))

if (( baseline_rows != 15 )); then
    echo "ERROR: Expected 15 E4 CPU configurations." >&2
    echo "Found: $baseline_rows" >&2
    exit 1
fi

echo
echo "E4 CPU baseline: VERIFIED"

# --------------------------------------------------
# 3. Fresh CMake configuration
# --------------------------------------------------

echo
echo "========================================"
echo " Configuring GCC NVPTX Offloading"
echo "========================================"

cmake --fresh -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON \
    -DCMAKE_CXX_COMPILER="$(command -v g++)" \
    -DMPI_CXX_COMPILER="$(command -v mpicxx)" \
    -DHEAT_ENABLE_GCC_NVPTX=ON

# --------------------------------------------------
# 4. Verify final linker flags BEFORE building
# --------------------------------------------------

LINK_FILE="$BUILD_DIR/CMakeFiles/stencil_hybrid_gpu.dir/link.txt"

if [[ ! -f "$LINK_FILE" ]]; then
    echo "ERROR: GPU linker command not generated." >&2
    exit 1
fi

echo
echo "GPU linker command:"
cat "$LINK_FILE"

if ! grep -q -- '-fopenmp' "$LINK_FILE"; then
    echo "ERROR: Missing -fopenmp in GPU final link." >&2
    exit 1
fi

if ! grep -q -- '-foffload=nvptx-none' "$LINK_FILE"; then
    echo "ERROR: Missing GCC NVPTX link option." >&2
    exit 1
fi

echo
echo "GPU linker flags: VERIFIED"

# --------------------------------------------------
# 5. Build E5 and validation executables
# --------------------------------------------------

echo
echo "========================================"
echo " Building E5 GPU Targets"
echo "========================================"

cmake --build "$BUILD_DIR" \
    --target stencil_hybrid_gpu test_gpu test_core test_mpi \
    --parallel 4

# --------------------------------------------------
# 6. Verify executable files
# --------------------------------------------------

for executable in \
    stencil_hybrid_gpu \
    test_gpu \
    test_core \
    test_mpi
do
    if [[ ! -x "$BUILD_DIR/$executable" ]]; then
        echo "ERROR: Missing executable: $executable" >&2
        exit 1
    fi
done

echo
echo "All required executables exist."

# --------------------------------------------------
# 7. Verify dynamic libraries
# --------------------------------------------------

for executable in stencil_hybrid_gpu test_gpu; do

    echo
    echo "Dynamic libraries: $executable"

    ldd "$BUILD_DIR/$executable" |
        grep -E 'libmpi|libgomp|not found' || true

    if ldd "$BUILD_DIR/$executable" | grep -q 'not found'; then
        echo "ERROR: Missing dynamic library in $executable." >&2
        exit 1
    fi

done

# --------------------------------------------------
# 8. CPU and MPI tests on login node
# --------------------------------------------------

echo
echo "========================================"
echo " Running CPU / MPI CTest"
echo "========================================"

ctest --test-dir "$BUILD_DIR" --output-on-failure

# GPU tests are intentionally NOT executed here.
# They will be executed on a PBS GPU node.

# --------------------------------------------------
# 9. Build completed
# --------------------------------------------------

echo
echo "========================================"
echo " E5 GCC NVPTX BUILD SUCCESSFUL"
echo "========================================"

echo "Build directory: $BUILD_DIR"
echo "Compiler: GCC 13.3.0"
echo "MPI: OpenMPI 5.0.3"
echo "GPU offloading: NVPTX"
echo
echo "GPU executable:"
ls -lh "$BUILD_DIR/stencil_hybrid_gpu"

echo
echo "GPU correctness executable:"
ls -lh "$BUILD_DIR/test_gpu"

echo
echo "Next step:"
echo "qsub scripts/vanda/debug.pbs"
