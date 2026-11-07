#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BUILD_DIR="${BUILD_DIR:-build/release}"

cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON
cmake --build "$BUILD_DIR" --parallel
ctest --test-dir "$BUILD_DIR" --output-on-failure

mkdir -p results/raw
if [[ -f results/raw/e4_hybrid.csv || -f results/raw/e4_hybrid_runs.csv ]]; then
    ARCHIVE_DIR="results/archive/e4_local_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$ARCHIVE_DIR"
    [[ ! -f results/raw/e4_hybrid.csv ]] || mv results/raw/e4_hybrid.csv "$ARCHIVE_DIR/"
    [[ ! -f results/raw/e4_hybrid_runs.csv ]] || mv results/raw/e4_hybrid_runs.csv "$ARCHIVE_DIR/"
fi

# Local runs may exceed the CPU slots visible to WSL. On Vanda, set
# E4_MPI_BINDING=vanda to bind each MPI rank to its requested CPU cores.
for configuration in \
    "2 16" "4 8" "8 4" "16 2" "32 1" \
    "4 16" "8 8" "16 4" "32 2" "64 1" \
    "8 16" "16 8" "32 4" "64 2" "128 1"; do
    read -r ranks threads <<< "$configuration"
    echo "[E4] Running with ${ranks} MPI ranks x ${threads} OpenMP threads"

    if [[ "${E4_MPI_BINDING:-local}" == "vanda" ]]; then

        # Distribute MPI ranks across 4 Vanda compute nodes.
        # Each node has 36 allocated CPU cores.
        if (( ranks == 2 )); then
            ranks_per_node=2
        else
            ranks_per_node=$((ranks / 4))
        fi

        echo "[E4] Mapping: ${ranks_per_node} ranks per node"
        echo "[E4] CPU cores per rank: ${threads}"

        mpirun \
            --map-by "ppr:${ranks_per_node}:node:PE=${threads}" \
            --bind-to core \
            --report-bindings \
            -np "$ranks" \
            "$BUILD_DIR/stencil_hybrid" --run "$ranks" "$threads"

    else
        mpirun --oversubscribe --bind-to none \
            -np "$ranks" \
            "$BUILD_DIR/stencil_hybrid" --run "$ranks" "$threads"
    fi
done

"$BUILD_DIR/stencil_hybrid" --summary
