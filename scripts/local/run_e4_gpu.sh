#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BUILD_DIR="${BUILD_DIR:-build/vanda_e4_gpu}"

for configuration in \
    "2 16" "4 8" "8 4" "16 2" "32 1" \
    "4 16" "8 8" "16 4" "32 2" "64 1" \
    "8 16" "16 8" "32 4" "64 2" "128 1"; do
    read -r ranks threads <<< "$configuration"
    echo "[E4 GPU] Running with ${ranks} MPI ranks x ${threads} threads/rank"

    if [[ "${E4_MPI_BINDING:-local}" == "vanda" ]]; then
        # Match the CPU E4 rank placement; all ranks on a node share its GPU.
        if (( ranks == 2 )); then
            ranks_per_node=2
        else
            ranks_per_node=$((ranks / 4))
        fi

        mpirun \
            --map-by "ppr:${ranks_per_node}:node:PE=${threads}" \
            --bind-to core \
            --report-bindings \
            -np "$ranks" \
            "$BUILD_DIR/stencil_hybrid_gpu" --run "$ranks" "$threads"
    else
        mpirun --oversubscribe --bind-to none \
            -np "$ranks" \
            "$BUILD_DIR/stencil_hybrid_gpu" --run "$ranks" "$threads"
    fi
done

"$BUILD_DIR/stencil_hybrid_gpu" --summary
