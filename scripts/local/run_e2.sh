#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BUILD_DIR="${BUILD_DIR:-build/release}"
RANKS_TO_RUN=("$@")
if [ ${#RANKS_TO_RUN[@]} -eq 0 ]; then
    RANKS_TO_RUN=(1 4 16 64 256)
fi

cmake -S . -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR"

for ranks in "${RANKS_TO_RUN[@]}"; do
    case "$ranks" in
        1|4|16|64|256) ;;
        *) echo "Unsupported E2 rank count: $ranks" >&2; exit 2 ;;
    esac

    echo "============================================================"
    echo "Running E2 with ${ranks} MPI ranks"
    echo "============================================================"
    mpirun -np "$ranks" "$BUILD_DIR/stencil_strong_mpi"
done

ALL_RESULTS_PRESENT=1
for ranks in 1 4 16 64 256; do
    if [ ! -f "results/raw/e2_rank_${ranks}.csv" ]; then
        ALL_RESULTS_PRESENT=0
        break
    fi
done

if [ "$ALL_RESULTS_PRESENT" -eq 1 ]; then
    python3 scripts/local/summarize_e2.py
else
    echo "Selected runs completed. Final E2 summary will be generated after all five rank configurations exist."
fi
