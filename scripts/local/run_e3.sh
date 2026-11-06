#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BUILD_DIR="${BUILD_DIR:-build/release}"
REQUESTED_RANKS=("$@")
if [ ${#REQUESTED_RANKS[@]} -eq 0 ]; then
    REQUESTED_RANKS=(1 4 16 64 256)
fi

for ranks in "${REQUESTED_RANKS[@]}"; do
    case "$ranks" in
        1|4|16|64|256) ;;
        *) echo "Unsupported E3 rank count: $ranks" >&2; exit 2 ;;
    esac
done

RANKS_TO_RUN=()
for supported_ranks in 1 4 16 64 256; do
    for requested_ranks in "${REQUESTED_RANKS[@]}"; do
        if [ "$requested_ranks" = "$supported_ranks" ]; then
            RANKS_TO_RUN+=("$supported_ranks")
            break
        fi
    done
done
if [ "${RANKS_TO_RUN[0]}" != 1 ]; then
    RANKS_TO_RUN=(1 "${RANKS_TO_RUN[@]}")
fi

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR"

mkdir -p results/raw
rm -f results/raw/e3_weak.csv

for ranks in "${RANKS_TO_RUN[@]}"; do
    echo "[E3] Running with ${ranks} MPI ranks"
    mpirun -np "$ranks" "$BUILD_DIR/stencil_weak_mpi"
done

"$BUILD_DIR/stencil_weak_mpi" --summary