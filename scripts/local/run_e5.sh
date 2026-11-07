#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BUILD_DIR="${BUILD_DIR:-build/vanda_gpu}"
EXECUTABLE="$BUILD_DIR/stencil_hybrid_gpu"
MODE="${1:-run}"

if [[ ! -x "$EXECUTABLE" ]]; then
    echo "Missing E5 GPU executable: $EXECUTABLE" >&2
    echo "Run scripts/local/build_e5.sh on a login node before submitting E5." >&2
    exit 1
fi

mapfile -t allocated_nodes < <(awk '!seen[$1]++ {print $1}' "${PBS_NODEFILE:?PBS_NODEFILE is not set}")
if (( ${#allocated_nodes[@]} != 1 )); then
    echo "E5 Group A requires exactly one allocated GPU node; found ${#allocated_nodes[@]}." >&2
    exit 1
fi

hostfile="$(mktemp "${TMPDIR:-/tmp}/e5_gpu_hosts.XXXXXX")"
trap 'rm -f -- "$hostfile"' EXIT
printf '%s slots=36\n' "${allocated_nodes[0]}" > "$hostfile"

if [[ "$MODE" == "--test" ]]; then
    echo "[E5] GPU reference preflight: one rank / one GPU"
    mpirun --hostfile "$hostfile" --map-by ppr:1:node --bind-to core \
        -np 1 "$EXECUTABLE" --test

    echo "[E5] GPU reference preflight: four ranks sharing one GPU"
    mpirun --hostfile "$hostfile" --map-by ppr:4:node:PE=1 --bind-to core \
        -np 4 "$EXECUTABLE" --test
    exit 0
fi

if [[ -e results/raw/e5_gpu.csv || -e results/raw/e5_gpu_runs.csv ]]; then
    echo "E5 GPU result file already exists. Move or remove both files explicitly before rerunning." >&2
    exit 1
fi

mapfile -t configurations < <("$EXECUTABLE" --list-configurations)
if (( ${#configurations[@]} != 5 )); then
    echo "Expected five E5 Group A configurations; found ${#configurations[@]}." >&2
    exit 1
fi

for entry in "${configurations[@]}"; do
    read -r index group cores ranks cpu_threads thread_limit <<< "$entry"
    if [[ "$group" != "A" ]] ||
       (( ranks * cpu_threads != cores || cores > 36 )); then
        echo "Invalid E5 resource mapping for configuration $index." >&2
        exit 1
    fi

    echo "[E5 GPU] Group ${group}: ${cores} CPU cores, one GPU,"
    echo "          ${ranks} MPI ranks, ${cpu_threads} CPU threads/rank,"
    echo "          GPU thread_limit=${thread_limit}"
    mpirun --hostfile "$hostfile" \
        --map-by "ppr:${ranks}:node:PE=${cpu_threads}" \
        --bind-to core --report-bindings \
        -np "$ranks" "$EXECUTABLE" --run "$index"
done

"$EXECUTABLE" --summary
