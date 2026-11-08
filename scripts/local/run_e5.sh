#!/usr/bin/env bash
set -euo pipefail

MODE="formal"
if (( $# == 1 )) && [[ "$1" == "--a1-only" ]]; then
    MODE="a1-only"
elif (( $# == 1 )) && [[ "$1" == "--a1-a2-only" ]]; then
    MODE="a1-a2-only"
elif (( $# != 0 )); then
    echo "Usage: bash scripts/local/run_e5.sh [--a1-only|--a1-a2-only]" >&2
    exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BUILD_DIR="${BUILD_DIR:-build/vanda_e5_gcc13}"
EXECUTABLE="$BUILD_DIR/stencil_hybrid_gpu"

if [[ ! -x "$EXECUTABLE" ]]; then
    echo "ERROR: Missing E5 GPU executable: $EXECUTABLE" >&2
    exit 1
fi

if [[ -e results/raw/e5_gpu.csv || -e results/raw/e5_gpu_runs.csv ]]; then
    echo "ERROR: Existing E5 result files detected." >&2
    echo "Archive debug/previous results before rerunning." >&2
    exit 1
fi

mapfile -t configurations < <("$EXECUTABLE" --list-configurations)
if (( ${#configurations[@]} != 5 )); then
    echo "ERROR: Expected five Group A configurations; found ${#configurations[@]}." >&2
    exit 1
fi

if [[ "$MODE" == "a1-only" ]]; then
    configurations=("${configurations[0]}")
    echo "E5 A1-only diagnostic run."
elif [[ "$MODE" == "a1-a2-only" ]]; then
    configurations=("${configurations[0]}" "${configurations[1]}")
    echo "E5 A1+A2 Direct GPU MPI integration run."
else
    echo "E5 FORMAL MODE: running all five Group A configurations."
fi

for entry in "${configurations[@]}"; do
    read -r index group cores ranks cpu_threads thread_limit <<< "$entry"

    if [[ "$group" != "A" ]] || \
       (( ranks * cpu_threads != cores || cores != 32 || cores > 36 )); then
        echo "ERROR: Invalid Group A resource mapping: $entry" >&2
        exit 1
    fi

    echo
echo "----------------------------------------"
    echo "[E5 GPU] Group $group, configuration $index"
    echo "CPU cores: $cores"
    echo "MPI ranks: $ranks"
    echo "CPU threads/rank: $cpu_threads"
    echo "GPU: 1"
    echo "GPU thread_limit: $thread_limit"
    echo "----------------------------------------"

    mpirun \
        --map-by "ppr:${ranks}:node:PE=${cpu_threads}" \
        --bind-to core \
        --report-bindings \
        -np "$ranks" \
        "$EXECUTABLE" --run "$index"
done

summary_rows=0
run_rows=0
if [[ -f results/raw/e5_gpu.csv ]]; then
    summary_rows=$(($(wc -l < results/raw/e5_gpu.csv) - 1))
fi
if [[ -f results/raw/e5_gpu_runs.csv ]]; then
    run_rows=$(($(wc -l < results/raw/e5_gpu_runs.csv) - 1))
fi

if [[ "$MODE" == "a1-only" ]]; then
    echo "A1 summary rows: $summary_rows/1"
    echo "A1 repetition rows: $run_rows/10"
    if (( summary_rows != 1 || run_rows != 10 )); then
        echo "ERROR: A1 output is incomplete." >&2
        exit 1
    fi
    echo "E5 A1-only run completed successfully."
elif [[ "$MODE" == "a1-a2-only" ]]; then
    echo "A1+A2 summary rows: $summary_rows/2"
    echo "A1+A2 repetition rows: $run_rows/20"
    if (( summary_rows != 2 || run_rows != 20 )); then
        echo "ERROR: A1+A2 output is incomplete." >&2
        exit 1
    fi
    echo "E5 A1+A2 integration run completed successfully."
else
    echo "Summary rows: $summary_rows/5"
    echo "Repetition rows: $run_rows/50"
    if (( summary_rows != 5 || run_rows != 50 )); then
        echo "ERROR: E5 result CSVs are incomplete." >&2
        exit 1
    fi
    "$EXECUTABLE" --summary
fi
