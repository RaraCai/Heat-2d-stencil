#!/usr/bin/env bash
set -euo pipefail

MODE="formal"
if (( $# == 1 )) && [[ "$1" == "--a1-only" ]]; then
    MODE="a1-only"
elif (( $# != 0 )); then
    echo "Usage: bash scripts/local/run_e5.sh [--a1-only]" >&2
    exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BUILD_DIR="${BUILD_DIR:-build/vanda_e5_gcc13}"
EXECUTABLE="${BUILD_DIR}/stencil_hybrid_gpu"

if [[ ! -x "$EXECUTABLE" ]]; then
    echo "ERROR: Missing E5 GPU executable: ${EXECUTABLE}" >&2
    echo "Run scripts/local/build_e5.sh on the login node first." >&2
    exit 1
fi

if [[ -z "${PBS_NODEFILE:-}" || ! -f "$PBS_NODEFILE" ]]; then
    echo "ERROR: PBS_NODEFILE is missing or invalid; run this script inside PBS." >&2
    exit 1
fi

mapfile -t allocated_nodes < <(awk '!seen[$1]++ {print $1}' "$PBS_NODEFILE")
if (( ${#allocated_nodes[@]} != 1 )); then
    echo "ERROR: E5 Group A requires exactly one allocated GPU node; found ${#allocated_nodes[@]}." >&2
    exit 1
fi

slots=$(wc -l < "$PBS_NODEFILE")
if (( slots < 32 )); then
    echo "ERROR: E5 Group A requires at least 32 allocated CPU slots; found ${slots}." >&2
    exit 1
fi

if [[ "${E4_MPI_BINDING:-}" != "vanda" ]]; then
    echo "ERROR: E4-compatible Vanda MPI binding is not enabled (E4_MPI_BINDING=vanda)." >&2
    exit 1
fi

if [[ "${OMP_TARGET_OFFLOAD:-}" != "MANDATORY" ]]; then
    echo "ERROR: OMP_TARGET_OFFLOAD must be MANDATORY." >&2
    exit 1
fi

if [[ -e results/raw/e5_gpu.csv || -e results/raw/e5_gpu_runs.csv ]]; then
    echo "ERROR: Existing E5 result files detected." >&2
    echo "Archive both CSV files before rerunning." >&2
    exit 1
fi

config_output="$("$EXECUTABLE" --list-configurations)"
mapfile -t configurations <<< "$config_output"
if (( ${#configurations[@]} != 5 )); then
    echo "ERROR: Expected five E5 Group A configurations; found ${#configurations[@]}." >&2
    exit 1
fi

for entry in "${configurations[@]}"; do
    read -r index group cores ranks cpu_threads thread_limit <<< "$entry"
    if [[ "$group" != "A" ]] ||
       (( ranks * cpu_threads != cores || cores > 36 || thread_limit < 1 )); then
        echo "ERROR: Invalid E5 resource mapping: ${entry}" >&2
        exit 1
    fi
done

if [[ "$MODE" == "a1-only" ]]; then
    configurations=("${configurations[0]}")
    echo "E5 DEBUG MODE: running A1 only (10 repetitions)."
else
    echo "E5 FORMAL MODE: running all five Group A configurations."
fi

echo "Allocated node: ${allocated_nodes[0]}"
echo "Allocated CPU slots: ${slots}"
echo "MPI launcher: $(command -v mpirun)"
echo "MPI runtime: $(mpirun --version | head -n 1)"
echo "Executable: ${EXECUTABLE}"

for entry in "${configurations[@]}"; do
    read -r index group cores ranks cpu_threads thread_limit <<< "$entry"

    echo
    echo "----------------------------------------"
    echo "[E5 GPU] Group ${group}, configuration ${index}"
    echo "CPU cores: ${cores}"
    echo "GPU: 1"
    echo "MPI ranks: ${ranks}"
    echo "OpenMP threads/rank: ${cpu_threads}"
    echo "GPU thread_limit: ${thread_limit}"
    echo "Mapping: ppr:${ranks}:node:PE=${cpu_threads}"
    echo "Binding: core"
    echo "----------------------------------------"

    # Same PPR + PE + core-binding policy as the successful E4 Vanda run.
    # PBS supplies the node allocation; do not create a second hostfile.
    mpirun \
        --map-by "ppr:${ranks}:node:PE=${cpu_threads}" \
        --bind-to core \
        --report-bindings \
        -np "$ranks" \
        "$EXECUTABLE" --run "$index"
done

if [[ "$MODE" == "formal" ]]; then
    echo
    echo "Generating complete five-configuration E5 summary..."
    "$EXECUTABLE" --summary
else
    echo
    echo "A1-only debug run finished; full five-configuration summary intentionally skipped."
fi
