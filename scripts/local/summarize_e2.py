#!/usr/bin/env python3
"""Combine per-rank E2 outputs and calculate both speedup definitions."""

import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "raw"
E1_CSV = RESULTS / "e1_serial.csv"
OUTPUT = RESULTS / "e2_strong_mpi.csv"


def read_serial_baseline(path: pathlib.Path) -> float:
    if not path.exists():
        raise FileNotFoundError(
            f"E1 baseline not found: {path}. Run E1 first or copy its CSV here."
        )
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"grid_size", "computation_mean_s"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("E1 CSV is missing grid_size/computation_mean_s columns.")
        for row in reader:
            if int(row["grid_size"]) == 4096:
                value = float(row["computation_mean_s"])
                if value <= 0:
                    raise ValueError("E1 4096-grid baseline must be positive.")
                return value
    raise ValueError("No 4096-grid row found in E1 CSV.")


def main() -> int:
    serial_baseline = read_serial_baseline(E1_CSV)
    rank_rows = []
    for ranks in (1, 4, 16, 64, 256):
        path = RESULTS / f"e2_rank_{ranks}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing E2 rank result: {path}")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1:
            raise ValueError(f"Expected one summary row in {path}")
        rank_rows.append(rows[0])

    mpi_one_runtime = float(rank_rows[0]["runtime_median_s"])
    if mpi_one_runtime <= 0:
        raise ValueError("MPI 1-rank baseline must be positive.")

    fields = [
        "global_nx", "global_ny", "ranks", "process_rows", "process_cols",
        "steps", "repetitions", "runtime_median_s", "runtime_min_s",
        "runtime_max_s", "halo_runtime_median_s", "speedup_mpi",
        "speedup_serial", "parallel_efficiency", "serial_baseline_s",
        "min_temperature", "max_temperature", "max_boundary_error",
        "validation_passed",
    ]

    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rank_rows:
            ranks = int(row["ranks"])
            runtime = float(row["runtime_median_s"])
            output = dict(row)
            output["speedup_mpi"] = mpi_one_runtime / runtime
            output["speedup_serial"] = serial_baseline / runtime
            output["parallel_efficiency"] = (mpi_one_runtime / runtime) / ranks
            output["serial_baseline_s"] = serial_baseline
            writer.writerow(output)

    print("\nE2 | MPI STRONG SCALING SUMMARY")
    print("-" * 104)
    print(f"{'Ranks':>7} {'Process Grid':>13} {'Runtime (s)':>15} "
          f"{'Halo (s)':>13} {'MPI Speedup':>14} "
          f"{'Serial Speedup':>16} {'Efficiency':>13} {'Status':>9}")
    print("-" * 104)
    for row in rank_rows:
        ranks = int(row["ranks"])
        runtime = float(row["runtime_median_s"])
        halo = float(row["halo_runtime_median_s"])
        mpi_speedup = mpi_one_runtime / runtime
        serial_speedup = serial_baseline / runtime
        efficiency = mpi_speedup / ranks
        process_grid = f"{row['process_rows']}x{row['process_cols']}"
        status = "PASS" if row["validation_passed"] == "1" else "FAIL"
        print(f"{ranks:7d} {process_grid:>13} {runtime:15.6f} "
              f"{halo:13.6f} {mpi_speedup:14.3f} "
              f"{serial_speedup:16.3f} {efficiency * 100:12.2f}% {status:>9}")
    print("-" * 104)
    print(f"E1 serial baseline : {serial_baseline:.6f} s (4096 x 4096 row from E1 CSV)")
    print(f"MPI 1-rank baseline: {mpi_one_runtime:.6f} s (E2 median)")
    print(f"E2 summary written : {OUTPUT}\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"E2 summary error: {error}", file=sys.stderr)
        raise SystemExit(1)
