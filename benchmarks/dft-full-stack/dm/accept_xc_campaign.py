#!/usr/bin/env python3
"""Validate every XC campaign result, then summarize accepted timing comparisons."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys


CELL_KEYS = ("case", "xc", "threads", "mode", "spin", "deriv", "points")
SHARED_KEYS = ("case", "xc", "xc_type", "threads", "method", "nao", "grid_points",
               "fixture_sha256", "coords_sha256", "weights_sha256", "dm_sha256",
               "input_sha256", "grid_cutoff", "numint_cutoff", "max_memory_mb", "affinity",
               "spin", "deriv", "output_shape", "calls", "warmups", "benchmark_script_sha256",
               "helper_script_sha256", "python", "numpy_version")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    import numpy as np
    batch = json.loads(args.batch.read_text())
    versions = {version["label"]: version for version in batch["versions"]}
    if args.baseline not in versions:
        raise ValueError("Baseline is not present in the batch")
    failures = []
    expected = batch["expected"]
    repeats = batch["repeats"]
    if repeats != 3:
        failures.append(f"Expected three fresh-process repeats, got {repeats}")
    if batch["status"] not in ("finished_unvalidated", "accepted"):
        failures.append(f'Nonterminal-success batch state: {batch["status"]}')
    if batch["completed"] != expected or len(batch["rows"]) != expected:
        failures.append(f'Incomplete batch: completed={batch["completed"]}, rows={len(batch["rows"])}, expected={expected}')
    cells = {tuple(cell.get(key) for key in CELL_KEYS) for cell in batch["selected_cells"]}
    if expected != len(cells) * len(versions) * repeats:
        failures.append("Expected task count does not match cells, versions and repeats")
    grouped = {}
    seen = set()
    for row in batch["rows"]:
        key = tuple(row.get(name) for name in CELL_KEYS)
        identity = (key, row["version"], row["repeat"])
        if identity in seen or key not in cells or row["version"] not in versions:
            failures.append(f"Duplicate or unexpected task: {identity}")
        seen.add(identity)
        if row["returncode"] != 0:
            failures.append(f'Nonzero return code: {row["result"]}')
        result_file = Path(row["result"])
        if not result_file.is_file():
            failures.append(f"Missing result: {result_file}")
            continue
        result = json.loads(result_file.read_text())
        version = versions[row["version"]]
        if result.get("status") != "succeeded":
            failures.append(f"Worker did not succeed: {result_file}")
        if result["source"]["commit"] != version["commit"] or row["commit"] != version["commit"]:
            failures.append(f"Wrong source commit: {result_file}")
        if result["source"]["tracked_changes"]:
            failures.append(f"Dirty source: {result_file}")
        if not Path(result["pyscf_import"]).is_relative_to(Path(version["source"]).resolve()):
            failures.append(f"Wrong import path: {result_file}")
        if not result.get("native_maps") or not all(item.get("sha256") for item in result["native_maps"]):
            failures.append(f"Missing native provenance: {result_file}")
        hashes = ("fixture_sha256", "benchmark_script_sha256")
        hashes += ("input_sha256",) if "mode" in result else ("coords_sha256", "weights_sha256", "dm_sha256")
        if not all(result.get(name) for name in hashes):
            failures.append(f"Missing fixture/input/harness provenance: {result_file}")
        samples = result["wall_seconds"]
        if len(samples) != batch["calls"] or any(not math.isfinite(value) or value <= 0 for value in samples):
            failures.append(f"Missing or invalid timing samples: {result_file}")
        if result["calls"] != batch["calls"] or result["warmups"] != batch["warmups"]:
            failures.append(f"Wrong repeat protocol: {result_file}")
        if not samples or result["median_wall_seconds"] != statistics.median(samples):
            failures.append(f"Incorrect process median: {result_file}")
        grouped.setdefault(key, {}).setdefault(row["version"], []).append((row, result))
    for cell in cells:
        for version in versions:
            actual = {row["repeat"] for row, _ in grouped.get(cell, {}).get(version, [])}
            if actual != set(range(repeats)):
                failures.append(f"Missing repeat set: {cell}, {version}, found={sorted(actual)}")
    comparisons = []
    validated = 0
    gates = {"nelec": 1e-8, "exc": 1e-8, "vxc": 1e-9}
    for key, group in grouped.items():
        if not group.get(args.baseline):
            continue
        baseline = sorted(group[args.baseline], key=lambda pair: pair[0]["repeat"])
        reference = baseline[0][1]
        reference_path = Path(reference["output_arrays"])
        if not reference_path.is_file():
            failures.append(f"Missing reference arrays: {reference_path}")
            continue
        with np.load(reference_path, allow_pickle=False) as archive:
            reference_arrays = {name: archive[name] for name in archive.files}
        mode = reference.get("mode", "xc")
        limits = gates if mode == "xc" else {"values": {"grid": 0., "ao": 1e-12, "libxc": 1e-10}[mode]}
        by_version = {}
        for label, records in group.items():
            local_failures = []
            errors = {name: 0. for name in limits}
            for row, result in records:
                context = row["result"]
                if any(result.get(name) != reference.get(name) for name in SHARED_KEYS):
                    local_failures.append(f"Input/settings mismatch: {context}")
                for name, value in result["max_repeat_absolute_error"].items():
                    if not math.isfinite(value) or value > limits[name]:
                        local_failures.append(f"Within-process numerical repeat error {name}={value}: {context}")
                arrays_path = Path(result["output_arrays"])
                if not arrays_path.is_file():
                    local_failures.append(f"Missing output arrays: {arrays_path}")
                    continue
                with np.load(arrays_path, allow_pickle=False) as arrays:
                    if set(arrays.files) != set(limits) or set(reference_arrays) != set(limits):
                        local_failures.append(f"Unexpected output fields: {context}")
                        continue
                    good = True
                    for name, limit in limits.items():
                        value, target = arrays[name], reference_arrays[name]
                        if value.shape != target.shape or not np.isfinite(value).all() or not np.isfinite(target).all():
                            local_failures.append(f"Nonfinite or mismatched array {name}: {context}")
                            good = False
                            continue
                        if value.ndim < 2:
                            error = float(np.max(np.abs(value - target)))
                        else:
                            error = max(float(np.max(np.abs(value[index] - target[index]))) for index in range(len(value)))
                        errors[name] = max(errors[name], error)
                        if error > limit:
                            local_failures.append(f"Numerical error {name}={error} > {limit}: {context}")
                            good = False
                    validated += int(good)
            failures.extend(local_failures)
            by_version[label] = {"errors": errors, "numerically_accepted": not local_failures,
                                 "seconds": [result["median_wall_seconds"] for _, result in records],
                                 "rss_mib": [result["peak_rss_mib"] for _, result in records]}
        for label, metrics in by_version.items():
            if label == args.baseline:
                continue
            base = by_version[args.baseline]
            good = metrics["numerically_accepted"] and base["numerically_accepted"]
            comparisons.append({**{name: value for name, value in zip(CELL_KEYS, key) if value is not None},
                                "baseline": args.baseline, "candidate": label,
                                "baseline_commit": versions[args.baseline]["commit"], "candidate_commit": versions[label]["commit"],
                                "numerically_accepted": good, "repeats": len(metrics["seconds"]),
                                "baseline_seconds_median": statistics.median(base["seconds"]),
                                "candidate_seconds_median": statistics.median(metrics["seconds"]),
                                "baseline_seconds_range": [min(base["seconds"]), max(base["seconds"])],
                                "candidate_seconds_range": [min(metrics["seconds"]), max(metrics["seconds"])],
                                "speedup": statistics.median(base["seconds"]) / statistics.median(metrics["seconds"]) if good else None,
                                "baseline_peak_rss_mib_median": statistics.median(base["rss_mib"]),
                                "candidate_peak_rss_mib_median": statistics.median(metrics["rss_mib"]),
                                "baseline_peak_rss_mib_range": [min(base["rss_mib"]), max(base["rss_mib"])],
                                "candidate_peak_rss_mib_range": [min(metrics["rss_mib"]), max(metrics["rss_mib"])],
                                "max_absolute_errors": metrics["errors"]})
    if validated != expected:
        failures.append(f"Numerical results validated {validated}/{expected}")
    summary = {"accepted": not failures, "expected": expected, "completed": batch["completed"],
               "numerically_validated": validated, "failures": failures, "comparisons": comparisons,
               "source_batch": str(args.batch.resolve()),
               "gates": gates, "micro_gates": {"grid": 0., "ao": 1e-12, "libxc": 1e-10},
               "statistic": "ordinary median of independent process medians; observed min/max ranges, not confidence intervals",
               "native_provenance": "Recorded paths and hashes checked present; caller must bind native build manifest to source commit"}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "acceptance.json").write_text(json.dumps(summary, indent=2) + "\n")
    if comparisons:
        columns = sorted(set().union(*(row.keys() for row in comparisons)))
        with (args.output / "comparison.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(comparisons)
    print(json.dumps({name: summary[name] for name in
                      ("accepted", "expected", "completed", "numerically_validated", "failures")}))
    return 0 if summary["accepted"] else 2


if __name__ == "__main__":
    sys.exit(main())
