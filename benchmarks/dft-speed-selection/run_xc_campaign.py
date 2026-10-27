#!/usr/bin/env python3
"""Run an interleaved XC comparison, saving every child and its full provenance."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def cpu_ticks(affinity):
    result = {}
    for line in Path("/proc/stat").read_text().splitlines():
        fields = line.split()
        if fields and fields[0] in {f"cpu{cpu}" for cpu in affinity}:
            result[fields[0]] = list(map(int, fields[1:]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol-xc.json"))
    parser.add_argument("--versions", nargs="+", required=True, metavar="LABEL=CHECKOUT")
    parser.add_argument("--python", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).with_name("fixtures"))
    parser.add_argument("--affinity", required=True, help="Ordered comma-separated physical CPU IDs; first N used")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--functionals", nargs="+")
    parser.add_argument("--threads", nargs="+", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Use a fresh output namespace; existing campaign evidence is immutable")
    protocol = json.loads(args.protocol.read_text())
    versions = []
    for spec in args.versions:
        label, source = spec.split("=", 1)
        source = Path(source).resolve()
        commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        versions.append({"label": label, "source": str(source), "commit": commit})
    affinity = list(map(int, args.affinity.split(",")))
    cells = [cell for cell in protocol["cells"]
             if (args.cases is None or cell["case"] in args.cases)
             and (args.functionals is None or cell["xc"] in args.functionals)
             and (args.threads is None or cell["threads"] in args.threads)]
    if not cells or max(cell["threads"] for cell in cells) > len(affinity):
        raise ValueError("No selected cells, or affinity has too few physical CPUs")
    args.output.mkdir(parents=True)
    batch = {"status": "running", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
             "versions": versions, "protocol": protocol, "selected_cells": cells,
             "affinity_order": affinity, "expected": len(cells) * len(versions) * args.repeats,
             "completed": 0, "rows": [], "repeats": args.repeats,
             "warmups": args.warmups, "calls": args.calls, "host": os.uname().nodename,
             "order": "Sequential fresh processes; version order rotates each replicate; same frozen fixtures"}
    ledger = args.output / "batch.json"
    def save():
        temporary = ledger.with_suffix(".tmp")
        temporary.write_text(json.dumps(batch, indent=2) + "\n")
        temporary.replace(ledger)
    save()
    worker = Path(__file__).with_name("xc_benchmark.py")
    try:
        for cell in cells:
            fixture = args.fixtures / (cell["case"] + ".npz")
            if not fixture.is_file():
                raise FileNotFoundError(f"Generate the baseline fixture before the campaign: {fixture}")
            cpus = affinity[:cell["threads"]]
            for repeat in range(args.repeats):
                offset = repeat % len(versions)
                for version in versions[offset:] + versions[:offset]:
                    name = f'{cell["case"]}-{cell["xc"].replace(",", "_")}-t{cell["threads"]}-{version["label"]}-r{repeat}'
                    output = args.output / (name + ".json")
                    command = [args.python, str(worker), "--source", version["source"],
                               "--commit", version["commit"], "--fixture", str(fixture),
                               "--xc", cell["xc"], "--threads", str(cell["threads"]),
                               "--affinity", ",".join(map(str, cpus)),
                               "--memory-mb", str(protocol["max_memory_mb"]),
                               "--warmups", str(args.warmups), "--calls", str(args.calls),
                               "--output", str(output)]
                    before_ticks = cpu_ticks(affinity)
                    before_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                    started = time.perf_counter()
                    print(f"START {name}", flush=True)
                    with output.with_suffix(".log").open("w") as logfile:
                        child = subprocess.run(command, stdout=logfile, stderr=subprocess.STDOUT,
                                               timeout=args.timeout, check=False)
                    wall = time.perf_counter() - started
                    after_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                    row = {**cell, "version": version["label"], "commit": version["commit"],
                           "repeat": repeat, "command": command, "result": str(output),
                           "returncode": child.returncode, "process_wall_seconds": wall,
                           "child_cpu_seconds": after_usage.ru_utime + after_usage.ru_stime
                                                - before_usage.ru_utime - before_usage.ru_stime,
                           "cpu_ticks_before": before_ticks, "cpu_ticks_after": cpu_ticks(affinity)}
                    batch["rows"].append(row)
                    if child.returncode or not output.is_file():
                        raise RuntimeError(f"Failed child {name}; inspect its retained log")
                    batch["completed"] += 1
                    save()
                    print(f'DONE {batch["completed"]}/{batch["expected"]} {name} {wall:.3f}s', flush=True)
        batch["status"] = "finished_unvalidated"
    except BaseException as error:
        batch["status"] = "failed"
        batch["error"] = repr(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
