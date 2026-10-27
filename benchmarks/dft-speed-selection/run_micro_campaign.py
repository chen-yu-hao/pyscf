#!/usr/bin/env python3
"""Run one frozen microbenchmark protocol against a branch and its parent."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time


def cpu_ticks(affinity):
    names = {f"cpu{cpu}" for cpu in affinity}
    result = {}
    for line in Path("/proc/stat").read_text().splitlines():
        fields = line.split()
        if fields and fields[0] in names:
            result[fields[0]] = list(map(int, fields[1:]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--versions", nargs=2, required=True, metavar="LABEL=CHECKOUT")
    parser.add_argument("--python", required=True)
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).with_name("fixtures"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--affinity", required=True, help="Ordered physical CPU IDs; first N used")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Use a fresh output directory; prior evidence is immutable")
    protocol = json.loads(args.protocol.read_text())
    cells = protocol["cells"]
    affinity = list(map(int, args.affinity.split(",")))
    if not cells or max(cell["threads"] for cell in cells) > len(affinity):
        raise ValueError("No cells selected, or too few CPUs in the affinity list")
    versions = []
    for spec in args.versions:
        label, directory = spec.split("=", 1)
        source = Path(directory).resolve()
        commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        versions.append({"label": label, "source": str(source), "commit": commit})
    if versions[0]["label"] == versions[1]["label"]:
        raise ValueError("Parent and candidate require distinct labels")
    args.output.mkdir(parents=True)
    worker = Path(__file__).with_name("micro_benchmark.py")
    batch = {
        "status": "running", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
        "versions": versions, "protocol": protocol, "selected_cells": cells,
        "affinity_order": affinity, "expected": len(cells) * len(versions) * 3,
        "completed": 0, "rows": [], "repeats": 3, "warmups": 1, "calls": 3,
        "host": os.uname().nodename,
        "order": "Sequential fresh processes; parent/candidate order rotates by replicate",
    }
    ledger = args.output / "batch.json"
    def save():
        temporary = ledger.with_suffix(".tmp")
        temporary.write_text(json.dumps(batch, indent=2) + "\n")
        temporary.replace(ledger)
    save()
    try:
        for cell in cells:
            fixture = args.fixtures / (cell["case"] + ".npz")
            if not fixture.is_file():
                raise FileNotFoundError(f"Missing frozen fixture: {fixture}")
            cpus = affinity[:cell["threads"]]
            for repeat in range(3):
                offset = repeat % 2
                for version in versions[offset:] + versions[:offset]:
                    name = f'{cell["mode"]}-{cell["case"]}-t{cell["threads"]}'
                    if "xc" in cell:
                        name += f'-{cell["xc"].replace(",", "_")}-s{cell["spin"]}-d{cell["deriv"]}'
                    name += f'-{version["label"]}-r{repeat}'
                    output = args.output / (name + ".json")
                    command = [args.python, str(worker), "--source", version["source"],
                               "--commit", version["commit"], "--mode", cell["mode"],
                               "--fixture", str(fixture), "--threads", str(cell["threads"]),
                               "--affinity", ",".join(map(str, cpus)), "--warmups", "1",
                               "--calls", "3", "--output", str(output)]
                    for option in ("xc", "spin", "deriv", "points"):
                        if option in cell:
                            command.extend(["--" + option, str(cell[option])])
                    row = {**cell, "version": version["label"], "commit": version["commit"],
                           "repeat": repeat, "command": command, "result": str(output),
                           "returncode": None, "cpu_ticks_before": cpu_ticks(affinity)}
                    batch["rows"].append(row)
                    save()
                    before_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                    started = time.perf_counter()
                    print(f"START {name}", flush=True)
                    with output.with_suffix(".log").open("w") as logfile:
                        child = subprocess.run(command, stdout=logfile, stderr=subprocess.STDOUT,
                                               timeout=args.timeout, check=False)
                    row["process_wall_seconds"] = time.perf_counter() - started
                    after_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                    row.update(returncode=child.returncode, cpu_ticks_after=cpu_ticks(affinity),
                               child_cpu_seconds=after_usage.ru_utime + after_usage.ru_stime
                                                 - before_usage.ru_utime - before_usage.ru_stime)
                    if child.returncode or not output.is_file():
                        raise RuntimeError(f"Failed child {name}; inspect its retained log")
                    batch["completed"] += 1
                    save()
                    print(f'DONE {batch["completed"]}/{batch["expected"]} {name} '
                          f'{row["process_wall_seconds"]:.3f}s', flush=True)
        batch["status"] = "finished_unvalidated"
    except BaseException as error:
        batch["status"] = "failed"
        batch["error"] = repr(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
