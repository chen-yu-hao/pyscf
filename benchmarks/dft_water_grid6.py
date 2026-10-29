"""Reproducible water-cluster grid-6 DFT smoke and SCF phase timings."""
import argparse
import json
import os
import platform
import time
from collections import defaultdict

import numpy as np
from pyscf import dft, gto, lib
from pyscf.scf import hf


def water_cluster(n):
    # Identical water monomers, arranged without hydrogen bonding to keep
    # assembly deterministic; n=1,2,4,6 exercises increasing AO/grid sizes.
    atoms = []
    for i in range(n):
        x = 4.0 * (i % 3)
        y = 4.0 * (i // 3)
        atoms.extend([
            ("O", (x, y, 0)),
            ("H", (x + 0.757, y, 0.587)),
            ("H", (x - 0.757, y, 0.587)),
        ])
    return atoms


def run(n, xc, threads, max_cycle):
    lib.num_threads(threads)
    mol = gto.M(atom=water_cluster(n), basis="def2-TZVPP", unit="Angstrom",
                verbose=0, max_memory=2300)
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = 6
    mf.conv_tol = 1e-8
    mf.max_cycle = max_cycle
    mf.max_memory = 2300
    stages = defaultdict(float)
    counts = defaultdict(int)

    def wrap(instance, method, name):
        original = getattr(instance, method)
        def timed(*a, **kw):
            start = time.perf_counter()
            try:
                return original(*a, **kw)
            finally:
                stages[name] += time.perf_counter() - start
                counts[name] += 1
        setattr(instance, method, timed)

    # get_veff includes XC + J/K; XC is measured separately by nr_rks.
    wrap(mf, "get_veff", "get_veff")
    wrap(mf, "get_jk", "get_jk")
    wrap(mf._numint, "nr_rks", "xc_native_numint")
    # Note NUMA workers bypass nr_rks on the coordinator.
    start = time.perf_counter()
    energy = float(mf.kernel())
    total = time.perf_counter() - start
    return dict(n_water=n, xc=xc, nao=mol.nao, grid_points=len(mf.grids.weights),
                threads=threads, numa=os.getenv("PYSCF_NUMA", "on"),
                converged=bool(mf.converged), cycles=int(getattr(mf, "cycles", -1)),
                energy_hartree=energy, wall_seconds=total,
                stages_seconds=dict(stages), call_counts=dict(counts),
                residual_seconds=max(0.0, total - stages["get_veff"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--xc", default="PBE")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--max-cycle", type=int, default=30)
    p.add_argument("--output", default="benchmark.json")
    args = p.parse_args()
    report = dict(platform=platform.platform(), processor=platform.processor(),
                  python=platform.python_version(), numpy=np.__version__,
                  omp_env=os.getenv("OMP_NUM_THREADS"),
                  cases=[], errors=[])
    for n in args.sizes:
        try:
            result = run(n, args.xc, args.threads, args.max_cycle)
            report["cases"].append(result)
            print(json.dumps(result), flush=True)
            if not result["converged"]:
                report["errors"].append("n=%d did not converge" % n)
        except Exception as exc:
            report["errors"].append("n=%d: %r" % (n, exc))
            print("ERROR", repr(exc), flush=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    if report["errors"]:
        raise SystemExit("; ".join(report["errors"]))


if __name__ == "__main__":
    main()
