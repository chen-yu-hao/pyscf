#!/usr/bin/env python3
"""Measure one submitted grid, LibXC, or AO change against its actual parent."""
import argparse
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time

from xc_benchmark import array_digest, checkout_info, digest, native_maps


def repeat_error(np, first, current):
    if first.shape != current.shape:
        raise RuntimeError("Output shape changed between repeats")
    if first.ndim == 1:
        return float(np.max(np.abs(first - current)))
    return max(float(np.max(np.abs(first[index] - current[index])))
               for index in range(len(first)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--threads", required=True, type=int)
    parser.add_argument("--affinity", required=True)
    parser.add_argument("--mode", required=True, choices=("grid", "libxc", "ao"))
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--xc", default="PBE")
    parser.add_argument("--spin", type=int, choices=(0, 1), default=0)
    parser.add_argument("--deriv", type=int, choices=(1, 2), default=1)
    parser.add_argument("--points", type=int)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--calls", type=int, default=3)
    args = parser.parse_args()
    args.source = args.source.resolve()
    source = checkout_info(args.source)
    if source["commit"] != args.commit or source["tracked_changes"]:
        raise RuntimeError(f"Source does not match requested clean commit: {source}")
    affinity = list(map(int, args.affinity.split(",")))
    if len(affinity) != args.threads:
        raise ValueError("Exactly one physical CPU per OpenMP thread is required")
    os.sched_setaffinity(0, affinity)
    os.environ.update(OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", BLIS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                      OMP_WAIT_POLICY="PASSIVE", GOMP_SPINCOUNT="0", NUMPY_MADVISE_HUGEPAGE="0")
    sys.path.insert(0, str(args.source))
    import numpy as np
    import pyscf
    from pyscf import dft, gto, lib
    from threadpoolctl import threadpool_info
    if not Path(pyscf.__file__).resolve().is_relative_to(args.source):
        raise RuntimeError("Wrong PySCF import")
    lib.num_threads(args.threads)
    meta = json.loads(args.fixture.with_suffix(".json").read_text())
    if digest(args.fixture) != meta["fixture_sha256"]:
        raise RuntimeError("Fixture changed")
    mol = gto.M(atom=meta["atoms"], basis=meta["basis"], spin=meta["spin"],
                charge=meta["charge"], unit=meta["unit"], verbose=0)
    details = {}
    if args.mode == "libxc":
        count = args.points or 524288
        xctype = dft.libxc.xc_type(args.xc)
        nvar = {"LDA": 1, "GGA": 4, "MGGA": 5}[xctype]
        phase = np.linspace(0, 16 * np.pi, count)
        rho = np.empty((args.spin + 1, nvar, count))
        fractions = (1.,) if args.spin == 0 else (.6, .4)
        for channel, fraction in enumerate(fractions):
            rho[channel, 0] = fraction * (1 + .4 * np.sin(phase))
            if nvar > 1:
                for direction in range(3):
                    rho[channel, direction + 1] = .1 * fraction * np.cos(phase + direction)
            if nvar == 5:
                density = rho[channel, 0]
                grad2 = (rho[channel, 1:4] ** 2).sum(axis=0)
                rho[channel, 4] = .3 * (3 * np.pi**2)**(2/3) * density**(5/3) + grad2 / (8*density)
        if args.spin == 0:
            rho = rho[0]
        inputs = rho
        calculate = lambda: dft.libxc.eval_xc1(args.xc, rho, spin=args.spin, deriv=args.deriv)
        details = {"xc": args.xc, "xc_type": xctype, "spin": args.spin, "deriv": args.deriv,
                   "input_kind": "deterministic positive smooth densities, gradients and admissible tau"}
    else:
        with np.load(args.fixture, allow_pickle=False) as fixture:
            coords = fixture["coords"]
        if args.mode == "grid":
            count = len(coords) if args.points is None else min(args.points, len(coords))
            permutation = np.random.default_rng(20261026).permutation(len(coords))[:count]
            coords = np.asarray(coords[permutation], order="C")
            calculate = lambda: dft.gen_grid.arg_group_grids(mol, coords)
            details = {"input_kind": "fixed permutation of frozen molecular grid; exact index order is checked",
                       "permutation_seed": 20261026}
        else:
            count = min(args.points or 20000, len(coords))
            offset = (len(coords) - count) // 2
            coords = np.asarray(coords[offset:offset + count], order="C")
            calculate = lambda: mol.eval_gto("GTOval_sph_deriv1", coords)
            details = {"input_kind": "contiguous central segment of frozen sorted molecular grid",
                       "offset": offset, "ao_evaluator": "GTOval_sph_deriv1"}
        inputs = coords
    before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    reference = None
    worst = 0.
    samples = []
    for iteration in range(args.warmups + args.calls):
        started = time.perf_counter()
        values = calculate()
        wall = time.perf_counter() - started
        if not np.isfinite(values).all():
            raise RuntimeError("Non-finite output")
        if reference is None:
            reference = values
        else:
            worst = max(worst, repeat_error(np, reference, values))
        if iteration >= args.warmups:
            samples.append(wall)
    after_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays = args.output.with_suffix(".npz")
    np.savez(arrays, values=values)
    result = {
        "status": "succeeded", "source": source, "mode": args.mode, **details,
        "benchmark_script_sha256": digest(Path(__file__)),
        "helper_script_sha256": digest(Path(__file__).with_name("xc_benchmark.py")),
        "pyscf_import": str(Path(pyscf.__file__).resolve()),
        "python": {"executable": sys.executable, "version": sys.version},
        "numpy_version": np.__version__, "threads": args.threads,
        "affinity": sorted(os.sched_getaffinity(0)), "threadpools": threadpool_info(),
        "case": meta["case"], "nao": mol.nao, "grid_points": count,
        "fixture": str(args.fixture), "fixture_sha256": meta["fixture_sha256"],
        "input_sha256": array_digest(inputs), "output_shape": list(values.shape),
        "warmups": args.warmups, "calls": args.calls, "wall_seconds": samples,
        "median_wall_seconds": statistics.median(samples),
        "peak_rss_mib": after_rss / 1024, "before_kernel_peak_rss_mib": before_rss / 1024,
        "max_repeat_absolute_error": {"values": worst}, "output_arrays": str(arrays),
        "native_maps": native_maps(),
        "timing_boundary": "Selected public call only; excludes input preparation and output checks",
        "rss_boundary": "Process high-water RSS including inputs and retained comparison outputs; not pure kernel allocation",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("mode", "threads", "median_wall_seconds", "peak_rss_mib")}))


if __name__ == "__main__":
    main()
