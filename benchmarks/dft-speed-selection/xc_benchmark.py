#!/usr/bin/env python3
"""Fixed-density XC benchmark; select the measured checkout with --source.

Fixture generation and the timed calculation run in separate fresh processes.
Only ordinary NumInt.nr_rks/nr_uks calls are timed.  No J/K or SCF is performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time


CASES = {
    "water12-rks": (12, False, "cc-pvdz", 288),
    "water11oh-uks": (12, True, "cc-pvdz", 283),
    "water20-rks": (20, False, "cc-pvtz", 1160),
    "water19oh-uks": (20, True, "cc-pvtz", 1146),
    "o2-uks": (0, True, "cc-pvtz", 60),
}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checkout_info(source):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(source), *args], text=True).strip()
    return {"root": str(source), "commit": git("rev-parse", "HEAD"),
            "tracked_changes": git("status", "--porcelain", "--untracked-files=no")}


def array_digest(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def geometry(case):
    count, radical, basis, nao = CASES[case]
    if not count:
        atoms = [["O", [0., 0., 0.]], ["O", [0., 0., 1.21]]]
        return atoms, basis, 2, nao
    atoms = []
    for index in range(count):
        x, y, z = (index % 5) * 3.4, (index // 5) * 3.4, 0.1 * (index % 3)
        atoms.extend([["O", [x, y, z]], ["H", [x, y - .757, z + .587]]])
        if not (radical and index == count - 1):
            atoms.append(["H", [x, y + .757, z + .587]])
    return atoms, basis, int(radical), nao


def native_maps():
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[-1].startswith("/") and ".so" in fields[-1]:
            path = fields[-1]
            if any(name in path.lower() for name in
                   ("pyscf", "libcint", "libxc", "libgomp", "libopenblas", "libmkl")):
                paths.add(path)
    return [{"path": path, "sha256": digest(path)} for path in sorted(paths)]


def make_fixture(args, np, gto, dft, scipy_linalg, source_info):
    atoms, basis, spin, expected_nao = geometry(args.case)
    mol = gto.M(atom=atoms, basis=basis, spin=spin, unit="Angstrom", verbose=0)
    assert mol.nao == expected_nao, (mol.nao, expected_nao)
    overlap = mol.intor("int1e_ovlp")
    hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    _, coeff = scipy_linalg.eigh(hcore, overlap)
    na, nb = mol.nelec
    if args.case.endswith("rks"):
        dm = 2 * coeff[:, :na].dot(coeff[:, :na].T)
    else:
        dm = np.array([coeff[:, :n].dot(coeff[:, :n].T) for n in (na, nb)])
    grids = dft.gen_grid.Grids(mol)
    grids.level = args.grid_level
    grids.build(with_non0tab=True)
    ni = dft.numint.NumInt()
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.fixture, dm=dm, overlap=overlap, coords=grids.coords,
             weights=grids.weights, non0tab=grids.non0tab)
    result = {
        "case": args.case, "atoms": atoms, "basis": basis, "spin": spin,
        "charge": 0, "unit": "Angstrom", "nao": mol.nao,
        "electrons": mol.nelectron, "spin_electrons": list(mol.nelec),
        "grid_level": args.grid_level, "grid_prune": "nwchem_prune (default)",
        "grid_cutoff": grids.cutoff, "numint_cutoff": ni.cutoff,
        "grid_points": len(grids.weights), "source": source_info,
        "fixture_sha256": digest(args.fixture),
        "coords_sha256": array_digest(grids.coords),
        "weights_sha256": array_digest(grids.weights),
        "dm_sha256": array_digest(dm),
        "density": "Frozen core-Hamiltonian occupied-orbital projector; not a converged SCF density",
        "overlap_electrons": np.einsum("...ij,ji->...", dm, overlap).tolist(),
    }
    args.fixture.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


def measure(args, np, gto, dft, source_info):
    meta = json.loads(args.fixture.with_suffix(".json").read_text())
    if digest(args.fixture) != meta["fixture_sha256"]:
        raise RuntimeError("Fixture has changed")
    data = np.load(args.fixture, allow_pickle=False)
    mol = gto.M(atom=meta["atoms"], basis=meta["basis"], spin=meta["spin"],
                charge=meta["charge"], unit=meta["unit"], verbose=0)
    grids = dft.gen_grid.Grids(mol)
    grids.level = meta["grid_level"]
    grids.cutoff = meta["grid_cutoff"]
    grids.coords, grids.weights = data["coords"], data["weights"]
    grids.non0tab = grids.screen_index = data["non0tab"]
    dm = data["dm"]
    ni = dft.numint.NumInt()
    ni.cutoff = meta["numint_cutoff"]
    calculate = ni.nr_rks if dm.ndim == 2 else ni.nr_uks
    samples = []
    reference = None
    max_repeat_error = {"nelec": 0., "exc": 0., "vxc": 0.}
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for iteration in range(args.warmups + args.calls):
        started = time.perf_counter()
        values = calculate(mol, grids, args.xc, dm, hermi=1, max_memory=args.memory_mb)
        wall = time.perf_counter() - started
        if not all(np.isfinite(value).all() for value in map(np.asarray, values)):
            raise RuntimeError("Non-finite NumInt output")
        if reference is None:
            reference = tuple(np.asarray(value).copy() for value in values)
        else:
            for name, value, expected in zip(max_repeat_error, values, reference):
                max_repeat_error[name] = max(max_repeat_error[name],
                                            float(np.max(np.abs(value - expected))))
        if iteration >= args.warmups:
            samples.append(wall)
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    output_arrays = args.output.with_suffix(".npz")
    np.savez(output_arrays, nelec=values[0], exc=values[1], vxc=values[2])
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
    except ImportError:
        pools = None
    result = {
        "status": "succeeded", "source": source_info,
        "benchmark_script_sha256": digest(Path(__file__)),
        "python": {"executable": sys.executable, "version": sys.version},
        "numpy_version": np.__version__,
        "pyscf_import": str(Path(sys.modules["pyscf"].__file__).resolve()),
        "case": meta["case"], "method": "RKS" if dm.ndim == 2 else "UKS",
        "xc": args.xc, "xc_type": ni._xc_type(args.xc), "nao": mol.nao,
        "threads": args.threads, "affinity": sorted(os.sched_getaffinity(0)),
        "max_memory_mb": args.memory_mb, "grid_points": len(grids.weights),
        "grid_cutoff": grids.cutoff, "numint_cutoff": ni.cutoff,
        "fixture": str(args.fixture), "fixture_sha256": meta["fixture_sha256"],
        "coords_sha256": meta["coords_sha256"], "weights_sha256": meta["weights_sha256"],
        "dm_sha256": meta["dm_sha256"], "density": meta["density"],
        "warmups": args.warmups, "calls": args.calls, "wall_seconds": samples,
        "median_wall_seconds": statistics.median(samples),
        "peak_rss_mib": rss_after / 1024, "before_xc_peak_rss_mib": rss_before / 1024,
        "max_repeat_absolute_error": max_repeat_error,
        "overlap_electrons": np.einsum("...ij,ji->...", dm, data["overlap"]).tolist(),
        "integrated_electrons": np.asarray(values[0]).tolist(), "exc_hartree": float(values[1]),
        "output_arrays": str(output_arrays), "native_maps": native_maps(),
        "threadpools": pools,
        "timing_boundary": "NumInt.nr_rks/nr_uks only; includes AO, density, LibXC and potential assembly; excludes import, fixture loading, grid build, J/K and SCF",
        "rss_boundary": "Linux process high-water RSS through warmup and timed calls; not a phase-local allocation measurement",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("case", "xc", "threads", "median_wall_seconds", "peak_rss_mib")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--make-fixture", action="store_true")
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--grid-level", type=int, default=2)
    parser.add_argument("--xc", default="PBE")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--affinity", required=True, help="Comma-separated physical OS CPU IDs")
    parser.add_argument("--memory-mb", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.source = args.source.resolve()
    source_info = checkout_info(args.source)
    if source_info["commit"] != args.commit or source_info["tracked_changes"]:
        raise RuntimeError(f"Source does not match clean requested commit: {source_info}")
    affinity = list(map(int, args.affinity.split(",")))
    if len(affinity) != args.threads:
        raise ValueError("Exactly one physical CPU per OpenMP thread is required")
    os.sched_setaffinity(0, affinity)
    for name, value in {"OMP_NUM_THREADS": str(args.threads), "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1",
                        "OMP_WAIT_POLICY": "PASSIVE", "GOMP_SPINCOUNT": "0",
                        "NUMEXPR_NUM_THREADS": "1", "NUMPY_MADVISE_HUGEPAGE": "0"}.items():
        os.environ[name] = value
    sys.path.insert(0, str(args.source))
    import numpy as np
    from scipy import linalg
    import pyscf
    from pyscf import dft, gto, lib
    if not Path(pyscf.__file__).resolve().is_relative_to(args.source):
        raise RuntimeError(f"Wrong PySCF import: {pyscf.__file__}")
    lib.num_threads(args.threads)
    if args.make_fixture:
        if args.case is None:
            parser.error("--case is required with --make-fixture")
        make_fixture(args, np, gto, dft, linalg, source_info)
    else:
        if args.output is None:
            parser.error("--output is required for measurement")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        measure(args, np, gto, dft, source_info)


if __name__ == "__main__":
    main()
