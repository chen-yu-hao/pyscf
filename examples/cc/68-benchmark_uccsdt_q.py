#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

'''Compare the UCCSDT(Q) C/OpenMP kernel with the Python tensor reference.

This uses synthetic amplitudes, not a converged molecular calculation.
For reproducible scaling, select physical CPU cores externally, for example:
    OPENBLAS_NUM_THREADS=1 taskset -c 0-3 python 68-benchmark_uccsdt_q.py
'''

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pyscf
from pyscf import lib
from pyscf.cc import uccsdt_q


def make_inputs(nocc, nvir):
    rng = np.random.default_rng(3049)
    nmo = nocc + nvir
    # Spin-specific real density factors give all Coulomb symmetries,
    # including pair exchange and the mixed-spin transpose relation.
    factors = []
    for _ in range(2):
        a = rng.normal(size=(6, nmo, nmo))
        factors.append((a + a.transpose(0, 2, 1)) * 0.02)

    def eri(a, b):
        return np.einsum('Lpr,Lqs->pqrs', a, b, optimize=True)

    energies = tuple(np.r_[np.linspace(-1.2, -0.5, nocc) - spin * 0.03,
                           np.linspace(0.2, 1.1, nvir) + spin * 0.02]
                     for spin in (0, 1))
    eris = SimpleNamespace(pppp=eri(factors[0], factors[0]),
                           pPpP=eri(factors[0], factors[1]),
                           PPPP=eri(factors[1], factors[1]),
                           mo_energy=energies, fock=tuple(np.diag(e) for e in energies))

    def doubles():
        a = rng.normal(size=(nocc, nocc, nvir, nvir)) * 0.01
        a -= a.swapaxes(0, 1).copy()
        return a - a.swapaxes(2, 3)

    t2 = (doubles(), rng.normal(size=(nocc, nvir, nocc, nvir)) * 0.01, doubles())
    same = (math.comb(nocc, 3), math.comb(nvir, 3))
    mixed = (math.comb(nocc, 2), math.comb(nvir, 2), nocc, nvir)
    t3 = tuple(rng.normal(size=s) * 0.001 for s in (same, mixed, mixed, same))
    mycc = SimpleNamespace(nocc=(nocc, nocc), nmo=(nmo, nmo),
                           do_tri_max_t=True, max_memory=4000, converged=True,
                           einsum_backend='numpy', stdout=io.StringIO(), verbose=0,
                           tamps=[(np.zeros((nocc, nvir)), np.zeros((nocc, nvir))), t2, t3])
    return mycc, eris


def timed(kernel, mycc, eris):
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    result = np.asarray(kernel(mycc, eris=eris, verbose=0))
    return result, time.process_time() - start_cpu, time.perf_counter() - start_wall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nocc', type=int, default=3, help='occupied orbitals per spin')
    parser.add_argument('--nvir', type=int, default=12, help='virtual orbitals per spin')
    parser.add_argument('--output', type=Path, help='also write the JSON report here')
    args = parser.parse_args()
    if min(args.nocc, args.nvir) < 2:
        parser.error('nocc and nvir must be at least 2 for quadruple excitations')
    mycc, eris = make_inputs(args.nocc, args.nvir)
    native_path = Path(uccsdt_q._native_library()._name).resolve()
    repo = Path(__file__).resolve().parents[2]
    report = {
        'python': platform.python_version(), 'numpy': np.__version__,
        'pyscf': pyscf.__version__, 'pyscf_source': pyscf.__file__,
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'dirty_worktree': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo)),
        'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        'native_build': {'path': str(native_path),
                         'sha256': hashlib.sha256(native_path.read_bytes()).hexdigest()},
        'source_sha256': {name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
                          for name in ('pyscf/cc/uccsdt_q.py',
                                       'pyscf/lib/ccsdt/uccsdt_q.c',
                                       'examples/cc/68-benchmark_uccsdt_q.py')},
        'seed': 3049, 'nocc': list(mycc.nocc), 'nvir': [args.nvir] * 2,
        'compact_tamps': True, 'max_memory_mb': mycc.max_memory, 'repeats': 3,
        'thread_environment': {key: os.environ.get(key) for key in
                               ('OMP_NUM_THREADS', 'OMP_PROC_BIND', 'OMP_PLACES',
                                'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        'native': [],
    }
    original_threads = lib.num_threads()
    try:
        lib.num_threads(1)
        # Warm the independent Python path, including its formula cache.
        uccsdt_q._kernel_python(mycc, eris=eris, verbose=0)
        reference, cpu, wall = timed(uccsdt_q._kernel_python, mycc, eris)
        report['python_reference'] = {'threads': 1, 'energies_eh': reference.tolist(),
                                      'cpu_seconds': cpu, 'wall_seconds': wall}
        for nthreads in (1, 2, 4):
            lib.num_threads(nthreads)
            uccsdt_q.kernel(mycc, eris=eris, verbose=0)
            runs = [timed(uccsdt_q.kernel, mycc, eris) for _ in range(3)]
            cpu = statistics.median(run[1] for run in runs)
            wall = statistics.median(run[2] for run in runs)
            for energy, _, _ in runs:
                np.testing.assert_allclose(energy, reference, atol=1e-11, rtol=1e-10)
            report['native'].append({
                'threads': nthreads, 'energies_eh': runs[-1][0].tolist(),
                'median_cpu_seconds': cpu, 'median_wall_seconds': wall,
                'cpu_over_wall': cpu / wall,
                'max_abs_reference_error_eh': max(float(np.max(np.abs(run[0] - reference)))
                                                 for run in runs),
                'samples': [{'cpu_seconds': run[1], 'wall_seconds': run[2]} for run in runs],
            })
    finally:
        lib.num_threads(original_threads)
    for row in report['native']:
        row['speedup_over_c_one_thread'] = report['native'][0]['median_wall_seconds'] / row['median_wall_seconds']
    output = json.dumps(report, indent=2) + '\n'
    print(output, end='')
    if args.output:
        args.output.write_text(output)


if __name__ == '__main__':
    main()
