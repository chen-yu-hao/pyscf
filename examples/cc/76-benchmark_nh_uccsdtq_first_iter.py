#!/usr/bin/env python
"""Measure one UCCSDTQ update for triplet NH/cc-pVDZ."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def worker(nthreads):
    import numpy as np
    from pyscf import cc, gto, lib, scf
    from pyscf.cc import _uccsdtq_native, _uccsdtq_terms, _uccsdtq_utils, uccsdtq

    lib.num_threads(1)
    mol = gto.M(atom='N 0 0 0; H 0 0 1.036', basis='cc-pvdz',
                charge=0, spin=2, cart=False, verbose=0)
    t0 = time.perf_counter()
    mf = scf.UHF(mol).set(conv_tol=1e-12).run()
    scf_seconds = time.perf_counter() - t0
    if not mf.converged:
        raise RuntimeError('NH UHF did not converge')

    mycc = cc.UCCSDTQ(mf, frozen=0, compact_tamps=True)
    t0 = time.perf_counter()
    eris = mycc.ao2mo()
    ao2mo_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, tamps = mycc.init_amps(eris)
    initialization_seconds = time.perf_counter() - t0

    source_paths = [uccsdtq.__file__, _uccsdtq_native.__file__,
                    _uccsdtq_terms.__file__, _uccsdtq_utils.__file__,
                    lib.load_library('libccsdt')._name]
    source_sha256 = {str(Path(x).resolve()): hashlib.sha256(Path(x).read_bytes()).hexdigest()
                     for x in source_paths}
    input_fingerprints = [float(lib.finger(x)) for x in
                          (mf.mo_coeff[0], mf.mo_coeff[1], eris.pppp,
                           eris.pPpP, eris.PPPP)]

    lib.num_threads(nthreads)
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    residual_norms = mycc.update_amps_(tamps, eris)
    first_iteration_seconds = time.perf_counter() - t0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        'molecule': 'NH',
        'geometry_angstrom': 1.036,
        'basis': 'cc-pVDZ',
        'charge': 0,
        'spin': 2,
        'multiplicity': 3,
        'electrons': mol.nelectron,
        'threads': nthreads,
        'runtime_threads': lib.num_threads(),
        'nocc': list(map(int, mycc.nocc)),
        'nmo': list(map(int, mycc.nmo)),
        'nvir': [int(mycc.nmo[i] - mycc.nocc[i]) for i in (0, 1)],
        'e_hf': float(mf.e_tot),
        'scf_seconds': scf_seconds,
        'ao2mo_seconds': ao2mo_seconds,
        'initialization_seconds': initialization_seconds,
        'first_iteration_seconds': first_iteration_seconds,
        'residual_norm': float(np.linalg.norm(residual_norms)),
        't4_unique_elements': int(sum(x.size for x in tamps[3])),
        'memory_estimate_bytes': {k: v for k, v in mycc.memory_estimate().items() if k != 'gb'},
        'rss_ru_maxrss_before_kib': int(rss_before),
        'rss_ru_maxrss_after_kib': int(rss_after),
        'input_fingerprints': input_fingerprints,
        'source_sha256': source_sha256,
        'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        'thread_environment': {k: os.environ.get(k) for k in
                               ('OMP_NUM_THREADS', 'OMP_PROC_BIND', 'OMP_PLACES',
                                'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', nargs='+', type=int, default=[1, 2, 4])
    parser.add_argument('--worker', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        print(json.dumps(worker(args.worker)))
        return
    rows = []
    for nthreads in args.threads:
        env = dict(os.environ, OMP_NUM_THREADS=str(nthreads),
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                               '--worker', str(nthreads)], env=env,
                              check=True, capture_output=True, text=True)
        rows.append(json.loads(proc.stdout))
    base = rows[0]
    for row in rows:
        if row['source_sha256'] != base['source_sha256']:
            raise RuntimeError('source/library hash changed during measurement')
        row['speedup_vs_one_thread'] = base['first_iteration_seconds'] / row['first_iteration_seconds']
    print(json.dumps({'case': 'NH/cc-pVDZ, triplet, one UCCSDTQ iteration',
                      'measurements': rows}, indent=2))


if __name__ == '__main__':
    main()
