#!/usr/bin/env python
"""Compare one RCCSDTQ and one UCCSDTQ update for H2 at 0.74 Angstrom."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def worker(method, nthreads, basis='cc-pvdz'):
    import numpy as np
    from pyscf import cc, gto, lib, scf

    lib.num_threads(1)
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis=basis, spin=0,
                charge=0, cart=False, verbose=0)
    t0 = time.perf_counter()
    mf = scf.RHF(mol).set(conv_tol=1e-13).run()
    scf_seconds = time.perf_counter() - t0
    assert mf.converged
    if method == 'RCCSDTQ':
        mycc = cc.RCCSDTQ(mf, frozen=0, compact_tamps=True)
    elif method == 'UCCSDTQ':
        mycc = cc.UCCSDTQ(mf.to_uhf(), frozen=0, compact_tamps=True)
    else:
        raise ValueError(method)

    t0 = time.perf_counter()
    eris = mycc.ao2mo()
    ao2mo_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, tamps = mycc.init_amps(eris)
    init_seconds = time.perf_counter() - t0
    if method == 'RCCSDTQ':
        # ccsdtq() normally creates these maps before entering its driver;
        # this benchmark deliberately calls exactly one updater directly.
        mycc.tri2block_map, mycc.tri2block_mask, mycc.tri2block_tp = mycc.setup_tri2block()
        mycc.blksize = min(mycc.blksize, mycc.nocc)
    # The direct updater must use the same block size as ccsdtq().  Omitting
    # this clamp pads H2's single occupied orbital to four on each T4 axis.
    sources = [Path(__file__).resolve(), Path(sys.modules[mycc.__class__.__module__].__file__),
               Path(lib.load_library('libccsdt')._name)]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    lib.num_threads(nthreads)
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    res_norm = mycc.update_amps_(tamps, eris)
    first_iteration_seconds = time.perf_counter() - t0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if hashes != {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}:
        raise RuntimeError('Source or native library changed during the measurement')
    return {
        'method': method,
        'geometry_angstrom': 0.74,
        'basis': basis,
        'threads': nthreads,
        'blksize': getattr(mycc, 'blksize', None),
        'einsum_backend': mycc.einsum_backend,
        'source_sha256': hashes,
        'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        'nocc': ([int(mycc.nocc)] if np.isscalar(mycc.nocc) else list(map(int, mycc.nocc))),
        'nmo': ([int(mycc.nmo)] if np.isscalar(mycc.nmo) else list(map(int, mycc.nmo))),
        'scf_seconds': scf_seconds,
        'ao2mo_seconds': ao2mo_seconds,
        'initialization_seconds': init_seconds,
        'first_iteration_seconds': first_iteration_seconds,
        'residual_norm': float(np.linalg.norm(res_norm)),
        'rss_ru_maxrss_before_kib': int(rss_before),
        'rss_ru_maxrss_after_kib': int(rss_after),
        'e_hf': float(mf.e_tot),
        'memory_estimate_bytes': ({k: v for k, v in mycc.memory_estimate().items() if k != 'gb'}
                                  if hasattr(mycc, 'memory_estimate') else None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', nargs='+', type=int, default=[1, 2, 4])
    parser.add_argument('--basis', default='cc-pvdz')
    parser.add_argument('--worker', nargs=3, metavar=('METHOD', 'THREADS', 'BASIS'))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker[0], int(args.worker[1]), args.worker[2])))
        return
    rows = []
    for method in ('RCCSDTQ', 'UCCSDTQ'):
        for nthreads in args.threads:
            env = dict(os.environ, OMP_NUM_THREADS=str(nthreads),
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                   '--worker', method, str(nthreads), args.basis],
                                  env=env, check=True, capture_output=True, text=True)
            rows.append(json.loads(proc.stdout))
    for method in ('RCCSDTQ', 'UCCSDTQ'):
        base = next(x for x in rows if x['method'] == method and x['threads'] == args.threads[0])
        for row in rows:
            if row['method'] == method:
                row['speedup_vs_first_thread'] = base['first_iteration_seconds'] / row['first_iteration_seconds']
    print(json.dumps({'case': 'H2/%s, first Jacobi update only' % args.basis,
                      'measurements': rows}, indent=2))


if __name__ == '__main__':
    main()
