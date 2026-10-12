#!/usr/bin/env python
"""Compare one RCCSD and UCCSD update for NH3/cc-pVDZ."""

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


GEOMETRY = ('N 0 0 0; H 0 0.9377 0.3816; '
            'H 0.8121 -0.4688 0.3816; H -0.8121 -0.4688 0.3816')


def _size(value):
    if isinstance(value, (tuple, list)):
        return sum(_size(x) for x in value)
    return int(value.nbytes)


def worker(method, nthreads):
    import numpy as np
    from pyscf import cc, gto, lib, scf

    lib.num_threads(1)
    mol = gto.M(atom=GEOMETRY, basis='cc-pvdz', charge=0, spin=0,
                cart=False, verbose=0)
    t0 = time.perf_counter()
    mf = scf.RHF(mol).set(conv_tol=1e-12).run()
    scf_seconds = time.perf_counter() - t0
    if method == 'RCCSD':
        mycc = cc.RCCSD(mf, frozen=0)
    elif method == 'UCCSD':
        # Pass RHF directly so the optimized closed-shell UCCSD path can be
        # selected; the resulting object still exposes unrestricted blocks.
        mycc = cc.UCCSD(mf, frozen=0)
    else:
        raise ValueError(method)
    t0 = time.perf_counter()
    eris = mycc.ao2mo()
    ao2mo_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, t1, t2 = mycc.init_amps(eris)
    initialization_seconds = time.perf_counter() - t0
    lib.num_threads(nthreads)
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    t1_new, t2_new = mycc.update_amps(t1, t2, eris)
    first_iteration_seconds = time.perf_counter() - t0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        'method': method,
        'molecule': 'NH3',
        'basis': 'cc-pVDZ',
        'charge': 0,
        'spin': 0,
        'threads': nthreads,
        'runtime_threads': lib.num_threads(),
        'nocc': ([int(mycc.nocc)] if np.isscalar(mycc.nocc)
                 else list(map(int, mycc.nocc))),
        'nmo': ([int(mycc.nmo)] if np.isscalar(mycc.nmo)
                else list(map(int, mycc.nmo))),
        'scf_seconds': scf_seconds,
        'ao2mo_seconds': ao2mo_seconds,
        'initialization_seconds': initialization_seconds,
        'first_iteration_seconds': first_iteration_seconds,
        'residual_norm': float(np.linalg.norm(np.asarray(t1_new).ravel())) +
                         float(np.linalg.norm(np.asarray(t2_new).ravel())),
        'input_amplitude_bytes': _size(t1) + _size(t2),
        'output_amplitude_bytes': _size(t1_new) + _size(t2_new),
        'rss_ru_maxrss_before_kib': int(rss_before),
        'rss_ru_maxrss_after_kib': int(rss_after),
        'e_hf': float(mf.e_tot),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', nargs='+', type=int, default=[1, 2, 4])
    parser.add_argument('--worker', nargs=2, metavar=('METHOD', 'THREADS'))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker[0], int(args.worker[1]))))
        return
    rows = []
    for method in ('RCCSD', 'UCCSD'):
        for nthreads in args.threads:
            env = dict(os.environ, OMP_NUM_THREADS=str(nthreads),
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                   '--worker', method, str(nthreads)], env=env,
                                  check=True, capture_output=True, text=True)
            rows.append(json.loads(proc.stdout))
    for method in ('RCCSD', 'UCCSD'):
        base = next(x for x in rows if x['method'] == method and x['threads'] == args.threads[0])
        for row in rows:
            if row['method'] == method:
                row['speedup_vs_one_thread'] = base['first_iteration_seconds'] / row['first_iteration_seconds']
    print(json.dumps({'case': 'NH3/cc-pVDZ, one CCSD update', 'measurements': rows}, indent=2))


if __name__ == '__main__':
    main()
