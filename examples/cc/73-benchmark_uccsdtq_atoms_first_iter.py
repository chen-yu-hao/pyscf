#!/usr/bin/env python
"""Measure exactly one UCCSDTQ update for N, F and O atoms in cc-pVDZ.

SCF, AO-to-MO integral construction and amplitude initialization are recorded
separately.  The reported iteration time is one call to ``update_amps_`` after
the initial amplitudes have been built; no convergence loop is entered.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


ATOMS = {
    # PySCF mol.spin is N_alpha - N_beta, i.e. multiplicity minus one.
    'N': ('N 0 0 0', 3),  # quartet, 5 electrons
    'F': ('F 0 0 0', 1),  # doublet, 9 electrons
    'O': ('O 0 0 0', 2),  # triplet, 8 electrons
}


def _worker(atom, nthreads):
    import numpy as np
    from pyscf import cc, gto, lib, scf
    from pyscf.cc import _uccsdtq_native, _uccsdtq_terms, _uccsdtq_utils, uccsdtq

    lib.num_threads(1)
    geometry, spin = ATOMS[atom]
    mol = gto.M(atom=geometry, basis='cc-pvdz', charge=0, spin=spin,
                cart=False, verbose=0)

    t0 = time.perf_counter()
    mf = scf.UHF(mol).set(conv_tol=1e-12).run()
    scf_seconds = time.perf_counter() - t0
    if not mf.converged:
        raise RuntimeError('%s UHF did not converge' % atom)

    mycc = cc.UCCSDTQ(mf, frozen=0, compact_tamps=True)
    t0 = time.perf_counter()
    eris = mycc.ao2mo()
    ao2mo_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, tamps = mycc.init_amps(eris)
    init_seconds = time.perf_counter() - t0

    # Freeze the input fingerprint before the single update.  The updater is
    # expected to mutate amplitudes in place, so this is intentionally a copy
    # only for provenance and does not enter the timed section.
    input_fingerprints = [float(lib.finger(x)) for x in
                          (mf.mo_coeff[0], mf.mo_coeff[1], eris.pppp,
                           eris.pPpP, eris.PPPP)]
    source_paths = [uccsdtq.__file__, _uccsdtq_native.__file__,
                    _uccsdtq_terms.__file__, _uccsdtq_utils.__file__,
                    lib.load_library('libccsdt')._name]
    source_hashes = {str(Path(x).resolve()): hashlib.sha256(Path(x).read_bytes()).hexdigest()
                     for x in source_paths}

    lib.num_threads(nthreads)
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    res_norm = mycc.update_amps_(tamps, eris)
    first_iter_seconds = time.perf_counter() - t0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    output_fingerprints = [float(lib.finger(x)) for rank in tamps for x in rank]
    memory = mycc.memory_estimate()
    return {
        'atom': atom,
        'basis': 'cc-pVDZ',
        'charge': 0,
        'spin': spin,
        'multiplicity': spin + 1,
        'electrons': mol.nelectron,
        'nocc': list(map(int, mycc.nocc)),
        'nmo': list(map(int, mycc.nmo)),
        'nvir': [int(mycc.nmo[i] - mycc.nocc[i]) for i in (0, 1)],
        'threads': nthreads,
        'runtime_threads': lib.num_threads(),
        'scf_seconds': scf_seconds,
        'ao2mo_seconds': ao2mo_seconds,
        'initialization_seconds': init_seconds,
        'first_iteration_seconds': first_iter_seconds,
        'residual_norm': float(np.linalg.norm(res_norm)),
        't4_unique_elements': int(sum(x.size for x in tamps[3])),
        'memory_estimate_bytes': {k: v for k, v in memory.items() if k != 'gb'},
        'rss_ru_maxrss_before_kib': int(rss_before),
        'rss_ru_maxrss_after_kib': int(rss_after),
        'input_fingerprints': input_fingerprints,
        'amplitude_fingerprints_after_update': output_fingerprints,
        'source_sha256': source_hashes,
        'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        'thread_environment': {k: os.environ.get(k) for k in
                               ('OMP_NUM_THREADS', 'OMP_PROC_BIND', 'OMP_PLACES',
                                'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atoms', nargs='+', choices=sorted(ATOMS), default=sorted(ATOMS))
    parser.add_argument('--threads', nargs='+', type=int, default=[1, 2, 4])
    parser.add_argument('--worker', nargs=2, metavar=('ATOM', 'THREADS'))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(_worker(args.worker[0], int(args.worker[1]))))
        return

    rows = []
    for atom in args.atoms:
        for nthreads in args.threads:
            env = dict(os.environ, OMP_NUM_THREADS=str(nthreads),
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                   '--worker', atom, str(nthreads)],
                                  env=env, check=True, capture_output=True, text=True)
            rows.append(json.loads(proc.stdout))

    groups = {}
    for row in rows:
        groups.setdefault(row['atom'], []).append(row)
    for atom, atom_rows in groups.items():
        reference = atom_rows[0]
        for row in atom_rows:
            if row['source_sha256'] != reference['source_sha256']:
                raise RuntimeError('source/library hash changed for %s' % atom)
            if max(abs(x-y) for x, y in zip(row['input_fingerprints'],
                                             reference['input_fingerprints'])) > 1e-10:
                raise RuntimeError('thread-dependent input for %s' % atom)
            row['speedup_vs_one_thread'] = reference['first_iteration_seconds'] / row['first_iteration_seconds']
    print(json.dumps({'case': 'neutral N/F/O atoms, cc-pVDZ, one UCCSDTQ iteration',
                      'measurements': rows}, indent=2))


if __name__ == '__main__':
    main()
