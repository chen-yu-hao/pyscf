#!/usr/bin/env python

'''Time UCCSDTQ residuals with fresh-process peak RSS and thread checks.

Run from a built PySCF checkout:
    python examples/cc/72-benchmark_uccsdtq.py --threads 1 2 4 --repeat 3

The default BeH/STO-3G case is deliberately small.  Results characterize
this case only, not asymptotic or production-size parallel scaling.
Peak RSS includes SCF, amplitude preparation, and residual evaluation.
'''

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


def worker(nthreads, repeat):
    import numpy as np
    from pyscf import cc, gto, lib, scf
    from pyscf.cc import uccsdtq, _uccsdtq_native, _uccsdtq_terms, _uccsdtq_utils as utils

    sources = [uccsdtq.__file__, _uccsdtq_native.__file__, _uccsdtq_terms.__file__, utils.__file__,
               lib.load_library('libccsdt')._name]

    def hashes():
        return {str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sources}

    def resident_bytes():
        if sys.platform.startswith('linux'):
            fields = {line.split(':')[0]: line.split(':')[1].strip()
                      for line in Path('/proc/self/status').read_text().splitlines() if ':' in line}
            return int(fields['VmRSS'].split()[0]) * 1024, int(fields['VmHWM'].split()[0]) * 1024
        value = int(lib.current_memory()[0] * 1e6)
        return value, value

    # Keep SCF and integral preparation identical between processes. Changing
    # their thread count can rotate exactly degenerate virtual subspaces,
    # which would change the *input* tensor fingerprints being compared.
    lib.num_threads(1)
    mol = gto.M(atom='Be 0 0 0; H 0 0 1.342', basis='sto-3g', spin=1, verbose=0)
    mf = scf.UHF(mol).set(conv_tol=1e-13).run()
    assert mf.converged
    mycc = cc.UCCSDTQ(mf, frozen=0, compact_tamps=True)
    eris = mycc.ao2mo()
    _, tamps = mycc.init_amps(eris)
    rng = np.random.default_rng(3049)
    # Fixed nonzero high-rank amplitudes exercise all native term families.
    t3 = [rng.normal(scale=1e-4, size=utils.block_shape(mycc, 3, b, True)) for b in range(4)]
    tamps[2] = tuple(t3)
    tamps[3] = [rng.normal(scale=1e-5, size=x.shape) for x in tamps[3]]
    input_fingerprints = [float(lib.finger(x)) for x in (mf.mo_coeff[0], mf.mo_coeff[1], eris.pppp,
                          eris.pPpP, eris.PPPP)]
    lib.num_threads(nthreads)
    baseline_rss, _ = resident_bytes()
    times = []
    result = None
    # One untimed call initializes the term plan and native bookkeeping.
    uccsdtq.residuals(mycc, tamps, eris)
    warm_rss, _ = resident_bytes()
    source_hashes = hashes()
    for _ in range(repeat):
        start = time.perf_counter()
        result = uccsdtq.residuals(mycc, tamps, eris)
        times.append(time.perf_counter() - start)
    if source_hashes != hashes():
        raise RuntimeError('Source files or native library changed during the measurement')
    final_rss, high_water = resident_bytes()
    resource_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    return {
        'threads': nthreads,
        'seconds': times,
        'median_seconds': statistics.median(times),
        'peak_rss_bytes': max(resource_peak, high_water, baseline_rss, warm_rss, final_rss),
        'getrusage_peak_rss_bytes': resource_peak,
        'proc_status_high_water_bytes': high_water,
        'rss_after_timing_bytes': final_rss,
        'rss_before_residual_bytes': baseline_rss,
        'rss_after_warmup_bytes': warm_rss,
        'runtime_threads': lib.num_threads(),
        'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        'thread_environment': {key: os.environ.get(key) for key in
                               ('OMP_NUM_THREADS', 'OMP_PROC_BIND', 'OMP_PLACES',
                                'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        'sha256': source_hashes,
        'residual_norms': [float(np.linalg.norm(block)) for rank in result for block in rank],
        'residual_fingerprints': [float(lib.finger(block)) for rank in result for block in rank],
        'input_fingerprints': input_fingerprints,
        'memory_estimate_bytes': {k: v for k, v in mycc.memory_estimate().items() if k != 'gb'},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', nargs='+', type=int, default=[1, 2, 4])
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--worker', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeat < 1 or any(n < 1 for n in args.threads):
        parser.error('threads and repeat must be positive')
    if args.worker is not None:
        print(json.dumps(worker(args.worker, args.repeat)))
        return
    rows = []
    for nthreads in args.threads:
        env = dict(os.environ, OMP_NUM_THREADS=str(nthreads), OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(nthreads),
                               '--repeat', str(args.repeat)], env=env, check=True, capture_output=True, text=True)
        rows.append(json.loads(proc.stdout))
    # Each measurement ran in a fresh process; RSS values are comparable.
    ref = rows[0]
    for row in rows:
        if row['sha256'] != ref['sha256']:
            raise RuntimeError('Source files or native library changed between worker processes')
        row['speedup_vs_first'] = ref['median_seconds'] / row['median_seconds']
        row['max_fingerprint_difference_vs_first'] = max(
            abs(x - y) for x, y in zip(row['residual_fingerprints'], ref['residual_fingerprints']))
        row['max_input_fingerprint_difference_vs_first'] = max(
            abs(x - y) for x, y in zip(row['input_fingerprints'], ref['input_fingerprints']))
    print(json.dumps({'case': 'BeH/STO-3G, five electrons, fixed amplitudes, compact T4',
                      'python': sys.executable, 'measurements': rows}, indent=2))
    if any(row['max_fingerprint_difference_vs_first'] > 1e-10 or
           row['max_input_fingerprint_difference_vs_first'] > 1e-10 for row in rows):
        raise RuntimeError('Thread-count input or residual check failed')


if __name__ == '__main__':
    main()
