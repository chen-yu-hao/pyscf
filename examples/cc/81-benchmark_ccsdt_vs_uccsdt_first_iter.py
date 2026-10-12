#!/usr/bin/env python
"""Compare compact RCCSDT and UCCSDT on the same first Jacobi update.

SCF, AO-to-MO transformation, initial amplitudes, and triangular maps are
prepared before timing.  Each timed operation starts from the same MP2
amplitudes and calls exactly one ``update_amps_``.  The script runs each
method in a fresh process so that thread settings and peak RSS are isolated.

Examples::

    ../.venv/bin/python examples/cc/81-benchmark_ccsdt_vs_uccsdt_first_iter.py \
        --threads 1 2 4 --repeat 3
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
CASES = {
    'h2-ccpvtz': {
        'atom': 'H 0 0 0; H 0 0 0.74', 'basis': 'cc-pvtz',
        'spin': 0, 'frozen': 0,
    },
    'nh3-ccpvdz': {
        # C3v equilibrium geometry (Angstrom), neutral closed shell.
        'atom': ('N 0 0 0; H 0 0 1.012; H 0.875 0 -0.337; '
                 'H -0.875 0 -0.337'),
        'basis': 'cc-pvdz', 'spin': 0, 'frozen': 0,
    },
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rss_bytes():
    fields = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            fields[key] = value.strip()
    current = int(fields['VmRSS'].split()[0]) * 1024
    high = int(fields['VmHWM'].split()[0]) * 1024
    # ru_maxrss is kB on Linux.  Keep the explicit unit conversion here.
    high = max(high, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    return current, high


def source_hashes():
    from pyscf import cc, lib
    paths = [
        Path(__file__).resolve(),
        Path(cc.rccsdt.__file__).resolve(),
        Path(cc.rccsdt_highm.__file__).resolve(),
        Path(cc.uccsdt.__file__).resolve(),
        Path(cc.uccsdt_highm.__file__).resolve(),
        Path(lib.load_library('libccsdt')._name).resolve(),
    ]
    return {str(path): sha(path) for path in paths}


def setup_maps(solver):
    """Initialize the compact T3 maps normally made by ``kernel``."""
    if not solver.do_tri_max_t:
        return
    if hasattr(solver, 'blksize'):
        solver.tri2block_map, solver.tri2block_mask, solver.tri2block_tp = solver.setup_tri2block()
        solver.blksize = min(solver.blksize, solver.nocc)
        solver.blksize_oovv = min(solver.blksize_oovv, solver.nocc)
        solver.blksize_oooo = min(solver.blksize_oooo, solver.nocc)
    else:
        attrs = (
            't2c_map_6f_oa', 't2c_mask_6f_oa', 't2c_map_2f_oa', 't2c_mask_2f_oa',
            't2c_map_6f_va', 't2c_mask_6f_va', 't2c_map_2f_va', 't2c_mask_2f_va',
            't2c_map_6f_ob', 't2c_mask_6f_ob', 't2c_map_2f_ob', 't2c_mask_2f_ob',
            't2c_map_6f_vb', 't2c_mask_6f_vb', 't2c_map_2f_vb', 't2c_mask_2f_vb',
        )
        for attr, value in zip(attrs, solver.setup_tri2block()):
            setattr(solver, attr, value)
        nocca, noccb = solver.nocc
        nmoa, nmob = solver.nmo
        nvira, nvirb = nmoa - nocca, nmob - noccb
        solver.blksize_o_aaa = min(solver.blksize_o_aaa, max(nocca, noccb))
        solver.blksize_v_aaa = min(solver.blksize_v_aaa, max(nvira, nvirb))
        solver.blksize_o_aab = min(solver.blksize_o_aab, max(nocca, noccb))
        solver.blksize_v_aab = min(solver.blksize_v_aab, max(nvira, nvirb))


def nested_bytes(value):
    if isinstance(value, dict):
        return sum(nested_bytes(x) for x in value.values())
    if isinstance(value, (tuple, list)):
        return sum(nested_bytes(x) for x in value)
    return int(getattr(value, 'nbytes', 0))


def worker(args):
    import numpy as np
    from pyscf import cc, gto, lib, scf

    lib.num_threads(args.nthreads)
    case = CASES[args.case]
    mol = gto.M(atom=case['atom'], basis=case['basis'], spin=case['spin'], charge=0,
                cart=False, verbose=0)
    scf_start = time.perf_counter()
    mf_r = scf.RHF(mol).run(conv_tol=1e-12, conv_tol_grad=1e-10)
    if not mf_r.converged:
        raise RuntimeError('RHF did not converge')
    scf_seconds = time.perf_counter() - scf_start
    mf = mf_r if args.method == 'RCCSDT' else mf_r.to_uhf()
    solver = (cc.RCCSDT(mf, frozen=case['frozen'], compact_tamps=True)
              if args.method == 'RCCSDT' else
              cc.UCCSDT(mf, frozen=case['frozen'], compact_tamps=True))
    solver.max_memory = 4000
    solver.einsum_backend = os.environ.get('CC_EINSUM_BACKEND', solver.einsum_backend)
    solver.verbose = 0
    ao2mo_start = time.perf_counter()
    eris = solver.ao2mo()
    ao2mo_seconds = time.perf_counter() - ao2mo_start
    setup_maps(solver)
    init_start = time.perf_counter()
    e_mp2, tamps = solver.init_amps(eris)
    init_seconds = time.perf_counter() - init_start
    tamps0 = copy.deepcopy(tamps)
    rss_before, peak_before = rss_bytes()
    times = []
    norms = []
    energies = []
    for _ in range(args.repeat + 1):
        trial = copy.deepcopy(tamps0)
        start = time.perf_counter()
        norm = solver.update_amps_(trial, eris)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        norms.append(float(np.linalg.norm(norm)))
        energies.append(float(solver.energy(trial, eris)))
    rss_after, peak_after = rss_bytes()
    paths = source_hashes()
    result = {
        'case': args.case, 'atom': case['atom'], 'basis': case['basis'],
        'method': args.method, 'threads': args.nthreads,
        'native_threads': lib.num_threads(), 'spin': case['spin'], 'frozen': case['frozen'],
        'nao': mol.nao_nr(), 'nocc': solver.nocc, 'nmo': solver.nmo,
        'nvir': (solver.nmo - solver.nocc if isinstance(solver.nocc, int)
                 else tuple(nmo - nocc for nmo, nocc in zip(solver.nmo, solver.nocc))),
        'scf_seconds': scf_seconds, 'ao2mo_seconds': ao2mo_seconds,
        'init_seconds': init_seconds, 'first_update_seconds': times[0],
        'repeat_update_seconds': times[1:],
        'median_update_seconds': statistics.median(times[1:]) if len(times) > 1 else times[0],
        'update_norms': norms, 'update_energies': energies,
        'e_hf': float(mf_r.e_tot), 'e_mp2': float(e_mp2),
        'amplitude_bytes': nested_bytes(tamps0), 'eri_bytes': nested_bytes(eris.__dict__),
        'rss_before_bytes': rss_before, 'rss_after_bytes': rss_after,
        'peak_rss_bytes': max(peak_before, peak_after),
        'source_sha256': paths,
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'thread_environment': {k: os.environ.get(k) for k in
                               ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / ('%s-%s-t%d.json' % (args.case, args.method, args.nthreads))
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--methods', nargs='+', choices=('RCCSDT', 'UCCSDT'),
                        default=('RCCSDT', 'UCCSDT'))
    parser.add_argument('--threads', nargs='+', type=int, default=(1, 2, 4))
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'audits/20261012-ccsdt-vs-uccsdt')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--method', choices=('RCCSDT', 'UCCSDT'))
    parser.add_argument('--nthreads', type=int)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
        return
    rows = []
    for case in args.cases:
        for method in args.methods:
            for nthreads in args.threads:
                command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                           '--case', case, '--method', method, '--nthreads', str(nthreads),
                           '--repeat', str(args.repeat), '--output', str(args.output)]
                env = dict(os.environ, OMP_NUM_THREADS=str(nthreads),
                           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                log_path = args.output / ('%s-%s-t%d.log' % (case, method, nthreads))
                args.output.mkdir(parents=True, exist_ok=True)
                with log_path.open('w') as stream:
                    try:
                        subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       timeout=args.timeout, check=True)
                        row = json.loads((args.output /
                                          ('%s-%s-t%d.json' % (case, method, nthreads))).read_text())
                        row['status'] = 'complete'
                    except subprocess.TimeoutExpired:
                        row = {'case': case, 'method': method, 'threads': nthreads,
                               'status': 'timeout', 'timeout_seconds': args.timeout}
                    except subprocess.CalledProcessError as err:
                        row = {'case': case, 'method': method, 'threads': nthreads,
                               'status': 'failed', 'returncode': err.returncode}
                rows.append(row)
                (args.output / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
                print(case, method, nthreads, row['status'], flush=True)


if __name__ == '__main__':
    main()
