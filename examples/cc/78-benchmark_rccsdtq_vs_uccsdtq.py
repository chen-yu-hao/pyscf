#!/usr/bin/env python
"""Compare native RCCSDTQ with UCCSDTQ on identical closed-shell references.

Fresh processes, common converged RCCSDT seeds, compact T4 and T1--T3 DIIS.
The complete CCSDTQ solve includes cold planning. Repeated Jacobi updates
start from identical converged amplitudes; copying is outside the timer.
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

CASES = {
    'be-sto3g': ('Be 0 0 0', 'sto-3g'),
    'h6-sto3g': ('; '.join('H 0 0 %.8f' % (.9*i) for i in range(6)), 'sto-3g'),
    'be-ccpvdz': ('Be 0 0 0', 'cc-pvdz'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(args):
    import numpy as np
    from pyscf import cc, gto, lib, scf
    from pyscf.cc import uccsdt
    lib.num_threads(1)
    atom, basis = CASES[args.case]
    mol = gto.M(atom=atom, basis=basis, spin=0, charge=0, cart=False, verbose=0)
    start = time.perf_counter()
    mf = scf.RHF(mol).run(conv_tol=1e-13, conv_tol_grad=1e-10)
    assert mf.converged
    scf_seconds = time.perf_counter()-start
    start = time.perf_counter()
    seed = cc.RCCSDT(mf, frozen=0, compact_tamps=False).run(
        conv_tol=1e-11, conv_tol_normt=1e-9, max_cycle=100)
    assert seed.converged
    seed_seconds = time.perf_counter()-start
    if args.method == 'RCCSDTQ':
        solver = cc.RCCSDTQ(mf, frozen=0, compact_tamps=True)
        tamps = copy.deepcopy(seed.tamps)
    else:
        solver = cc.UCCSDTQ(mf.to_uhf(), frozen=0, compact_tamps=True)
        solver.quadruples_backend = 'auto' if args.method == 'U-auto' else 'blocked'
        tamps = uccsdt.tamps_rhf2uhf(solver, seed.tamps)
    solver.max_memory = 4000
    solver.conv_tol = 1e-10
    solver.conv_tol_normt = 1e-8
    solver.max_cycle = 100
    solver.do_diis_max_t = False
    solver.diis_space = 6
    start = time.perf_counter()
    eris = solver.ao2mo()
    ao2mo_seconds = time.perf_counter()-start
    paths = [Path(__file__).resolve(), Path(lib.load_library('libccsdt')._name)]
    root = Path(cc.__file__).resolve().parent
    paths += [root / (name+'.py') for name in ('rccsdt','rccsdt_highm','rccsdtq','rccsdtq_highm',
        'uccsdt','uccsdt_highm','uccsdtq','_uccsdtq_native','_uccsdtq_blocked','_uccsdtq_terms','_uccsdtq_utils')]
    hashes = {str(p): sha(p) for p in paths}
    lib.num_threads(args.nthreads)
    native_threads = lib.load_library('libccsdt').UCCSDTQomp_threads()
    assert native_threads == args.nthreads
    iteration_seconds = []
    original = solver.update_amps_
    def timed_update(t, e):
        start = time.perf_counter()
        result = original(t, e)
        iteration_seconds.append(time.perf_counter()-start)
        return result
    solver.update_amps_ = timed_update
    start = time.perf_counter()
    solver.kernel(tamps=tamps, eris=eris)
    solve_seconds = time.perf_counter()-start
    solver.update_amps_ = original
    if not solver.converged:
        raise RuntimeError('CCSDTQ did not converge in 100 iterations')
    converged_energy = float(solver.e_tot)
    warm = []
    update_energies = []
    for _ in range(args.repeat):
        trial = copy.deepcopy(solver.tamps)
        start = time.perf_counter()
        original(trial, eris)
        warm.append(time.perf_counter()-start)
        update_energies.append(float(solver.energy(trial, eris)+solver.e_hf))
    if hashes != {str(p): sha(p) for p in paths}:
        raise RuntimeError('Sources/library changed during measurement')
    def blocks(amplitudes):
        return [a for rank in amplitudes for a in (rank if isinstance(rank, (list, tuple)) else (rank,))]
    seed_arrays = [np.asarray(mf.mo_coeff), np.asarray(mf.mo_occ)] + [np.asarray(x) for x in seed.tamps]
    fingerprint = hashlib.sha256()
    for array in seed_arrays:
        fingerprint.update(str(array.shape).encode())
        fingerprint.update(np.ascontiguousarray(array).tobytes())
    status = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:')):
            status[line.split(':')[0]] = int(line.split()[1])*1024
    result = dict(case=args.case, atom=atom, basis=basis, frozen=0, spin=0,
        method=args.method, backend=getattr(solver, '_quadruples_backend_used', 'restricted'),
        threads=args.nthreads, native_threads=native_threads, nocc=solver.nocc, nmo=solver.nmo,
        scf_seconds=scf_seconds, seed_seconds=seed_seconds, ao2mo_seconds=ao2mo_seconds,
        solve_seconds=solve_seconds, iteration_seconds=iteration_seconds,
        warm_update_seconds=warm, warm_median_seconds=statistics.median(warm),
        warm_update_energies=update_energies, e_tot=converged_energy, e_seed=float(seed.e_tot),
        converged=bool(solver.converged), cycles=solver.cycles,
        amplitude_bytes=sum(x.nbytes for x in blocks(solver.tamps)),
        t4_bytes=sum(x.nbytes for x in blocks([solver.tamps[3]])),
        peak_rss_bytes=max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, status['VmHWM']),
        input_sha256=fingerprint.hexdigest(), source_sha256=hashes,
        conv_tol=solver.conv_tol, conv_tol_normt=solver.conv_tol_normt,
        do_diis_max_t=False, diis_space=6,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        thread_environment={k: os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    (args.output/('%s-%s-t%d.json' % (args.case, args.method, args.nthreads))).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--methods', nargs='+', choices=['RCCSDTQ','U-auto','U-blocked'], default=['RCCSDTQ','U-auto','U-blocked'])
    parser.add_argument('--threads', nargs='+', type=int, default=[1,4])
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--output', type=Path, default=Path('audits/20261012-rccsdtq-comparison'))
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--method', choices=['RCCSDTQ','U-auto','U-blocked'])
    parser.add_argument('--nthreads', type=int)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    rows = []
    for case in args.cases:
        for threads in args.threads:
            for method in args.methods:
                command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--case', case,
                    '--method', method, '--nthreads', str(threads), '--repeat', str(args.repeat), '--output', str(args.output)]
                env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                log = args.output/('%s-%s-t%d.log' % (case,method,threads))
                with log.open('w') as stream:
                    try:
                        subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       timeout=args.timeout, check=True)
                        row=json.loads(log.with_suffix('.json').read_text())
                    except subprocess.TimeoutExpired:
                        row=dict(case=case,method=method,threads=threads,status='timeout',limit=args.timeout)
                    except subprocess.CalledProcessError as err:
                        row=dict(case=case,method=method,threads=threads,status='failed',returncode=err.returncode)
                rows.append(row)
                (args.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
                print(case,method,threads,row.get('status','complete'),flush=True)
    if any(r.get('status') == 'failed' for r in rows):
        raise RuntimeError('One or more benchmark workers failed; see logs')

if __name__ == '__main__':
    main()
