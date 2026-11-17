#!/usr/bin/env python
'''
Timing benchmark: JointSCF against independent ("direct") SCF calculations.

Every measurement runs in a fresh Python process with a fixed number of
threads.  The timed section ("wall") starts from the Mole objects and ends
when all configurations are converged; it includes the grids, the integrals
and the SCF iterations.  Importing PySCF and building the Mole objects is
excluded for all methods.  The time to launch worker processes (process pool
of the independent calculations, workers of JointSCF) is recorded separately
as "startup".  Results are appended to a JSON file, so an interrupted run
can be continued by calling the script again.

Usage:
    python run_benchmark.py [--cases a,b] [--methods x,y] [--repeats 3]
                            [--nproc 4] [--out results.json]
'''

import os
import sys
import json
import time
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

def method_table(nproc):
    n = nproc
    return {
        # independent calculations, one after the other
        'direct_1t':        dict(kind='direct', omp=1),
        f'direct_{n}t':     dict(kind='direct', omp=n),
        # independent calculations in n processes with one thread each
        f'direct_{n}p':     dict(kind='pool', omp=1, nproc=n),
        'sync_1w':          dict(kind='joint', schedule='sync', omp=1, workers=1),
        f'sync_{n}w':       dict(kind='joint', schedule='sync', omp=n, workers=n),
        'cascade_1w':       dict(kind='joint', schedule='cascade', omp=1, workers=1),
        f'cascade_{n}w':    dict(kind='joint', schedule='cascade', omp=n, workers=n),
        'transferable_polish_1w':      dict(kind='joint', schedule='transferable', polish=True, omp=1, workers=1),
        f'transferable_polish_{n}w':   dict(kind='joint', schedule='transferable', polish=True, omp=n, workers=n),
        'transferable_nopolish_1w':    dict(kind='joint', schedule='transferable', polish=False, omp=1, workers=1),
        f'transferable_nopolish_{n}w': dict(kind='joint', schedule='transferable', polish=False, omp=n, workers=n),
        # the same with the AO values on the grids kept in memory
        'direct_1t_cache':     dict(kind='direct', omp=1, cache=True),
        f'direct_{n}p_cache':  dict(kind='pool', omp=1, nproc=n, cache=True),
        'cascade_1w_cache':    dict(kind='joint', schedule='cascade', omp=1, workers=1, cache=True),
        f'cascade_{n}w_cache': dict(kind='joint', schedule='cascade', omp=n, workers=n, cache=True),
        # reference energies, converged to 1e-12 (not a timing)
        'reference':        dict(kind='pool', omp=1, nproc=n, conv_tol=1e-12),
    }


######## child process ########

def _make_mf(mol, xc, level, cache, conv_tol):
    from pyscf import dft
    mf = dft.RKS(mol, xc=xc)
    mf.grids.level = level
    if cache:
        mf._numint = dft.numint.CachedNumInt()
    if conv_tol is not None:
        mf.conv_tol = conv_tol
    return mf

def _solve_direct(mol, xc, level, cache, conv_tol):
    mf = _make_mf(mol, xc, level, cache, conv_tol)
    nfock = [0]
    get_veff = mf.get_veff
    def counting_get_veff(*args, **kwargs):
        nfock[0] += 1
        return get_veff(*args, **kwargs)
    mf.get_veff = counting_get_veff
    e_tot = mf.kernel()
    return float(e_tot), bool(mf.converged), nfock[0]

def _pool_init():
    from pyscf import lib
    lib.num_threads(1)

def _pool_warmup(_):
    from pyscf import gto
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    _solve_direct(mol, 'pbe', 0, False, None)
    return os.getpid()

def _pool_task(args):
    from pyscf import gto
    idx, molstr, xc, level, cache, conv_tol = args
    mol = gto.loads(molstr)
    mol.verbose = 0
    return (idx,) + _solve_direct(mol, xc, level, cache, conv_tol)

def child(case, spec):
    import resource
    from pyscf import gto, lib
    from pyscf.scf import jointscf
    sys.path.insert(0, HERE)
    import ensembles

    lib.num_threads(spec['omp'])
    cache = spec.get('cache', False)
    conv_tol = spec.get('conv_tol')
    mols, xc, level = ensembles.build_case(case)
    # load the libraries and compile nothing inside the timed section
    _solve_direct(gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0),
                  'pbe', 0, False, None)
    out = {'nconf': len(mols), 'nao': int(mols[0].nao), 'xc': xc, 'grid_level': level}

    if spec['kind'] == 'direct':
        t0 = time.perf_counter()
        res = [_solve_direct(mol, xc, level, cache, conv_tol) for mol in mols]
        out['wall'] = time.perf_counter() - t0
        out['startup'] = 0.
    elif spec['kind'] == 'pool':
        import multiprocessing
        ctx = multiprocessing.get_context('spawn')
        t0 = time.perf_counter()
        with ctx.Pool(spec['nproc'], initializer=_pool_init) as pool:
            pool.map(_pool_warmup, range(4 * spec['nproc']), chunksize=1)
            tasks = [(i, mol.dumps(), xc, level, cache, conv_tol)
                     for i, mol in enumerate(mols)]
            out['startup'] = time.perf_counter() - t0
            t0 = time.perf_counter()
            res = sorted(pool.imap_unordered(_pool_task, tasks, chunksize=1))
            out['wall'] = time.perf_counter() - t0
        res = [r[1:] for r in res]
    else:
        t0 = time.perf_counter()
        jmf = jointscf.JointSCF([_make_mf(mol, xc, level, cache, conv_tol) for mol in mols])
        jmf.schedule = spec['schedule']
        jmf.max_workers = spec['workers']
        for attr in ('polish', 'molecule_batch', 'correction_rank', 'latent_rank',
                     'optimizer', 'max_shared_fock', 'seed'):
            if attr in spec:
                setattr(jmf, attr, spec[attr])
        jmf.kernel()
        # Launching the worker processes is reported separately, as it is
        # for the process pool of the independent calculations.
        out['startup'] = jmf.timings['startup']
        out['wall'] = time.perf_counter() - t0 - out['startup']
        res = list(zip(jmf.e_tot.tolist(), jmf.converged.tolist(), jmf.nfock.tolist()))
        out['levels'] = jmf.levels.tolist()
        out['promoted_at'] = jmf.promoted_at.tolist()
        out['promotions'] = [getattr(m, 'promotions', []) for m in jmf.models]
        out['batch_history'] = jmf.scf_history
        out['ndescriptor'] = [int(m.z.shape[1]) for m in jmf.models]
        out['nfock_shared'] = jmf.nfock_shared.tolist()
        out['nfock_polish'] = jmf.nfock_polish.tolist()
        out['e_err_est'] = jmf.e_err_est.tolist()

    out['e_tot'] = [r[0] for r in res]
    out['converged'] = [r[1] for r in res]
    out['nfock'] = [r[2] for r in res]
    out['maxrss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
    if spec['kind'] == 'pool':
        out['maxrss_mb'] = None
    json.dump(out, sys.stdout, default=float)


######## parent process ########

def environment_info():
    import platform
    info = {'platform': platform.platform(), 'python': sys.version.split()[0],
            'ncpu': os.cpu_count()}
    try:
        with open('/proc/cpuinfo') as f:
            models = [line.split(':')[1].strip() for line in f if line.startswith('model name')]
        info['cpu'] = models[0] if models else None
        with open('/proc/meminfo') as f:
            info['mem_gb'] = round(int(f.readline().split()[1]) / 1024**2, 1)
    except OSError:
        pass
    import numpy, scipy, pyscf
    info['numpy'] = numpy.__version__
    info['scipy'] = scipy.__version__
    info['pyscf'] = pyscf.__version__
    try:
        info['git'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
    except Exception:
        pass
    return info

def run_one(case, method, spec):
    env = dict(os.environ)
    env.setdefault('PYSCF_EXT_PATH', '')
    env['OMP_NUM_THREADS'] = str(spec['omp'])
    if spec['omp'] == 1:
        # the BLAS library bundled with numpy/scipy
        env['OPENBLAS_NUM_THREADS'] = '1'
    else:
        env.pop('OPENBLAS_NUM_THREADS', None)
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), '--child', case, json.dumps(spec)],
        env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'{case} {method} failed:\n{proc.stderr[-2000:]}')
    return json.loads(proc.stdout)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', default=None)
    parser.add_argument('--methods', default=None)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--nproc', type=int, default=os.cpu_count())
    parser.add_argument('--out', default=os.path.join(HERE, 'results.json'))
    parser.add_argument('--child', nargs=2, default=None)
    args = parser.parse_args()
    if args.child:
        return child(args.child[0], json.loads(args.child[1]))

    sys.path.insert(0, HERE)
    import ensembles
    table = method_table(args.nproc)
    cases = args.cases.split(',') if args.cases else list(ensembles.CASES)
    methods = args.methods.split(',') if args.methods else list(table)
    # methods which are measured only once
    once = {'reference', f'direct_{args.nproc}t', 'sync_1w'}

    results = {'environment': environment_info(), 'runs': {}}
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
    runs = results['runs']

    for case in cases:
        for repeat in range(args.repeats):
            for method in methods:
                if repeat > 0 and method in once:
                    continue
                key = f'{case}/{method}/{repeat}'
                if key in runs:
                    continue
                t0 = time.time()
                runs[key] = run_one(case, method, table[method])
                r = runs[key]
                print('%-20s %-18s rep %d  wall %8.2f s  nfock %6.2f  conv %d/%d  (%.0f s)' % (
                    case, method, repeat, r['wall'], sum(r['nfock']) / r['nconf'],
                    sum(r['converged']), r['nconf'], time.time() - t0), flush=True)
                with open(args.out + '.tmp', 'w') as f:
                    json.dump(results, f)
                os.replace(args.out + '.tmp', args.out)

if __name__ == '__main__':
    main()
