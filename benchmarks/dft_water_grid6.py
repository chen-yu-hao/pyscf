"""Paired cold-SCF water benchmarks; profiling is a separate, untimed sample."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from unittest import mock


def water_cluster(n):
    # Deterministic performance geometries, NOT optimized hydrogen-bonded clusters.
    atoms = []
    for i in range(n):
        x, y = 4.0 * (i % 3), 4.0 * (i // 3)
        atoms += [('O', (x, y, 0)), ('H', (x+.757, y, .587)),
                  ('H', (x-.757, y, .587))]
    return atoms


def worker(cfg):
    root = Path(cfg['root']).resolve()
    sys.path.insert(0, str(root))
    import numpy as np
    import pyscf
    from pyscf import dft, gto, lib
    from pyscf.dft import numa, numint
    from threadpoolctl import threadpool_info, threadpool_limits
    if not Path(pyscf.__file__).resolve().is_relative_to(root):
        raise RuntimeError('Imported the wrong PySCF checkout')
    lib.num_threads(cfg['threads'])
    mol = gto.M(atom=water_cluster(cfg['n']), basis='def2-TZVPP',
                unit='Angstrom', verbose=0, max_memory=2300)
    mf = dft.RKS(mol)
    mf.xc, mf.grids.level = cfg['xc'], 6
    mf.conv_tol, mf.max_cycle, mf.max_memory = 1e-8, 50, 2300
    stages, counts = defaultdict(float), defaultdict(int)

    def instrument(stack, obj, attr, label):
        original = getattr(obj, attr)
        def timed(*args, **kwargs):
            t = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                stages[label] += time.perf_counter()-t
                counts[label] += 1
        replacement = timed
        if isinstance(obj, type) and isinstance(inspect.getattr_static(obj, attr), staticmethod):
            replacement = staticmethod(timed)
        stack.enter_context(mock.patch.object(obj, attr, replacement))

    with threadpool_limits(limits=1, user_api='blas'):
        with ExitStack() as stack:
            if cfg['profile']:
                # Never replace ni.nr_rks or add callables to ni.__dict__:
                # either would disable the production NUMA eligibility check.
                instrument(stack, numa, 'nr_rks', 'xc_total_including_numa')
                for method in ('get_veff', 'get_jk', 'eig'):
                    instrument(stack, mf, method, method)
                instrument(stack, mf.grids, 'build', 'grid_build')
                for method in ('eval_ao', 'eval_rho1', 'eval_rho2', 'eval_xc_eff'):
                    instrument(stack, numint.NumInt, method, method)
                instrument(stack, numint, '_xc_vmat', 'xc_matrix_assembly')
            start = time.perf_counter()
            energy = float(mf.kernel())
            wall = time.perf_counter()-start
        dm = np.asarray(mf.make_rdm1())
        fixed_dm = dm
        if cfg.get('reference'):
            with np.load(cfg['reference']) as ref:
                fixed_dm = ref['dm'].copy()
        # Verification is outside the SCF timer. All repetitions use the first
        # baseline density, NOT separately converged densities for the Vxc check.
        ne, exc, vxc = mf._numint.nr_rks(mol, mf.grids, mf.xc, fixed_dm,
                                        hermi=1, max_memory=2300)
        pools = threadpool_info()
    if not all(np.isfinite(x).all() for x in (energy, dm, ne, exc, vxc)):
        raise RuntimeError('Nonfinite SCF/XC result')
    digest = hashlib.sha256()
    for arr in (mf.grids.coords, mf.grids.weights):
        digest.update(np.ascontiguousarray(arr).data)
    arrays = str(Path(cfg['result']).with_suffix('.npz'))
    np.savez(arrays, dm=dm, vxc=vxc, ne=ne, exc=exc)
    sha = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                  text=True).strip()
    return dict(variant=cfg['variant'], source_sha=sha, source_file=pyscf.__file__,
                n_water=cfg['n'], xc=cfg['xc'], nao=mol.nao, threads=cfg['threads'],
                grid_level=6, grid_points=len(mf.grids.weights), grid_sha256=digest.hexdigest(),
                converged=bool(mf.converged), cycles=int(getattr(mf, 'cycles', -1)),
                energy_hartree=energy, scf_seconds=wall, profiled=cfg['profile'],
                stages_seconds=dict(stages), stage_calls=dict(counts), arrays=arrays,
                numa=os.getenv('PYSCF_NUMA'), tile_screen=os.getenv('PYSCF_DFT_VMAT_TILE_SCREEN'),
                used_incore_eri=mf._eri is not None, threadpools=pools,
                python=platform.python_version(), numpy=np.__version__,
                affinity=sorted(os.sched_getaffinity(0)))


def launch(cfg, timeout):
    config = Path(cfg['result']).with_suffix('.config.json')
    config.write_text(json.dumps(cfg))
    env = dict(os.environ, PYTHONPATH=cfg['root'], OMP_NUM_THREADS=str(cfg['threads']),
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1',
               OMP_DYNAMIC='FALSE', OMP_MAX_ACTIVE_LEVELS='1', PYSCF_NUMA=cfg['numa'],
               PYSCF_DFT_VMAT_TILE_SCREEN='1' if cfg['variant']=='candidate' else '0')
    started = time.perf_counter()
    with Path(cfg['result']).with_suffix('.log').open('w') as log:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(config)],
                       cwd=cfg['root'], env=env, stdout=log, stderr=subprocess.STDOUT,
                       timeout=timeout, check=True)
    value = json.loads(Path(cfg['result']).read_text())
    value['process_seconds'] = time.perf_counter()-started
    return value


def verify(reference, candidate):
    import numpy as np
    if not reference['converged'] or not candidate['converged']:
        raise AssertionError('SCF did not converge')
    if reference['grid_sha256'] != candidate['grid_sha256']:
        raise AssertionError('Different grids: fixed-density comparison is invalid')
    delta = abs(reference['energy_hartree']-candidate['energy_hartree'])
    if delta > 1e-8:
        raise AssertionError(f'Energy difference {delta:.3e} exceeds 1e-8 Eh')
    with np.load(reference['arrays']) as a, np.load(candidate['arrays']) as b:
        errors = {key: float(np.max(np.abs(a[key]-b[key]))) for key in ('ne', 'exc', 'vxc')}
        for key in errors:
            np.testing.assert_allclose(a[key], b[key], atol=1e-9, rtol=1e-10)
    return dict(energy_abs_hartree=delta, fixed_density_max_abs=errors)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', help=argparse.SUPPRESS)
    p.add_argument('--baseline-root')
    p.add_argument('--candidate-root', default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--sizes', type=int, nargs='+', default=[1, 2, 4])
    p.add_argument('--xc', nargs='+', default=['PBE', 'MN15'])
    p.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4])
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--numa', choices=['off', 'on', 'auto'], default='off')
    p.add_argument('--timeout', type=float, default=1800)
    p.add_argument('--output', default='benchmark-results')
    a = p.parse_args()
    if a.worker:
        cfg = json.loads(Path(a.worker).read_text())
        Path(cfg['result']).write_text(json.dumps(worker(cfg), indent=2))
        return
    if not a.baseline_root or a.repeat < 1 or min(a.sizes+a.threads) < 1:
        p.error('Provide --baseline-root and positive sizes/threads/repeat')
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    roots = dict(baseline=str(Path(a.baseline_root).resolve()),
                 candidate=str(Path(a.candidate_root).resolve()))
    report = dict(schema=2, baseline_root=roots['baseline'], candidate_root=roots['candidate'],
                  platform=platform.platform(), cases=[], errors=[])
    for xc in a.xc:
        for n in a.sizes:
            for threads in a.threads:
                key = f'{xc}-n{n}-t{threads}'
                trials, reference = [], None
                try:
                    for rep in range(a.repeat+int(a.profile)):
                        profiling = rep == a.repeat
                        order = ['baseline', 'candidate'] if rep % 2 == 0 else ['candidate', 'baseline']
                        for variant in order:
                            cfg = dict(root=roots[variant], variant=variant, xc=xc, n=n,
                                       threads=threads, numa=a.numa, profile=profiling,
                                       reference=reference['arrays'] if reference else None,
                                       result=str(out/f'{key}-{rep}-{variant}.json'))
                            result = launch(cfg, a.timeout)
                            if reference is None:
                                reference = result
                            result['verification'] = verify(reference, result)
                            trials.append(result)
                            print(json.dumps({k: result[k] for k in
                                              ('variant', 'n_water', 'xc', 'threads', 'scf_seconds', 'profiled')}), flush=True)
                    medians = {v: statistics.median(r['scf_seconds'] for r in trials
                               if r['variant']==v and not r['profiled']) for v in roots}
                    report['cases'].append(dict(case=key, trials=trials, median_s=medians,
                                                speedup=medians['baseline']/medians['candidate']))
                except Exception as error:
                    report['errors'].append(dict(case=key, error=repr(error), partial_trials=trials))
                (out/'summary.json').write_text(json.dumps(report, indent=2))
    if report['errors']:
        raise SystemExit('Benchmark failures; inspect summary.json and per-case logs')


if __name__ == '__main__':
    main()
