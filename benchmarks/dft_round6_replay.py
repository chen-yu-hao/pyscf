"""Saved-density XC replay: no new SCF and no whole-SCF speedup claim."""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from unittest import mock

import dft_water_grid6 as harness

BASE_SHA = '819a7013451536b60e3c968f71987288f7910047'
MODES = ('raw', 'coalesce', 'scatter')


def worker(cfg):
    import numpy as np
    root = Path(cfg['root']).resolve()
    sys.path.insert(0, str(root))
    import pyscf
    from pyscf import dft, gto, lib
    from pyscf.dft import numint
    from threadpoolctl import threadpool_info, threadpool_limits
    assert Path(pyscf.__file__).resolve().is_relative_to(root)
    sha = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    assert sha == cfg['expected_sha']
    record = json.loads(Path(cfg['reference_json']).read_text())
    with np.load(cfg['reference_npz'], allow_pickle=False) as ref:
        fixed_dm = ref['fixed_dm'].copy()
        expected = {k: ref[k].copy() for k in ('ne', 'exc', 'vxc')}
    assert record['source_sha'] == 'b5b4f1c711b63c1dbb4fb409e979f53852601eca'
    assert record['xc'] == 'MN15' and record['n_water'] == cfg['n']
    lib.num_threads(2)
    mol = gto.M(atom=harness.water_cluster(cfg['n']), basis='def2-TZVPP',
                unit='Angstrom', verbose=0, max_memory=2300)
    mf = dft.RKS(mol)
    mf.xc, mf.grids.level, mf.max_memory = 'MN15', 6, 2300
    stages, counts, dispatch = defaultdict(float), defaultdict(int), Counter()

    def instrument(stack, obj, name, label):
        original = getattr(obj, name)
        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                stages[label] += time.perf_counter()-start
                counts[label] += 1
        replacement = timed
        if isinstance(obj, type) and isinstance(inspect.getattr_static(obj, name), staticmethod):
            replacement = staticmethod(timed)
        stack.enter_context(mock.patch.object(obj, name, replacement))

    original_c = numint.libdft.VXCxc_vmat
    def counted(*args):
        ret = original_c(*args)
        dispatch[f"{'sparse' if bool(args[8]) else 'dense'}_ncomp{args[7].value}_returned{ret}"] += 1
        return ret

    with threadpool_limits(limits=1, user_api='blas'):
        # Reconstruct the initial-density pruning, NOT a new self-consistent run.
        overlap = mf.get_ovlp(mol)
        dm0 = mf.get_init_guess(mol, mf.init_guess, s1e=overlap)
        mf.initialize_grids(mol, dm0)
        digest = hashlib.sha256()
        for a in (mf.grids.coords, mf.grids.weights):
            digest.update(np.ascontiguousarray(a).data)
        assert digest.hexdigest() == record['grid_sha256'], 'Saved/reference grid differs'
        assert len(mf.grids.weights) == record['grid_points']
        with ExitStack() as stack:
            if cfg['profile']:
                for name in ('eval_ao', 'eval_rho1', 'eval_rho2', 'eval_xc_eff'):
                    instrument(stack, numint.NumInt, name, name)
                instrument(stack, numint, '_xc_vmat', 'xc_matrix_assembly')
                stack.enter_context(mock.patch.object(numint.libdft, 'VXCxc_vmat', counted))
            start = time.perf_counter()
            ne, exc, vxc = mf._numint.nr_rks(mol, mf.grids, mf.xc, fixed_dm,
                                            hermi=1, max_memory=2300)
            elapsed = time.perf_counter()-start
        pools = threadpool_info()
    values = dict(ne=ne, exc=exc, vxc=vxc)
    errors = {}
    for key, value in values.items():
        assert np.isfinite(value).all()
        np.testing.assert_allclose(value, expected[key], atol=1e-9, rtol=1e-10)
        errors[key] = float(np.max(np.abs(value-expected[key])))
    arrays = str(Path(cfg['result']).with_suffix('.npz'))
    np.savez(arrays, fixed_dm=fixed_dm, **values)
    return dict(mode=cfg['mode'], n_water=cfg['n'], source_sha=sha,
                reference_json=cfg['reference_json'], basis='def2-TZVPP', xc='MN15',
                grid_level=6, grid_points=len(mf.grids.weights), grid_sha256=digest.hexdigest(),
                nao=mol.nao, threads=2, profiled=cfg['profile'], xc_seconds=elapsed,
                stages_seconds=dict(stages), stage_calls=dict(counts), c_dispatch_counts=dict(dispatch),
                arrays=arrays, verification=errors, no_new_scf=True,
                energy_gate_status='not evaluated: saved-density XC diagnostic only',
                threadpools=pools, numa=os.getenv('PYSCF_NUMA'),
                coalesce=os.getenv('PYSCF_DFT_MGGA_COALESCE'),
                scatter_runs=os.getenv('PYSCF_DFT_MGGA_SCATTER_RUNS'),
                dense_blas=os.getenv('PYSCF_DFT_DENSE_BLAS'),
                mgga_pack_simd=os.getenv('PYSCF_DFT_MGGA_PACK_SIMD'),
                native_hashes={n: hashlib.sha256((root/'pyscf/lib'/n).read_bytes()).hexdigest()
                               for n in ('libdft.so', 'libcvhf.so', 'libcgto.so')},
                diagnostic_scope='One saved-density XC replay, NOT molecular SCF timing')


def launch(cfg):
    path = Path(cfg['result']).with_suffix('.config.json')
    path.write_text(json.dumps(cfg))
    env = dict(os.environ, PYTHONPATH=cfg['root'], OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1',
               OMP_DYNAMIC='FALSE', OMP_MAX_ACTIVE_LEVELS='1', PYSCF_NUMA='off',
               PYSCF_DFT_DENSE_BLAS='1', PYSCF_DFT_MGGA_PACK_SIMD='1',
               PYSCF_DFT_VMAT_TILE_SCREEN='0', PYSCF_JK_BLOCK_BOUND='0',
               PYSCF_JK_BLOCK_BOUND_STATS='0', PYSCF_DFT_MGGA_COALESCE_STATS='0',
               PYSCF_DFT_MGGA_COALESCE=str(int(cfg['mode']=='coalesce')),
               PYSCF_DFT_MGGA_SCATTER_RUNS=str(int(cfg['mode']=='scatter')))
    start = time.perf_counter()
    with Path(cfg['result']).with_suffix('.log').open('w') as log:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(path)],
                       cwd=cfg['root'], env=env, stdout=log, stderr=subprocess.STDOUT,
                       timeout=600, check=True)
    return json.loads(Path(cfg['result']).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker')
    p.add_argument('--baseline-root')
    p.add_argument('--candidate-root')
    p.add_argument('--candidate-sha')
    p.add_argument('--reference-dir', type=Path)
    p.add_argument('--output', type=Path, default=Path('replay-results'))
    a = p.parse_args()
    if a.worker:
        cfg = json.loads(Path(a.worker).read_text())
        Path(cfg['result']).write_text(json.dumps(worker(cfg), indent=2))
        return
    import numpy as np
    a.output = a.output.resolve(); a.output.mkdir(parents=True, exist_ok=True)
    report = dict(scope=__doc__, cases=[], errors=[], expected_records=24)
    for n in (4, 8):
        rows = []
        try:
            refbase = a.reference_dir.resolve()/f'full/MN15-n{n}-t2-0-baseline'
            for rep in range(4):
                order = MODES[rep % 3:] + MODES[:rep % 3]
                for mode in order:
                    root = a.candidate_root if mode=='scatter' else a.baseline_root
                    cfg = dict(root=str(Path(root).resolve()), n=n, mode=mode,
                               expected_sha=a.candidate_sha if mode=='scatter' else BASE_SHA,
                               profile=rep==3, reference_json=str(refbase)+'.json',
                               reference_npz=str(refbase)+'.npz',
                               result=str(a.output/f'MN15-n{n}-{rep}-{mode}.json'))
                    row = launch(cfg); rows.append(row)
                    print(json.dumps({k: row[k] for k in ('mode','n_water','xc_seconds','profiled','verification')}), flush=True)
            # Reopen every saved output and compare actual arrays, not just flags.
            with np.load(str(refbase)+'.npz', allow_pickle=False) as ref:
                for row in rows:
                    with np.load(row['arrays'], allow_pickle=False) as val:
                        assert all(np.isfinite(val[k]).all() for k in val.files)
                        np.testing.assert_array_equal(val['fixed_dm'], ref['fixed_dm'])
                        for key in ('ne','exc','vxc'):
                            np.testing.assert_allclose(val[key], ref[key], atol=1e-9, rtol=1e-10)
            med = {m: statistics.median(r['xc_seconds'] for r in rows if r['mode']==m and not r['profiled']) for m in MODES}
            assert len(rows)==12
            for mode in MODES:
                assert sum(r['mode']==mode and not r['profiled'] for r in rows)==3
                profiles = [r for r in rows if r['mode']==mode and r['profiled']]
                assert len(profiles)==1 and profiles[0]['c_dispatch_counts'].get('sparse_ncomp4_returned1',0)>0
            report['cases'].append(dict(n_water=n, trials=rows, median_xc_s=med,
                scatter_speedup_vs_raw=med['raw']/med['scatter'], coalesce_speedup_vs_raw=med['raw']/med['coalesce']))
        except Exception as exc:
            report['errors'].append(dict(n_water=n,error=repr(exc),partial_trials=rows))
        (a.output/'summary.json').write_text(json.dumps(report,indent=2))
    if report['errors']: raise SystemExit('Replay failed: inspect saved logs; no threshold changes allowed')
    assert len(report['cases'])==2 and sum(len(c['trials']) for c in report['cases'])==24

if __name__=='__main__':
    main()
