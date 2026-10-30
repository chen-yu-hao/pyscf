"""Paired MGGA exact-screen coalescing against the validated round-three kernel.

Both sides enable dense BLAS and SIMD MGGA packing; JK block bounds and AO
cache are disabled. Diagnostics run ONLY in post-SCF fixed-density verification.
"""
import ctypes
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from unittest import mock
import dft_water_grid6 as harness

old_verify = harness.verify


def worker(cfg):
    import numpy as np
    root = Path(cfg['root']).resolve()
    sys.path.insert(0, str(root))
    import pyscf
    from pyscf import dft, gto, lib, scf
    from pyscf.dft import numa, numint
    from threadpoolctl import threadpool_info, threadpool_limits
    if not Path(pyscf.__file__).resolve().is_relative_to(root):
        raise RuntimeError('Wrong PySCF checkout')
    if cfg['numa'] != 'off':
        raise ValueError('This controlled experiment requires NUMA off')
    lib.num_threads(cfg['threads'])
    mol = gto.M(atom=harness.water_cluster(cfg['n']), basis='def2-TZVPP',
                unit='Angstrom', verbose=0, max_memory=2300)
    mf = dft.RKS(mol)
    mf.xc, mf.grids.level = cfg['xc'], 6
    mf.conv_tol, mf.max_cycle, mf.max_memory = 1e-8, 50, 2300
    stages, counts, dispatch = defaultdict(float), defaultdict(int), Counter()

    def instrument(stack, obj, attr, label):
        original = getattr(obj, attr)
        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                stages[label] += time.perf_counter()-start
                counts[label] += 1
        replacement = timed
        if isinstance(obj, type) and isinstance(inspect.getattr_static(obj, attr), staticmethod):
            replacement = staticmethod(timed)
        stack.enter_context(mock.patch.object(obj, attr, replacement))

    original_c = numint.libdft.VXCxc_vmat
    def counted(*args):
        ret = original_c(*args)
        dispatch[f"{'sparse' if bool(args[8]) else 'dense'}_ncomp{args[7].value}_returned{ret}"] += 1
        return ret

    with threadpool_limits(limits=1, user_api='blas'):
        with ExitStack() as stack:
            if cfg['profile']:
                instrument(stack, numa, 'nr_rks', 'xc_total_including_numa')
                for name in ('get_veff', 'get_jk', 'eig'):
                    instrument(stack, mf, name, name)
                instrument(stack, mf.grids, 'build', 'grid_build')
                for name in ('eval_ao', 'eval_rho1', 'eval_rho2', 'eval_xc_eff'):
                    instrument(stack, numint.NumInt, name, name)
                instrument(stack, numint, '_xc_vmat', 'xc_matrix_assembly')
                stack.enter_context(mock.patch.object(numint.libdft, 'VXCxc_vmat', counted))
            start = time.perf_counter()
            energy = float(mf.kernel())
            wall = time.perf_counter()-start
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        dm = np.asarray(mf.make_rdm1())
        fixed_dm = dm.copy()
        if cfg.get('reference'):
            with np.load(cfg['reference'], allow_pickle=False) as ref:
                fixed_dm = ref['dm'].copy()
        diagnostic = None
        counter = getattr(numint.libdft, 'VXCmgga_coalesce_stats', None)
        values = (ctypes.c_ulonglong*4)()
        if counter is not None:
            counter.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_int]
            counter.restype = None
            counter(values, 1)
        # SCF and independent profile timings above contain NO diagnostic counters.
        with mock.patch.dict(os.environ, {'PYSCF_DFT_MGGA_COALESCE_STATS': '1'}):
            ne, exc, vxc = mf._numint.nr_rks(mol, mf.grids, mf.xc, fixed_dm,
                                            hermi=1, max_memory=2300)
        if counter is not None:
            counter(values, 1)
            diagnostic = dict(zip(('raw_blocks', 'batches', 'joined', 'widest'), map(int, values)))
        jk_mf = scf.RHF(mol)
        jk_mf.max_memory = 2300
        jk_mf._is_mem_enough = lambda: False
        vj, vk = jk_mf.get_jk(mol, fixed_dm, hermi=1)
        pools = threadpool_info()
    for value in (energy, dm, fixed_dm, ne, exc, vxc, vj, vk):
        if not np.isfinite(value).all():
            raise AssertionError('Nonfinite SCF or fixed-density result')
    digest = hashlib.sha256()
    for arr in (mf.grids.coords, mf.grids.weights):
        digest.update(np.ascontiguousarray(arr).data)
    arrays = str(Path(cfg['result']).with_suffix('.npz'))
    np.savez(arrays, dm=dm, fixed_dm=fixed_dm, ne=ne, exc=exc, vxc=vxc, vj=vj, vk=vk)
    sha = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    return dict(variant=cfg['variant'], source_sha=sha, source_file=pyscf.__file__,
                n_water=cfg['n'], xc=cfg['xc'], basis='def2-TZVPP', nao=mol.nao,
                threads=cfg['threads'], grid_level=6, grid_points=len(mf.grids.weights),
                grid_sha256=digest.hexdigest(), converged=bool(mf.converged),
                cycles=int(getattr(mf, 'cycles', -1)), energy_hartree=energy,
                scf_seconds=wall, profiled=cfg['profile'], stages_seconds=dict(stages),
                stage_calls=dict(counts), c_dispatch_counts=dict(dispatch),
                dispatch_count_scope='profile SCF only', coalesce_diagnostic=diagnostic,
                diagnostic_scope='untimed common-density XC verification; includes all-screened batches',
                timed_diagnostic_counters=False, arrays=arrays, numa=os.getenv('PYSCF_NUMA'),
                dense_blas=os.getenv('PYSCF_DFT_DENSE_BLAS'),
                tile_screen=os.getenv('PYSCF_DFT_VMAT_TILE_SCREEN'),
                mgga_pack_simd=os.getenv('PYSCF_DFT_MGGA_PACK_SIMD'),
                jk_block_bound=os.getenv('PYSCF_JK_BLOCK_BOUND'),
                coalesce=os.getenv('PYSCF_DFT_MGGA_COALESCE'), used_ao_cache=False,
                used_incore_eri=mf._eri is not None, max_rss_kib_at_scf_end=rss,
                native_hashes={name:hashlib.sha256((root/'pyscf/lib'/name).read_bytes()).hexdigest()
                               for name in ('libdft.so', 'libcvhf.so', 'libcgto.so')},
                threadpools=pools, python=platform.python_version(), numpy=np.__version__,
                affinity=sorted(os.sched_getaffinity(0)), round=5)


def verify(reference, candidate):
    import numpy as np
    report = old_verify(reference, candidate)
    with np.load(reference['arrays'], allow_pickle=False) as a, np.load(candidate['arrays'], allow_pickle=False) as b:
        np.testing.assert_array_equal(a['fixed_dm'], b['fixed_dm'])
        errors = {}
        for key in ('vj', 'vk'):
            np.testing.assert_allclose(b[key], a[key], atol=1e-9, rtol=1e-10)
            errors[key] = float(np.max(np.abs(a[key]-b[key])))
    report['fixed_density_jk_max_abs'] = errors
    return report


def launch(cfg, timeout):
    config = Path(cfg['result']).with_suffix('.config.json')
    config.write_text(json.dumps(cfg))
    env = dict(os.environ, PYTHONPATH=cfg['root'], OMP_NUM_THREADS=str(cfg['threads']),
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1',
               OMP_DYNAMIC='FALSE', OMP_MAX_ACTIVE_LEVELS='1', PYSCF_NUMA='off',
               PYSCF_DFT_DENSE_BLAS='1', PYSCF_DFT_MGGA_PACK_SIMD='1',
               PYSCF_DFT_VMAT_TILE_SCREEN='0', PYSCF_JK_BLOCK_BOUND='0',
               PYSCF_JK_BLOCK_BOUND_STATS='0', PYSCF_DFT_MGGA_COALESCE_STATS='0',
               PYSCF_DFT_MGGA_COALESCE=str(int(cfg['variant']=='candidate')))
    start = time.perf_counter()
    with Path(cfg['result']).with_suffix('.log').open('w') as log:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(config)],
                       cwd=cfg['root'], env=env, stdout=log, stderr=subprocess.STDOUT,
                       timeout=timeout, check=True)
    result = json.loads(Path(cfg['result']).read_text())
    result['process_seconds'] = time.perf_counter()-start
    return result


if __name__ == '__main__':
    harness.worker, harness.verify, harness.launch = worker, verify, launch
    harness.main()
