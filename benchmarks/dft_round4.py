"""Round 4: isolated collocation-cache / native-ISA / Qcint comparisons.

Baseline is validated round two. Both sides keep its dense BLAS enabled; round
three candidates stay OFF. Default PySCF grid blocking is NOT changed. Every
SCF starts in a fresh process. Cached AO arrays are released before fixed-density
XC/J/K verification; no density-dependent result is cached.
"""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import ctypes
import hashlib
import importlib.util
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


def worker(cfg):
    root = Path(cfg['root']).resolve()
    sys.path.insert(0, str(root))
    import numpy as np
    import pyscf
    from pyscf import dft, gto, lib, scf
    from pyscf.dft import numa, numint
    from threadpoolctl import threadpool_info, threadpool_limits
    if not Path(pyscf.__file__).resolve().is_relative_to(root):
        raise RuntimeError('Wrong PySCF checkout')
    lib.num_threads(cfg['threads'])
    mol = gto.M(atom=harness.water_cluster(cfg['n']), basis='def2-TZVPP',
                unit='Angstrom', verbose=0, max_memory=2300)
    mf = dft.RKS(mol)
    mf.xc, mf.grids.level = cfg['xc'], 6
    mf.conv_tol, mf.max_cycle, mf.max_memory = 1e-8, 50, 2300
    stages, counts = defaultdict(float), defaultdict(int)
    cache_stats = None

    def instrument(stack, obj, attr, label):
        original = getattr(obj, attr)
        def timed(*a, **kw):
            start = time.perf_counter()
            try:
                return original(*a, **kw)
            finally:
                stages[label] += time.perf_counter() - start
                counts[label] += 1
        stack.enter_context(mock.patch.object(obj, attr, timed))

    with threadpool_limits(limits=1, user_api='blas'):
        with ExitStack() as outer:
            cache = None
            if cfg['variant'] == 'candidate' and cfg['comparison'] in ('cache', 'combined'):
                # The adapter is part of the candidate source, never injected into baseline.
                from pyscf.dft.ao_cache import cached_ao
                cache = outer.enter_context(cached_ao(mf, cfg['cache_mb']))
            with ExitStack() as profile:
                if cfg['profile']:
                    instrument(profile, numa, 'nr_rks', 'xc_total')
                    for method in ('get_veff', 'get_jk', 'eig'):
                        instrument(profile, mf, method, method)
                    instrument(profile, mf.grids, 'build', 'grid_build')
                    for method in ('eval_ao', 'eval_rho1', 'eval_rho2', 'eval_xc_eff'):
                        instrument(profile, mf._numint, method, method)
                    instrument(profile, numint, '_xc_vmat', 'xc_matrix_assembly')
                start = time.perf_counter()
                energy = float(mf.kernel())
                wall = time.perf_counter()-start
            if cache is not None:
                cache_stats = cache.snapshot()
            scf_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        dm = np.asarray(mf.make_rdm1())
        fixed_dm = dm.copy()
        if cfg.get('reference'):
            with np.load(cfg['reference']) as ref:
                fixed_dm = ref['dm'].copy()
        ne, exc, vxc = mf._numint.nr_rks(mol, mf.grids, mf.xc, fixed_dm,
                                        hermi=1, max_memory=2300)
        vj, vk = scf.hf.get_jk(mol, fixed_dm, hermi=1)
        pools = threadpool_info()
    for arr in (energy, dm, fixed_dm, ne, exc, vxc, vj, vk):
        if not np.isfinite(arr).all():
            raise RuntimeError('Nonfinite SCF/XC/J/K result')
    digest = hashlib.sha256()
    for arr in (mf.grids.coords, mf.grids.weights):
        digest.update(np.ascontiguousarray(arr).data)
    arrays = str(Path(cfg['result']).with_suffix('.npz'))
    np.savez(arrays, dm=dm, fixed_dm=fixed_dm, ne=ne, exc=exc, vxc=vxc, vj=vj, vk=vk)
    hashes = {}
    for name in ('libdft.so', 'libcvhf.so', 'deps/lib/libcint.so', 'deps/lib/libxc.so'):
        path = root/'pyscf/lib'/name
        if path.exists():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    cmake = root/'build-native/CMakeCache.txt'
    build_options = {}
    if cmake.exists():
        for line in cmake.read_text().splitlines():
            if line.startswith(('USE_QCINT:', 'BUILD_MARCH_NATIVE:', 'CMAKE_C_FLAGS_RELEASE:')):
                key, value = line.split('=', 1)
                build_options[key.split(':')[0]] = value
    return dict(round=4, comparison=cfg['comparison'], variant=cfg['variant'],
                source_sha=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
                source_file=pyscf.__file__, n_water=cfg['n'], xc=cfg['xc'], nao=mol.nao,
                threads=cfg['threads'], basis='def2-TZVPP', grid_level=6,
                grid_points=len(mf.grids.weights), grid_sha256=digest.hexdigest(),
                converged=bool(mf.converged), cycles=int(getattr(mf, 'cycles', -1)),
                energy_hartree=energy, scf_seconds=wall, profiled=cfg['profile'],
                stages_seconds=dict(stages), stage_calls=dict(counts), arrays=arrays,
                ao_cache=cache_stats, cache_stats_scope='SCF only; arrays released before verification',
                max_rss_kib_at_scf_end=scf_rss_kib, used_incore_eri=mf._eri is not None,
                numa=os.getenv('PYSCF_NUMA'), dense_blas=os.getenv('PYSCF_DFT_DENSE_BLAS'),
                tile_screen=os.getenv('PYSCF_DFT_VMAT_TILE_SCREEN'),
                mgga_simd=os.getenv('PYSCF_DFT_MGGA_PACK_SIMD'), jk_bound=os.getenv('PYSCF_JK_BLOCK_BOUND'),
                native_hashes=hashes, build_options=build_options, threadpools=pools,
                python=platform.python_version(), numpy=np.__version__,
                affinity=sorted(os.sched_getaffinity(0)))


original_verify = harness.verify

def verify(reference, candidate):
    import numpy as np
    result = original_verify(reference, candidate)
    with np.load(reference['arrays']) as a, np.load(candidate['arrays']) as b:
        np.testing.assert_array_equal(a['fixed_dm'], b['fixed_dm'])
        result['fixed_density_jk_max_abs'] = {}
        for key in ('vj', 'vk'):
            np.testing.assert_allclose(a[key], b[key], atol=1e-9, rtol=1e-10)
            result['fixed_density_jk_max_abs'][key] = float(np.max(np.abs(a[key]-b[key])))
    cache = candidate['ao_cache']
    if cache is not None:
        if cache['peak_bytes'] > cache['max_bytes']:
            raise AssertionError('Retained AO budget exceeded')
    return result


def launch(cfg, timeout):
    cfg = dict(cfg, comparison=OPTIONS.comparison, cache_mb=OPTIONS.cache_mb)
    config = Path(cfg['result']).with_suffix('.config.json')
    config.write_text(json.dumps(cfg))
    env = dict(os.environ, PYTHONPATH=cfg['root'], OMP_NUM_THREADS=str(cfg['threads']),
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1',
               OMP_DYNAMIC='FALSE', OMP_MAX_ACTIVE_LEVELS='1', PYSCF_NUMA='off',
               PYSCF_DFT_DENSE_BLAS='1', PYSCF_DFT_VMAT_TILE_SCREEN='0',
               PYSCF_DFT_MGGA_PACK_SIMD='0', PYSCF_JK_BLOCK_BOUND='0')
    start = time.perf_counter()
    with Path(cfg['result']).with_suffix('.log').open('w') as log:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(config)],
                       cwd=cfg['root'], env=env, stdout=log, stderr=subprocess.STDOUT,
                       timeout=timeout, check=True)
    value = json.loads(Path(cfg['result']).read_text())
    value['process_seconds'] = time.perf_counter()-start
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--comparison', choices=['cache', 'native', 'qcint', 'combined'], default='cache')
    parser.add_argument('--cache-mb', type=float, default=512.)
    OPTIONS, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    harness.worker, harness.launch, harness.verify = worker, launch, verify
    harness.main()
