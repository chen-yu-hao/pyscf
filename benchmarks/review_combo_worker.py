"""One fresh-process, cold-cache SCF measurement for a declared option profile.

The controller supplies --config JSON and audits complete arrays across profiles.
This module imports no numerical package until the requested checkout is selected.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import ctypes
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
from unittest import mock

from review_upstream import (
    array_sha, backend_hashes, file_sha, native_libraries, source_sha,
    water_cluster,
)


def _counter_function(library, name):
    function = getattr(library, name, None)
    if function is not None:
        function.argtypes = [ctypes.c_void_p, ctypes.c_int]
        function.restype = None
    return function


def _read_counts(function, names, reset=False):
    if function is None:
        return dict(available=False)
    values = (ctypes.c_ulonglong * len(names))()
    function(ctypes.cast(values, ctypes.c_void_p), int(reset))
    return dict(available=True, **dict(zip(names, map(int, values))))


def worker(cfg):
    root = Path(cfg['root']).resolve()
    sys.path.insert(0, str(root))
    import numpy as np
    import pyscf
    from pyscf import dft, gto, lib, scf
    from pyscf.dft import numint
    from pyscf.scf import _vhf
    from threadpoolctl import threadpool_info, threadpool_limits

    output = Path(cfg['output']).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    actual_sha = source_sha(root)
    if actual_sha != cfg['sha']:
        raise RuntimeError(f"Wrong source SHA: {actual_sha} != {cfg['sha']}")
    if not Path(pyscf.__file__).resolve().is_relative_to(root):
        raise RuntimeError('Wrong PySCF import: ' + str(pyscf.__file__))
    for key, expected in cfg['flags'].items():
        if os.environ.get(key) != str(expected):
            raise RuntimeError(f'Incorrect effective environment: {key}')
    if os.environ.get('PYSCF_NUMA', '').lower() != 'off':
        raise ValueError('The native option search requires PYSCF_NUMA=off')
    for key in ('PYSCF_DFT_MGGA_COALESCE_STATS', 'PYSCF_JK_BLOCK_BOUND_STATS'):
        if os.environ.get(key, '0') != '0':
            raise ValueError(f'Timed diagnostic counters must be disabled: {key}')
    effective_flags = {key: os.environ.get(key) for key in cfg['flags']}

    threads = int(cfg['threads'])
    blas_threads = int(cfg.get('blas_threads', 1))
    max_memory = float(cfg.get('max_memory', 2300))
    ao_mb = float(cfg.get('ao_cache_mb', 0))
    if threads < 1 or blas_threads < 1:
        raise ValueError('Thread counts must be positive')
    if not math.isfinite(max_memory) or max_memory <= 0:
        raise ValueError('max_memory must be finite and positive')
    if not math.isfinite(ao_mb) or ao_mb < 0:
        raise ValueError('ao_cache_mb must be finite and nonnegative')
    force_direct = bool(cfg.get('force_direct', False))
    cache_options = cfg.get('cache_options')
    if ao_mb:
        from pyscf.dft.ao_cache import cached_ao
    if cache_options is not None:
        from pyscf.scf.eri_cache import cached_eri

    lib.num_threads(threads)
    mol = gto.M(atom=water_cluster(cfg['n']), basis=cfg['basis'], unit='Angstrom',
                verbose=0, max_memory=max_memory)
    mf = dft.RKS(mol)
    mf.xc = cfg['xc']
    mf.grids.level = cfg['grid_level']
    mf.conv_tol, mf.max_cycle, mf.max_memory = 1e-9, 80, max_memory
    mf.direct_scf_tol = 1e-13
    if force_direct:
        mf._is_mem_enough = lambda: False

    coalesce_names = ('raw_blocks', 'batches', 'joined', 'widest')
    bound_names = ('visits', 'rejects')
    coalesce_fn = _counter_function(numint.libdft, 'VXCmgga_coalesce_stats')
    bound_fn = _counter_function(_vhf.libcvhf, 'CVHFround3_bound_stats')
    dispatch = Counter()
    block_shapes = Counter()
    ao_cache = eri_cache = None
    ao_entry_memory = eri_entry_memory = None
    with threadpool_limits(limits=blas_threads, user_api='blas'), \
            threadpool_limits(limits=threads, user_api='openmp'), ExitStack() as life:
        memory_before_scf = lib.current_memory()[0]
        started = time.perf_counter()
        # A fixed scope order makes headroom accounting reproducible. Neither
        # cache is prefilled; both construction and cold filling are timed.
        if ao_mb:
            ao_entry_memory = lib.current_memory()[0]
            ao_cache = life.enter_context(cached_ao(mf, max_cache_mb=ao_mb))
        if cache_options is not None:
            eri_entry_memory = lib.current_memory()[0]
            eri_cache = life.enter_context(cached_eri(mf, **cache_options))
        energy = float(mf.kernel())
        seconds = time.perf_counter() - started
        rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        memory_after_scf = lib.current_memory()[0]
        used_incore_eri = mf._eri is not None
        timed_ao = ao_cache.snapshot() if ao_cache is not None else None
        timed_eri = eri_cache.stats() if eri_cache is not None else None
        if not mf.converged or not np.isfinite(energy):
            raise AssertionError('SCF did not converge to a finite energy')
        if timed_eri:
            for cache in timed_eri['caches']:
                assert all(cache[key] == 0 for key in ('calls', 'hits', 'misses', 'cost_bypass'))
                assert cache['data_bytes'] <= cache['allocated_bytes'] <= cache['max_bytes']
        if timed_ao:
            assert timed_ao['retained_bytes'] <= timed_ao['max_bytes']
            assert timed_ao['peak_bytes'] <= timed_ao['max_bytes']
        dm = np.asarray(mf.make_rdm1())
        fixed_dm = dm.copy()
        if cfg.get('reference_arrays'):
            with np.load(cfg['reference_arrays'], allow_pickle=False) as reference:
                fixed_dm = reference['dm'].copy()

        # All native counter, dispatch, and block-shape observations below are
        # after the SCF timer and its RSS/cache snapshots. Upstream may have no
        # fused-XC or experimental-counter symbols at all.
        _read_counts(coalesce_fn, coalesce_names, reset=True)
        with ExitStack() as diagnostics:
            diagnostics.enter_context(mock.patch.dict(
                os.environ, {'PYSCF_DFT_MGGA_COALESCE_STATS': '1'}))
            original = getattr(numint.libdft, 'VXCxc_vmat', None)
            if original is not None:
                def counted(*args):
                    result = int(original(*args))
                    kind = 'sparse' if bool(args[8]) else 'dense'
                    ncomp = int(args[7].value)
                    dispatch[f'{kind}-ncomp{ncomp}-returned{result}'] += 1
                    return result
                diagnostics.enter_context(mock.patch.object(numint.libdft, 'VXCxc_vmat', counted))
            original_block_loop = mf._numint.block_loop

            def observed_blocks(*args, **kwargs):
                for ao, mask, weight, coords in original_block_loop(*args, **kwargs):
                    block_shapes[f'{tuple(ao.shape)};bytes={ao.nbytes};mask={mask is not None}'] += 1
                    yield ao, mask, weight, coords

            diagnostics.enter_context(mock.patch.object(mf._numint, 'block_loop', observed_blocks))
            ne, exc, vxc = mf._numint.nr_rks(
                mol, mf.grids, mf.xc, fixed_dm, hermi=1, max_memory=max_memory)
        coalesce_stats = _read_counts(coalesce_fn, coalesce_names)
        diagnostic_ao = ao_cache.snapshot() if ao_cache is not None else None

        if eri_cache is not None:
            eri_cache.stats(diagnostics=True, reset=True)
        _read_counts(bound_fn, bound_names, reset=True)
        with mock.patch.dict(os.environ, {'PYSCF_JK_BLOCK_BOUND_STATS': '1'}):
            vj, vk = mf.get_jk(mol, fixed_dm, hermi=1)
        bound_jk = _read_counts(bound_fn, bound_names)
        diagnostic_eri = eri_cache.stats(diagnostics=False) if eri_cache is not None else None

        # PBE uses J-only in its timed kernel. Observe that wrapper separately
        # rather than infer its activation from the J+K validation call.
        hybrid = bool(mf._numint.libxc.is_hybrid_xc(mf.xc))
        bound_j_only = None
        if not hybrid:
            _read_counts(bound_fn, bound_names, reset=True)
            with mock.patch.dict(os.environ, {'PYSCF_JK_BLOCK_BOUND_STATS': '1'}):
                vj_only, _ = mf.get_jk(mol, fixed_dm, hermi=1, with_j=True, with_k=False)
            bound_j_only = _read_counts(bound_fn, bound_names)
            np.testing.assert_allclose(vj, vj_only, atol=1e-9, rtol=1e-10)

        # A distinct optimizer, with block-bound disabled, is an independent
        # check of cached integrals and of optional J/K pre-rejection.
        independent_jk = cache_options is not None or cfg['flags'].get('PYSCF_JK_BLOCK_BOUND') == '1'
        if independent_jk:
            uncached = scf.RHF(mol)
            uncached.max_memory = max_memory
            uncached.direct_scf_tol = mf.direct_scf_tol
            uncached._is_mem_enough = lambda: False
            with mock.patch.dict(os.environ, {'PYSCF_JK_BLOCK_BOUND': '0',
                                             'PYSCF_JK_BLOCK_BOUND_STATS': '0'}):
                vj_uncached, vk_uncached = uncached.get_jk(mol, fixed_dm, hermi=1)
            np.testing.assert_allclose(vj, vj_uncached, atol=1e-9, rtol=1e-10)
            np.testing.assert_allclose(vk, vk_uncached, atol=1e-9, rtol=1e-10)
        else:
            vj_uncached, vk_uncached = vj.copy(), vk.copy()
        pools = threadpool_info()
        effective_threads = lib.num_threads()
        assert effective_threads == threads
        for pool in pools:
            if pool['user_api'] == 'blas':
                assert pool['num_threads'] == blas_threads
            if pool['user_api'] == 'openmp':
                assert pool['num_threads'] == threads

    values = dict(dm=dm, fixed_dm=fixed_dm, ne=ne, exc=exc, vxc=vxc,
                  vj=vj, vk=vk, vj_uncached=vj_uncached, vk_uncached=vk_uncached)
    if not all(np.isfinite(value).all() for value in values.values()):
        raise AssertionError('Non-finite molecular diagnostic result')
    if not (np.isfinite(mf.grids.coords).all() and np.isfinite(mf.grids.weights).all()):
        raise AssertionError('Non-finite grid')
    if not math.isfinite(seconds) or seconds <= 0:
        raise AssertionError('Invalid SCF duration')
    arrays = output.with_suffix('.npz')
    np.savez_compressed(arrays, **values)
    grid_digest = array_sha(mf.grids.coords, mf.grids.weights)
    grid_arrays = output.parent / f'grid-{grid_digest}.npz'
    if grid_arrays.exists():
        with np.load(grid_arrays, allow_pickle=False) as saved_grid:
            np.testing.assert_array_equal(mf.grids.coords, saved_grid['coords'])
            np.testing.assert_array_equal(mf.grids.weights, saved_grid['weights'])
    else:
        np.savez_compressed(grid_arrays, coords=mf.grids.coords, weights=mf.grids.weights)
    libraries = native_libraries()
    for pool in pools:
        if pool.get('filepath') and Path(pool['filepath']).is_file():
            pool['sha256'] = file_sha(pool['filepath'])
    blas_counts = sorted({pool['num_threads'] for pool in pools if pool['user_api'] == 'blas'})
    eri_caches = (timed_eri or {}).get('caches', [])
    eri_diag_caches = (diagnostic_eri or {}).get('caches', [])
    scope_status = dict(
        ao=dict(requested=bool(ao_mb), requested_mib=ao_mb,
                budget_bytes=(timed_ao or {}).get('max_bytes', 0),
                admissions_in_scf=(timed_ao or {}).get('admissions', 0),
                hits_in_scf=(timed_ao or {}).get('hits', 0),
                retained_bytes_at_scf_end=(timed_ao or {}).get('retained_bytes', 0)),
        eri=dict(requested=cache_options is not None,
                 budget_bytes=(timed_eri or {}).get('budget_bytes', 0),
                 cache_count=len(eri_caches), enabled_count=sum(c.get('enabled', False) for c in eri_caches),
                 entries_at_scf_end=sum(c['entries'] for c in eri_caches),
                 diagnostic_hits=sum(c['hits'] for c in eri_diag_caches),
                 filter_queries=sum(c.get('saturation_filter', {}).get('queries', 0) for c in eri_diag_caches),
                 inactive_because_incore=cache_options is not None and used_incore_eri),
        kernel_jk_mode='J+K' if hybrid else 'J-only',
    )
    result = dict(
        config=cfg, source_sha=actual_sha, source_file=str(pyscf.__file__),
        effective_flags=effective_flags, build_profile=cfg.get('build_profile'),
        energy_hartree=energy, scf_seconds=seconds, converged=bool(mf.converged),
        cycles=int(getattr(mf, 'cycles', -1)), nao=mol.nao,
        max_rss_kib_at_scf_end=rss_kib, memory_mb_before_scf=memory_before_scf,
        memory_mb_after_scf=memory_after_scf, used_incore_eri=used_incore_eri,
        ao_cache_entry_memory_mb=ao_entry_memory, eri_cache_entry_memory_mb=eri_entry_memory,
        effective_threads=effective_threads, requested_blas_threads=blas_threads,
        effective_blas_threads=blas_counts, threadpools=pools,
        grid_sha256=grid_digest, grid_arrays=grid_arrays.name,
        grid_arrays_sha256=file_sha(grid_arrays), grid_points=len(mf.grids.weights),
        fixed_dm_sha256=array_sha(fixed_dm), arrays=arrays.name, arrays_sha256=file_sha(arrays),
        native_libraries=libraries, backend_hashes=backend_hashes(libraries),
        timed_cache=timed_eri, cache_diagnostics=diagnostic_eri,
        timed_ao_cache=timed_ao, ao_cache_diagnostics=diagnostic_ao, scope_status=scope_status,
        post_scf_xc_dispatch=dict(dispatch), post_scf_block_shapes=dict(block_shapes),
        post_scf_coalesce=coalesce_stats,
        post_scf_jk_bound=dict(jk=bound_jk, j_only=bound_j_only),
        uncached_jk_verified=independent_jk, independent_jk_verified=independent_jk,
        timed_diagnostic_counters=False, ao_intrinsic_cache_counters=bool(ao_mb), profiled=False,
        python=platform.python_version(), platform=platform.platform(),
        package_versions={name: importlib.metadata.version(name)
                          for name in ('numpy', 'scipy', 'h5py', 'threadpoolctl')},
        affinity=sorted(os.sched_getaffinity(0)),
        scientific_settings=dict(conv_tol=1e-9, max_cycle=80, direct_scf_tol=1e-13,
                                 max_memory_mb=max_memory, force_direct=force_direct),
        timing_scope='cold AO/ERI scope setup plus complete mf.kernel; imports, molecule and checks excluded',
        block_shape_scope='post-SCF common-density diagnostic; timed SCF is not instrumented',
        thread_scope='global OpenMP and BLAS runtime settings; fused XC may cap its local team to its workspace',
    )
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    try:
        return worker(cfg)
    except Exception as error:
        if cfg.get('output'):
            failure = Path(cfg['output']).resolve().with_suffix('.failure.json')
            failure.parent.mkdir(parents=True, exist_ok=True)
            failure.write_text(json.dumps(dict(config=cfg, passed=False,
                                              error=f'{type(error).__name__}: {error}',
                                              traceback=traceback.format_exc()), indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
