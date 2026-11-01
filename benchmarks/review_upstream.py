"""Matched fresh-process SCF comparison and independent ERI-policy SCF gates.

The performance suite compares a pinned upstream checkout, candidate defaults,
and candidate dense-BLAS/SIMD options.  Other experimental options remain off.
The ERI suite is a correctness gate, not a large-system performance claim.
Only the controller imports this file before selecting a source checkout.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
from unittest import mock


UPSTREAM_SHA = '1c6d4a88393557da61a639d078d1aed2813912dd'
FLAGS = {
    'PYSCF_NUMA': 'off',
    'PYSCF_DFT_DENSE_BLAS': '0',
    'PYSCF_DFT_MGGA_PACK_SIMD': '0',
    'PYSCF_DFT_VMAT_TILE_SCREEN': '0',
    'PYSCF_DFT_MGGA_COALESCE': '0',
    'PYSCF_DFT_MGGA_COALESCE_STATS': '0',
    'PYSCF_DFT_MGGA_SCATTER_RUNS': '0',
    'PYSCF_JK_BLOCK_BOUND': '0',
    'PYSCF_JK_BLOCK_BOUND_STATS': '0',
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'BLIS_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
    'OMP_DYNAMIC': 'FALSE',
    'OMP_MAX_ACTIVE_LEVELS': '1',
}


def water_cluster(n):
    """Same deterministic, non-optimized geometry as the earlier experiments."""
    atoms = []
    for i in range(n):
        x, y = 4.0 * (i % 3), 4.0 * (i // 3)
        atoms.extend([('O', (x, y, 0.0)), ('H', (x + .757, y, .587)),
                      ('H', (x - .757, y, .587))])
    return atoms


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(*arrays):
    import numpy as np
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def source_sha(root):
    return subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()


def native_libraries():
    """Hash the libraries actually mapped by the worker, including symlink targets."""
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) != 6 or not fields[5].startswith('/'):
            continue
        path = Path(fields[5])
        if any(path.name.startswith(prefix) for prefix in
               ('libcint.', 'libxc.', 'libxc_itrf.', 'libdft.', 'libcvhf.',
                'libcgto.', 'libnp_helper.')) and path.is_file():
            paths.add(path.resolve())
    return {str(path): file_sha(path) for path in sorted(paths)}


def backend_hashes(libraries):
    return {prefix: sorted({sha for path, sha in libraries.items()
                           if Path(path).name.startswith(prefix)})
            for prefix in ('libcint.', 'libxc.')}


def worker(cfg):
    root = Path(cfg['root']).resolve()
    sys.path.insert(0, str(root))
    import numpy as np
    import pyscf
    from pyscf import dft, gto, lib, scf
    from pyscf.dft import numint
    from threadpoolctl import threadpool_info, threadpool_limits

    if not Path(pyscf.__file__).resolve().is_relative_to(root):
        raise RuntimeError('Wrong PySCF import: ' + str(pyscf.__file__))
    actual_sha = source_sha(root)
    if actual_sha != cfg['sha']:
        raise RuntimeError(f"Wrong source SHA: {actual_sha} != {cfg['sha']}")
    for key, expected in cfg['flags'].items():
        if os.environ.get(key) != expected:
            raise RuntimeError(f'Incorrect effective environment: {key}')

    lib.num_threads(cfg['threads'])
    mol = gto.M(atom=water_cluster(cfg['n']), basis=cfg['basis'], unit='Angstrom',
                verbose=0, max_memory=2300)
    mf = dft.RKS(mol)
    mf.xc, mf.grids.level = cfg['xc'], cfg['grid_level']
    mf.conv_tol, mf.max_cycle, mf.max_memory = 1e-9, 80, 2300
    mf.direct_scf_tol = 1e-13
    if cfg['force_direct']:
        # Only the separately labelled small-system correctness gate forces this.
        mf._is_mem_enough = lambda: False

    cache_options = cfg['cache_options']
    if cache_options is not None:
        from pyscf.scf.eri_cache import cached_eri
    output = Path(cfg['output'])
    dispatch = Counter()
    cache = None
    with threadpool_limits(limits=1, user_api='blas'), \
            threadpool_limits(limits=cfg['threads'], user_api='openmp'), ExitStack() as life:
        # Cold scope allocation, native allocation/fill, and the complete SCF
        # are inside this timer. Imports, molecule construction and validation
        # are outside. There is no warm-up or profiling inside timed work.
        started = time.perf_counter()
        if cache_options is not None:
            cache = life.enter_context(cached_eri(mf, **cache_options))
        energy = float(mf.kernel())
        seconds = time.perf_counter() - started
        rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        timed_cache = cache.stats() if cache else None
        dm = np.asarray(mf.make_rdm1())
        fixed_dm = dm.copy()
        if cfg['reference_arrays']:
            with np.load(cfg['reference_arrays'], allow_pickle=False) as reference:
                fixed_dm = reference['dm'].copy()

        # Observe the fork's C XC path only after timing. Upstream need not
        # provide numa.py, _xc_vmat, or VXCxc_vmat.
        with ExitStack() as diagnostics:
            original = getattr(numint.libdft, 'VXCxc_vmat', None)
            if original is not None:
                def counted(*args):
                    result = original(*args)
                    path = 'sparse' if bool(args[8]) else 'dense'
                    dispatch[f'{path}-ncomp{args[7].value}-returned{result}'] += 1
                    return result
                diagnostics.enter_context(mock.patch.object(numint.libdft, 'VXCxc_vmat', counted))
            ne, exc, vxc = mf._numint.nr_rks(
                mol, mf.grids, mf.xc, fixed_dm, hermi=1, max_memory=2300)
        if cache:
            cache.stats(diagnostics=True, reset=True)
        vj, vk = mf.get_jk(mol, fixed_dm, hermi=1)
        cache_diagnostics = cache.stats(diagnostics=False) if cache else None
        # A distinct optimizer independently verifies the cached path; this
        # validation is outside the timer and the captured SCF peak RSS.
        if cache:
            uncached = scf.RHF(mol)
            uncached.max_memory = 2300
            uncached.direct_scf_tol = mf.direct_scf_tol
            uncached._is_mem_enough = lambda: False
            vj_uncached, vk_uncached = uncached.get_jk(mol, fixed_dm, hermi=1)
            np.testing.assert_allclose(vj, vj_uncached, atol=1e-9, rtol=1e-10)
            np.testing.assert_allclose(vk, vk_uncached, atol=1e-9, rtol=1e-10)
        else:
            vj_uncached, vk_uncached = vj.copy(), vk.copy()
        pools = threadpool_info()
        effective_threads = lib.num_threads()

    values = dict(dm=dm, fixed_dm=fixed_dm, ne=ne, exc=exc, vxc=vxc, vj=vj, vk=vk,
                  vj_uncached=vj_uncached, vk_uncached=vk_uncached)
    if (not np.isfinite(energy) or not all(np.isfinite(v).all() for v in values.values())
            or not np.isfinite(mf.grids.coords).all() or not np.isfinite(mf.grids.weights).all()):
        raise AssertionError('Non-finite molecular result')
    if not mf.converged:
        raise AssertionError('SCF did not converge')
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
    result = dict(
        config=cfg, source_sha=actual_sha, source_file=str(pyscf.__file__),
        energy_hartree=energy, scf_seconds=seconds, converged=bool(mf.converged),
        cycles=int(getattr(mf, 'cycles', -1)), nao=mol.nao,
        max_rss_kib_at_scf_end=rss_kib, used_incore_eri=mf._eri is not None,
        effective_threads=effective_threads, threadpools=pools,
        grid_sha256=grid_digest, grid_arrays=grid_arrays.name,
        grid_arrays_sha256=file_sha(grid_arrays),
        grid_points=len(mf.grids.weights), fixed_dm_sha256=array_sha(fixed_dm),
        arrays=arrays.name, arrays_sha256=file_sha(arrays),
        native_libraries=libraries, backend_hashes=backend_hashes(libraries),
        timed_cache=timed_cache, cache_diagnostics=cache_diagnostics,
        post_scf_xc_dispatch=dict(dispatch), uncached_jk_verified=cache is not None,
        timed_diagnostic_counters=False, profiled=False,
        python=platform.python_version(), platform=platform.platform(),
        package_versions={name: importlib.metadata.version(name)
                          for name in ('numpy', 'scipy', 'h5py', 'threadpoolctl')},
        affinity=sorted(os.sched_getaffinity(0)),
        timing_scope='cold cache setup plus mf.kernel; imports, molecule and checks excluded',
    )
    output.write_text(json.dumps(result, indent=2) + '\n')


def validate_case(directory, case, performance):
    import numpy as np
    records = [json.loads((directory / name).read_text()) for name in case['records']]
    if len(records) != len(case['variants']) * case['repeats']:
        raise AssertionError('Incomplete case')
    reference = records[0]
    maxima = {key: 0.0 for key in ('energy', 'ne', 'exc', 'vxc', 'vj', 'vk')}
    with np.load(directory / reference['arrays'], allow_pickle=False) as arrays:
        common = {key: arrays[key].copy() for key in arrays.files}
    assert reference['config']['variant'] == case['variants'][0]
    expected_backend = reference['backend_hashes']
    assert all(expected_backend[prefix] for prefix in ('libcint.', 'libxc.'))
    def blas_hashes(record):
        return sorted({p['sha256'] for p in record['threadpools'] if p['user_api'] == 'blas'})
    assert blas_hashes(reference)
    grid_path = directory / reference['grid_arrays']
    assert file_sha(grid_path) == reference['grid_arrays_sha256']
    with np.load(grid_path, allow_pickle=False) as saved_grid:
        assert array_sha(saved_grid['coords'], saved_grid['weights']) == reference['grid_sha256']
    for record in records:
        cfg = record['config']
        assert cfg['case'] == case['case']
        assert cfg['variant'] in case['variants']
        for key in ('n', 'xc', 'basis', 'grid_level', 'threads', 'force_direct'):
            assert cfg[key] == reference['config'][key]
        expected_flags = dict(FLAGS, OMP_NUM_THREADS=str(cfg['threads']))
        if not performance or cfg['variant'] in ('candidate-optimized', 'candidate-eri'):
            expected_flags.update(PYSCF_DFT_DENSE_BLAS='1', PYSCF_DFT_MGGA_PACK_SIMD='1')
        if cfg['variant'] == 'coalesce':
            expected_flags['PYSCF_DFT_MGGA_COALESCE'] = '1'
        if cfg['variant'] == 'scatter':
            expected_flags['PYSCF_DFT_MGGA_SCATTER_RUNS'] = '1'
        assert cfg['flags'] == expected_flags
        expects_cache = cfg['variant'] in ('candidate-eri', 'legacy', 'fast', 'filtered')
        assert (cfg['cache_options'] is not None) == expects_cache
        if expects_cache:
            assert cfg['cache_options']['fast_lookup'] == (cfg['variant'] in ('fast', 'filtered'))
            assert cfg['cache_options']['saturation_filter'] == (cfg['variant'] == 'filtered')
        assert record['source_sha'] == cfg['sha']
        assert record['converged'] and not record['profiled']
        assert record['timed_diagnostic_counters'] is False
        assert record['effective_threads'] == cfg['threads']
        assert record['backend_hashes'] == expected_backend
        assert blas_hashes(record) == blas_hashes(reference)
        assert record['package_versions'] == reference['package_versions']
        assert record['python'] == reference['python']
        assert record['affinity'] == reference['affinity']
        assert record['grid_sha256'] == reference['grid_sha256']
        assert record['grid_arrays'] == reference['grid_arrays']
        assert record['grid_arrays_sha256'] == reference['grid_arrays_sha256']
        assert record['used_incore_eri'] == reference['used_incore_eri']
        assert record['config']['flags']['PYSCF_NUMA'] == 'off'
        assert record['nao'] == reference['nao']
        assert np.isfinite(record['scf_seconds']) and record['scf_seconds'] > 0
        for pool in record['threadpools']:
            if pool['user_api'] == 'blas':
                assert pool['num_threads'] == 1
            if pool['user_api'] == 'openmp':
                assert pool['num_threads'] == cfg['threads']
        energy_error = abs(record['energy_hartree'] - reference['energy_hartree'])
        assert np.isfinite(energy_error) and energy_error <= 1e-8
        maxima['energy'] = max(maxima['energy'], energy_error)
        path = directory / record['arrays']
        assert file_sha(path) == record['arrays_sha256']
        with np.load(path, allow_pickle=False) as arrays:
            assert all(np.isfinite(arrays[key]).all() for key in arrays.files)
            np.testing.assert_array_equal(arrays['fixed_dm'], common['fixed_dm'])
            assert array_sha(arrays['fixed_dm']) == record['fixed_dm_sha256']
            for key in ('ne', 'exc', 'vxc', 'vj', 'vk'):
                np.testing.assert_allclose(arrays[key], common[key], atol=1e-9, rtol=1e-10)
                maxima[key] = max(maxima[key], float(np.max(np.abs(arrays[key] - common[key]))))
            if cfg['cache_options'] is not None:
                assert record['uncached_jk_verified']
                for key in ('vj', 'vk'):
                    np.testing.assert_allclose(arrays[key], arrays[key + '_uncached'],
                                               atol=1e-9, rtol=1e-10)
        if record['timed_cache']:
            for cache in record['timed_cache']['caches']:
                assert cache['calls'] == cache['hits'] == cache['misses'] == cache['cost_bypass'] == 0
                assert cache['data_bytes'] <= cache['allocated_bytes'] <= cache['max_bytes']
            if cfg['force_direct'] or cfg['n'] >= 4:
                assert not record['used_incore_eri']
                assert record['timed_cache']['caches']
                assert sum(c['hits'] for c in record['cache_diagnostics']['caches']) > 0
        if cfg['require_filter_activation'] and cfg['variant'] == 'filtered':
            assert sum(c['saturation_filter']['queries']
                       for c in record['cache_diagnostics']['caches']) > 0
        if cfg['variant'] != 'upstream':
            assert sum(count for name, count in record['post_scf_xc_dispatch'].items()
                       if name.endswith('-returned1')) > 0, 'Candidate XC native path never executed'
            if cfg['n'] == 1:
                assert sum(count for name, count in record['post_scf_xc_dispatch'].items()
                           if name.startswith('dense-') and name.endswith('-returned1')) > 0
        if cfg['variant'] in ('coalesce', 'scatter'):
            assert record['post_scf_xc_dispatch'].get('sparse-ncomp4-returned1', 0) > 0, \
                'The intended sparse MGGA native path did not execute'
    samples = {}
    for variant in case['variants']:
        selected = [r for r in records if r['config']['variant'] == variant]
        assert sorted(r['config']['repeat'] for r in selected) == list(range(case['repeats']))
        samples[variant] = [r['scf_seconds'] for r in selected]
    medians = {variant: statistics.median(values) for variant, values in samples.items()}
    result = dict(case=case['case'], records=len(records), passed=True, max_abs=maxima,
                  performance_claims=performance, samples_seconds=samples,
                  median_seconds=medians,
                  rss_mib={variant: [r['max_rss_kib_at_scf_end'] / 1024 for r in records
                                     if r['config']['variant'] == variant]
                           for variant in case['variants']})
    if performance:
        result['speedup_vs_upstream'] = {v: medians['upstream'] / seconds
                                         for v, seconds in medians.items()}
        result['time_saved_percent_vs_upstream'] = {
            v: 100 * (1 - seconds / medians['upstream']) for v, seconds in medians.items()}
    return result


def audit(output):
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert not manifest['errors']
    assert len(manifest['cases']) == manifest['expected_cases']
    for case in manifest['cases']:
        for name in case['records']:
            record = json.loads((output / name).read_text())
            expected_sha = (manifest['upstream_sha'] if record['config']['variant'] == 'upstream'
                            else manifest['candidate_sha'])
            assert record['source_sha'] == record['config']['sha'] == expected_sha
    results = [validate_case(output, case, manifest['suite'] == 'performance')
               for case in manifest['cases']]
    result = dict(passed=True, suite=manifest['suite'], cases=results,
                  total_records=sum(c['records'] for c in results),
                  upstream_sha=manifest['upstream_sha'], candidate_sha=manifest['candidate_sha'],
                  gates=dict(energy_atol=1e-8, array_atol=1e-9, array_rtol=1e-10),
                  scope='SCF timings' if manifest['suite'] == 'performance'
                  else 'ERI-policy complete-SCF correctness gate; not a performance claim')
    (output / 'acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidate = Path(args.candidate_root).resolve()
    upstream = Path(args.upstream_root).resolve() if args.upstream_root else candidate
    assert source_sha(candidate) == args.candidate_sha
    if args.suite == 'performance':
        assert source_sha(upstream) == args.upstream_sha
        cases = [(n, xc, 'def2-TZVPP', 6, False, False)
                 for n in (1, 4) for xc in ('PBE', 'MN15')]
        if args.include_n8:
            cases.append((8, 'MN15', 'def2-TZVPP', 6, False, False))
        variants = ['upstream', 'candidate-default', 'candidate-optimized']
        if args.include_eri_variant:
            variants.append('candidate-eri')
        repeats = args.repeats
    else:
        cases = [(2, 'MN15', 'cc-pvdz', 2, True, False)]
        if args.saturation_gate:
            cases.append((4, 'MN15', 'def2-TZVPP', 6, False, True))
        variants = ['control', 'legacy', 'fast', 'filtered']
        repeats = 1
    manifest = dict(suite=args.suite, upstream_sha=args.upstream_sha,
                    candidate_sha=args.candidate_sha, expected_cases=len(cases), cases=[], errors=[])
    manifest_path = output / 'manifest.json'
    for n, xc, basis, grid, force_direct, saturation in cases:
        name = f'{xc}-n{n}-t{args.threads}-{basis}-grid{grid}'
        case_variants = variants + ['coalesce', 'scatter'] if saturation else variants
        case = dict(case=name, variants=case_variants, repeats=repeats, records=[])
        manifest['cases'].append(case)
        reference = None
        for repeat in range(repeats):
            # Rotate the order to balance monotonic runner drift. First record
            # is always the common-density reference for this case.
            order = (case_variants[repeat % len(case_variants):]
                     + case_variants[:repeat % len(case_variants)])
            for variant in order:
                upstream_variant = variant == 'upstream'
                root = upstream if upstream_variant else candidate
                sha = args.upstream_sha if upstream_variant else args.candidate_sha
                flags = dict(FLAGS, OMP_NUM_THREADS=str(args.threads))
                optimized = variant in ('candidate-optimized', 'candidate-eri') or args.suite == 'eri-gate'
                if optimized:
                    flags.update(PYSCF_DFT_DENSE_BLAS='1', PYSCF_DFT_MGGA_PACK_SIMD='1')
                if variant == 'coalesce':
                    flags['PYSCF_DFT_MGGA_COALESCE'] = '1'
                if variant == 'scatter':
                    flags['PYSCF_DFT_MGGA_SCATTER_RUNS'] = '1'
                options = None
                if variant in ('legacy', 'fast', 'filtered', 'candidate-eri'):
                    options = dict(max_mb=64 if saturation else 256 if args.suite == 'performance' else 1,
                                   min_primitive_work=64 if saturation or args.suite == 'performance' else 0,
                                   fast_lookup=variant in ('fast', 'filtered'),
                                   saturation_filter=variant == 'filtered')
                filename = f'{name}-{repeat}-{variant}.json'
                cfg = dict(case=name, root=str(root), sha=sha, variant=variant, repeat=repeat,
                           n=n, xc=xc, basis=basis, grid_level=grid, threads=args.threads,
                           force_direct=force_direct, require_filter_activation=saturation,
                           cache_options=options, flags=flags, reference_arrays=reference,
                           output=str(output / filename))
                config = output / filename.replace('.json', '.config.json')
                config.write_text(json.dumps(cfg, indent=2) + '\n')
                manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
                env = dict(os.environ, **flags, PYTHONPATH=str(root))
                try:
                    with (output / filename.replace('.json', '.log')).open('w') as log:
                        subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                        '--worker-config', str(config)], cwd=root, env=env,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       timeout=args.worker_timeout, check=True)
                    case['records'].append(filename)
                    if reference is None:
                        reference = str((output / filename).with_suffix('.npz'))
                    record = json.loads((output / filename).read_text())
                    print(json.dumps(dict(case=name, repeat=repeat, variant=variant,
                                          seconds=record['scf_seconds'], energy=record['energy_hartree'],
                                          rss_mib=record['max_rss_kib_at_scf_end'] / 1024)), flush=True)
                except Exception as error:
                    manifest['errors'].append(f'{filename}: {type(error).__name__}: {error}')
                    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
                    raise
                manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        # Reopen each case immediately so an error fails before later expensive work.
        validate_case(output, case, args.suite == 'performance')
    return audit(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-config')
    parser.add_argument('--audit')
    parser.add_argument('--suite', choices=('performance', 'eri-gate'), default='performance')
    parser.add_argument('--upstream-root')
    parser.add_argument('--candidate-root')
    parser.add_argument('--upstream-sha', default=UPSTREAM_SHA)
    parser.add_argument('--candidate-sha')
    parser.add_argument('--output')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--worker-timeout', type=int, default=600)
    parser.add_argument('--include-n8', action='store_true')
    parser.add_argument('--include-eri-variant', action='store_true')
    parser.add_argument('--saturation-gate', action='store_true')
    args = parser.parse_args()
    if args.worker_config:
        return worker(json.loads(Path(args.worker_config).read_text()))
    if args.audit:
        return audit(args.audit)
    if not args.candidate_root or not args.candidate_sha or not args.output:
        parser.error('--candidate-root, --candidate-sha, and --output are required')
    if args.suite == 'performance' and not args.upstream_root:
        parser.error('--upstream-root is required for the performance suite')
    if args.repeats < 1 or args.threads < 1:
        parser.error('repeats and threads must be positive')
    return run(args)


if __name__ == '__main__':
    main()
