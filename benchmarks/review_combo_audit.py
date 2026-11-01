"""Independent, fail-closed audit of an ordinary combination-search SCF record.

Run on the measurement runner while its pinned checkouts and libraries still
exist. Every call reopens the record NPZ and grid NPZ. No PySCF import is needed.
The reference is the first common-density record for this molecular case, not
the latest upstream repetition. Timings are never used to waive correctness.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np


ENERGY_ATOL = 1e-8
ARRAY_ATOL = 1e-9
ARRAY_RTOL = 1e-10
MIB = 1024 ** 2
# The Python observation precedes the context's own current_memory() call.
# This is an observation tolerance, not a relaxation of either requested budget.
HEADROOM_OBSERVATION_BYTES = 2 * MIB
SWITCHES = (
    'PYSCF_DFT_DENSE_BLAS', 'PYSCF_DFT_MGGA_PACK_SIMD',
    'PYSCF_DFT_VMAT_TILE_SCREEN', 'PYSCF_DFT_MGGA_COALESCE',
    'PYSCF_DFT_MGGA_COALESCE_STATS', 'PYSCF_DFT_MGGA_SCATTER_RUNS',
    'PYSCF_JK_BLOCK_BOUND', 'PYSCF_JK_BLOCK_BOUND_STATS',
)
BLAS_FLAGS = ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS')
REQUIRED_FLAGS = set(SWITCHES + BLAS_FLAGS + (
    'PYSCF_NUMA', 'OMP_NUM_THREADS', 'OMP_DYNAMIC', 'OMP_MAX_ACTIVE_LEVELS'))
REQUIRED_ARRAYS = {'dm', 'fixed_dm', 'ne', 'exc', 'vxc', 'vj', 'vk',
                   'vj_uncached', 'vk_uncached'}
AO_FIELDS = ('calls', 'hits', 'misses', 'bypasses', 'admissions', 'invalidations',
             'retained_bytes', 'peak_bytes', 'max_bytes', 'entries')
ERI_FIELDS = ('calls', 'hits', 'misses', 'cost_bypass', 'data_bytes',
              'allocated_bytes', 'max_bytes', 'entries')
DISPATCH = re.compile(r'^(dense|sparse)-ncomp(1|2|4)-returned([01])$')


class AuditError(AssertionError):
    """The measurement is ineligible for ranking."""


def _require(condition, message):
    if not condition:
        raise AuditError(message)


def _integer(value, label, minimum=0):
    _require(isinstance(value, int) and not isinstance(value, bool)
             and value >= minimum, f'{label}: expected integer >= {minimum}')
    return value


def _number(value, label, minimum=0, positive=False):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(value) and value >= minimum
             and (not positive or value > 0), f'{label}: invalid finite number')
    return value


def _sha(value, label, length=64):
    _require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{%d}' % length, value),
             f'{label}: invalid hash')
    return value


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(MIB), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _exact_array(actual, expected, label):
    _require(actual.shape == expected.shape and actual.dtype.str == expected.dtype.str,
             f'{label}: shape or dtype differs')
    _require(np.ascontiguousarray(actual).tobytes() == np.ascontiguousarray(expected).tobytes(),
             f'{label}: bytes differ')


def _allclose(actual, expected, label):
    _require(actual.shape == expected.shape, f'{label}: shapes differ')
    try:
        np.testing.assert_allclose(actual, expected, atol=ARRAY_ATOL, rtol=ARRAY_RTOL)
    except AssertionError as error:
        raise AuditError(f'{label}: numerical tolerance failed') from error
    return float(np.max(np.abs(actual - expected)))


def _artifact(record_path, name):
    _require(isinstance(name, str) and bool(name), 'Missing artifact name')
    path = (record_path.parent / name).resolve()
    _require(path.is_relative_to(record_path.parent), f'Artifact escapes record directory: {name}')
    _require(path.is_file(), f'Missing artifact: {path}')
    return path


def _read_npz(path, required):
    with np.load(path, allow_pickle=False) as archive:
        _require(required <= set(archive.files), f'{path.name}: missing arrays')
        values = {key: archive[key] for key in archive.files}
    for key, value in values.items():
        _require(np.issubdtype(value.dtype, np.number) and value.size > 0,
                 f'{path.name}/{key}: empty or nonnumeric array')
        _require(bool(np.isfinite(value).all()), f'{path.name}/{key}: nonfinite array')
    return values


def _profile(record):
    configured = record['config'].get('build_profile') or 'portable'
    _require(isinstance(configured, str), 'Invalid build_profile')
    _require((record.get('build_profile') or 'portable') == configured,
             'Recorded build_profile disagrees with config')
    return configured


def _validate_record(path, expected_cfg=None):
    record = json.loads(path.read_text())
    cfg = record['config']
    if expected_cfg is not None:
        for key, value in expected_cfg.items():
            _require(key in cfg and cfg[key] == value, f'Config mismatch: {key}')
    root = Path(cfg['root']).resolve()
    _require(root.is_dir(), f'Missing pinned source checkout: {root}')
    _require(Path(cfg['output']).resolve() == path, 'Config output is not this record')
    expected_sha = _sha(cfg['sha'], 'config.sha', length=40)
    _require(record['source_sha'] == expected_sha, 'Source SHA disagrees with config')
    actual_sha = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    _require(actual_sha == expected_sha, 'Pinned checkout HEAD changed')
    source_file = Path(record['source_file']).resolve()
    _require(source_file == (root / 'pyscf' / '__init__.py').resolve() and source_file.is_file(),
             'PySCF import did not come from the selected checkout')
    _profile(record)

    threads = _integer(cfg['threads'], 'threads', minimum=1)
    blas = _integer(cfg['blas_threads'], 'blas_threads', minimum=1)
    _integer(cfg['n'], 'n', minimum=1)
    _integer(cfg['grid_level'], 'grid_level')
    _number(cfg['max_memory'], 'max_memory', positive=True)
    _number(cfg['ao_cache_mb'], 'ao_cache_mb')
    _require(isinstance(cfg['force_direct'], bool), 'force_direct must be boolean')
    flags = cfg['flags']
    _require(isinstance(flags, dict) and REQUIRED_FLAGS <= set(flags), 'Incomplete effective flag contract')
    _require(all(isinstance(k, str) and isinstance(v, str) for k, v in flags.items()),
             'Flags must use string keys and values')
    _require(record['effective_flags'] == flags, 'Observed environment differs from declared flags')
    _require(all(flags[key] in ('0', '1') for key in SWITCHES), 'Invalid binary option flag')
    _require(flags['PYSCF_NUMA'] == 'off', 'NUMA must be explicitly off')
    _require(flags['OMP_NUM_THREADS'] == str(threads), 'OMP flag does not match requested threads')
    _require(all(flags[key] == str(blas) for key in BLAS_FLAGS), 'BLAS flags do not match requested threads')
    _require(flags['OMP_DYNAMIC'] == 'FALSE' and flags['OMP_MAX_ACTIVE_LEVELS'] == '1',
             'OpenMP dynamic/nested team policy changed')
    for key in ('PYSCF_DFT_MGGA_COALESCE_STATS', 'PYSCF_JK_BLOCK_BOUND_STATS'):
        _require(flags[key] == '0', f'Diagnostic counter enabled during timing: {key}')
    if 'PYSCF_DFT_VMAT_MAX_MEMORY_MB' in flags:
        _require(flags['PYSCF_DFT_VMAT_MAX_MEMORY_MB'].isdigit()
                 and int(flags['PYSCF_DFT_VMAT_MAX_MEMORY_MB']) > 0, 'Invalid Vmat workspace budget')
    _require(record['timed_diagnostic_counters'] is False and record['profiled'] is False,
             'Profiled or counter-instrumented timing is not rankable')
    _require(record['ao_intrinsic_cache_counters'] == bool(cfg['ao_cache_mb']),
             'AO intrinsic counter provenance mismatch')
    _require(record['converged'] is True and math.isfinite(record['energy_hartree']), 'Unconverged/nonfinite energy')
    _number(record['scf_seconds'], 'scf_seconds', positive=True)
    _number(record['max_rss_kib_at_scf_end'], 'SCF RSS', positive=True)
    _number(record['memory_mb_before_scf'], 'memory before SCF')
    _number(record['memory_mb_after_scf'], 'memory after SCF')
    _integer(record['cycles'], 'cycles', minimum=-1)
    nao = _integer(record['nao'], 'nao', minimum=1)
    points = _integer(record['grid_points'], 'grid_points', minimum=1)
    _require(isinstance(record['used_incore_eri'], bool), 'Missing in-core strategy observation')
    if cfg['force_direct']:
        _require(not record['used_incore_eri'], 'Forced-direct worker used in-core ERIs')
    settings = dict(conv_tol=1e-9, max_cycle=80, direct_scf_tol=1e-13,
                    max_memory_mb=cfg['max_memory'], force_direct=cfg['force_direct'])
    _require(record['scientific_settings'] == settings, 'Scientific convergence/settings changed')

    _require(record['effective_threads'] == threads, 'Effective OpenMP threads differ')
    _require(record['requested_blas_threads'] == blas, 'Requested BLAS thread provenance differs')
    pools = record['threadpools']
    blas_pools = [p for p in pools if p['user_api'] == 'blas']
    omp_pools = [p for p in pools if p['user_api'] == 'openmp']
    _require(bool(blas_pools), 'No BLAS runtime observation')
    _require(threads == 1 or bool(omp_pools), 'No OpenMP runtime observation for a threaded run')
    _require(all(p['num_threads'] == blas for p in blas_pools), 'Effective BLAS runtime threads differ')
    _require(all(p['num_threads'] == threads for p in omp_pools), 'Effective OpenMP runtime threads differ')
    _require(record['effective_blas_threads'] == [blas], 'Effective BLAS summary differs')
    for pool in blas_pools:
        _sha(pool['sha256'], 'BLAS library hash')
        _require(_file_sha(pool['filepath']) == pool['sha256'], 'BLAS library hash changed')
    affinity = record['affinity']
    _require(isinstance(affinity, list) and affinity
             and all(isinstance(cpu, int) and not isinstance(cpu, bool) and cpu >= 0 for cpu in affinity)
             and affinity == sorted(set(affinity)), 'Invalid CPU affinity')
    _require(isinstance(record['python'], str) and bool(record['python']), 'Missing Python provenance')
    for name in ('numpy', 'scipy', 'h5py', 'threadpoolctl'):
        _require(isinstance(record['package_versions'].get(name), str)
                 and bool(record['package_versions'][name]), f'Missing package version: {name}')
    libraries = record['native_libraries']
    _require(isinstance(libraries, dict) and bool(libraries), 'Missing loaded native libraries')
    for name, digest in libraries.items():
        _sha(digest, 'Native library hash')
        _require(_file_sha(name) == digest, f'Native library changed: {name}')
    backend = {prefix: sorted({digest for name, digest in libraries.items()
                               if Path(name).name.startswith(prefix)})
               for prefix in ('libcint.', 'libxc.')}
    _require(all(backend.values()) and record['backend_hashes'] == backend, 'Backend provenance mismatch')

    arrays_path = _artifact(path, record['arrays'])
    grid_path = _artifact(path, record['grid_arrays'])
    _require(_file_sha(arrays_path) == _sha(record['arrays_sha256'], 'Array archive hash'), 'Array archive hash mismatch')
    _require(_file_sha(grid_path) == _sha(record['grid_arrays_sha256'], 'Grid archive hash'), 'Grid archive hash mismatch')
    arrays = _read_npz(arrays_path, REQUIRED_ARRAYS)
    grid = _read_npz(grid_path, {'coords', 'weights'})
    for key in ('dm', 'fixed_dm', 'vxc', 'vj', 'vk', 'vj_uncached', 'vk_uncached'):
        _require(arrays[key].shape == (nao, nao), f'{key}: unexpected RKS matrix shape')
    _require(arrays['ne'].size == arrays['exc'].size == 1, 'Expected scalar RKS Ne and Exc')
    _require(grid['coords'].shape == (points, 3) and grid['weights'].shape == (points,), 'Invalid grid shapes')
    _require(_array_sha(grid['coords'], grid['weights']) == record['grid_sha256'], 'Grid data hash mismatch')
    _require(_array_sha(arrays['fixed_dm']) == record['fixed_dm_sha256'], 'Fixed-DM hash mismatch')
    return record, arrays, grid, arrays_path


def _budget(record, field, requested_mib, fraction, budget):
    _integer(budget, 'Cache budget')
    _require(budget <= int(requested_mib * MIB), 'Cache exceeds requested byte budget')
    entry_memory = _number(record[field], field)
    nominal = max(0, record['config']['max_memory'] - entry_memory) * 1e6 * fraction
    _require(budget <= nominal + HEADROOM_OBSERVATION_BYTES, 'Cache exceeds observed memory headroom cap')
    return dict(requested_mib=requested_mib, budget_bytes=budget,
                entry_memory_mb=entry_memory, nominal_headroom_cap_bytes=nominal,
                headroom_observation_tolerance_bytes=HEADROOM_OBSERVATION_BYTES)


def _ao_stats(stats, label):
    _require(isinstance(stats, dict), f'{label}: missing cache snapshot')
    for key in AO_FIELDS:
        _integer(stats[key], f'{label}.{key}')
    _require(stats['calls'] == stats['hits'] + stats['misses'] + stats['bypasses'], f'{label}: call accounting mismatch')
    _require(stats['admissions'] <= stats['misses'] and stats['entries'] <= min(512, stats['admissions']),
             f'{label}: admission/entry accounting mismatch')
    _require(stats['retained_bytes'] <= stats['peak_bytes'] <= stats['max_bytes'], f'{label}: byte budget exceeded')


def _eri_stats(stats, label, options, timed=False):
    _require(isinstance(stats, dict), f'{label}: missing scope snapshot')
    budget = _integer(stats['budget_bytes'], f'{label}.budget_bytes')
    caches = stats['caches']
    _require(isinstance(caches, list) and len(caches) <= 1, f'{label}: replicated ERI scope budget')
    for cache in caches:
        _require(isinstance(cache['enabled'], bool), f'{label}: invalid enabled state')
        for key in ERI_FIELDS:
            _integer(cache[key], f'{label}.{key}')
        _require(cache['data_bytes'] <= cache['allocated_bytes'] <= cache['max_bytes'] <= budget,
                 f'{label}: byte budget exceeded')
        if timed:
            _require(all(cache[key] == 0 for key in ERI_FIELDS[:4]), f'{label}: timed counters enabled')
        lookup = cache.get('lookup', {})
        filtering = cache.get('saturation_filter', {})
        for name, fields in (('lookup', lookup), ('filter', filtering)):
            for key, value in fields.items():
                _integer(value, f'{label}.{name}.{key}')
        if cache['enabled']:
            _require(lookup.get('fast_lookup') == int(options.get('fast_lookup', False)),
                     f'{label}: wrong ERI lookup policy')
            _require(bool(filtering.get('bytes', 0)) == options.get('saturation_filter', False),
                     f'{label}: wrong saturation-filter policy')
        if timed:
            _require(all(value == 0 for key, value in lookup.items() if key != 'fast_lookup'),
                     f'{label}: lookup counters enabled during timing')
            _require(all(value == 0 for key, value in filtering.items() if key != 'bytes'),
                     f'{label}: filter counters enabled during timing')
        if filtering:
            _require(filtering['bytes'] <= cache['allocated_bytes'], f'{label}: filter exceeds allocated budget')
            _require(filtering['recompute_bypasses'] <= filtering['negatives'] <= filtering['queries'],
                     f'{label}: inconsistent filter accounting')
    return caches


def _cache_audit(record, arrays):
    cfg = record['config']
    ao = dict(requested=bool(cfg['ao_cache_mb']), active_in_scf=False)
    if ao['requested']:
        timed, diagnostic = record['timed_ao_cache'], record['ao_cache_diagnostics']
        _ao_stats(timed, 'timed AO'); _ao_stats(diagnostic, 'diagnostic AO')
        _require(timed['max_bytes'] == diagnostic['max_bytes'], 'AO budget changed within scope')
        for key in ('calls', 'hits', 'misses', 'bypasses', 'admissions', 'invalidations', 'peak_bytes'):
            _require(diagnostic[key] >= timed[key], f'AO cumulative counter decreased: {key}')
        ao.update(_budget(record, 'ao_cache_entry_memory_mb', cfg['ao_cache_mb'], .5, timed['max_bytes']))
        ao.update(snapshot=timed, active_in_scf=timed['hits'] > 0 and timed['admissions'] > 0,
                  hit_fraction=timed['hits'] / timed['calls'] if timed['calls'] else 0.0)
    else:
        _require(record['timed_ao_cache'] is None and record['ao_cache_diagnostics'] is None,
                 'Undeclared AO cache scope')

    options = cfg['cache_options']
    eri = dict(requested=options is not None, attached_in_scf=False, populated_in_scf=False,
               diagnostic_reuse_observed=False, filter_queries_observed=False,
               timed_hits=None, inactive_because_incore=options is not None and record['used_incore_eri'])
    if options is not None:
        _require(isinstance(options, dict) and set(options) <= {'max_mb', 'min_primitive_work', 'fast_lookup', 'saturation_filter'},
                 'Invalid ERI cache option keys')
        requested = _number(options.get('max_mb', 256.), 'ERI max_mb')
        minimum = _integer(options.get('min_primitive_work', 64), 'ERI admission work')
        _require(minimum <= 2**32 - 1, 'ERI admission work exceeds uint32')
        for key in ('fast_lookup', 'saturation_filter'):
            _require(isinstance(options.get(key, False), bool), f'Invalid {key}')
        _require(not options.get('saturation_filter', False) or options.get('fast_lookup', False),
                 'Saturation filter requires fast lookup')
        timed = _eri_stats(record['timed_cache'], 'timed ERI', options, timed=True)
        diagnostic = _eri_stats(record['cache_diagnostics'], 'diagnostic ERI', options)
        _require(record['timed_cache']['budget_bytes'] == record['cache_diagnostics']['budget_bytes'], 'ERI budget changed')
        eri.update(_budget(record, 'eri_cache_entry_memory_mb', requested, .25, record['timed_cache']['budget_bytes']))
        hits = sum(c['hits'] for c in diagnostic)
        queries = sum(c.get('saturation_filter', {}).get('queries', 0) for c in diagnostic)
        entries = sum(c['entries'] for c in timed)
        eri.update(attached_in_scf=any(c['enabled'] for c in timed), populated_in_scf=entries > 0,
                   entries_at_scf_end=entries, diagnostic_hits=hits, diagnostic_reuse_observed=hits > 0,
                   filter_queries=queries, filter_queries_observed=queries > 0,
                   allocated_bytes_at_scf_end=sum(c['allocated_bytes'] for c in timed))
    else:
        _require(record['timed_cache'] is None and record['cache_diagnostics'] is None, 'Undeclared ERI cache scope')
    independent = options is not None or cfg['flags']['PYSCF_JK_BLOCK_BOUND'] == '1'
    _require(record['uncached_jk_verified'] == record['independent_jk_verified'] == independent,
             'Independent J/K verification provenance mismatch')
    errors = {}
    if independent:
        for key in ('vj', 'vk'):
            errors[key] = _allclose(arrays[key], arrays[key + '_uncached'], f'Independent uncached {key}')
    return dict(ao=ao, eri=eri), errors


def _activation(record):
    flags = record['config']['flags']
    dispatch = record['post_scf_xc_dispatch']
    counts = dict(dense=0, sparse=0, mgga=0, sparse_mgga=0, native_success=0, native_return_zero=0)
    for key, value in dispatch.items():
        match = DISPATCH.fullmatch(key)
        _require(match is not None, f'Invalid dispatch counter: {key}')
        _integer(value, 'dispatch count')
        kind, component, done = match.groups()
        if done == '0':
            counts['native_return_zero'] += value
            continue
        counts[kind] += value; counts['native_success'] += value
        if component == '4':
            counts['mgga'] += value
            if kind == 'sparse': counts['sparse_mgga'] += value
    coal = record['post_scf_coalesce']
    _require(isinstance(coal.get('available'), bool), 'Missing coalescing diagnostic availability')
    if coal['available']:
        for key in ('raw_blocks', 'batches', 'joined', 'widest'):
            _integer(coal[key], f'coalescing.{key}')
        _require(coal['raw_blocks'] == coal['batches'] + coal['joined'] and coal['widest'] <= 4,
                 'Inconsistent coalescing diagnostic')
        if flags['PYSCF_DFT_MGGA_COALESCE'] == '0':
            _require(coal['joined'] == 0 and coal['widest'] <= 1, 'Undeclared coalescing occurred')
        if coal['joined']:
            _require(counts['sparse_mgga'] > 0, 'Coalescing lacks eligible native call')
    mapping = {'dense_blas': ('PYSCF_DFT_DENSE_BLAS', 'dense'),
               'mgga_simd': ('PYSCF_DFT_MGGA_PACK_SIMD', 'mgga'),
               'tile_screen': ('PYSCF_DFT_VMAT_TILE_SCREEN', 'sparse'),
               'coalesce': ('PYSCF_DFT_MGGA_COALESCE', 'sparse_mgga'),
               'scatter': ('PYSCF_DFT_MGGA_SCATTER_RUNS', 'sparse_mgga')}
    result = {}
    for name, (flag, path) in mapping.items():
        requested = flags[flag] == '1'
        result[name] = dict(requested=requested, eligible_native_call_observed=requested and counts[path] > 0,
                            inactive_for_observed_path=requested and counts[path] == 0)
    result['coalesce']['actual_join_observed'] = bool(coal.get('joined', 0))
    result['coalesce']['diagnostic'] = coal
    bound = record['post_scf_jk_bound']
    for label, counter in bound.items():
        if counter is None: continue
        _require(isinstance(counter.get('available'), bool), f'Invalid {label} bound availability')
        if counter['available']:
            _integer(counter['visits'], f'{label}.visits'); _integer(counter['rejects'], f'{label}.rejects')
            _require(counter['rejects'] <= counter['visits'], 'Invalid J/K bound accounting')
    for key, value in record['post_scf_block_shapes'].items():
        _require(isinstance(key, str), 'Invalid diagnostic block shape')
        _integer(value, 'block shape count', minimum=1)
    result.update(dispatch=dispatch, counts=counts, jk_bound=bound,
                  scope='Post-SCF common-density observations; not counters from timed SCF',
                  limitation='Eligible dispatch does not prove every tile used an option or that a tile was skipped')
    return result


def audit_record(record_path, reference_path, expected_cfg=None):
    """Reopen and audit one record; raise AuditError on every invalid gate.

    Thread/BLAS counts, cache settings, memory limits and direct/in-core strategy
    may intentionally differ. The returned strategy_difference exposes them.
    Backend hashes must match when build_profile matches. A different build
    profile is reported explicitly and never silently treated as a runtime-only
    option change. No failure is converted into a successful ranking record.
    """
    path, ref_path = Path(record_path).resolve(), Path(reference_path).resolve()
    record, arrays, grid, arrays_path = _validate_record(path, expected_cfg)
    if path == ref_path:
        reference, ref_arrays, ref_grid, ref_arrays_path = record, arrays, grid, arrays_path
    else:
        reference, ref_arrays, ref_grid, ref_arrays_path = _validate_record(ref_path)
    cfg, ref_cfg = record['config'], reference['config']
    for key in ('n', 'xc', 'basis', 'grid_level'):
        _require(cfg[key] == ref_cfg[key], f'Molecular reference mismatch: {key}')
    _require(record['nao'] == reference['nao'], 'Reference AO count differs')
    for key in ('python', 'package_versions', 'affinity'):
        _require(record[key] == reference[key], f'Environment differs: {key}')
    same_profile = _profile(record) == _profile(reference)
    blas_hashes = lambda r: sorted({p['sha256'] for p in r['threadpools'] if p['user_api'] == 'blas'})
    if same_profile:
        _require(record['backend_hashes'] == reference['backend_hashes'], 'Backend hashes differ within build profile')
        _require(blas_hashes(record) == blas_hashes(reference), 'BLAS hashes differ within build profile')
    for key in ('coords', 'weights'):
        _exact_array(grid[key], ref_grid[key], f'Common grid {key}')
    _require(record['grid_sha256'] == reference['grid_sha256'], 'Reference grid hash differs')
    _exact_array(arrays['fixed_dm'], ref_arrays['dm'], 'Common fixed density matrix')
    if cfg.get('reference_arrays'):
        _require(Path(cfg['reference_arrays']).resolve() == ref_arrays_path, 'Wrong common-DM archive configured')
    else:
        _require(path == ref_path, 'Non-reference worker lacks common-DM input')
    energy_error = abs(record['energy_hartree'] - reference['energy_hartree'])
    _require(energy_error <= ENERGY_ATOL, 'Total SCF energy tolerance failed')
    errors = dict(energy=energy_error)
    for key in ('ne', 'exc', 'vxc', 'vj', 'vk'):
        errors[key] = _allclose(arrays[key], ref_arrays[key], f'Common-density {key}')
    caches, independent_errors = _cache_audit(record, arrays)
    activation = _activation(record)
    strategy_keys = ('threads', 'blas_threads', 'max_memory', 'force_direct')
    differences = {key: dict(reference=ref_cfg[key], candidate=cfg[key])
                   for key in strategy_keys if cfg[key] != ref_cfg[key]}
    if record['used_incore_eri'] != reference['used_incore_eri']:
        differences['used_incore_eri'] = dict(reference=reference['used_incore_eri'], candidate=record['used_incore_eri'])
    if not same_profile:
        differences['build_profile'] = dict(reference=_profile(reference), candidate=_profile(record))
    return dict(passed=True, record=str(path), reference=str(ref_path), source_sha=record['source_sha'],
                case={key: cfg[key] for key in ('n', 'xc', 'basis', 'grid_level')},
                build_profile=_profile(record), same_build_profile_as_reference=same_profile,
                backend_hashes=record['backend_hashes'], blas_hashes=blas_hashes(record),
                seconds=record['scf_seconds'], rss_mib=record['max_rss_kib_at_scf_end'] / 1024,
                cycles=record['cycles'], energy_hartree=record['energy_hartree'],
                max_abs=errors, independent_jk_max_abs=independent_errors,
                cacheactive=caches, activation=activation, strategy_difference=differences,
                threads=cfg['threads'], blas_threads=cfg['blas_threads'], used_incore_eri=record['used_incore_eri'],
                raw_arrays_reopened=True, raw_grid_arrays_reopened=True,
                gates=dict(energy_atol=ENERGY_ATOL, array_atol=ARRAY_ATOL, array_rtol=ARRAY_RTOL))
