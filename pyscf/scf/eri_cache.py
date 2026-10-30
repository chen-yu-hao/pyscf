# Copyright 2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Optional, bounded FP64 integral reuse for molecular direct RHF/RKS.

Only density-independent, ordinary real spherical Coulomb ERIs are retained.
Native screening and contraction remain unchanged. Cache lookup is in C, not a
Python per-quartet callback. No cache is enabled by importing this module.
"""
from contextlib import contextmanager
import ctypes
import math
import threading
import numpy as np
from pyscf import lib, gto
from pyscf.scf import _vhf

_PTR = ctypes.c_void_p
_lib = _vhf.libcvhf
_lib.CVHFeri_cache_create.restype = _PTR
_lib.CVHFeri_cache_create.argtypes = [_PTR, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    _PTR, _PTR, _PTR, _PTR, ctypes.c_size_t, ctypes.c_uint]
_lib.CVHFeri_cache_release.argtypes = [_PTR]
_lib.CVHFeri_cache_release.restype = None
_lib.CVHFeri_cache_stats.argtypes = [_PTR, _PTR, ctypes.c_int, ctypes.c_int]
_lib.CVHFeri_cache_stats.restype = None
_fast_create = getattr(_lib, 'CVHFeri_cache_create_fast', None)
_lookup_stats = getattr(_lib, 'CVHFeri_cache_lookup_stats', None)
_filter_create = getattr(_lib, 'CVHFeri_cache_create_filtered', None)
_filter_stats = getattr(_lib, 'CVHFeri_cache_filter_stats', None)
if _fast_create is not None:
    _fast_create.restype = _PTR
    _fast_create.argtypes = _lib.CVHFeri_cache_create.argtypes
if _lookup_stats is not None:
    _lookup_stats.argtypes = [_PTR, _PTR]
    _lookup_stats.restype = None

if _filter_create is not None:
    _filter_create.restype = _PTR
    _filter_create.argtypes = _lib.CVHFeri_cache_create.argtypes
if _filter_stats is not None:
    _filter_stats.argtypes = [_PTR, _PTR]
    _filter_stats.restype = None


class _IntegralCache:
    def __init__(self, opt, mol, max_bytes, min_work=64, fast_lookup=False, saturation_filter=False):
        # Keep the optimizer alive until the native registry entry is removed.
        self.opt = opt
        self.ptr = None
        self.buffers = None
        self.lock = threading.RLock()
        if opt._intor != 'int2e_sph' or mol._env[gto.PTR_RANGE_OMEGA] != 0:
            return
        atm = np.ascontiguousarray(mol._atm, dtype=np.int32)
        bas = np.ascontiguousarray(mol._bas, dtype=np.int32)
        env = np.ascontiguousarray(mol._env, dtype=np.float64)
        loc = np.ascontiguousarray(mol.ao_loc_nr(), dtype=np.int32)
        self.buffers = (atm, bas, env, loc, mol)
        create = (_filter_create if saturation_filter else
                  _fast_create if fast_lookup else _lib.CVHFeri_cache_create)
        self.ptr = create(
            opt._cintopt, len(atm), len(bas), len(env), atm.ctypes.data,
            bas.ctypes.data, env.ctypes.data, loc.ctypes.data, max_bytes, min_work)

    def stats(self, diagnostics=None, reset=False):
        with self.lock:
            if not self.ptr:
                return dict(enabled=False, calls=0, hits=0, misses=0, cost_bypass=0,
                            data_bytes=0, allocated_bytes=0, max_bytes=0, entries=0)
            lookup = None
            if _lookup_stats is not None:
                lookup = np.zeros(5, dtype=np.uint64)
                _lookup_stats(self.ptr, lookup.ctypes.data)
            filtering = None
            if _filter_stats is not None:
                filtering = np.zeros(4, dtype=np.uint64)
                _filter_stats(self.ptr, filtering.ctypes.data)
            out = np.zeros(8, dtype=np.uint64)
            _lib.CVHFeri_cache_stats(self.ptr, out.ctypes.data,
                                    -1 if diagnostics is None else int(diagnostics), int(reset))
            names = ('calls', 'hits', 'misses', 'cost_bypass', 'data_bytes',
                     'allocated_bytes', 'max_bytes', 'entries')
            result = dict(enabled=True, **dict(zip(names, map(int, out))))
            if lookup is not None:
                result['lookup'] = dict(zip(('probes', 'empty_stops', 'reserve_failures',
                                             'tombstone_reuses', 'fast_lookup'), map(int, lookup)))
            if filtering is not None:
                result['saturation_filter'] = dict(zip(('queries', 'negatives',
                                                        'recompute_bypasses', 'bytes'),
                                                       map(int, filtering)))
            return result

    def close(self):
        with self.lock:
            if self.ptr:
                _lib.CVHFeri_cache_release(self.ptr)
                self.ptr = None
            self.opt = None
            self.buffers = None


class _Scope:
    def __init__(self, mf, max_mb, min_work, fast_lookup, saturation_filter):
        self.mf = mf
        available = max(0., float(mf.max_memory)-lib.current_memory()[0])
        self.max_bytes = int(min(max_mb*1024*1024, .25*available*1e6))
        self.min_work = min_work
        self.fast_lookup = fast_lookup
        self.saturation_filter = saturation_filter
        self.caches = []
        self.original_init = mf.init_direct_scf
        self.lock = threading.RLock()

    def attach(self, opt, mol):
        with self.lock:
            # At most one full-range cache per scope; do not replicate budget
            # when range-separated or replaced optimizers are constructed.
            if self.caches or self.max_bytes < 65536:
                return opt
            if not isinstance(opt, _vhf._VHFOpt) or opt._intor != 'int2e_sph':
                return opt
            if mol._env[gto.PTR_RANGE_OMEGA] == 0:
                self.caches.append(_IntegralCache(opt, mol, self.max_bytes, self.min_work, self.fast_lookup, self.saturation_filter))
            return opt

    def init(self, mol=None):
        if mol is None:
            mol = self.mf.mol
        return self.attach(self.original_init(mol), mol)

    def stats(self, diagnostics=None, reset=False):
        with self.lock:
            return dict(budget_bytes=self.max_bytes,
                        caches=[c.stats(diagnostics, reset) for c in self.caches],
                        scope='Counters only when explicitly enabled; bytes retained across SCF')

    def close(self):
        with self.lock:
            for c in self.caches:
                c.close()


@contextmanager
def cached_eri(mf, max_mb=256., min_primitive_work=64, fast_lookup=False,
               saturation_filter=False):
    """Temporarily reuse a bounded subset of real full-range spherical ERIs.

    ``max_mb`` is an allocation budget (MiB), capped to one quarter of current
    PySCF memory headroom; it is NOT a total-process RSS limit. The cache only
    attaches if the normal SCF selects direct J/K, leaving in-core selection
    untouched. Its cold-fill cost is included in the ordinary SCF timing.

    ``fast_lookup=True`` opts into precomputed shell metadata and early-stop
    hash lookup. Failed publications retain reusable tombstones, preventing
    concurrent collisions from hiding previously cached entries. No integral
    or density cutoff changes. The original policy remains the default.

    ``saturation_filter=True`` (requires ``fast_lookup=True``) uses a compact
    membership filter only when a new nonzero integral cannot fit. A definite
    cache miss is recomputed normally without scanning the large hash index.
    Possible hits still require an exact key check. Filter memory comes out of
    the same budget; cold filling includes atomic filter updates. Default OFF.

    ``min_primitive_work`` affects admission only, never integral screening.
    Concurrent calculations with different optimizers have independent caches.
    Concurrent kernels on the *same* SCF object and nested scopes are unsupported.
    Geometry/basis/environment mismatches bypass the native cache entirely.
    """
    from pyscf.scf.hf import RHF
    from pyscf.dft.rks import RKS
    if type(mf) not in (RHF, RKS) or mf.mol.cart or not mf.direct_scf:
        raise ValueError('cached_eri requires ordinary direct spherical RHF or RKS')
    if not math.isfinite(max_mb) or max_mb < 0:
        raise ValueError('max_mb must be finite and nonnegative')
    if not isinstance(min_primitive_work, int) or not 0 <= min_primitive_work <= 2**32-1:
        raise ValueError('min_primitive_work must be a uint32 integer')
    if '_eri_cache_scope' in mf.__dict__:
        raise RuntimeError('nested integral-cache scopes are not supported')
    if not isinstance(fast_lookup, bool):
        raise TypeError('fast_lookup must be a bool')
    if fast_lookup and (_fast_create is None or _lookup_stats is None):
        raise RuntimeError('fast_lookup requires rebuilding the native libcvhf library')
    if not isinstance(saturation_filter, bool):
        raise TypeError('saturation_filter must be a bool')
    if saturation_filter and not fast_lookup:
        raise ValueError('saturation_filter requires fast_lookup=True')
    if saturation_filter and (_filter_create is None or _filter_stats is None):
        raise RuntimeError('saturation_filter requires rebuilding libcvhf')
    scope = _Scope(mf, max_mb, min_primitive_work, fast_lookup, saturation_filter)
    had_init = 'init_direct_scf' in mf.__dict__
    saved_init = mf.__dict__.get('init_direct_scf')
    mf.__dict__['_eri_cache_scope'] = scope
    mf.init_direct_scf = scope.init
    try:
        opt = mf._opt.get(None)
        if opt is not None:
            scope.attach(opt, mf.mol)
        yield scope
    finally:
        if had_init:
            mf.init_direct_scf = saved_init
        else:
            mf.__dict__.pop('init_direct_scf', None)
        mf.__dict__.pop('_eri_cache_scope', None)
        scope.close()
