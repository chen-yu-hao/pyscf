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


class _IntegralCache:
    def __init__(self, opt, mol, max_bytes, min_work=64):
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
        self.ptr = _lib.CVHFeri_cache_create(
            opt._cintopt, len(atm), len(bas), len(env), atm.ctypes.data,
            bas.ctypes.data, env.ctypes.data, loc.ctypes.data, max_bytes, min_work)

    def stats(self, diagnostics=None, reset=False):
        with self.lock:
            if not self.ptr:
                return dict(enabled=False, calls=0, hits=0, misses=0, cost_bypass=0,
                            data_bytes=0, allocated_bytes=0, max_bytes=0, entries=0)
            out = np.zeros(8, dtype=np.uint64)
            _lib.CVHFeri_cache_stats(self.ptr, out.ctypes.data,
                                    -1 if diagnostics is None else int(diagnostics), int(reset))
            names = ('calls', 'hits', 'misses', 'cost_bypass', 'data_bytes',
                     'allocated_bytes', 'max_bytes', 'entries')
            return dict(enabled=True, **dict(zip(names, map(int, out))))

    def close(self):
        with self.lock:
            if self.ptr:
                _lib.CVHFeri_cache_release(self.ptr)
                self.ptr = None
            self.opt = None
            self.buffers = None


class _Scope:
    def __init__(self, mf, max_mb, min_work):
        self.mf = mf
        available = max(0., float(mf.max_memory)-lib.current_memory()[0])
        self.max_bytes = int(min(max_mb, .25*available)*1024*1024)
        self.min_work = min_work
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
                self.caches.append(_IntegralCache(opt, mol, self.max_bytes, self.min_work))
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
def cached_eri(mf, max_mb=256., min_primitive_work=64):
    """Temporarily reuse a bounded subset of real full-range spherical ERIs.

    ``max_mb`` is an allocation budget (MiB), capped to one quarter of current
    PySCF memory headroom; it is NOT a total-process RSS limit. The cache only
    attaches if the normal SCF selects direct J/K, leaving in-core selection
    untouched. Its cold-fill cost is included in the ordinary SCF timing.

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
    scope = _Scope(mf, max_mb, min_primitive_work)
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
