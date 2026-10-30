# Copyright 2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Opt-in bounded reuse of *density-independent* molecular AO collocation.

Independent implementation of memoization, inspired by the collocation-cache
idea documented in Psi4. No external source code is incorporated. Native-only:
use PYSCF_NUMA=off. No grid pruning, AO cutoff, density or XC values are changed.
The byte budget covers retained AO arrays (not total process RSS).
"""
from contextlib import contextmanager
import hashlib
import math
import os
import threading
import numpy as np


def _digest_arrays(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        if value is None:
            digest.update(b'none;')
            continue
        arr = np.asarray(value)
        if arr.dtype.hasobject:
            raise TypeError('Object arrays cannot be used as collocation keys')
        digest.update(repr((arr.shape, arr.dtype.str)).encode())
        digest.update(np.ascontiguousarray(arr).data)
    return digest.digest()


class AOBlockCache:
    """Content-keyed, bounded, scan-resistant AO cache with copy-out semantics.

    When capacity is reached, keep admitted blocks instead of LRU eviction:
    repeatedly scanning a grid larger than memory must not evict every hit.
    Molecule content changes invalidate the cache. Coordinate/mask/cutoff or
    derivative changes cannot hit an old entry. Entries never alias out buffers.
    One cache belongs to one calculation; concurrent mutation of a Mole object
    is unsupported, as with the underlying molecular integrator.
    """
    def __init__(self, evaluator, max_bytes, max_entries=512):
        if isinstance(max_bytes, bool) or int(max_bytes) != max_bytes or max_bytes < 0:
            raise ValueError('max_bytes must be a nonnegative integer')
        if max_entries < 1:
            raise ValueError('max_entries must be positive')
        self.evaluator = evaluator
        self.max_bytes = int(max_bytes)
        self.max_entries = int(max_entries)
        self._entries = {}
        self._molecule = None
        self._lock = threading.RLock()
        self._stats = dict(calls=0, hits=0, misses=0, bypasses=0, admissions=0,
                           invalidations=0, retained_bytes=0, peak_bytes=0)

    def snapshot(self):
        with self._lock:
            return dict(self._stats, max_bytes=self.max_bytes,
                        entries=len(self._entries))

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._molecule = None
            self._stats['retained_bytes'] = 0

    @staticmethod
    def _copy_out(value, out):
        if out is None:
            return np.array(value, copy=True, order='K')
        ngrids, nao = value.shape[-2:]
        comp = 1 if value.ndim == 2 else value.shape[0]
        # Identical layout to gto.eval_gto: derivative components, each F order.
        target = np.ndarray((comp, nao, ngrids), dtype=np.float64,
                            buffer=out).transpose(0, 2, 1)
        if value.ndim == 2:
            target = target[0]
        np.copyto(target, value, casting='no')
        return target

    def __call__(self, mol, coords, deriv=0, shls_slice=None, non0tab=None,
                 cutoff=None, out=None, verbose=None):
        kwargs = dict(deriv=deriv, shls_slice=shls_slice, non0tab=non0tab,
                      cutoff=cutoff, out=out, verbose=verbose)
        with self._lock:
            self._stats['calls'] += 1
            if not self.max_bytes or not isinstance(deriv, (int, np.integer)):
                self._stats['bypasses'] += 1
                return self.evaluator(mol, coords, **kwargs)
            molecule = (_digest_arrays(mol._atm, mol._bas, mol._env), bool(mol.cart))
            if self._molecule is not None and molecule != self._molecule:
                self._entries.clear()
                self._stats['retained_bytes'] = 0
                self._stats['invalidations'] += 1
            self._molecule = molecule
            key = (molecule, _digest_arrays(coords, non0tab), int(deriv),
                   None if shls_slice is None else tuple(shls_slice), cutoff)
            entry = self._entries.get(key)
            if entry is not None:
                self._stats['hits'] += 1
                return self._copy_out(entry, out)
            self._stats['misses'] += 1
            value = self.evaluator(mol, coords, **kwargs)
            available = self.max_bytes - self._stats['retained_bytes']
            if (isinstance(value, np.ndarray) and value.dtype == np.float64
                    and value.ndim in (2, 3) and value.nbytes <= available
                    and len(self._entries) < self.max_entries):
                try:
                    saved = np.array(value, copy=True, order='K')
                except MemoryError:
                    return value  # Never change a successful integral result.
                saved.flags.writeable = False
                self._entries[key] = saved
                self._stats['admissions'] += 1
                self._stats['retained_bytes'] += saved.nbytes
                self._stats['peak_bytes'] = max(self._stats['peak_bytes'],
                                               self._stats['retained_bytes'])
            return value


@contextmanager
def cached_ao(mf, max_cache_mb=256):
    """Temporarily retain AO blocks during a native molecular RKS/UKS SCF.

    max_cache_mb is an AO-data budget in MiB, additionally limited to half the
    current PySCF memory headroom. Copies and ordinary integral work buffers
    are not included. Context exit restores the evaluator and releases arrays,
    including on SCF failure. Default PySCF behavior is completely unchanged
    unless this context is entered. Use the same block policy in comparisons.
    """
    from pyscf import gto, lib
    from pyscf.dft import numint
    if os.environ.get('PYSCF_NUMA', '').lower() != 'off':
        raise ValueError('cached_ao currently requires explicit PYSCF_NUMA=off')
    if not math.isfinite(max_cache_mb) or max_cache_mb < 0:
        raise ValueError('max_cache_mb must be finite and nonnegative')
    ni = mf._numint
    if type(ni) is not numint.NumInt or type(mf.mol) is not gto.Mole:
        raise TypeError('Only an unmodified molecular NumInt/Mole is supported')
    if ni.eval_ao is not numint.eval_ao or 'eval_gto' in mf.mol.__dict__:
        raise TypeError('Custom AO evaluators and nested caches are not supported')
    # MiB for the retained data; PySCF max_memory/current_memory use decimal MB.
    headroom = max(0., float(mf.max_memory) - lib.current_memory()[0]) * 1e6
    budget = int(min(max_cache_mb * 1024**2, headroom * .5))
    cache = AOBlockCache(ni.eval_ao, budget)
    missing = object()
    previous = ni.__dict__.get('eval_ao', missing)
    ni.eval_ao = cache
    try:
        yield cache
    finally:
        if previous is missing:
            del ni.eval_ao
        else:
            ni.eval_ao = previous
        cache.clear()
