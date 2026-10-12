# Copyright 2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Tiled compact UCCSDT triples residuals.

Each spin-resolved seed from :mod:`uccsdt_highm` is contracted once per tile.
The C gather reads compact amplitudes directly, and the C writeback applies
unnormalized occupied and virtual antisymmetrizers into compact storage.
Neither all-spin full T3 arrays nor full antisymmetrized residuals are made.
"""

import ctypes
import itertools
import math
from collections import OrderedDict
from functools import lru_cache

import numpy as np
from pyscf import lib
from pyscf.cc.rccsdt import _einsum, _get_pytblis, _libccsdt

# Coefficients include the antisymmetrizer normalization used in
# uccsdt_highm.compute_r3_uhf. Input and output labels follow its interleaved
# majority-spin convention; these are the same 56 equations, without the
# explicit permutation-expanded contractions of the old compact backend.
_SEEDS = (
    (
        ('bcdk,ijad->ijkabc', 'W_vvvo', 't2aa', 0.25),
        ('lcjk,ilab->ijkabc', 'W_ovoo', 't2aa', -0.25),
        ('cd,ijkabd->ijkabc', 'F_vv', 't3aaa', 0.08333333333333333),
        ('lk,ijlabc->ijkabc', 'F_oo', 't3aaa', -0.08333333333333333),
        ('abde,ijkdec->ijkabc', 'W_vvvv', 't3aaa', 0.041666666666666664),
        ('lmij,lmkabc->ijkabc', 'W_oooo', 't3aaa', 0.041666666666666664),
        ('alid,ljkdbc->ijkabc', 'W_voov', 't3aaa', 0.25),
        ('alid,jkbcld->ijkabc', 'W_vOoV', 't3aab', 0.25),
    ),
    (
        ('bcdk,ijad->ijabkc', 'W_vVvO', 't2aa', 0.5),
        ('bcjd,iakd->ijabkc', 'W_vVoV', 't2ab', 1.0),
        ('abdi,jdkc->ijabkc', 'W_vvvo', 't2ab', -0.5),
        ('lcjk,ilab->ijabkc', 'W_oVoO', 't2aa', -0.5),
        ('aljk,iblc->ijabkc', 'W_vOoO', 't2ab', 1.0),
        ('laij,lbkc->ijabkc', 'W_ovoo', 't2ab', 0.5),
        ('cd,ijabkd->ijabkc', 'F_VV', 't3aab', 0.25),
        ('ad,ijbdkc->ijabkc', 'F_vv', 't3aab', -0.5),
        ('lk,ijablc->ijabkc', 'F_OO', 't3aab', -0.25),
        ('li,jlabkc->ijabkc', 'F_oo', 't3aab', 0.5),
        ('abde,ijdekc->ijabkc', 'W_vvvv', 't3aab', 0.125),
        ('bced,ijaekd->ijabkc', 'W_vVvV', 't3aab', 0.5),
        ('lmij,lmabkc->ijabkc', 'W_oooo', 't3aab', 0.125),
        ('lmik,ljabmc->ijabkc', 'W_oOoO', 't3aab', 0.5),
        ('alid,ljdbkc->ijabkc', 'W_voov', 't3aab', 1.0),
        ('alid,lkdcjb->ijabkc', 'W_vOoV', 't3bba', 1.0),
        ('lcid,ljabkd->ijabkc', 'W_oVoV', 't3aab', -0.5),
        ('aldk,ijdblc->ijabkc', 'W_vOvO', 't3aab', -0.5),
        ('lcdk,ijlabd->ijabkc', 'W_oVvO', 't3aaa', 0.25),
        ('clkd,ijabld->ijabkc', 'W_VOOV', 't3aab', 0.25),
    ),
    (
        ('cbkd,ijad->ijabkc', 'W_vVoV', 't2bb', 0.5),
        ('cbdj,kdia->ijabkc', 'W_vVvO', 't2ab', 1.0),
        ('abdi,kcjd->ijabkc', 'W_VVVO', 't2ab', -0.5),
        ('clkj,ilab->ijabkc', 'W_vOoO', 't2bb', -0.5),
        ('lakj,lcib->ijabkc', 'W_oVoO', 't2ab', 1.0),
        ('laij,kclb->ijabkc', 'W_OVOO', 't2ab', 0.5),
        ('cd,ijabkd->ijabkc', 'F_vv', 't3bba', 0.25),
        ('ad,ijbdkc->ijabkc', 'F_VV', 't3bba', -0.5),
        ('lk,ijablc->ijabkc', 'F_oo', 't3bba', -0.25),
        ('li,jlabkc->ijabkc', 'F_OO', 't3bba', 0.5),
        ('abde,ijdekc->ijabkc', 'W_VVVV', 't3bba', 0.125),
        ('cbde,ijaekd->ijabkc', 'W_vVvV', 't3bba', 0.5),
        ('lmij,lmabkc->ijabkc', 'W_OOOO', 't3bba', 0.125),
        ('mlki,ljabmc->ijabkc', 'W_oOoO', 't3bba', 0.5),
        ('alid,ljdbkc->ijabkc', 'W_VOOV', 't3bba', 1.0),
        ('ladi,lkdcjb->ijabkc', 'W_oVvO', 't3aab', 1.0),
        ('cldi,ljabkd->ijabkc', 'W_vOvO', 't3bba', -0.5),
        ('lakd,ijdblc->ijabkc', 'W_oVoV', 't3bba', -0.5),
        ('clkd,ijlabd->ijabkc', 'W_vOoV', 't3bbb', 0.25),
        ('clkd,ijabld->ijabkc', 'W_voov', 't3bba', 0.25),
    ),
    (
        ('bcdk,ijad->ijkabc', 'W_VVVO', 't2bb', 0.25),
        ('lcjk,ilab->ijkabc', 'W_OVOO', 't2bb', -0.25),
        ('cd,ijkabd->ijkabc', 'F_VV', 't3bbb', 0.08333333333333333),
        ('lk,ijlabc->ijkabc', 'F_OO', 't3bbb', -0.08333333333333333),
        ('abde,ijkdec->ijkabc', 'W_VVVV', 't3bbb', 0.041666666666666664),
        ('lmij,lmkabc->ijkabc', 'W_OOOO', 't3bbb', 0.041666666666666664),
        ('alid,ljkdbc->ijkabc', 'W_VOOV', 't3bbb', 0.25),
        ('ladi,jkbcld->ijkabc', 'W_oVvO', 't3bba', 0.25),
    ),
)

_T3_NAMES = ('t3aaa', 't3aab', 't3bba', 't3bbb')
_T2_NAMES = ('t2aa', 't2ab', 't2bb')
_INT_PTR = ctypes.POINTER(ctypes.c_int64)
_DOUBLE_PTR = ctypes.POINTER(ctypes.c_double)


@lru_cache(None)
def _kernels():
    functions = (_libccsdt.UCCSDT_t3_gather, _libccsdt.UCCSDT_t3_pack)
    for function in functions:
        function.argtypes = [_DOUBLE_PTR, _DOUBLE_PTR, ctypes.c_int,
                             _INT_PTR, _INT_PTR, _INT_PTR, _INT_PTR]
        function.restype = None
    return functions


def _full_shapes(mycc):
    oa, ob = mycc.nocc
    va, vb = (n-o for n, o in zip(mycc.nmo, mycc.nocc))
    return ((oa, oa, oa, va, va, va), (oa, oa, va, va, ob, vb),
            (ob, ob, vb, vb, oa, va), (ob, ob, ob, vb, vb, vb))


def _ranges(labels, output, starts, shape, full_shape):
    position = {c: i for i, c in enumerate(output)}
    return tuple((starts[position[c]], shape[position[c]]) if c in position
                 else (0, n) for c, n in zip(labels, full_shape))


def _working_bytes(shape, seeds, arrays, full_shapes, tblis=False):
    """Conservative tile + contraction-copy bound, excluding the gather cache."""
    output_size = math.prod(shape)
    worst = 0
    for equation, name, source, alpha in seeds:
        left, right = equation.split('->')[0].split(',')
        output = equation.split('->')[1]
        input_shape = full_shapes[_T3_NAMES.index(source)] if source in _T3_NAMES else arrays[source].shape
        nr = math.prod(n for _, n in _ranges(right, output, (0,)*6, shape, input_shape))
        nl = math.prod(n for _, n in _ranges(left, output, (0,)*6, shape, arrays[name].shape))
        if tblis:
            # TBLIS consumes strided operands and accumulates into the tile.
            # Reserve both input panels conservatively, but there are no
            # full result/scaling/transpose copies as in the NumPy path.
            worst = max(worst, output_size + nr + nl)
        else:
            # Result, scaled result, tile, and contiguous BLAS operands.
            worst = max(worst, 3*output_size + 2*nr + nl)
    return 8*worst


def _prepare(mycc, imds, t2, t3, spin_blocks, workspace_bytes):
    shapes = _full_shapes(mycc)
    # An unavailable optional backend falls back to NumPy in _einsum and
    # must retain the more conservative NumPy temporary-space estimate.
    tblis = mycc.einsum_backend == 'pytblis' and _get_pytblis() is not None
    for block, (array, dims) in enumerate(zip(t3, shapes)):
        if block in (0, 3):
            expected = (math.comb(dims[0], 3), math.comb(dims[3], 3))
        else:
            expected = (math.comb(dims[0], 2), math.comb(dims[2], 2), dims[4], dims[5])
        if array.dtype != np.float64 or array.shape != expected:
            raise ValueError('Compact T3 blocks must have the expected shape and float64 dtype')
    arrays = dict(zip(_T2_NAMES, t2))
    nonzero = {name: bool(np.any(array)) for name, array in zip(_T3_NAMES, t3)}
    if workspace_bytes is None:
        # The dispatcher may copy uncomputed RHF partner blocks after exact
        # symmetry reuse, so reserve all four output arrays even for subsets.
        output_bytes = sum(block.nbytes for block in t3)
        available = max(0, (mycc.max_memory-lib.current_memory()[0])*1e6-output_bytes)
        workspace_bytes = int(min(64*1024**2, available*.8))
    workspace_bytes = int(workspace_bytes)
    cache_bytes = workspace_bytes//4
    plans = {}
    for block in spin_blocks:
        if not t3[block].size:
            continue
        seeds = tuple(seed for seed in _SEEDS[block]
                      if seed[2] not in nonzero or nonzero[seed[2]])
        for _, name, _, _ in seeds:
            arrays[name] = getattr(imds, name)
        tile = list(shapes[block])
        bound = _working_bytes(tile, seeds, arrays, shapes, tblis)
        while bound + cache_bytes > workspace_bytes:
            choices = []
            for axis, size in enumerate(tile):
                if size > 1:
                    trial = tile.copy()
                    trial[axis] = (size+1)//2
                    choices.append((_working_bytes(trial, seeds, arrays, shapes, tblis), axis, trial))
            if not choices:
                raise MemoryError('Insufficient workspace for one UCCSDT T3 residual tile')
            bound, _, tile = min(choices)
        # Adjacent contractions consume identical gathered operands. This
        # remains effective when only one large input fits in the cache.
        def gather_key(seed):
            equation, _, source, _ = seed
            if source not in _T3_NAMES:
                return source, ()
            right = equation.split('->')[0].split(',')[1]
            output = equation.split('->')[1]
            ranges = _ranges(right, output, (0,)*6, tile, shapes[_T3_NAMES.index(source)])
            return source, ranges
        seeds = tuple(sorted(seeds, key=gather_key))
        plans[block] = {'shape': tuple(tile), 'work_bytes': bound,
                        'cache_bytes': cache_bytes, 'seeds': seeds}
    return shapes, arrays, plans, workspace_bytes


def workspace_plan(mycc, imds, t2, t3, spin_blocks=(0, 1, 2, 3), workspace_bytes=None):
    """Return the tile/copy/cache upper bounds used by :func:`compute_r3`.

    The bound covers temporary NumPy data, not existing amplitudes, integrals,
    residual output, or small BLAS/runtime allocations. ``workspace_bytes`` is
    an internal test override; normal calls use the remaining ``max_memory``.
    """
    _, _, plans, budget = _prepare(mycc, imds, t2, t3, spin_blocks, workspace_bytes)
    return {'budget_bytes': budget,
            'blocks': {block: {key: value for key, value in plan.items() if key != 'seeds'}
                       for block, plan in plans.items()}}


def _native(function, destination, source, block, dims, ranges):
    starts = np.asarray([start for start, size in ranges], dtype=np.int64)
    sizes = np.asarray([size for start, size in ranges], dtype=np.int64)
    strides = np.asarray(source.strides, dtype=np.int64)//source.itemsize
    function(destination.ctypes.data_as(_DOUBLE_PTR), source.ctypes.data_as(_DOUBLE_PTR),
             ctypes.c_int(block in (1, 2)), dims.ctypes.data_as(_INT_PTR),
             starts.ctypes.data_as(_INT_PTR), sizes.ctypes.data_as(_INT_PTR),
             strides.ctypes.data_as(_INT_PTR))


def compute_r3(mycc, imds, t2, t3, spin_blocks=(0, 1, 2, 3), workspace_bytes=None):
    """Compute compact, fully antisymmetrized T3 residuals with bounded tiles.

    Skipped spin blocks return ``None`` so a caller that has established exact
    RHF spin symmetry can copy the partner result. Every requested empty block
    is a zero array with the original packed shape. Input intermediates are
    retained for the caller, matching the legacy compact backend.
    """
    shapes, arrays, plans, budget = _prepare(mycc, imds, t2, t3, spin_blocks, workspace_bytes)
    gather, pack = _kernels()
    stats = {'gather_calls': 0, 'cache_hits': 0, 'contraction_calls': 0, 'tiles': 0,
             'cache_peak_bytes': 0, 'workspace_bytes': budget,
             'workspace_bound_bytes': max((p['work_bytes']+p['cache_bytes'] for p in plans.values()), default=0),
             'spin_blocks': tuple(spin_blocks)}
    mycc._t3_fused_stats = stats
    dims = tuple(np.asarray(shape, dtype=np.int64) for shape in shapes)
    residuals = [None]*4
    for block in spin_blocks:
        residuals[block] = np.zeros_like(t3[block], order='C')
        if block not in plans:
            continue
        plan = plans[block]
        tile_shape = plan['shape']
        for starts in itertools.product(*(range(0, n, step) for n, step in zip(shapes[block], tile_shape))):
            stats['tiles'] += 1
            shape = tuple(min(step, n-start) for start, step, n in zip(starts, tile_shape, shapes[block]))
            tile = np.zeros(shape)
            cache = OrderedDict()
            cache_used = 0
            for equation, name, source, alpha in plan['seeds']:
                left, right = equation.split('->')[0].split(',')
                output = equation.split('->')[1]
                lhs_ranges = _ranges(left, output, starts, shape, arrays[name].shape)
                lhs = arrays[name][tuple(slice(start, start+n) for start, n in lhs_ranges)]
                if source in _T3_NAMES:
                    source_block = _T3_NAMES.index(source)
                    rhs_ranges = _ranges(right, output, starts, shape, shapes[source_block])
                    key = (source_block, rhs_ranges)
                    if key in cache:
                        stats['cache_hits'] += 1
                        rhs = cache.pop(key)
                        cache[key] = rhs
                    else:
                        rhs_shape = tuple(n for _, n in rhs_ranges)
                        nbytes = 8*math.prod(rhs_shape)
                        while cache and cache_used + nbytes > plan['cache_bytes']:
                            _, old = cache.popitem(last=False)
                            cache_used -= old.nbytes
                            del old
                        rhs = np.empty(rhs_shape)
                        _native(gather, rhs, t3[source_block], source_block,
                                dims[source_block], rhs_ranges)
                        stats['gather_calls'] += 1
                        if nbytes <= plan['cache_bytes']:
                            cache[key] = rhs
                            cache_used += nbytes
                            stats['cache_peak_bytes'] = max(stats['cache_peak_bytes'], cache_used)
                else:
                    rhs_ranges = _ranges(right, output, starts, shape, arrays[source].shape)
                    rhs = arrays[source][tuple(slice(start, start+n) for start, n in rhs_ranges)]
                _einsum(mycc.einsum_backend, equation, lhs, rhs, out=tile, alpha=alpha, beta=1.)
                stats['contraction_calls'] += 1
                # An uncached operand must not survive into the next gather.
                del rhs
            cache.clear()
            _native(pack, residuals[block], tile, block,
                    dims[block], tuple(zip(starts, shape)))
            # Do not retain this output while allocating the next tile.
            del tile
    return residuals
