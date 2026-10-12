# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Full iterative CCSDTQ for real unrestricted Hartree--Fock references.

T1, T2 and T3 use the full native UCCSDT spin blocks.  T4 has five blocks,
aaaa, aaab, aabb, bbba, bbbb; same-spin occupied and virtual groups are
stored as strictly increasing combinations in this module.  The highm
variant stores full T4.  All ranks enter the connected CC equations and
are iterated to convergence; e_tot is the CCSDTQ total energy.
'''

from itertools import combinations
from math import factorial, prod

import numpy as np

from pyscf import __config__, lib
from pyscf.lib import logger
from pyscf.cc import rccsdt, uccsdt, uccsdt_highm
from pyscf.cc import _uccsdtq_utils as utils


def _check_memory(mycc, size, what):
    available = max(0, mycc.max_memory - lib.current_memory()[0]) * 1e6
    if size > available:
        raise MemoryError('UCCSDTQ %s need at least %.1f MB; %.1f MB remains in max_memory'
                          % (what, size / 1e6, available / 1e6))


def init_amps(mycc, eris=None):
    size = 8 * sum(prod(utils.block_shape(mycc, rank, b, rank == 4 and mycc.do_tri_max_t))
                   for rank in range(1, 5) for b in range(rank + 1))
    _check_memory(mycc, size, 'initial amplitudes')
    return uccsdt.init_amps_uhf(mycc, eris)


def _quadruples_backend(mycc):
    from math import comb
    backend = mycc.quadruples_backend
    if backend not in ('auto', 'direct', 'blocked'):
        raise ValueError('quadruples_backend must be auto, direct or blocked')
    if backend == 'auto':
        oa, ob, va, vb = utils._dims(mycc)
        unique = sum(comb(oa, a)*comb(va, a)*comb(ob, 4-a)*comb(vb, 4-a)
                     for a in range(5))
        # A conservative crossover heuristic: tiny matrix products cost more
        # to dispatch than the direct scalar contractions. This estimates the
        # latter's repeated four-dummy-index work, not total solver FLOPs.
        work = unique*((oa+ob)*(va+vb))**2
        # Small scalar problems parallelize before the blocked matrices are
        # large enough for threaded BLAS. Raise the crossover with the team.
        backend = 'blocked' if work >= 1_000_000*lib.num_threads() else 'direct'
    if backend != getattr(mycc, '_quadruples_backend_used', None):
        logger.info(mycc, 'UCCSDTQ quadruples contraction backend: %s', backend)
        mycc._quadruples_backend_used = backend
    if backend == 'blocked':
        from pyscf.cc import _uccsdtq_blocked
        return _uccsdtq_blocked
    from pyscf.cc import _uccsdtq_native
    return _uccsdtq_native


def residuals(mycc, tamps, eris):
    '''Return raw connected R1--R4 in native UHF amplitude layouts.

    These include the diagonal Fock terms.  No energy denominator or level
    shift is applied here.  T4 feedback is added after the UCCSDT residuals
    have been antisymmetrized, so it is not antisymmetrized a second time.
    '''
    backend = _quadruples_backend(mycc)

    t1, t2, t3, t4 = tamps
    size = sum(t.nbytes for blocks in tamps for t in blocks)
    size += eris.pppp.nbytes + eris.pPpP.nbytes + eris.PPPP.nbytes
    _check_memory(mycc, size, 'residuals and dressed integrals')
    imds = uccsdt._IMDS()
    uccsdt.update_t1_fock_eris_uhf(mycc, imds, t1, eris)
    uccsdt.intermediates_t1t2_uhf(mycc, imds, t2)
    r1, r2 = uccsdt.compute_r1r2_uhf(mycc, imds, t2)
    uccsdt_highm.r1r2_add_t3_uhf_(mycc, imds, r1, r2, t3)
    uccsdt.antisymmetrize_r2_uhf_(r2)

    uccsdt.intermediates_t3_uhf(mycc, imds, t2)
    uccsdt_highm.intermediates_t3_add_t3_uhf(mycc, imds, t3)
    r3 = uccsdt_highm.compute_r3_uhf(mycc, imds, t2, t3)
    uccsdt_highm.antisymmetrize_r3_uhf_(r3)

    r4 = tuple(np.zeros_like(t, order='C') for t in t4)
    backend.residuals(mycc, imds, t2, t3, t4, out=(r2, r3, r4))
    return [r1, r2, r3, r4]


def _t4_axis_energies(mycc, block, mo_energy):
    na, nb = 4 - block, block
    first = 0 if na >= nb else 1
    counts = (max(na, nb), min(na, nb))
    energies = []
    for spin, count in zip((first, 1 - first), counts):
        if not count:
            continue
        no = mycc.nocc[spin]
        occ = np.asarray(mo_energy[spin][:no])
        vir = -np.asarray(mo_energy[spin][no:]) - mycc.level_shift
        if mycc.do_tri_max_t:
            for e in (occ, vir):
                idx = np.asarray(list(combinations(range(e.size), count)), dtype=np.intp)
                idx = idx.reshape(-1, count)
                energies.append(e[idx].sum(axis=1))
        else:
            energies.extend([occ] * count + [vir] * count)
    return energies


def r4_divide_e_(mycc, r4, mo_energy):
    '''Divide by sum(e_occ)-sum(e_vir)-4*level_shift in bounded chunks.'''
    # Limit temporary denominators to 131072 doubles, even for full T4.
    chunk = 131072
    for block, residual in enumerate(r4):
        axes = _t4_axis_energies(mycc, block, mo_energy)
        flat = residual.reshape(-1)
        strides = [int(np.prod(residual.shape[k+1:], dtype=np.int64))
                   for k in range(residual.ndim)]
        for start in range(0, flat.size, chunk):
            index = np.arange(start, min(start + chunk, flat.size), dtype=np.int64)
            denom = np.zeros(index.size)
            for e, stride in zip(axes, strides):
                denom += e[(index // stride) % e.size]
            flat[start:start + chunk] /= denom
    return r4


def update_amps_uccsdtq_(mycc, tamps, eris):
    '''Perform an in-place Jacobi update of all four excitation ranks.'''
    time0 = logger.process_clock(), logger.perf_counter()
    r1, r2, r3, r4 = residuals(mycc, tamps, eris)
    uccsdt.r1r2_divide_e_uhf_(mycc, r1, r2, eris.mo_energy)
    uccsdt_highm.r3_divide_e_uhf_(mycc, r3, eris.mo_energy)
    r4_divide_e_(mycc, r4, eris.mo_energy)
    norms = []
    for rank, (amplitudes, residual) in enumerate(zip(tamps, (r1, r2, r3, r4)), 1):
        for block, (t, r) in enumerate(zip(amplitudes, residual)):
            norm = np.linalg.norm(r)
            if rank == 4 and mycc.do_tri_max_t:
                # Match the full-tensor norm, including same-spin copies.
                norm *= factorial(4 - block) * factorial(block)
            norms.append(norm)
            t += r
    logger.timer_debug1(mycc, 'UCCSDTQ residual and update', *time0)
    return norms


def _prepare_tamps(mycc, tamps):
    if tamps is None:
        return None
    if not 2 <= len(tamps) <= 4:
        raise ValueError('UCCSDTQ requires initial T1/T2, optionally T3 and T4')
    result = []
    for rank, blocks in enumerate(tamps, 1):
        if len(blocks) != rank + 1:
            raise ValueError('T%d requires %d spin blocks' % (rank, rank + 1))
        compact = rank == 4 and mycc.do_tri_max_t
        converted = []
        for b, block in enumerate(blocks):
            block = np.asarray(block)
            if np.iscomplexobj(block):
                raise NotImplementedError('UCCSDTQ requires real amplitudes')
            shape = utils.block_shape(mycc, rank, b, compact)
            if block.shape != shape:
                _check_memory(mycc, 8 * prod(shape), 'amplitude conversion')
                convert = utils._pack_block if compact else utils._unpack_block
                block = convert(mycc, block, rank, b)
            converted.append(np.asarray(block, dtype=np.float64, order='C'))
        result.append(tuple(converted))
    for rank in range(len(result) + 1, 5):
        shapes = [utils.block_shape(mycc, rank, b, rank == 4 and mycc.do_tri_max_t)
                  for b in range(rank + 1)]
        _check_memory(mycc, 8 * sum(prod(shape) for shape in shapes), 'initial amplitudes')
        result.append(tuple(np.zeros(shape) for shape in shapes))
    return result


def tamps_tri2full(mycc, tamps):
    return [utils.unpack_blocks(mycc, blocks, rank)
            for rank, blocks in enumerate(tamps, 1)]


def tamps_full2tri(mycc, tamps):
    return list(tamps[:3]) + [utils.pack_t4(mycc, tamps[3])]


class UCCSDTQ(uccsdt.UCCSDT):
    __doc__ = __doc__
    cc_order = 4
    do_diis_max_t = getattr(__config__, 'cc_uccsdtq_UCCSDTQ_do_diis_max_t', False)
    conv_tol = getattr(__config__, 'cc_uccsdtq_UCCSDTQ_conv_tol', 1e-7)
    conv_tol_normt = getattr(__config__, 'cc_uccsdtq_UCCSDTQ_conv_tol_normt', 1e-6)
    quadruples_backend = getattr(__config__, 'cc_uccsdtq_UCCSDTQ_quadruples_backend', 'auto')
    _keys = uccsdt.UCCSDT._keys | {'quadruples_backend'}

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        uccsdt.UCCSDT.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.tamps = [None] * 4

    @property
    def t4(self):
        return self.tamps[3]

    @t4.setter
    def t4(self, value):
        self.tamps[3] = value

    update_amps_ = update_amps_uccsdtq_
    init_amps = init_amps
    amplitudes_to_vector = utils.amplitudes_to_vector
    vector_to_amplitudes = utils.vector_to_amplitudes
    vector_size = utils.vector_size
    memory_estimate = utils.memory_estimate
    memory_estimate_log = utils.memory_estimate_log
    dump_chk = utils.dump_chk
    restore_from_diis_ = utils.restore_from_diis_
    tamps_tri2full = tamps_tri2full
    tamps_full2tri = tamps_full2tri

    def kernel(self, tamps=None, eris=None):
        return self.ccsdtq(tamps, eris)

    def ccsdtq(self, tamps=None, eris=None):
        if any(np.iscomplexobj(c) for c in self.mo_coeff):
            raise NotImplementedError('UCCSDTQ requires real orbitals and integrals')
        if self.iterative_damping != 1.0:
            raise NotImplementedError('UCCSDTQ iterative damping is not implemented')
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()
        self.e_hf = self.get_e_hf()
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        if any(np.iscomplexobj(x) for x in (eris.fock[0], eris.fock[1],
                                           eris.pppp, eris.pPpP, eris.PPPP)):
            raise NotImplementedError('UCCSDTQ requires real orbitals and integrals')
        if tamps is None and all(t is not None for t in self.tamps):
            tamps = self.tamps
        tamps = _prepare_tamps(self, tamps)
        self.memory_estimate_log()
        self.converged, self.e_corr, self.tamps = rccsdt.kernel(
            self, eris, tamps, max_cycle=self.max_cycle, tol=self.conv_tol,
            tolnormt=self.conv_tol_normt, verbose=self.verbose, callback=self.callback)
        self._finalize()
        return self.e_corr, self.tamps

    def ccsdt_q(self, tamps=None, eris=None):
        raise NotImplementedError('The perturbative (Q) correction requires a CCSDT solver')
