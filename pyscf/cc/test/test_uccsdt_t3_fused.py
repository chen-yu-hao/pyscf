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

"""Packed T3 contractions against the independent full residual equations."""

import copy
from itertools import combinations
import unittest
from unittest.mock import patch

import numpy as np

from pyscf import cc, gto, lib, scf
from pyscf.cc import uccsdt, uccsdt_highm
from pyscf.cc import _uccsdt_t3 as fused


def _setup(mycc):
    names = tuple('t2c_%s_%sf_%s' % (kind, fold, space)
                  for space in ('oa', 'va', 'ob', 'vb')
                  for fold in (6, 2) for kind in ('map', 'mask'))
    for name, value in zip(names, mycc.setup_tri2block()):
        setattr(mycc, name, value)
    for spin in ('aaa', 'aab'):
        setattr(mycc, 'blksize_o_' + spin, min(3, max(mycc.nocc)))
        setattr(mycc, 'blksize_v_' + spin,
                min(3, max(m - o for m, o in zip(mycc.nmo, mycc.nocc))))
    return mycc


def _random_amplitudes(mycc, eris, seed=382):
    rng = np.random.default_rng(seed)
    _, tamps = mycc.init_amps(eris)
    for blocks in tamps:
        for a in blocks:
            a[:] = rng.normal(scale=.003, size=a.shape)
    for a in (tamps[1][0], tamps[1][2]):
        a[:] = a - a.swapaxes(0, 1)
        a[:] = a - a.swapaxes(2, 3)
    return tamps


def _intermediates(mycc, eris, tamps):
    imds = uccsdt._IMDS()
    uccsdt.update_t1_fock_eris_uhf(mycc, imds, tamps[0], eris)
    uccsdt.intermediates_t1t2_uhf(mycc, imds, tamps[1])
    uccsdt.intermediates_t3_uhf(mycc, imds, tamps[1])
    uccsdt.intermediates_t3_add_t3_tri_uhf(mycc, imds, tamps[2])
    return imds


def _pack_reference(full):
    # Select canonical elements independently of the new scatter kernel.
    packed = []
    for spin, a in enumerate(full):
        if spin in (0, 3):
            holes = list(combinations(range(a.shape[0]), 3))
            particles = list(combinations(range(a.shape[3]), 3))
            out = np.empty((len(holes), len(particles)))
            for i, h in enumerate(holes):
                for j, p in enumerate(particles):
                    out[i, j] = a[h + p]
        else:
            holes = list(combinations(range(a.shape[0]), 2))
            particles = list(combinations(range(a.shape[2]), 2))
            out = np.empty((len(holes), len(particles)) + a.shape[4:])
            for i, h in enumerate(holes):
                for j, p in enumerate(particles):
                    out[i, j] = a[h + p]
        packed.append(out)
    return packed


def _full_reference(mycc, imds, tamps):
    full_t3 = mycc.tamps_tri2full(tamps[2])
    raw = uccsdt_highm.compute_r3_uhf(mycc, copy.deepcopy(imds), tamps[1], full_t3)
    return _pack_reference(uccsdt_highm.antisymmetrize_r3_uhf_(raw))


def _legacy_reference(mycc, imds, tamps):
    # Call each old block explicitly; the public dispatcher selects the new
    # backend in production and is deliberately not the reference here.
    work = copy.deepcopy(imds)
    aaa = uccsdt.compute_r3aaa_tri_uhf(mycc, work, tamps[1], tamps[2])
    bbb = uccsdt.compute_r3bbb_tri_uhf(mycc, work, tamps[1], tamps[2])
    aab = uccsdt.compute_r3aab_tri_uhf(mycc, work, tamps[1], tamps[2])
    bba = uccsdt.compute_r3bba_tri_uhf(mycc, work, tamps[1], tamps[2])
    return [aaa, aab, bba, bbb]


def _symmetric_case():
    mol = gto.M(atom=[('H', (0, 0, .8*i)) for i in range(6)],
                basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12).to_uhf()
    assert mf.converged
    mycc = _setup(cc.UCCSDT(mf))
    eris = mycc.ao2mo()
    tamps = _random_amplitudes(mycc, eris)
    t1, t2, t3 = tamps
    t1[1][:] = t1[0]
    t2[2][:] = t2[0]
    t2[1][:] = (t2[1] + t2[1].transpose(2, 3, 0, 1))*.5
    t3[2][:] = t3[1]
    t3[3][:] = t3[0]
    imds = _intermediates(mycc, eris, tamps)
    # Give the exact-equality guard a strictly spin-symmetric case. Real
    # AO transformations may otherwise leave harmless rounding differences.
    for left, right in (('F_oo', 'F_OO'), ('F_vv', 'F_VV'),
                        ('W_oooo', 'W_OOOO'), ('W_voov', 'W_VOOV'),
                        ('W_vvvv', 'W_VVVV'), ('W_vvvo', 'W_VVVO'),
                        ('W_ovoo', 'W_OVOO')):
        setattr(imds, right, getattr(imds, left).copy())
    for left, right in (('W_vOoV', 'W_oVvO'), ('W_vVvO', 'W_vVoV'),
                        ('W_oVoO', 'W_vOoO'), ('W_oVoV', 'W_vOvO')):
        setattr(imds, right, getattr(imds, left).transpose(1, 0, 3, 2).copy())
    for name in ('W_oOoO', 'W_vVvV'):
        a = getattr(imds, name)
        a[:] = (a + a.transpose(1, 0, 3, 2))*.5
    return mycc, tamps, imds


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom='O 0 0 0; H 0 0 .97', basis='6-31g', spin=1, verbose=0)
        with lib.with_omp_threads(1):
            cls.mf = scf.UHF(mol).run(conv_tol=1e-12)
        assert cls.mf.converged

    def setUp(self):
        self.threads = lib.num_threads()
        lib.num_threads(1)

    def tearDown(self):
        lib.num_threads(self.threads)

    def assert_blocks_close(self, actual, expected):
        for spin, (a, b) in enumerate(zip(actual, expected)):
            with self.subTest(spin=spin):
                np.testing.assert_allclose(a, b, atol=2e-12, rtol=2e-12)

    def _case(self, rotate=False):
        mo = self.mf.mo_coeff.copy()
        if rotate:
            rng = np.random.default_rng(91)
            for spin, occupied in enumerate(self.mf.mo_occ):
                no = np.count_nonzero(occupied)
                for sl in (slice(None, no), slice(no, None)):
                    block = mo[spin, :, sl]
                    q, _ = np.linalg.qr(rng.normal(size=(block.shape[1],) * 2))
                    mo[spin, :, sl] = block @ q
        # Both a frozen occupied orbital and an unequal frozen virtual space.
        frozen = [[0, mo.shape[-1] - 1], [0]]
        mycc = _setup(cc.UCCSDT(self.mf, frozen=frozen, mo_coeff=mo))
        eris = mycc.ao2mo()
        tamps = _random_amplitudes(mycc, eris)
        self.assertTrue(all(a.size for a in tamps[2]))
        self.assertNotEqual(*mycc.nocc)
        self.assertNotEqual(*mycc.nmo)
        imds = _intermediates(mycc, eris, tamps)
        return mycc, eris, tamps, imds

    def test_all_spin_blocks_nonzero_against_full_and_legacy(self):
        for rotate in (False, True):
            with self.subTest(noncanonical=rotate):
                mycc, _, tamps, imds = self._case(rotate)
                expected = _full_reference(mycc, imds, tamps)
                self.assert_blocks_close(_legacy_reference(mycc, imds, tamps), expected)
                for backend in ('numpy', 'pyscf'):
                    mycc.einsum_backend = backend
                    for threads in (1, 4):
                        lib.num_threads(threads)
                        with self.subTest(backend=backend, threads=threads):
                            actual = fused.compute_r3(mycc, imds, tamps[1], tamps[2])
                            self.assert_blocks_close(actual, expected)

    def test_exact_spin_reuse_and_rejections(self):
        mycc, tamps, imds = _symmetric_case()
        self.assertTrue(uccsdt._can_reuse_rhf_t3(mycc, imds, tamps[1], tamps[2]))
        eris = mycc.ao2mo()
        self.assertTrue(uccsdt._can_reuse_rhf_t3(mycc, imds, tamps[1], tamps[2], eris))
        eris.mo_coeff = [a.copy() for a in eris.mo_coeff]
        eris.mo_energy = [a.copy() for a in eris.mo_energy]
        for a in (eris.mo_coeff[1], eris.mo_energy[1]):
            old = a.flat[0]
            a.flat[0] = np.nextafter(old, np.inf)
            self.assertFalse(uccsdt._can_reuse_rhf_t3(
                mycc, imds, tamps[1], tamps[2], eris))
            a.flat[0] = old
        del eris.mo_coeff
        self.assertFalse(uccsdt._can_reuse_rhf_t3(mycc, imds, tamps[1], tamps[2], eris))
        expected = _full_reference(mycc, imds, tamps)
        with patch.object(fused, 'compute_r3', wraps=fused.compute_r3) as call:
            actual = uccsdt.compute_r3_tri_uhf(mycc, copy.deepcopy(imds),
                                             tamps[1], tamps[2])
        self.assertEqual(call.call_args.kwargs['spin_blocks'], (0, 1))
        self.assert_blocks_close(actual, expected)

        self.assertFalse(np.shares_memory(actual[0], actual[3]))
        self.assertFalse(np.shares_memory(actual[1], actual[2]))

        # A one-ULP asymmetry must disable reuse just like an orbital phase
        # difference. The general unrestricted equations remain available.
        arrays = [(a, 0) for a in (mycc.mo_coeff[1], mycc.mo_occ[1], tamps[1][2],
                                  tamps[2][2], tamps[2][3], imds.F_OO, imds.W_vOoV)]
        arrays.append((tamps[1][1], 1))
        for a, index in arrays:
            with self.subTest(shape=a.shape):
                old = a.flat[index]
                a.flat[index] = np.nextafter(old, np.inf)
                self.assertFalse(uccsdt._can_reuse_rhf_t3(
                    mycc, imds, tamps[1], tamps[2]))
                a.flat[index] = old
        mycc.mo_coeff[1, :, 0] *= -1
        self.assertFalse(uccsdt._can_reuse_rhf_t3(mycc, imds, tamps[1], tamps[2]))
        mycc.mo_coeff[1, :, 0] *= -1
        mycc.frozen = [[4], [5]]
        self.assertFalse(uccsdt._can_reuse_rhf_t3(mycc, imds, tamps[1], tamps[2]))
        mycc.frozen = None
        tamps[1][1].flat[1] += .01
        with patch.object(fused, 'compute_r3', wraps=fused.compute_r3) as call:
            actual = uccsdt.compute_r3_tri_uhf(mycc, copy.deepcopy(imds),
                                             tamps[1], tamps[2])
        self.assertEqual(call.call_args.kwargs['spin_blocks'], (0, 1, 2, 3))
        self.assert_blocks_close(actual, _full_reference(mycc, imds, tamps))
        tamps[1][1].flat[1] -= .01
        mycc.t3_spin_reuse = False
        with patch.object(fused, 'compute_r3', wraps=fused.compute_r3) as call:
            actual = uccsdt.compute_r3_tri_uhf(mycc, copy.deepcopy(imds),
                                             tamps[1], tamps[2])
        self.assertEqual(call.call_args.kwargs.get('spin_blocks', (0, 1, 2, 3)),
                         (0, 1, 2, 3))
        self.assert_blocks_close(actual, expected)

    def test_full_update_noncanonical_frozen(self):
        mycc, eris, tamps, _ = self._case(True)
        full = cc.UCCSDT(self.mf, frozen=mycc.frozen,
                        mo_coeff=mycc.mo_coeff, compact_tamps=False)
        expected = copy.deepcopy(tamps[:2]) + [mycc.tamps_tri2full(tamps[2])]
        full.update_amps_(expected, eris)
        for backend in ('fused', 'legacy'):
            with self.subTest(backend=backend):
                mycc.t3_backend = backend
                actual = copy.deepcopy(tamps)
                mycc.update_amps_(actual, eris)
                self.assert_blocks_close(actual[0], expected[0])
                self.assert_blocks_close(actual[1], expected[1])
                self.assert_blocks_close(mycc.tamps_tri2full(actual[2]), expected[2])
                self.assertAlmostEqual(mycc.energy(actual, eris),
                                       full.energy(expected, eris), 12)

    def test_readonly_strided_inputs_and_subset(self):
        mycc, _, tamps, imds = self._case(True)
        expected = _full_reference(mycc, imds, tamps)
        saved = {}
        for name, a in vars(imds).items():
            if isinstance(a, np.ndarray):
                a = np.asfortranarray(a)
                a.flags.writeable = False
                setattr(imds, name, a)
                saved[name] = a.copy()
        layouts = (np.asfortranarray, lambda a: a[::-1].copy()[::-1])
        for layout in layouts:
            inputs = [[layout(a) for a in rank] for rank in tamps[1:]]
            for rank in inputs:
                for a in rank:
                    a.flags.writeable = False
            actual = fused.compute_r3(mycc, imds, *inputs, spin_blocks=(0, 1))
            self.assert_blocks_close(actual[:2], expected[:2])
            self.assertIsNone(actual[2])
            self.assertIsNone(actual[3])
        for name, a in saved.items():
            np.testing.assert_array_equal(getattr(imds, name), a)

    def test_workspace_bound_and_boundary_tiles(self):
        mycc, _, tamps, imds = self._case(False)
        # A deliberately small but feasible workspace forces partial boundary
        # tiles in every spin sector. The planner's
        # reported temporary bound must remain below the caller's budget.
        plan = fused.workspace_plan(mycc, imds, tamps[1], tamps[2],
                                    workspace_bytes=64000)
        self.assertEqual(plan['budget_bytes'], 64000)
        self.assertTrue(plan['blocks'])
        shapes = [a.shape for a in mycc.tamps_tri2full(tamps[2])]
        for block, info in plan['blocks'].items():
            with self.subTest(block=block):
                self.assertLessEqual(info['work_bytes'] + info['cache_bytes'],
                                     plan['budget_bytes'])
                self.assertTrue(any(n % size for n, size in zip(shapes[block], info['shape'])))
        expected = _full_reference(mycc, imds, tamps)
        actual = fused.compute_r3(mycc, imds, tamps[1], tamps[2],
                                  workspace_bytes=64000)
        self.assert_blocks_close(actual, expected)
        with self.assertRaises(MemoryError):
            fused.compute_r3(mycc, imds, tamps[1], tamps[2], workspace_bytes=1)

    def test_max_memory_reserves_residual_outputs(self):
        mycc, _, tamps, imds = self._case(False)
        output_bytes = sum(a.nbytes for a in tamps[2])
        expected = _full_reference(mycc, imds, tamps)
        mycc.max_memory = 100.1
        with patch.object(fused.lib, 'current_memory', return_value=(100., 0.)):
            plan = fused.workspace_plan(mycc, imds, tamps[1], tamps[2])
            self.assertLessEqual(plan['budget_bytes'] + output_bytes, 100000)
            for info in plan['blocks'].values():
                self.assertLessEqual(info['work_bytes'] + info['cache_bytes'],
                                     plan['budget_bytes'])
            actual = fused.compute_r3(mycc, imds, tamps[1], tamps[2])
            self.assert_blocks_close(actual, expected)
            # The residual arrays alone exhaust the available memory.
            mycc.max_memory = 100. + output_bytes / 2e6
            with self.assertRaises(MemoryError):
                fused.compute_r3(mycc, imds, tamps[1], tamps[2])


if __name__ == '__main__':
    unittest.main()
