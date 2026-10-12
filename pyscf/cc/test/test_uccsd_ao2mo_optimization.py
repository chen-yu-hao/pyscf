# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.

"""Independent integral checks for identical-orbital UCCSD AO2MO reuse."""

import itertools
import unittest
from unittest import mock

import numpy as np
from pyscf import ao2mo, cc, gto, lib, scf
from pyscf.cc import uccsd


BLOCKS = ('oooo', 'ovoo', 'ovov', 'oovv', 'ovvo', 'ovvv', 'vvvv',
          'OOOO', 'OVOO', 'OVOV', 'OOVV', 'OVVO', 'OVVV', 'VVVV',
          'ooOO', 'ovOO', 'ovOV', 'ooVV', 'ovVO', 'ovVV', 'vvVV',
          'OVoo', 'OOvv', 'OVvo', 'OVvv')


def setUpModule():
    global mf
    mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)


def tearDownModule():
    global mf
    del mf


class KnownValues(unittest.TestCase):
    def compare_outcore(self, mycc, expected_transforms):
        with mock.patch.object(ao2mo, 'full', wraps=ao2mo.full) as transform:
            actual = uccsd._make_eris_incore(mycc)
        self.assertEqual(transform.call_count, expected_transforms)
        reference = uccsd._make_eris_outcore(mycc)
        try:
            for name in BLOCKS:
                with self.subTest(block=name):
                    np.testing.assert_allclose(getattr(actual, name),
                                               np.asarray(getattr(reference, name)),
                                               atol=2e-12, rtol=2e-12)
        finally:
            reference.feri.close()
        return actual

    def test_equal_orbitals_preserve_independent_spin_blocks(self):
        eris = self.compare_outcore(cc.UCCSD(mf), 1)
        for left, right in itertools.combinations(BLOCKS, 2):
            self.assertFalse(np.shares_memory(getattr(eris, left),
                                             getattr(eris, right)), (left, right))
        # Public ERI blocks remain writable without changing another spin block.
        original = {name: getattr(eris, name).copy() for name in BLOCKS}
        for name in BLOCKS:
            block = getattr(eris, name)
            block.flat[0] += 1
            for other in BLOCKS:
                if other != name:
                    np.testing.assert_array_equal(getattr(eris, other), original[other])
            block[...] = original[name]

    def test_symmetric_and_asymmetric_frozen_spaces(self):
        for frozen, ntransforms in ((1, 1), ([[0], [0]], 1),
                                     ([[0], []], 2), ([[0], [1]], 2)):
            with self.subTest(frozen=frozen):
                self.compare_outcore(cc.UCCSD(mf, frozen=frozen), ntransforms)

    def test_rotated_beta_orbitals_use_distinct_transformations(self):
        mycc = cc.UCCSD(mf)
        mycc.mo_coeff = np.array(mycc.mo_coeff, copy=True)
        angle = .17
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        mycc.mo_coeff[1, :, 5:7] = mycc.mo_coeff[1, :, 5:7] @ rotation
        self.compare_outcore(mycc, 2)

    def test_equal_orbitals_with_unequal_occupations(self):
        mycc = cc.UCCSD(mf)
        mycc.mo_occ = np.array(mycc.mo_occ, copy=True)
        mycc.mo_occ[0, 5] = 1
        mycc.mo_occ[1, 4] = 0
        self.assertEqual(mycc.nocc, (6, 4))
        self.compare_outcore(mycc, 1)

    def test_custom_ao2mo_callback_retains_three_calls_and_full_layout(self):
        mycc = cc.UCCSD(mf)
        reference = uccsd._make_eris_incore(mycc)

        def transform(coeff):
            if isinstance(coeff, np.ndarray):
                coeff = (coeff,) * 4
            return ao2mo.general(mycc._scf._eri, coeff, compact=False)

        callback = mock.Mock(side_effect=transform)
        actual = uccsd._make_eris_incore(mycc, ao2mofn=callback)
        self.assertEqual(callback.call_count, 3)
        nvira, nvirb = (nmo - nocc for nmo, nocc in zip(mycc.nmo, mycc.nocc))
        for name in BLOCKS:
            ref = getattr(reference, name)
            if name in ('ovvv', 'OVVV', 'ovVV', 'OVvv'):
                ref = getattr(reference, 'get_' + name)()
            elif name in ('vvvv', 'VVVV'):
                ref = ao2mo.restore(1, ref, nvira if name == 'vvvv' else nvirb)
            elif name == 'vvVV':
                idxa = lib.square_mat_in_trilu_indices(nvira).ravel()
                idxb = lib.square_mat_in_trilu_indices(nvirb).ravel()
                ref = ref[idxa[:, None], idxb[None, :]].reshape(
                    nvira, nvira, nvirb, nvirb)
            with self.subTest(block=name):
                self.assertEqual(getattr(actual, name).ndim, 4)
                np.testing.assert_allclose(getattr(actual, name), ref,
                                           atol=2e-12, rtol=2e-12)


if __name__ == '__main__':
    unittest.main()
