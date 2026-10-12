# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.

"""Regression tests for the closed-shell UCCSD contraction fast path."""

import unittest
from unittest import mock
import numpy as np

from pyscf import cc, gto, scf
from pyscf.cc import ccsd, uccsd


class KnownValues(unittest.TestCase):
    def test_same_spin_zero_pairs_and_supplied_diagonal(self):
        rng = np.random.default_rng(38)
        for no in (0, 1, 4):
            for with_diagonal in (False, True):
                tensor = rng.normal(size=(no, no, 3, 3))
                if not with_diagonal:
                    tensor[np.arange(no), np.arange(no)] = 0
                weights = rng.normal(size=(3, 3, 3, 3))
                def contraction(mycc, amplitudes, direct, out):
                    return np.einsum('xac,acbd->xbd', amplitudes, weights)
                reference = ccsd._unpack_t2_tril(
                    contraction(None, tensor[np.tril_indices(no)], False, None),
                    no, 3, None, 'jiba')
                counted = mock.Mock(wraps=contraction)
                actual = uccsd._contract_same_spin_vvvv(None, tensor, counted)
                np.testing.assert_allclose(actual, reference, atol=1e-13)
                if not with_diagonal and no < 2:
                    counted.assert_not_called()
                elif no:
                    pairs = counted.call_args.args[1].shape[0]
                    self.assertEqual(pairs, no*(no+1)//2 if with_diagonal else no*(no-1)//2)

    def test_rohf_is_not_tagged_as_rhf(self):
        mol = gto.M(atom='N', basis='sto-3g', spin=3, verbose=0)
        self.assertFalse(cc.UCCSD(scf.ROHF(mol).run())._closed_shell)

    def test_changed_beta_integrals_do_not_reuse_alpha_contraction(self):
        mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0',
                    basis='sto-3g', spin=0, verbose=0)
        mf = scf.RHF(mol).run()
        optimized = cc.UCCSD(mf)
        eris = optimized.ao2mo()
        _, t1, t2 = optimized.init_amps(eris)
        rng = np.random.default_rng(912)
        delta = rng.normal(scale=.01, size=eris.VVVV.shape)
        eris.VVVV += delta + delta.T
        general = cc.UCCSD(mf.to_uhf())
        reference = general.update_amps(t1, t2, eris)
        with mock.patch.object(eris, '_contract_VVVV_t2', wraps=eris._contract_VVVV_t2) as beta:
            actual = optimized.update_amps(t1, t2, eris)
            self.assertEqual(beta.call_count, 1)
        for rank_a, rank_b in zip(actual, reference):
            for a, b in zip(rank_a, rank_b):
                np.testing.assert_allclose(a, b, atol=2e-13, rtol=2e-13)

    def test_rhf_derived_vvvv_reuse_matches_general_uhf(self):
        mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0',
                    basis='sto-3g', spin=0, verbose=0)
        mf = scf.RHF(mol).run()
        optimized = cc.UCCSD(mf)
        self.assertTrue(optimized._closed_shell)
        eris = optimized.ao2mo()
        _, t1, t2 = optimized.init_amps(eris)

        general = cc.UCCSD(mf.to_uhf())
        self.assertFalse(general._closed_shell)
        optimized_update = optimized.update_amps(t1, t2, eris)
        general_update = general.update_amps(t1, t2, eris)
        for actual, reference in zip(optimized_update[0], general_update[0]):
            np.testing.assert_allclose(actual, reference, atol=2e-13, rtol=2e-13)
        for actual, reference in zip(optimized_update[1], general_update[1]):
            np.testing.assert_allclose(actual, reference, atol=2e-13, rtol=2e-13)

    def test_spin_breaking_amplitudes_use_general_path(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 0.74',
                    basis='sto-3g', spin=0, verbose=0)
        mf = scf.RHF(mol).run()
        optimized = cc.UCCSD(mf)
        eris = optimized.ao2mo()
        _, t1, t2 = optimized.init_amps(eris)
        t2 = tuple(np.array(x, copy=True) for x in t2)
        t2[2][0, 0, 0, 0] += 1e-4
        general = cc.UCCSD(mf.to_uhf())
        actual = optimized.update_amps(t1, t2, eris)
        reference = general.update_amps(t1, t2, eris)
        for a, b in zip(actual[0], reference[0]):
            np.testing.assert_allclose(a, b, atol=2e-13, rtol=2e-13)
        for a, b in zip(actual[1], reference[1]):
            np.testing.assert_allclose(a, b, atol=2e-13, rtol=2e-13)

        t1 = tuple(np.array(x, copy=True) for x in t1)
        t1[1][0, 0] += 1e-4
        actual = optimized.update_amps(t1, t2, eris)
        reference = general.update_amps(t1, t2, eris)
        for a, b in zip(actual[0], reference[0]):
            np.testing.assert_allclose(a, b, atol=2e-13, rtol=2e-13)
        for a, b in zip(actual[1], reference[1]):
            np.testing.assert_allclose(a, b, atol=2e-13, rtol=2e-13)


if __name__ == '__main__':
    unittest.main()
