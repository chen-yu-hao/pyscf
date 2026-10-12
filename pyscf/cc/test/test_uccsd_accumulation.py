# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.

"""Validate UCCSD contraction accumulation against spin-orbital CCSD."""

import unittest
from unittest import mock
import numpy as np

from pyscf import cc, gto, lib, scf


class KnownValues(unittest.TestCase):
    def test_polarized_update_matches_gccsd(self):
        mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='sto-3g',
                    spin=1, verbose=0)
        mf = scf.UHF(mol).run()
        mycc = cc.UCCSD(mf)
        eris = mycc.ao2mo()
        _, t1, t2 = mycc.init_amps(eris)
        rng = np.random.default_rng(4)
        t1 = tuple(.03 * rng.normal(size=x.shape) for x in t1)
        t2 = list(.03 * rng.normal(size=x.shape) for x in t2)
        for k in (0, 2):
            t2[k] -= t2[k].transpose(1, 0, 2, 3).copy()
            t2[k] -= t2[k].transpose(0, 1, 3, 2).copy()
        t2 = tuple(np.asfortranarray(x) for x in t2)
        before = tuple(x.copy() for x in t1 + t2)
        # Force the guarded GEMM path on this small unequal-spin system.
        # An independent GCCSD update below remains the equation reference.
        with mock.patch.object(lib.numpy_helper, 'EINSUM_MAX_SIZE', 0), \
                mock.patch.object(lib.numpy_helper, 'EINSUM_BACKEND', None):
            with mock.patch.object(lib.numpy_helper, 'dot', wraps=lib.numpy_helper.dot) as dot:
                actual = mycc.update_amps(t1, t2, eris)
                self.assertTrue(any(call.kwargs.get('c') is not None for call in dot.call_args_list))

        gcc = cc.GCCSD(mf.to_ghf())
        geris = gcc.ao2mo()
        gt1 = gcc.spatial2spin(t1, geris.orbspin)
        gt2 = gcc.spatial2spin(t2, geris.orbspin)
        reference = tuple(gcc.spin2spatial(x, geris.orbspin)
                          for x in gcc.update_amps(gt1, gt2, geris))
        for actual_rank, expected_rank in zip(actual, reference):
            for a, b in zip(actual_rank, expected_rank):
                np.testing.assert_allclose(a, b, atol=2e-13, rtol=2e-13)
        for a, b in zip(t1 + t2, before):
            np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
