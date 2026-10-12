# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.

"""Regression tests for UCCSD denominator application."""

import unittest
from unittest import mock
import numpy as np

from pyscf import cc, gto, lib, scf
from pyscf.cc import uccsd


class KnownValues(unittest.TestCase):
    def test_sliced_denominator_matches_full(self):
        rng = np.random.default_rng(472)
        for shape in ((3, 2, 4, 5), (3, 3, 4, 4), (0, 2, 4, 5),
                      (3, 0, 4, 5), (3, 2, 0, 5), (3, 2, 4, 0)):
            for complex_amplitudes in (False, True):
                with self.subTest(shape=shape, complex=complex_amplitudes):
                    ni, nj, na, nb = shape
                    eia = -1 - rng.random((ni, na))
                    ejb = -1 - rng.random((nj, nb))
                    t2 = rng.random(shape)
                    if complex_amplitudes:
                        t2 = t2 + 1j * rng.random(shape)
                    expected = t2 / lib.direct_sum('ia+jb->ijab', eia, ejb)
                    uccsd._divide_by_doubles_denominator(t2, eia, ejb)
                    np.testing.assert_array_equal(t2, expected)

    def test_sliced_denominators_match_full_update(self):
        mol = gto.M(atom='O 0 0 0; H 0 0 1', basis='sto-3g',
                    spin=1, verbose=0)
        mycc = cc.UCCSD(scf.UHF(mol).run())
        eris = mycc.ao2mo()
        _, t1, t2 = mycc.init_amps(eris)

        def full_denominator(t2, eia, ejb):
            t2 /= lib.direct_sum('ia+jb->ijab', eia, ejb)

        for level_shift in (0, .2):
            with self.subTest(level_shift=level_shift):
                mycc.level_shift = level_shift
                actual = mycc.update_amps(t1, t2, eris)
                with mock.patch.object(uccsd, '_divide_by_doubles_denominator',
                                       side_effect=full_denominator):
                    reference = mycc.update_amps(t1, t2, eris)
                for actual_rank, reference_rank in zip(actual, reference):
                    for a, b in zip(actual_rank, reference_rank):
                        np.testing.assert_allclose(a, b, atol=2e-13, rtol=2e-13)


if __name__ == '__main__':
    unittest.main()
