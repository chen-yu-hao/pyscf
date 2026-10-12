# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.

"""Check direct residual accumulation against the independent RCCSD equations."""

import unittest
from unittest import mock
import numpy as np

from pyscf import gto, lib, scf
from pyscf.cc import ccsd, rccsd


def setUpModule():
    global mf, old_threads
    old_threads = lib.num_threads()
    lib.num_threads(1)
    mol = gto.M(atom='N 0 0 0; H 0 0 .99; H .94 0 -.33; H -.47 .81 -.33',
                basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)


def tearDownModule():
    lib.num_threads(old_threads)
    global mf
    del mf


class KnownValues(unittest.TestCase):
    def check_update(self, random_amplitudes, incore, blocked, async_io):
        optimized = ccsd.CCSD(mf)
        optimized.incore_complete = incore
        optimized.async_io = async_io
        eris = optimized.ao2mo()
        reference = rccsd.RCCSD(mf)
        reference_eris = reference.ao2mo()
        _, t1, t2 = optimized.init_amps(eris)
        if random_amplitudes:
            rng = np.random.default_rng(241)
            t1 = rng.normal(scale=.02, size=t1.shape)
            t2 = rng.normal(scale=.02, size=t2.shape)
            t2 += t2.transpose(1, 0, 3, 2)
        t1_saved, t2_saved = t1.copy(), t2.copy()
        expected = reference.update_amps(t1, t2, reference_eris)
        if blocked:
            # Ten virtual orbitals give blocks of 4, 4, 2. The final output
            # slices are strided, so they must retain correct fallback updates.
            self.assertEqual(t1.shape[1], 10)
            optimized.max_memory = 0
        with mock.patch.object(ccsd, 'BLKMIN', 4):
            actual = optimized.update_amps(t1, t2, eris)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, atol=2e-13, rtol=2e-13)
        np.testing.assert_array_equal(t1, t1_saved)
        np.testing.assert_array_equal(t2, t2_saved)

    def test_first_update(self):
        self.check_update(False, True, False, False)

    def test_nonzero_amplitudes(self):
        self.check_update(True, True, False, False)

    def test_tail_blocks(self):
        for incore in (False, True):
            for async_io in (False, True):
                with self.subTest(incore=incore, async_io=async_io):
                    self.check_update(True, incore, True, async_io)


if __name__ == '__main__':
    unittest.main()
