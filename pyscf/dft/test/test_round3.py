"""Exact-path regression: unchanged arrays and unchanged direct J/K cutoffs."""
import ctypes
import os
import unittest
from unittest import mock
import numpy as np
from pyscf import gto, scf, lib
from pyscf.scf import _vhf
from pyscf.dft import numint
from pyscf.dft.test.test_numint_mgga_vmat import KnownValues


class Round3Tests(unittest.TestCase):
    def test_packed_mgga_is_bitwise_equal(self):
        helper = KnownValues()
        for ng in (8, 56, 120, 224, 904):
            for threads in (1, 2, 4):
                ao, wv, mask, pairs, loc, initial = helper._inputs(ng)
                for screened in (False, True):
                    for reverse in (False, True):
                        values = ao[::-1] if reverse else ao
                        screen = mask if screened else None
                        with self.subTest(ng=ng, threads=threads, sparse=screened, reverse=reverse), \
                                lib.with_omp_threads(threads):
                            with mock.patch.dict(os.environ, {'PYSCF_DFT_MGGA_PACK_SIMD': '0'}):
                                old = numint._mgga_vmat(values, wv, 20, screen, pairs, loc, initial.copy())
                            with mock.patch.dict(os.environ, {'PYSCF_DFT_MGGA_PACK_SIMD': '1'}):
                                new = numint._mgga_vmat(values, wv, 20, screen, pairs, loc, initial.copy())
                            np.testing.assert_array_equal(new, old)

    def test_direct_jk_current_and_delta_densities(self):
        mol = gto.M(atom='O 0 0 0; H .757 0 .587; H -.757 0 .587; '
                        'O 4 0 0; H 4.757 0 .587; H 3.243 0 .587',
                    basis='def2-svp', verbose=0)
        mf = scf.RHF(mol)
        mf._is_mem_enough = lambda: False
        dm = mf.get_init_guess()
        counter = _vhf.libcvhf.CVHFround3_bound_stats
        counter.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_int]
        counts = (ctypes.c_ulonglong * 2)()
        counter(counts, 1)
        rng = np.random.default_rng(3001)
        delta = rng.normal(size=dm.shape)
        delta = 1e-5*(delta+delta.T)
        for threads in (1, 2):
            with lib.with_omp_threads(threads):
                for density in (dm, delta, np.zeros_like(dm)):
                    for with_j, with_k in ((True, True), (True, False), (False, True)):
                        with mock.patch.dict(os.environ, {'PYSCF_JK_BLOCK_BOUND': '0'}):
                            old = mf.get_jk(mol, density, with_j=with_j, with_k=with_k)
                        with mock.patch.dict(os.environ, {'PYSCF_JK_BLOCK_BOUND': '1',
                                                         'PYSCF_JK_BLOCK_BOUND_STATS': '1'}):
                            new = mf.get_jk(mol, density, with_j=with_j, with_k=with_k)
                        for a, b in zip(old, new):
                            if a is None:
                                self.assertIsNone(b)
                            else:
                                np.testing.assert_allclose(a, b, atol=1e-11, rtol=1e-12)
        counter(counts, 0)
        self.assertGreater(counts[0], 0)
        self.assertGreater(counts[1], 0)
        self.assertIsNone(mf._eri)


if __name__ == '__main__':
    unittest.main()
