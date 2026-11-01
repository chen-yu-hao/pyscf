"""Opt-in MGGA coalescing must retain every old screening decision."""
import ctypes
import os
import unittest
from unittest import mock
import numpy as np
from pyscf import lib
from pyscf.dft import numint


class KnownValues(unittest.TestCase):
    def setUp(self):
        self.threads = lib.num_threads()
        self.environment = mock.patch.dict(os.environ, {
            'PYSCF_DFT_DENSE_BLAS': '1', 'PYSCF_DFT_MGGA_PACK_SIMD': '1',
            'PYSCF_DFT_VMAT_TILE_SCREEN': '0',
            'PYSCF_DFT_MGGA_COALESCE_STATS': '0'})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.addCleanup(lib.num_threads, self.threads)

    @staticmethod
    def call(ao, wv, screen, pair, loc, initial, flag, threads=1, ncomp=4):
        lib.num_threads(threads)
        out = initial.copy()
        def ptr(a):
            return None if a is None else a.ctypes.data_as(ctypes.c_void_p)
        with mock.patch.dict(os.environ, {'PYSCF_DFT_MGGA_COALESCE': flag}):
            ret = numint.libdft.VXCxc_vmat(
                ptr(out), ptr(ao), ptr(wv), ctypes.c_int(len(out)),
                ctypes.c_int(ao.shape[-1]), ctypes.c_int(len(loc)-1),
                ctypes.c_int(20), ctypes.c_int(ncomp), ptr(screen), ptr(pair),
                ptr(loc), ctypes.c_ssize_t(ao.strides[0]//8))
        return ret, out

    def test_equivalent_and_distinct_masks(self):
        rng = np.random.default_rng(20261030)
        for sizes in ([1, 3, 5, 7], [70, 3, 70, 3]):
            loc = np.asarray([0]+list(np.cumsum(sizes)), dtype=np.int32)
            ns, n = len(sizes), int(loc[-1])
            for g in (1, 55, 56, 57, 223, 224, 225, 449, 896):
                ao = rng.normal(size=(8, n, g))[::2]
                if g == 449:
                    ao = ao[::-1]
                wv = rng.normal(size=(5, g))
                wv[:, ::7] = 0
                initial = rng.normal(size=(n, n))
                pair = rng.integers(0, 2, (ns, ns), dtype=np.uint8)
                pair |= pair.T
                for mode in ('same', 'equivalent', 'gaps', 'threshold'):
                    screen = rng.integers(20, 51, ((g+55)//56, ns), dtype=np.uint8)
                    if mode == 'same':
                        screen[:] = 25
                    elif mode == 'gaps':
                        screen[:, ::2] = 0
                    elif mode == 'threshold':
                        screen[:] = rng.choice([0, 1, 9, 10, 11, 20, 255], screen.shape)
                    saved_ao, saved_wv = ao.copy(), wv.copy()
                    for threads in (1, 2, 4):
                        with self.subTest(n=n, g=g, mode=mode, threads=threads):
                            rb, baseline = self.call(ao, wv, screen, pair, loc, initial, '0', threads)
                            rc, candidate = self.call(ao, wv, screen, pair, loc, initial, '1', threads)
                            self.assertEqual((rb, rc), (1, 1))
                            np.testing.assert_allclose(candidate, baseline, atol=4e-11, rtol=2e-12)
                            np.testing.assert_array_equal(ao, saved_ao)
                            np.testing.assert_array_equal(wv, saved_wv)

    def test_unaffected_dense_and_other_functionals(self):
        rng = np.random.default_rng(90)
        ao = rng.normal(size=(4, 67, 449))
        wv = rng.normal(size=(5, 449))
        loc = np.asarray([0, 3, 10, 67], dtype=np.int32)
        pair = np.ones((3, 3), dtype=np.uint8)
        screen = np.full((9, 3), 25, dtype=np.uint8)
        for mask, nc in ((None, 4), (None, 2), (screen, 2), (screen, 1)):
            for flag in ('1', '', 'true', '01', '2'):
                a = self.call(ao, wv, mask, pair, loc, np.zeros((67, 67)), '0', ncomp=nc)
                b = self.call(ao, wv, mask, pair, loc, np.zeros((67, 67)), flag, ncomp=nc)
                self.assertEqual(a[0], b[0])
                np.testing.assert_array_equal(a[1], b[1])

    def test_asymmetric_fallback_does_not_write(self):
        ao = np.ones((4, 8, 112)); wv = np.ones((5, 112))
        loc = np.asarray([0, 3, 8], dtype=np.int32)
        mask = np.full((2, 2), 25, dtype=np.uint8)
        pair = np.asarray([[1, 1], [0, 1]], dtype=np.uint8)
        initial = np.arange(64, dtype=float).reshape(8, 8)
        ret, out = self.call(ao, wv, mask, pair, loc, initial, '1')
        self.assertEqual(ret, 0)
        np.testing.assert_array_equal(out, initial)

    def test_budget_reverts_to_original_packing(self):
        # Expanded 224-grid packing does not fit 1 MiB, original 56-grid does.
        ao = np.ones((4, 160, 448)); wv = np.ones((5, 448))
        loc = np.asarray([0, 80, 160], dtype=np.int32)
        mask = np.full((8, 2), 25, dtype=np.uint8)
        pair = np.ones((2, 2), dtype=np.uint8)
        initial = np.zeros((160, 160))
        with mock.patch.dict(os.environ, {'PYSCF_DFT_VMAT_MAX_MEMORY_MB': '1'}):
            a = self.call(ao, wv, mask, pair, loc, initial, '0')
            b = self.call(ao, wv, mask, pair, loc, initial, '1')
        self.assertEqual((a[0], b[0]), (1, 1))
        np.testing.assert_array_equal(a[1], b[1])

    def test_diagnostics_are_separate_and_resettable(self):
        ao = np.ones((4, 16, 448)); wv = np.ones((5, 448))
        loc = np.asarray([0, 3, 8, 16], dtype=np.int32)
        mask = np.full((8, 3), 25, dtype=np.uint8)
        pair = np.ones((3, 3), dtype=np.uint8)
        fn = numint.libdft.VXCmgga_coalesce_stats
        fn.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_int]
        fn.restype = None
        counts = (ctypes.c_ulonglong*4)()
        fn(counts, 1)
        with mock.patch.dict(os.environ, {'PYSCF_DFT_MGGA_COALESCE_STATS': '1'}):
            self.call(ao, wv, mask, pair, loc, np.zeros((16, 16)), '1')
        fn(counts, 1)
        self.assertEqual(list(counts), [8, 2, 6, 4])
        fn(counts, 0)
        self.assertEqual(list(counts), [0, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()
