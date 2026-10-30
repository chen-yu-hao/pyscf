"""Regression checks for opt-in dense XC contractions; no molecular shortcuts."""
import os
import unittest
from unittest import mock

import numpy as np
from pyscf import lib
from pyscf.dft import numint
from threadpoolctl import threadpool_limits


class DenseBlasTests(unittest.TestCase):
    def setUp(self):
        self.limit = threadpool_limits(limits=1, user_api='blas')
        self.addCleanup(self.limit.restore_original_limits)

    @staticmethod
    def inputs(nao, ngrids):
        rng = np.random.default_rng(20261030 + nao + ngrids)
        ao = rng.normal(size=(8, nao, ngrids))[::2].transpose(0, 2, 1)
        weights = rng.normal(size=(5, ngrids))
        weights[:, ::7] = 0
        loc = np.arange(nao+1, dtype=np.int32)
        initial = rng.normal(size=(nao, nao))
        return ao, weights, loc, initial

    @staticmethod
    def formula(ao, weights, xctype):
        a = ao[0]
        result = a.T @ (weights[0, :, None] * a)
        if xctype != 'LDA':
            for c in range(1, 4):
                cross = a.T @ (weights[c, :, None] * ao[c])
                result += cross + cross.T
                if xctype == 'MGGA':
                    result += .5 * ao[c].T @ (weights[4, :, None] * ao[c])
        return result

    @staticmethod
    def evaluate(ao, weights, loc, initial, xctype, flag, screen=None, pairs=None):
        values = ao[0] if xctype == 'LDA' else ao
        wv = weights[0] if xctype == 'LDA' else weights[:4] if xctype == 'GGA' else weights
        with mock.patch.dict(os.environ, {'PYSCF_DFT_DENSE_BLAS': str(flag)}):
            return numint._xc_vmat(values, wv, 20, screen, pairs, loc,
                                   initial.copy(), xctype=xctype, hermi=1)

    def test_dense_boundaries_and_negative_stride(self):
        for nao in (5, 67, 145):
            for ngrids in (1, 895, 896, 897, 1793):
                ao, weights, loc, initial = self.inputs(nao, ngrids)
                ao = ao[::-1]
                original_ao, original_weights = ao.copy(), weights.copy()
                for xctype in ('LDA', 'GGA', 'MGGA'):
                    reference = initial + self.formula(ao, weights, xctype)
                    for threads in (1, 4):
                        with self.subTest(nao=nao, ngrids=ngrids, xc=xctype, threads=threads), lib.with_omp_threads(threads):
                            old = self.evaluate(ao, weights, loc, initial, xctype, 0)
                            new = self.evaluate(ao, weights, loc, initial, xctype, 1)
                            np.testing.assert_allclose(new, reference, atol=3e-11, rtol=2e-12)
                            np.testing.assert_allclose(new, old, atol=3e-11, rtol=2e-12)
                            repeated = self.evaluate(ao, weights, loc, initial, xctype, 1)
                            np.testing.assert_array_equal(new, repeated)
                            self.assertEqual(lib.num_threads(), threads)
                np.testing.assert_array_equal(ao, original_ao)
                np.testing.assert_array_equal(weights, original_weights)

    def test_screened_path_unchanged(self):
        ao, weights, loc, initial = self.inputs(67, 120)
        rng = np.random.default_rng(30)
        screen = rng.choice([0, 8, 14, 25], size=(3, 67)).astype(np.uint8)
        pairs = rng.integers(0, 2, size=(67, 67), dtype=np.uint8)
        pairs |= pairs.T.copy()
        for xctype in ('LDA', 'GGA', 'MGGA'):
            for threads in (1, 4):
                with self.subTest(xc=xctype, threads=threads), lib.with_omp_threads(threads):
                    old = self.evaluate(ao, weights, loc, initial, xctype, 0, screen, pairs)
                    new = self.evaluate(ao, weights, loc, initial, xctype, 1, screen, pairs)
                    np.testing.assert_array_equal(new, old)

    def test_small_workspace_fallback(self):
        ao, weights, loc, initial = self.inputs(236, 897)
        with mock.patch.dict(os.environ, {'PYSCF_DFT_VMAT_MAX_MEMORY_MB': '1'}):
            for xctype in ('LDA', 'GGA', 'MGGA'):
                reference = initial + self.formula(ao, weights, xctype)
                actual = self.evaluate(ao, weights, loc, initial, xctype, 1)
                np.testing.assert_allclose(actual, reference, atol=3e-11, rtol=2e-12)

    def test_default_and_malformed_switch_are_off(self):
        ao, weights, loc, initial = self.inputs(67, 897)
        for xctype in ('LDA', 'GGA', 'MGGA'):
            reference = self.evaluate(ao, weights, loc, initial, xctype, 0)
            for value in ('', 'false', '11', 'garbage'):
                actual = self.evaluate(ao, weights, loc, initial, xctype, value)
                np.testing.assert_array_equal(actual, reference)
            with mock.patch.dict(os.environ):
                os.environ.pop('PYSCF_DFT_DENSE_BLAS', None)
                values = ao[0] if xctype == 'LDA' else ao
                wv = weights[0] if xctype == 'LDA' else weights[:4] if xctype == 'GGA' else weights
                actual = numint._xc_vmat(values, wv, 20, None, None, loc,
                                         initial.copy(), xctype=xctype)
                np.testing.assert_array_equal(actual, reference)


if __name__ == '__main__':
    unittest.main(verbosity=2)
