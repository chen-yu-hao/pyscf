#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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

import unittest
import numpy
from pyscf import lib
from pyscf.dft import numint


class KnownValues(unittest.TestCase):
    def setUp(self):
        self.rng = numpy.random.default_rng(17)
        self.ao_loc = numpy.asarray([0] + list(numpy.cumsum([1, 3, 5] * 15)),
                                    dtype=numpy.int32)
        self.nbas = len(self.ao_loc) - 1
        self.nao = self.ao_loc[-1]

    def _inputs(self, ngrids):
        # Leave gaps between components to exercise the component stride.
        ao = self.rng.normal(size=(8, self.nao, ngrids))[::2].transpose(0, 2, 1)
        wv = self.rng.normal(size=ngrids)
        nblocks = (ngrids + numint.BLKSIZE - 1) // numint.BLKSIZE
        screen = self.rng.choice([0, 8, 14, 25], size=(nblocks, self.nbas)).astype(numpy.uint8)
        pairs = self.rng.integers(0, 2, size=(self.nbas, self.nbas), dtype=numpy.uint8)
        pairs = numpy.asarray(pairs | pairs.T, order='C')
        return ao, wv, screen, pairs

    def _reference(self, bra, ket, wv, screen=None, pairs=None):
        if wv is None:
            wv = numpy.ones(bra.shape[1])
        out = numpy.zeros((self.nao, self.nao))
        shell = numpy.repeat(numpy.arange(self.nbas), numpy.diff(self.ao_loc))
        for g0 in range(0, bra.shape[1], numint.BLKSIZE):
            g1 = g0 + numint.BLKSIZE
            block = numpy.einsum('xgi,xgj,g->ij', bra[1:4, g0:g1],
                                 ket[1:4, g0:g1], wv[g0:g1])
            if screen is not None:
                si = screen[g0 // numint.BLKSIZE].astype(int)
                active = ((si[:, None] > 0) & (si[None, :] > 0) &
                          (si[:, None] + si[None, :] >= 20) & (pairs != 0))
                block *= active[shell[:, None], shell[None, :]]
            out += block
        return out

    def test_screened_tau(self):
        # Full two-block tiles, a partial tile, and the partial-only path.
        for ngrids in (224, 120, 56):
            with self.subTest(ngrids=ngrids), lib.with_omp_threads(2):
                ao, wv, screen, pairs = self._inputs(ngrids)
                expected = self._reference(ao, ao, wv, screen, pairs)
                out = self.rng.normal(size=(self.nao, self.nao))
                out += out.T
                initial = out.copy()
                actual = numint._tau_dot_sparse(ao, ao, wv, 20, screen,
                                                pairs, self.ao_loc, out=out)
                self.assertIs(actual, out)
                numpy.testing.assert_allclose(actual, initial + expected,
                                              atol=1e-11, rtol=1e-13)

    def test_component_stride_and_allocation(self):
        ao, wv, screen, pairs = self._inputs(224)
        # Reverse the derivative stride, which remains contiguous per component.
        ao = ao[::-1]
        expected = self._reference(ao, ao, wv, screen, pairs)
        with lib.with_omp_threads(1):
            actual = numint._tau_dot_sparse(ao, ao, wv, 20, screen, pairs, self.ao_loc)
        self.assertEqual(actual.shape, (self.nao, self.nao))
        numpy.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-13)

    def test_dense_fallback(self):
        for ngrids, no_screen, no_pairs in ((55, False, False),
                                            (120, True, False),
                                            (120, False, True)):
            with self.subTest(ngrids=ngrids, no_screen=no_screen, no_pairs=no_pairs):
                ao, wv, screen, pairs = self._inputs(ngrids)
                ket = ao[::-1]
                expected = self._reference(ao, ket, wv)
                actual = numint._tau_dot_sparse(
                    ao, ket, wv, 20, None if no_screen else screen,
                    None if no_pairs else pairs, self.ao_loc)
                numpy.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-13)

    def test_dense_accumulation(self):
        for ngrids in (1, 257, 2049):
            ao, wv, screen, pairs = self._inputs(ngrids)
            ket = ao[::-1]
            initial = self.rng.normal(size=(self.nao, self.nao))
            expected = initial + self._reference(ao, ket, wv)
            for threads in (1, 4):
                with self.subTest(ngrids=ngrids, threads=threads), lib.with_omp_threads(threads):
                    out = initial.copy()
                    actual = numint._tau_dot_sparse(ao, ket, wv, 20, None,
                                                    pairs, self.ao_loc, out)
                    self.assertIs(actual, out)
                    numpy.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-13)
                    repeated = numint._tau_dot_sparse(ao, ket, wv, 20, None,
                                                      pairs, self.ao_loc, initial.copy())
                    numpy.testing.assert_array_equal(actual, repeated)

    def test_dense_signed_gram(self):
        for weight_sign in (0, 1, -1):
            ao, wv, screen, pairs = self._inputs(1025)
            if weight_sign:
                wv = abs(wv) * weight_sign
            wv[::7] = 0
            initial = self.rng.normal(size=(self.nao, self.nao))
            expected = initial + self._reference(ao, ao, wv)
            for threads in (1, 4):
                with self.subTest(weight_sign=weight_sign, threads=threads), lib.with_omp_threads(threads):
                    actual = numint._tau_dot_sparse(ao, ao, wv, 20, None,
                                                    pairs, self.ao_loc, initial.copy())
                    numpy.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-13)

    def test_dense_scratch_limit_fallback(self):
        nao = 1024
        ao = self.rng.normal(size=(4, nao, 1)).transpose(0, 2, 1)
        ket = ao[::-1]
        wv = numpy.array([-0.4])
        expected = numpy.einsum('xgi,xgj,g->ij', ao[1:4], ket[1:4], wv)
        with lib.with_omp_threads(8):
            actual = numint._tau_dot_sparse(ao, ket, wv, 20, None, None, None)
        numpy.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-13)


if __name__ == '__main__':
    unittest.main()
