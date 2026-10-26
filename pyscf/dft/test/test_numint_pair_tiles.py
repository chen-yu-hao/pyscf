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
    def check_pair_tiles(self, widths):
        rng = numpy.random.default_rng(23)
        ao_loc = numpy.asarray([0] + list(numpy.cumsum(widths)), dtype=numpy.int32)
        nao = ao_loc[-1]
        nbas = len(widths)
        shell = numpy.repeat(numpy.arange(nbas), widths)
        for ngrids in (56, 120, 224):
            bra = rng.normal(size=(4, nao, ngrids)).transpose(0, 2, 1)
            ket = rng.normal(size=(4, nao, ngrids)).transpose(0, 2, 1)
            weight = rng.normal(size=ngrids)
            screen = rng.choice([0, 8, 14, 25],
                                size=((ngrids+55)//56, nbas)).astype(numpy.uint8)
            pairs = rng.integers(0, 2, size=(nbas, nbas), dtype=numpy.uint8)
            pairs = numpy.asarray(pairs | pairs.T, order='C')
            for kind in ('weighted', 'unweighted', 'tau'):
                for hermi in ((0, 1) if kind != 'tau' else (1,)):
                    right = bra if hermi else ket
                    wv = None if kind == 'unweighted' else weight
                    initial = rng.normal(size=(nao, nao))
                    if hermi:
                        initial += initial.T
                    expected = initial.copy()
                    components = slice(1, 4) if kind == 'tau' else slice(0, 1)
                    for g0 in range(0, ngrids, numint.BLKSIZE):
                        g1 = min(g0 + numint.BLKSIZE, ngrids)
                        si = screen[g0 // numint.BLKSIZE].astype(int)
                        active = ((si[:, None] > 0) & (si[None, :] > 0) &
                                  (si[:, None] + si[None, :] >= 20) & (pairs != 0))
                        weights = numpy.ones(g1-g0) if wv is None else wv[g0:g1]
                        block = numpy.einsum('xgi,xgj,g->ij',
                                             bra[components, g0:g1],
                                             right[components, g0:g1], weights)
                        expected += block * active[shell[:, None], shell[None, :]]
                    results = []
                    for threads in (1, 4):
                        with self.subTest(ngrids=ngrids, kind=kind, hermi=hermi,
                                          threads=threads), lib.with_omp_threads(threads):
                            out = initial.copy()
                            if kind == 'tau':
                                actual = numint._tau_dot_sparse(
                                    bra, right, wv, 20, screen, pairs, ao_loc, out)
                            else:
                                actual = numint._dot_ao_ao_sparse(
                                    bra[0], right[0], wv, 20, screen, pairs,
                                    ao_loc, hermi, out)
                            self.assertIs(actual, out)
                            numpy.testing.assert_allclose(actual, expected,
                                                          atol=1e-11, rtol=1e-13)
                            results.append(actual)
                    numpy.testing.assert_array_equal(*results)

    def test_multiple_pair_tiles(self):
        # 288 AOs form nine 32-AO tiles and 45 Hermitian output tasks.
        self.check_pair_tiles([1, 3, 5, 7] * 18)

    def test_shell_wider_than_tile(self):
        # A Cartesian l=8 shell has 45 AOs and must remain intact.
        self.check_pair_tiles([1, 45, 3, 5, 7, 15, 1])


if __name__ == '__main__':
    unittest.main()
