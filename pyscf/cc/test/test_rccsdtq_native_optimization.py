# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independent permutation/stride checks for restricted T4 block unpacking."""
from itertools import combinations_with_replacement, permutations, product
from types import SimpleNamespace
import unittest

import numpy as np
from pyscf import lib
from pyscf.cc import rccsdt, rccsdtq


def _reference(t4, nocc, bounds, blockshape, triples=False, holes=()):
    """Stable occupied sorting and NumPy transposes, without C map indices."""
    nvir = t4.shape[-1]
    result = np.full(tuple(blockshape)+(nvir,)*4, -1729.)
    canonical = {indices: row for row, indices in enumerate(
        combinations_with_replacement(range(nocc), 4))}
    for occupied in product(*(range(start, stop) for start, stop in bounds)):
        if occupied in holes:
            continue
        sorted_axes = np.argsort(occupied, kind='stable')
        transpose = np.argsort(sorted_axes)
        row = canonical[tuple(sorted(occupied))]
        block = t4[row].transpose(transpose)
        if triples:
            block = block + block.transpose(1, 2, 0, 3) + block.transpose(1, 3, 2, 0)
        local = tuple(i-start for i, (start, _) in zip(occupied, bounds))
        result[local] = block
    return result


class KnownValues(unittest.TestCase):
    def setUp(self):
        self.threads = lib.num_threads()
        lib.num_threads(1)
        self.nocc = 4
        self.map, self.mask, _ = rccsdt.setup_tri2block_rhf(
            SimpleNamespace(nocc=self.nocc, cc_order=4))

    def tearDown(self):
        lib.num_threads(self.threads)

    def _check(self, nvir, bounds, blockshape, triples, holes=()):
        rng = np.random.default_rng(811+nvir)
        count = len(tuple(combinations_with_replacement(range(self.nocc), 4)))
        t4 = rng.normal(size=(count,)+(nvir,)*4)
        original = t4.copy()
        mask = self.mask.copy()
        for indices in holes:
            mask[(slice(None),)+indices] = False
        expected = _reference(t4, self.nocc, bounds, blockshape, triples, holes)
        function = rccsdtq.unpack_t4_tri2block_triples_ if triples else rccsdtq.unpack_t4_tri2block_
        for threads in (1, 4):
            lib.num_threads(threads)
            result = np.full(expected.shape, -1729.)
            returned = function(t4, result, self.map, mask,
                                *(index for bound in bounds for index in bound),
                                self.nocc, nvir, *blockshape)
            self.assertIs(returned, result)
            np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(t4, original)

    def test_all_24_permutations_and_repeated_occupied_indices(self):
        # All distinct occupied permutations map through each of the 24
        # native permutation codes; the whole domain also includes repeats.
        owners = set()
        for occupied in permutations(range(4)):
            selected = np.flatnonzero(self.mask[(slice(None),)+occupied])
            self.assertEqual(len(selected), 1)
            owners.add(int(selected[0]))
        self.assertEqual(owners, set(range(24)))
        self.assertTrue(np.all(self.mask.sum(axis=0) == 1))
        for nvir in (1, 3, 5):
            for triples in (False, True):
                with self.subTest(nvir=nvir, triples=triples):
                    self._check(nvir, ((0, 4),)*4, (4,)*4, triples)

    def test_nonzero_starts_padding_and_mask_holes(self):
        bounds = ((1, 4), (0, 2), (2, 4), (1, 3))
        blockshape = (4, 3, 3, 4)
        holes = ((1, 0, 2, 1), (3, 1, 3, 2))
        for nvir in (1, 4):
            for triples in (False, True):
                with self.subTest(nvir=nvir, triples=triples):
                    self._check(nvir, bounds, blockshape, triples, holes)

    def test_identity_copy_preserves_bits(self):
        # This region has only the identity occupied permutation. Include
        # signed zero and nonfinite values: a plain unpack is exactly a copy.
        t4 = np.arange(35*16, dtype=np.float64).reshape(35, 2, 2, 2, 2)
        t4[0].flat[:4] = [0., -0., np.inf, np.nan]
        output = np.zeros((1, 1, 1, 1, 2, 2, 2, 2))
        rccsdtq.unpack_t4_tri2block_(t4, output, self.map, self.mask,
                                  0, 1, 0, 1, 0, 1, 0, 1, 4, 2, 1, 1, 1, 1)
        np.testing.assert_array_equal(output.reshape(-1).view(np.uint64), t4[0].reshape(-1).view(np.uint64))


if __name__ == '__main__':
    unittest.main()
