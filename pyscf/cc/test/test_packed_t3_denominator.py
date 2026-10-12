#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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

import itertools
import math
from types import SimpleNamespace
import unittest

import numpy as np
from pyscf import lib
from pyscf.cc import uccsdt, uccsdt_highm


def make_case(nocc=(5, 3), nvir=(6, 4), level_shift=.23):
    rng = np.random.default_rng(517)
    cc = SimpleNamespace(nocc=nocc, nmo=tuple(o + v for o, v in zip(nocc, nvir)),
                         level_shift=level_shift, cc_order=3)
    energies = tuple(np.r_[-rng.uniform(.4, 2., o), rng.uniform(.1, 3., v)]
                     for o, v in zip(nocc, nvir))
    oa, ob = nocc
    va, vb = nvir
    shapes = ((math.comb(oa, 3), math.comb(va, 3)),
              (math.comb(oa, 2), math.comb(va, 2), ob, vb),
              (math.comb(ob, 2), math.comb(vb, 2), oa, va),
              (math.comb(ob, 3), math.comb(vb, 3)))
    return cc, [rng.standard_normal(shape) for shape in shapes], energies


def explicit_result(cc, blocks, energies):
    """Denominator from the physical spin-labelled occupied/virtual energies."""
    result = [block.copy() for block in blocks]
    for block, spin, rank in zip(result, (0, 0, 1, 1), (3, 2, 2, 3)):
        nocc, nvir = cc.nocc[spin], cc.nmo[spin] - cc.nocc[spin]
        em = energies[spin]
        for oi, holes in enumerate(itertools.combinations(range(nocc), rank)):
            for vi, particles in enumerate(itertools.combinations(range(nvir), rank)):
                base = sum(em[i] for i in holes) - sum(em[nocc + a] for a in particles)
                if rank == 3:
                    block[oi, vi] /= base - 3 * cc.level_shift
                else:
                    other = energies[1 - spin]
                    for k, c in np.ndindex(block.shape[2:]):
                        denominator = base + other[k] - other[cc.nocc[1 - spin] + c]
                        block[oi, vi, k, c] /= denominator - 3 * cc.level_shift
    return result


def setup_maps(cc):
    names = ('t2c_map_6f_oa', 't2c_mask_6f_oa', 't2c_map_2f_oa', 't2c_mask_2f_oa',
             't2c_map_6f_va', 't2c_mask_6f_va', 't2c_map_2f_va', 't2c_mask_2f_va',
             't2c_map_6f_ob', 't2c_mask_6f_ob', 't2c_map_2f_ob', 't2c_mask_2f_ob',
             't2c_map_6f_vb', 't2c_mask_6f_vb', 't2c_map_2f_vb', 't2c_mask_2f_vb')
    for name, value in zip(names, uccsdt.setup_tri2block_t3_uhf(cc)):
        setattr(cc, name, value)


class KnownValues(unittest.TestCase):
    def setUp(self):
        self.old_threads = lib.num_threads()
        lib.num_threads(1)

    def tearDown(self):
        lib.num_threads(self.old_threads)

    def test_unequal_spin_spaces(self):
        for shift in (0., .23):
            for threads in (1, 4):
                with self.subTest(shift=shift, threads=threads):
                    cc, blocks, energies = make_case(level_shift=shift)
                    expected = explicit_result(cc, blocks, energies)
                    identities = [id(block) for block in blocks]
                    lib.num_threads(threads)
                    result = uccsdt.r3_tri_divide_e_uhf_(cc, blocks, energies)
                    self.assertIs(result, blocks)
                    self.assertEqual(identities, [id(block) for block in result])
                    for actual, reference in zip(result, expected):
                        np.testing.assert_allclose(actual, reference, atol=2e-16, rtol=5e-16)

    def test_small_and_empty_sectors(self):
        for nocc, nvir in (((0, 0), (4, 3)), ((1, 2), (3, 4)), ((2, 3), (1, 2)),
                            ((3, 1), (3, 0)), ((4, 3), (5, 6))):
            with self.subTest(nocc=nocc, nvir=nvir):
                cc, blocks, energies = make_case(nocc, nvir)
                expected = explicit_result(cc, blocks, energies)
                uccsdt.r3_tri_divide_e_uhf_(cc, blocks, energies)
                for actual, reference in zip(blocks, expected):
                    np.testing.assert_allclose(actual, reference, atol=2e-16, rtol=5e-16)

    def test_full_tensor_and_norm(self):
        cc, blocks, energies = make_case((4, 3), (5, 4))
        setup_maps(cc)
        full = list(uccsdt.tamps_tri2full_uhf(cc, blocks))
        uccsdt_highm.r3_divide_e_uhf_(cc, full, energies)
        uccsdt.r3_tri_divide_e_uhf_(cc, blocks, energies)
        for packed, reference in zip(blocks, uccsdt.tamps_full2tri_uhf(cc, full)):
            np.testing.assert_array_equal(packed, reference)
        for packed, dense, multiplicity in zip(blocks, full, (36, 4, 4, 36)):
            self.assertAlmostEqual(np.linalg.norm(dense),
                                   np.linalg.norm(packed) * np.sqrt(multiplicity), 12)

    def test_strided_output_and_energies(self):
        cc, blocks, energies = make_case((4, 3), (4, 5))
        blocks = [np.stack((b, b), axis=-1)[..., 0] for b in blocks]
        energies = tuple(np.repeat(e, 2)[::2] for e in energies)
        expected = explicit_result(cc, blocks, energies)
        uccsdt.r3_tri_divide_e_uhf_(cc, blocks, energies)
        for actual, reference in zip(blocks, expected):
            np.testing.assert_allclose(actual, reference, atol=2e-16, rtol=5e-16)

    def test_spin_exchange(self):
        cc, blocks, energies = make_case()
        swapped_cc = SimpleNamespace(nocc=cc.nocc[::-1], nmo=cc.nmo[::-1],
                                     level_shift=cc.level_shift)
        swapped = [block.copy() for block in blocks[::-1]]
        uccsdt.r3_tri_divide_e_uhf_(cc, blocks, energies)
        uccsdt.r3_tri_divide_e_uhf_(swapped_cc, swapped, energies[::-1])
        for original, flipped in zip(blocks, swapped[::-1]):
            np.testing.assert_array_equal(original, flipped)

    def test_zero_denominator_is_not_regularized(self):
        cc, blocks, energies = make_case((3, 3), (3, 3), level_shift=0.)
        energies = tuple(np.zeros_like(e) for e in energies)
        blocks = [np.ones_like(block) for block in blocks]
        uccsdt.r3_tri_divide_e_uhf_(cc, blocks, energies)
        for block in blocks:
            self.assertTrue(np.isposinf(block).all())


if __name__ == '__main__':
    unittest.main()
