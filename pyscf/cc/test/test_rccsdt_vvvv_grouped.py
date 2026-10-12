# Copyright 2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

import copy
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pyscf.cc import rccsdt


def _random_residual_case(nocc=5, nvir=5):
    rng = np.random.default_rng(1808)
    mycc = SimpleNamespace(nocc=nocc, nmo=nocc+nvir, cc_order=3,
                           blksize=3, blksize_oovv=2, blksize_oooo=3,
                           stdout=sys.stdout, verbose=0, einsum_backend='numpy')
    mycc.tri2block_map, mycc.tri2block_mask, mycc.tri2block_tp = rccsdt.setup_tri2block_rhf(mycc)
    shapes = {'F_oo': (nocc, nocc), 'F_vv': (nvir, nvir),
              'W_oooo': (nocc,)*4, 'W_vvvv': (nvir,)*4,
              'W_ovvo': (nocc, nvir, nvir, nocc),
              'W_ovov': (nocc, nvir, nocc, nvir),
              'W_vooo': (nvir, nocc, nocc, nocc),
              'W_vvvo': (nvir, nvir, nvir, nocc)}
    imds = SimpleNamespace(**{name: rng.normal(size=shape) for name, shape in shapes.items()})
    t2 = rng.normal(size=(nocc, nocc, nvir, nvir))
    ntri = nocc*(nocc+1)*(nocc+2)//6
    t3 = rng.normal(size=(ntri, nvir, nvir, nvir))
    return mycc, imds, t2, t3


class KnownValues(unittest.TestCase):
    def test_grouped_vvvv_matches_packed_loop(self):
        rng = np.random.default_rng(1807)
        nocc, nvir = 5, 6
        ntri = nocc * (nocc + 1) * (nocc + 2) // 6
        w = rng.normal(size=(nvir, nvir, nvir, nvir))
        t3 = rng.normal(size=(ntri, nvir, nvir, nvir))
        initial = rng.normal(size=t3.shape)
        expected = initial.copy()
        index = 0
        for i in range(nocc):
            for j in range(i, nocc):
                for k in range(j, nocc):
                    rccsdt._einsum('numpy', 'abde,dec->abc', w, t3[index],
                                   out=expected[index], beta=1.)
                    rccsdt._einsum('numpy', 'acde,dbe->abc', w, t3[index],
                                   out=expected[index], beta=1.)
                    rccsdt._einsum('numpy', 'bcde,ade->abc', w, t3[index],
                                   out=expected[index], beta=1.)
                    index += 1
        actual = initial.copy()
        rccsdt._r3_vvvv_tri_grouped(
            lambda script, *args, **kwargs: rccsdt._einsum('numpy', script, *args, **kwargs),
            w, t3, actual)
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)

    def test_grouped_empty_output(self):
        w = np.zeros((3, 3, 3, 3))
        t3 = np.zeros((0, 3, 3, 3))
        out = np.zeros_like(t3)
        rccsdt._r3_vvvv_tri_grouped(
            lambda script, *args, **kwargs: rccsdt._einsum('numpy', script, *args, **kwargs),
            w, t3, out)
        self.assertEqual(out.size, 0)

    def test_full_residual_pytblis(self):
        pytblis = rccsdt._get_pytblis()
        if pytblis is None:
            self.skipTest('pytblis is optional')
        for nocc in (5, 7):
            with self.subTest(nocc=nocc):
                mycc, imds, t2, t3 = _random_residual_case(nocc, 4)
                expected = rccsdt.compute_r3_tri(mycc, copy.deepcopy(imds), t2, t3)
                mycc.einsum_backend = 'pytblis'
                t2.flags.writeable = t3.flags.writeable = False
                with patch.object(rccsdt, '_r3_vvvv_tri_grouped',
                                  wraps=rccsdt._r3_vvvv_tri_grouped) as grouped:
                    actual = rccsdt.compute_r3_tri(mycc, copy.deepcopy(imds), t2, t3)
                grouped.assert_called_once()
                np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-12)

    def test_missing_pytblis_retains_unbatched_path(self):
        mycc, imds, t2, t3 = _random_residual_case(3, 3)
        expected = rccsdt.compute_r3_tri(mycc, copy.deepcopy(imds), t2, t3)
        mycc.einsum_backend = 'pytblis'
        with patch.object(rccsdt, '_get_pytblis', return_value=None), \
                patch.object(rccsdt, '_r3_vvvv_tri_grouped') as grouped:
            actual = rccsdt.compute_r3_tri(mycc, copy.deepcopy(imds), t2, t3)
        grouped.assert_not_called()
        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
