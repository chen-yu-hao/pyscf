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

"""RCCSDTQ-local accumulation, including edge-block views and aliasing."""

import unittest
from unittest.mock import patch

import numpy as np
from pyscf.cc import rccsdt, rccsdtq, rccsdtq_highm, uccsdt


class KnownValues(unittest.TestCase):
    def test_rank_four_accumulation_and_input_preservation(self):
        rng = np.random.default_rng(910)
        left = rng.normal(size=(2, 2, 2, 3))
        right = rng.normal(size=(2, 2, 3, 2, 2, 2))
        left_original, right_original = left.copy(), right.copy()
        expression = 'abej,iklecd->ijklabcd'
        contraction = np.einsum(expression, left, right, optimize=True)
        for backend in ('numpy', 'pyscf'):
            for alpha in (1., -1., .5, -1./6.):
                for layout in ('contiguous', 'tail', 'negative'):
                    with self.subTest(backend=backend, alpha=alpha, layout=layout):
                        if layout == 'contiguous':
                            out = rng.normal(size=contraction.shape)
                        else:
                            backing = rng.normal(size=(4, 4, 4, 4, 2, 2, 2, 2))
                            out = backing[:2, :3, :2, :3]
                            if layout == 'negative':
                                out = out[..., ::-1]
                            self.assertFalse(out.flags.c_contiguous)
                        expected = alpha * contraction + out.copy()
                        result = rccsdtq._einsum(backend, expression, left, right,
                                               out=out, alpha=alpha, beta=1.)
                        self.assertIs(result, out)
                        np.testing.assert_allclose(result, expected, atol=2e-12, rtol=2e-12)
                        np.testing.assert_array_equal(left, left_original)
                        np.testing.assert_array_equal(right, right_original)

    def test_aliasing_output_preserves_original_operand_semantics(self):
        rng = np.random.default_rng(911)
        for backend in ('numpy', 'pyscf'):
            for layout in ('same', 'transpose', 'negative'):
                for alpha in (1., -1., .25):
                    with self.subTest(backend=backend, layout=layout, alpha=alpha):
                        left, right = rng.normal(size=(2, 4, 4))
                        out = left if layout == 'same' else left.T if layout == 'transpose' else left[:, ::-1]
                        self.assertTrue(np.shares_memory(left, out))
                        right_original = right.copy()
                        expected = alpha * (left.copy() @ right) + out.copy()
                        returned = rccsdtq._einsum(backend, 'ij,jk->ik', left, right,
                                                 out=out, alpha=alpha, beta=1.)
                        self.assertIs(returned, out)
                        np.testing.assert_allclose(out, expected, atol=2e-12, rtol=2e-12)
                        np.testing.assert_array_equal(right, right_original)

    def test_fallback_beta_and_returned_contraction(self):
        rng = np.random.default_rng(912)
        left = rng.normal(size=(4, 3))[:, ::-1]
        right = rng.normal(size=(3, 5))[::-1]
        product = left @ right
        for backend in ('numpy', 'pyscf'):
            for beta in (0., -.5, .25, 2.):
                with self.subTest(backend=backend, beta=beta):
                    out = rng.normal(size=(5, 4)).T
                    expected = -.25 * product + beta * out.copy()
                    returned = rccsdtq._einsum(backend, 'ij,jk->ik', left, right,
                                             out=out, alpha=-.25, beta=beta)
                    self.assertIs(returned, out)
                    np.testing.assert_allclose(out, expected, atol=2e-12, rtol=2e-12)
            returned = rccsdtq._einsum(backend, 'ij,jk->ik', left, right, alpha=.25, beta=1.)
            np.testing.assert_allclose(returned, .25 * product, atol=2e-12, rtol=2e-12)

    def test_pytblis_dispatch_keeps_native_out_contract(self):
        left, right, out = np.ones((2, 2)), np.ones((2, 2)), np.zeros((2, 2))
        with patch.object(rccsdtq, '_base_einsum', return_value=out) as original:
            returned = rccsdtq._einsum('pytblis', 'ij,jk->ik', left, right,
                                     out=out, alpha=.25, beta=1.)
        self.assertIs(returned, out)
        original.assert_called_once_with('pytblis', 'ij,jk->ik', left, right,
                                         out=out, alpha=.25, beta=1.)

    def test_optimized_helper_is_local_to_rccsdtq(self):
        self.assertIs(rccsdtq._base_einsum, rccsdt._einsum)
        self.assertIs(rccsdtq_highm._einsum, rccsdtq._einsum)
        self.assertIs(uccsdt._einsum, rccsdt._einsum)
        self.assertIsNot(rccsdtq._einsum, rccsdt._einsum)


if __name__ == '__main__':
    unittest.main()
