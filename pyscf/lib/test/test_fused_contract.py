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
from unittest import mock
import warnings
import numpy as np
from pyscf.lib import numpy_helper as nh


class KnownValues(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(24)
        self.backend = mock.patch.object(nh, 'EINSUM_BACKEND', None)
        self.threshold = mock.patch.object(nh, 'EINSUM_MAX_SIZE', 0)
        self.backend.start()
        self.threshold.start()
        self.addCleanup(self.backend.stop)
        self.addCleanup(self.threshold.stop)

    def check_contract(self, script, a, b, out, fused, alpha=-0.5, beta=1.25):
        expected = alpha * np.einsum(script, a, b) + beta * out.copy()
        with mock.patch.object(nh, 'dot', wraps=nh.dot) as dot:
            actual = nh.contract(script, a, b, out=out, alpha=alpha, beta=beta)
        self.assertIs(actual, out)
        np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)
        self.assertEqual(dot.call_count, 1)
        target = dot.call_args.kwargs.get('c')
        self.assertEqual(target is not None, fused)
        if fused:
            self.assertTrue(np.shares_memory(target, out))

    def test_rank_four_accumulation(self):
        a = self.rng.random((3, 4, 5))
        b = self.rng.random((5, 6, 7))
        for beta in (0, 1, -0.3):
            out = self.rng.random((3, 4, 6, 7))
            self.check_contract('ijk,klm->ijlm', a, b, out, True, beta=beta)

    def test_fused_accumulation_with_warnings_as_errors(self):
        a = self.rng.random((3, 4, 5))
        b = self.rng.random((5, 6, 7))
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            out = self.rng.random((3, 4, 6, 7))
            self.check_contract('ijk,klm->ijlm', a, b, out, True)
            for order in ('C', 'F'):
                out = np.array(self.rng.random((12, 42)), order=order)
                self.check_contract('ik,kj->ij', a.reshape(12, 5),
                                    b.reshape(5, 42), out, True)

    def test_transposed_and_fortran_matrix(self):
        a = self.rng.random((3, 4))
        b = self.rng.random((4, 5))
        for order in ('C', 'F'):
            out = np.array(self.rng.random((5, 3)), order=order)
            self.check_contract('ik,kj->ji', a, b, out, True)

    def test_permuted_tensor_view(self):
        a = self.rng.random((3, 4, 5))
        b = self.rng.random((5, 6, 7))
        out = self.rng.random((3, 4, 6, 7)).transpose(0, 2, 1, 3)
        self.check_contract('ijk,klm->iljm', a, b, out, True)
        self.check_contract('ijk,klm->iljm', a, b, out.copy(), False)

    def test_strided_output(self):
        a = self.rng.random((3, 4))
        b = self.rng.random((4, 5))
        for out in (self.rng.random((3, 10))[:, ::2],
                    self.rng.random((3, 5))[::-1]):
            self.check_contract('ik,kj->ij', a, b, out, False)

    def test_complex_accumulation(self):
        a = self.rng.random((3, 4)) + 1j * self.rng.random((3, 4))
        b = self.rng.random((4, 5)) + 1j * self.rng.random((4, 5))
        for order in ('C', 'F'):
            out = np.array(self.rng.random((3, 5)) + 0.4j, order=order)
            self.check_contract('ik,kj->ij', a, b, out, True,
                                alpha=-0.5+0.2j, beta=0.7-0.3j)

    def test_dtype_fallback(self):
        a = self.rng.random((3, 4))
        b = self.rng.random((4, 5))
        out = self.rng.random((3, 5)) + 0.4j
        self.check_contract('ik,kj->ij', a, b, out, False, alpha=0.2j)
        self.check_contract('ik,kj->ij', a+0.1j, b, out, False)
        self.check_contract('ik,kj->ij', a.astype(np.float32), b,
                            out.real.copy(), False)
        with mock.patch.object(nh, 'dot', wraps=nh.dot) as dot:
            with self.assertRaises(TypeError):
                nh.contract('ik,kj->ij', a, b, out=out.real.copy(),
                            alpha=0.2j, beta=1)
        self.assertIsNone(dot.call_args.kwargs.get('c'))

    def test_aliasing_fallback(self):
        # A requires a copy to flatten its free axes; still check its original
        # storage for aliasing before sending the output buffer to BLAS.
        a = self.rng.random((2, 3, 4)).transpose(1, 0, 2)
        b = self.rng.random((4, 4))
        self.check_contract('ijk,kl->ijl', a, b, a, False)
        a = self.rng.random((4, 4))
        self.check_contract('ik,kj->ij', a, b, b, False)

    def test_unaligned_and_readonly_output(self):
        a = self.rng.random((3, 4))
        b = self.rng.random((4, 5))
        out = np.ndarray((3, 5), buffer=np.zeros(121, dtype=np.uint8),
                         dtype=np.float64, offset=1)
        self.check_contract('ik,kj->ij', a, b, out, False)
        out.flags.writeable = False
        with mock.patch.object(nh, 'dot', wraps=nh.dot) as dot:
            with self.assertRaises(ValueError):
                nh.contract('ik,kj->ij', a, b, out=out)
        self.assertIsNone(dot.call_args.kwargs.get('c'))

    def test_zero_dimensions_and_numpy_path(self):
        for threshold in (0, 500):
            with mock.patch.object(nh, 'EINSUM_MAX_SIZE', threshold):
                for shapes in (((3, 0), (0, 5)), ((0, 4), (4, 5))):
                    a, b = (np.zeros(shape) for shape in shapes)
                    out = np.ones((a.shape[0], b.shape[1]))
                    expected = 0.3 * out.copy()
                    actual = nh.contract('ik,kj->ij', a, b, out=out,
                                         alpha=-0.5, beta=0.3)
                    self.assertIs(actual, out)
                    np.testing.assert_array_equal(actual, expected)
        with mock.patch.object(nh, 'EINSUM_MAX_SIZE', 500), \
                mock.patch.object(nh, 'dot', wraps=nh.dot) as dot:
            a = self.rng.random((3, 4))
            b = self.rng.random((4, 5))
            out = np.ones((3, 5))
            nh.contract('ik,kj->ij', a, b, out=out, alpha=-0.5, beta=0.3)
            np.testing.assert_allclose(out, -0.5 * np.einsum('ik,kj->ij', a, b) + 0.3)
            dot.assert_not_called()


if __name__ == '__main__':
    unittest.main()
