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
import numpy as np
from pyscf.lib import numpy_helper

try:
    import pytblis
except (ImportError, OSError):
    pytblis = None


@unittest.skipIf(pytblis is None, 'pytblis is not installed')
class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = pytblis.get_num_threads()
        pytblis.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        pytblis.set_num_threads(cls.threads)

    def setUp(self):
        self.rng = np.random.default_rng(18)
        for patcher in (mock.patch.object(numpy_helper, 'EINSUM_BACKEND', 'pytblis'),
                        mock.patch.object(numpy_helper, 'pytblis', pytblis, create=True)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_strided_rank4_accumulation(self):
        a = self.rng.random((4, 5, 13, 12))
        b = self.rng.random((13, 12, 7, 6))
        storage = self.rng.random((4, 5, 14, 12))
        before = storage.copy()
        out = storage[:, :, ::2, ::2]
        expected = .5 * np.einsum('ijef,efab->ijab', a, b) + out
        result = numpy_helper.einsum('ijef,efab->ijab', a, b,
                                     out=out, alpha=.5, beta=1)
        self.assertIs(result, out)
        np.testing.assert_allclose(out, expected, rtol=1e-13, atol=1e-13)
        before[:, :, ::2, ::2] = expected
        np.testing.assert_allclose(storage, before, rtol=1e-13, atol=1e-13)

    def test_complex_accumulation(self):
        a = self.rng.random((24, 25)) + 1j * self.rng.random((24, 25))
        b = self.rng.random((25, 26)) + 1j * self.rng.random((25, 26))
        out = self.rng.random((24, 26)).astype(complex)
        expected = (.5 + .2j) * (a @ b) - .3j * out
        numpy_helper.einsum('ik,kj->ij', a, b, out=out, alpha=.5+.2j, beta=-.3j)
        np.testing.assert_allclose(out, expected, rtol=1e-13, atol=1e-13)

    def test_dtype_fallback(self):
        a = self.rng.random((24, 25))
        b = self.rng.random((25, 26))
        out = self.rng.random((24, 26)).astype(complex)
        expected = .5j * (a @ b) + out
        numpy_helper.einsum('ik,kj->ij', a, b, out=out, alpha=.5j, beta=1)
        np.testing.assert_allclose(out, expected, rtol=1e-13, atol=1e-13)

    def test_mixed_real_complex(self):
        a = self.rng.random((24, 25)) + 1j * self.rng.random((24, 25))
        b = self.rng.random((25, 26))
        out = self.rng.random((24, 26)).astype(complex)
        expected = .5 * (a @ b) + out
        numpy_helper.einsum('ik,kj->ij', a, b, out=out, alpha=.5, beta=1)
        np.testing.assert_allclose(out, expected, rtol=1e-13, atol=1e-13)

    def test_small_array_fallback(self):
        a = self.rng.random((3, 4))
        b = self.rng.random((4, 5))
        out = self.rng.random((3, 5))
        expected = .5 * (a @ b) + out
        numpy_helper.einsum('ik,kj->ij', a, b, out=out, alpha=.5, beta=1)
        np.testing.assert_allclose(out, expected, rtol=1e-13, atol=1e-13)

    def test_partial_trace_accumulation(self):
        a = self.rng.random((24, 25))
        b = self.rng.random((26, 25))
        out = self.rng.random(25)
        expected = .5 * np.einsum('ij,kj->j', a, b) + out.copy()
        native_contract = pytblis.contract
        supported = numpy_helper._pytblis_supports_partial_trace(native_contract)
        with mock.patch.object(pytblis, 'contract', autospec=True, side_effect=native_contract) as contract:
            result = numpy_helper.einsum('ij,kj->j', a, b, out=out, alpha=.5, beta=1)
        self.assertIs(result, out)
        if supported:
            self.assertTrue(contract.call_args.kwargs['allow_partial_trace'])
        else:
            contract.assert_not_called()
        np.testing.assert_allclose(out, expected, rtol=1e-13, atol=1e-13)

    def test_legacy_contract_partial_trace_fallback(self):
        def legacy_contract(script, a, b, alpha=1, beta=0, out=None):
            raise AssertionError('Legacy TBLIS cannot handle a partial trace')

        a = self.rng.random((24, 25))
        b = self.rng.random((26, 25))
        out = self.rng.random(25)
        expected = .5 * np.einsum('ij,kj->j', a, b) + out.copy()
        with mock.patch.object(pytblis, 'contract', legacy_contract):
            result = numpy_helper.einsum('ij,kj->j', a, b, out=out, alpha=.5, beta=1)
        self.assertIs(result, out)
        np.testing.assert_allclose(out, expected, rtol=1e-13, atol=1e-13)

    def test_legacy_multioperand_partial_trace_fallback(self):
        def legacy_contract(script, a, b, alpha=1, beta=0, out=None):
            raise AssertionError('Legacy TBLIS cannot handle a partial trace')

        a = self.rng.random((1, 4, 3))
        b = self.rng.random((3, 5))
        c = self.rng.random((5, 4))
        expected = np.einsum('iab,bc,cd->ad', a, b, c)
        with mock.patch.object(pytblis, 'contract', legacy_contract), \
                mock.patch.object(pytblis, 'einsum', side_effect=AssertionError('Legacy TBLIS')):
            result = numpy_helper.einsum('iab,bc,cd->ad', a, b, c)
        np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-13)

    def test_custom_contract(self):
        a = self.rng.random((3, 4))
        b = self.rng.random((4, 5))
        custom = mock.Mock(wraps=numpy_helper.contract)
        result = numpy_helper.einsum('ik,kj->ij', a, b, alpha=.5, _contract=custom)
        np.testing.assert_allclose(result, .5 * (a @ b))
        custom.assert_called_once()

    def test_default_multioperand(self):
        a = self.rng.random((24, 25))
        b = self.rng.random((25, 26))
        c = self.rng.random((26, 27))
        expected = np.einsum('ik,kl,lj->ij', a, b, c, optimize=True)
        result = numpy_helper.einsum('ik,kl,lj->ij', a, b, c, optimize=True)
        np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-13)


if __name__ == '__main__':
    unittest.main()
