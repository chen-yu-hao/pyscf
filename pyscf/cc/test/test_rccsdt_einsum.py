# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import unittest
from unittest.mock import patch

import numpy as np

from pyscf.cc import rccsdt
from pyscf.cc import uccsdt
from types import SimpleNamespace


class KnownValues(unittest.TestCase):
    def test_three_operands_and_empty_contractions(self):
        rng = np.random.default_rng(423)
        cases = [('ij,jk,kl->il', (rng.normal(size=(3, 4)),
                                  rng.normal(size=(4, 5)), rng.normal(size=(5, 3)))),
                 ('ij,jk->ik', (np.empty((3, 0)), np.empty((0, 3))))]
        for expression, tensors in cases:
            product = np.einsum(expression, *tensors, optimize=True)
            for backend in ('numpy', 'pyscf'):
                for alpha in (1., -1., .25):
                    for beta in (0., 1., -.5):
                        out = rng.normal(size=product.shape).T
                        expected = alpha * product + beta * out.copy()
                        with self.subTest(expression=expression, backend=backend,
                                          alpha=alpha, beta=beta):
                            result = rccsdt._einsum(backend, expression, *tensors,
                                                   out=out, alpha=alpha, beta=beta)
                            self.assertIs(result, out)
                            np.testing.assert_allclose(out, expected, atol=2e-13, rtol=2e-13)

    def test_accumulation_aliases_and_strides(self):
        rng = np.random.default_rng(424)
        for backend in ('numpy', 'pyscf'):
            for alpha in (1., -1., .5):
                for beta in (0., 1., -.5):
                    for layout in ('copy', 'transpose', 'reverse', 'operand'):
                        left = rng.normal(size=(4, 4))
                        right = rng.normal(size=(4, 4))
                        out = left if layout == 'operand' else rng.normal(size=(4, 4))
                        if layout == 'transpose':
                            out = out.T
                        elif layout == 'reverse':
                            out = out[:, ::-1]
                        expected = alpha * (left @ right) + beta * out.copy()
                        before_right = right.copy()
                        with self.subTest(backend=backend, alpha=alpha, beta=beta, layout=layout):
                            actual = rccsdt._einsum(backend, 'ij,jk->ik', left, right,
                                                   out=out, alpha=alpha, beta=beta)
                            self.assertIs(actual, out)
                            np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
                            np.testing.assert_array_equal(right, before_right)

    def test_direct_output_and_accumulation(self):
        rng = np.random.default_rng(421)
        left = rng.normal(size=(3, 4))
        right = rng.normal(size=(4, 5))
        product = left @ right
        for backend in ('numpy', 'pyscf'):
            with self.subTest(backend=backend):
                out = np.zeros((3, 5))
                returned = rccsdt._einsum(backend, 'ij,jk->ik', left, right,
                                          out=out, alpha=1., beta=0.)
                self.assertIs(returned, out)
                np.testing.assert_allclose(out, product, atol=1e-13, rtol=1e-13)

                out.fill(.37)
                returned = rccsdt._einsum(backend, 'ij,jk->ik', left, right,
                                          out=out, alpha=-.25, beta=1.)
                self.assertIs(returned, out)
                np.testing.assert_allclose(out, -.25*product + .37,
                                           atol=1e-13, rtol=1e-13)

                alias_left = rng.normal(size=(3, 3))
                alias_right = rng.normal(size=(3, 3))
                expected = alias_left @ alias_right
                returned = rccsdt._einsum(backend, 'ij,jk->ik', alias_left, alias_right,
                                           out=alias_left, alpha=1., beta=0.)
                self.assertIs(returned, alias_left)
                np.testing.assert_allclose(alias_left, expected, atol=1e-13, rtol=1e-13)

    def test_disjoint_interleaved_views_use_conservative_alias_check(self):
        rng = np.random.default_rng(429)
        backends = ['numpy', 'pyscf']
        if rccsdt._get_pytblis() is not None:
            backends.append('pytblis')
        for backend in backends:
            for beta in (0., 1., -.5):
                with self.subTest(backend=backend, beta=beta):
                    backing = rng.normal(size=(4, 8))
                    left, out = backing[:, ::2], backing[:, 1::2]
                    right = rng.normal(size=(4, 4))
                    self.assertFalse(np.shares_memory(left, out))
                    self.assertTrue(np.may_share_memory(left, out))
                    left_original = left.copy()
                    expected = -.25 * (left @ right) + beta * out.copy()
                    with patch.object(rccsdt.np, 'shares_memory',
                                      side_effect=AssertionError('Exact alias checks are unnecessary')):
                        returned = rccsdt._einsum(backend, 'ij,jk->ik', left, right,
                                                   out=out, alpha=-.25, beta=beta)
                    self.assertIs(returned, out)
                    np.testing.assert_allclose(out, expected, atol=2e-13, rtol=2e-13)
                    np.testing.assert_array_equal(left, left_original)

    def test_pyscf_accumulation_with_automatic_pytblis_backend(self):
        try:
            import pytblis
        except ImportError:
            self.skipTest('pytblis is optional')
        from pyscf.lib import numpy_helper
        rng = np.random.default_rng(425)
        with patch.object(numpy_helper, 'EINSUM_BACKEND', 'pytblis'), \
                patch.object(numpy_helper, 'pytblis', pytblis, create=True):
            for alias in (False, True):
                left, right = rng.normal(size=(2, 24, 24))
                out = left if alias else rng.normal(size=(24, 24)).T
                expected = .25 * (left.copy() @ right) + out.copy()
                actual = rccsdt._einsum('pyscf', 'ij,jk->ik', left, right,
                                       out=out, alpha=.25, beta=1.)
                self.assertIs(actual, out)
                np.testing.assert_allclose(out, expected, atol=2e-12, rtol=2e-12)

    def test_low_rank_fock_dressing_matches_dense_maps(self):
        rng = np.random.default_rng(422)
        for nocc, nvir in ((2, 3), (5, 7)):
            fock = rng.normal(size=(nocc+nvir, nocc+nvir))
            t1 = rng.normal(scale=.03, size=(nocc, nvir))
            obj = SimpleNamespace(nocc=nocc, nmo=nocc+nvir)
            x, y = rccsdt.update_xy(obj, t1)
            expected = x @ fock @ y.T
            actual = rccsdt._dress_fock(fock, t1, nocc)
            np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)

        oa, ob, va, vb = 2, 3, 4, 5
        fa = rng.normal(size=(oa+va, oa+va))
        fb = rng.normal(size=(ob+vb, ob+vb))
        t1a = rng.normal(scale=.03, size=(oa, va))
        t1b = rng.normal(scale=.03, size=(ob, vb))
        obj = SimpleNamespace(nocc=(oa, ob), nmo=(oa+va, ob+vb))
        xa, xb, ya, yb = uccsdt.update_xy_uhf(obj, (t1a, t1b))
        np.testing.assert_allclose(rccsdt._dress_fock(fa, t1a, oa), xa @ fa @ ya.T,
                                   atol=2e-13, rtol=2e-13)
        np.testing.assert_allclose(rccsdt._dress_fock(fb, t1b, ob), xb @ fb @ yb.T,
                                   atol=2e-13, rtol=2e-13)

    def test_pytblis_no_output_scaling_and_complex_promotion(self):
        try:
            import pytblis
        except ImportError:
            self.skipTest('pytblis is optional')
        if not rccsdt.lib.numpy_helper._pytblis_supports_partial_trace(pytblis.contract):
            self.skipTest('pytblis lacks partial-trace support')
        rng = np.random.default_rng(426)
        left = np.asfortranarray(rng.normal(size=(3, 5)))
        right = rng.normal(size=(5, 4))[:, ::-1]
        pairs = ((left, right), (left + .2j, right + .3j),
                 (left, right + .3j), (left + .2j, right))
        contract = pytblis.contract
        for a, b in pairs:
            a.flags.writeable = False
            b.flags.writeable = False
            product = np.einsum('ik,kj->ij', a, b)
            for alpha in (-1., .5, 0., 1.):
                native_results = []

                def record(*args, **kwargs):
                    result = contract(*args, **kwargs)
                    native_results.append(result)
                    return result

                with self.subTest(dtype_a=a.dtype, dtype_b=b.dtype, alpha=alpha):
                    with patch.object(pytblis, 'contract', autospec=True, side_effect=record) as call:
                        actual = rccsdt._einsum('pytblis', 'ik,kj->ij', a, b, alpha=alpha)
                    np.testing.assert_allclose(actual, alpha * product, atol=2e-13, rtol=2e-13)
                    self.assertEqual(call.call_args.kwargs['alpha'], alpha)
                    self.assertIs(actual, native_results[0])
            # A complex coefficient may promote real operands, or require
            # the non-native scaling fallback for mixed real/complex inputs.
            actual = rccsdt._einsum('pytblis', 'ik,kj->ij', a, b, alpha=.25+.5j)
            self.assertTrue(np.iscomplexobj(actual))
            np.testing.assert_allclose(actual, (.25+.5j)*product, atol=2e-13, rtol=2e-13)

    def test_pytblis_empty_axes_and_aliases(self):
        try:
            import pytblis
        except ImportError:
            self.skipTest('pytblis is optional')
        for shapes in (((3, 0), (0, 4)), ((0, 3), (3, 4))):
            left, right = (np.empty(shape) for shape in shapes)
            expected = np.einsum('ik,kj->ij', left, right)
            for alpha in (-1., .5, 0.):
                with self.subTest(shapes=shapes, alpha=alpha):
                    actual = rccsdt._einsum('pytblis', 'ik,kj->ij', left, right, alpha=alpha)
                    np.testing.assert_array_equal(actual, alpha * expected)
                    out = np.full(expected.shape, .75)
                    actual = rccsdt._einsum('pytblis', 'ik,kj->ij', left, right,
                                           out=out, alpha=alpha, beta=.5)
                    np.testing.assert_array_equal(actual, alpha * expected + .375)
        rng = np.random.default_rng(427)
        for alpha, beta in ((.5, 1.), (-1., .25), (0., .5)):
            left, right = rng.normal(size=(2, 12, 12))
            expected = alpha * (left @ right) + beta * left
            actual = rccsdt._einsum('pytblis', 'ik,kj->ij', left, right,
                                   out=left, alpha=alpha, beta=beta)
            self.assertIs(actual, left)
            np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)

    def test_pytblis_multiple_operands_and_partial_traces(self):
        if rccsdt._get_pytblis() is None:
            self.skipTest('pytblis is optional')
        rng = np.random.default_rng(428)
        cases = (
            ('ij,jk,kl->il', tuple(rng.normal(size=(4, 4)) for _ in range(3))),
            ('iab,bc->ac', (rng.normal(size=(1, 4, 3)), rng.normal(size=(3, 4)))),
            ('iab,bc,cd->ad', (rng.normal(size=(1, 4, 3)),
                              rng.normal(size=(3, 5)), rng.normal(size=(5, 4)))),
            ('iab->ab', (rng.normal(size=(1, 4, 4)),)))
        for script, tensors in cases:
            tensors = tuple(tensor[..., ::-1] for tensor in tensors)
            product = np.einsum(script, *tensors, optimize=True)
            for alpha in (1., -.5, .25+.5j):
                with self.subTest(script=script, alpha=alpha):
                    actual = rccsdt._einsum('pytblis', script, *tensors, alpha=alpha)
                    np.testing.assert_allclose(actual, alpha * product, atol=2e-13, rtol=2e-13)
                    for beta in (0., 1., -.5):
                        out = rng.normal(size=product.shape).astype(actual.dtype).T
                        expected = alpha * product + beta * out.copy()
                        returned = rccsdt._einsum('pytblis', script, *tensors,
                                                  out=out, alpha=alpha, beta=beta)
                        self.assertIs(returned, out)
                        np.testing.assert_allclose(out, expected, atol=2e-13, rtol=2e-13)

        left, middle, right = rng.normal(size=(3, 4, 4))
        expected = -.5 * (left.copy() @ middle @ right) + .25 * left
        actual = rccsdt._einsum('pytblis', 'ij,jk,kl->il', left, middle, right,
                               out=left, alpha=-.5, beta=.25)
        self.assertIs(actual, left)
        np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)

    def test_pytblis_legacy_partial_trace_fallback(self):
        pytblis = rccsdt._get_pytblis()
        if pytblis is None:
            self.skipTest('pytblis is optional')

        def legacy_contract(script, a, b, alpha=1., beta=0., out=None):
            self.fail('A backend without partial-trace support must use NumPy')

        legacy = SimpleNamespace(contract=legacy_contract, einsum=pytblis.einsum)
        left = np.arange(12., dtype=float).reshape(1, 4, 3)
        right = np.arange(12., dtype=float).reshape(3, 4)
        out = np.full((4, 4), .5)
        expected = -.25 * np.einsum('iab,bc->ac', left, right) + out
        with patch.object(rccsdt, '_get_pytblis', return_value=legacy):
            actual = rccsdt._einsum('pytblis', 'iab,bc->ac', left, right,
                                   out=out, alpha=-.25, beta=1.)
        self.assertIs(actual, out)
        np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)


if __name__ == '__main__':
    unittest.main()
