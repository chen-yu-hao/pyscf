#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import unittest
from unittest import mock
import numpy
from pyscf import dft, gto, lib
from pyscf.dft import numint


def legacy_vmat(ao, wv, nbins, screen_index, pair_mask, ao_loc, out=None):
    # Present the same logical values in the packed layout expected by legacy scaling.
    ao = numpy.asarray(ao.transpose(0, 2, 1), order='C').transpose(0, 2, 1)
    weights = wv.copy()
    weights[0] *= .5
    weights[4] *= .5
    aow = numint._scale_ao_sparse(ao[:4], weights[:4], screen_index, ao_loc)
    mat = numint._dot_ao_ao_sparse(ao[0], aow, None, nbins, screen_index,
                                   pair_mask, ao_loc, hermi=0)
    tau = numint._tau_dot_sparse(ao, ao, weights[4], nbins, screen_index,
                                pair_mask, ao_loc)
    value = lib.hermi_sum(mat) + tau
    if out is None:
        return value
    out += value
    return out


def dense_formula(ao, wv):
    result = numpy.einsum('gi,g,gj->ij', ao[0], wv[0], ao[0])
    for c in range(1, 4):
        cross = numpy.einsum('gi,g,gj->ij', ao[0], wv[c], ao[c])
        result += cross + cross.T
        result += .5*numpy.einsum('gi,g,gj->ij', ao[c], wv[4], ao[c])
    return result


def legacy_xc_vmat(ao, wv, nbins, screen_index, pair_mask, ao_loc, out=None,
                   xctype='MGGA', hermi=1):
    if xctype == 'MGGA':
        return legacy_vmat(ao, wv, nbins, screen_index, pair_mask, ao_loc, out)
    if xctype == 'LDA':
        ao = numpy.asarray(ao, order='F')
        value = numint._dot_ao_ao_sparse(ao, ao, wv, nbins, screen_index,
                                         pair_mask, ao_loc, hermi=hermi)
    else:
        ao = numpy.asarray(ao.transpose(0, 2, 1), order='C').transpose(0, 2, 1)
        weights = wv.copy()
        weights[0] *= .5
        aow = numint._scale_ao_sparse(ao, weights, screen_index, ao_loc)
        value = numint._dot_ao_ao_sparse(ao[0], aow, None, nbins, screen_index,
                                         pair_mask, ao_loc, hermi=0)
        value = lib.hermi_sum(value)
    if out is None:
        return value
    out += value
    return out


class KnownValues(unittest.TestCase):
    def _inputs(self, ngrids, loc=None):
        rng = numpy.random.default_rng(29)
        if loc is None:
            loc = numpy.asarray([0] + list(numpy.cumsum([1, 3, 5] * 8)), dtype=numpy.int32)
        nao = loc[-1]
        nbas = len(loc)-1
        ao = rng.normal(size=(8, nao, ngrids))[::2].transpose(0, 2, 1)
        wv = rng.normal(size=(5, ngrids))
        wv[:, ::7] = 0
        screen = rng.choice([0, 8, 14, 25], size=((ngrids+55)//56, nbas)).astype(numpy.uint8)
        pairs = rng.integers(0, 2, size=(nbas, nbas), dtype=numpy.uint8)
        pairs = numpy.asarray(pairs | pairs.T, order='C')
        initial = rng.normal(size=(nao, nao))
        return ao, wv, screen, pairs, loc, initial

    def test_dense_formula_and_accumulation(self):
        for ngrids in (1, 55, 224, 257):
            ao, wv, screen, pairs, loc, initial = self._inputs(ngrids)
            ao = ao[::-1]
            reference = initial + dense_formula(ao, wv)
            original_weights = wv.copy()
            for threads in (1, 4):
                with self.subTest(ngrids=ngrids, threads=threads), lib.with_omp_threads(threads):
                    out = initial.copy()
                    result = numint._mgga_vmat(ao, wv, 20, None, pairs, loc, out)
                    self.assertIs(result, out)
                    numpy.testing.assert_allclose(result, reference, atol=3e-11, rtol=2e-12)
                    numpy.testing.assert_array_equal(wv, original_weights)

    def test_screened_legacy(self):
        for ngrids in (8, 56, 120, 224):
            ao, wv, screen, pairs, loc, initial = self._inputs(ngrids)
            reference = legacy_vmat(ao, wv, 20, screen, pairs, loc, initial.copy())
            direct = initial.copy()
            shell = numpy.repeat(numpy.arange(len(loc)-1), numpy.diff(loc))
            for start in range(0, ngrids, numint.BLKSIZE):
                stop = min(start+numint.BLKSIZE, ngrids)
                bins = screen[start//numint.BLKSIZE].astype(int)
                allowed = ((bins[:, None] > 0) & (bins[None, :] > 0)
                           & (bins[:, None] + bins[None, :] >= 20) & (pairs != 0))
                direct += (dense_formula(ao[:, start:stop], wv[:, start:stop])
                           * allowed[shell[:, None], shell[None, :]])
            for threads in (1, 4):
                with self.subTest(ngrids=ngrids, threads=threads), lib.with_omp_threads(threads):
                    actual = numint._mgga_vmat(ao, wv, 20, screen, pairs, loc, initial.copy())
                    numpy.testing.assert_allclose(actual, reference, atol=3e-11, rtol=2e-12)
                    numpy.testing.assert_allclose(actual, direct, atol=3e-11, rtol=2e-12)
                    repeated = numint._mgga_vmat(ao, wv, 20, screen, pairs, loc, initial.copy())
                    numpy.testing.assert_array_equal(actual, repeated)

    def test_conservative_bin_pruning(self):
        ao, wv, screen, pairs, loc, initial = self._inputs(120)
        pairs[:] = 1
        for all_small in (True, False):
            screen[:] = 8 if all_small else 2
            if not all_small:
                screen[:, -1] = 15
            reference = legacy_vmat(ao, wv, 20, screen, pairs, loc, initial.copy())
            actual = numint._mgga_vmat(ao, wv, 20, screen, pairs, loc, initial.copy())
            numpy.testing.assert_allclose(actual, reference, atol=3e-11, rtol=2e-12)

    def test_lda_gga_formulas(self):
        for ngrids in (55, 120):
            ao, wv, screen, pairs, loc, initial = self._inputs(ngrids)
            for xctype in ('LDA', 'GGA'):
                values = ao[0] if xctype == 'LDA' else ao
                weights = wv[0] if xctype == 'LDA' else wv[:4]
                if xctype == 'LDA':
                    direct = numpy.einsum('gi,g,gj->ij', ao[0], wv[0], ao[0])
                else:
                    no_tau = wv.copy()
                    no_tau[4] = 0
                    direct = dense_formula(ao, no_tau)
                actual = numint._xc_vmat(values, weights, 20, None, pairs, loc,
                                          initial.copy(), xctype=xctype)
                numpy.testing.assert_allclose(actual, initial+direct, atol=3e-11, rtol=2e-12)
                reference = legacy_xc_vmat(values, weights, 20, screen, pairs, loc,
                                            initial.copy(), xctype=xctype)
                actual = numint._xc_vmat(values, weights, 20, screen, pairs, loc,
                                          initial.copy(), xctype=xctype)
                numpy.testing.assert_allclose(actual, reference, atol=3e-11, rtol=2e-12)

    def test_weighted_partition_all_modes(self):
        ao, wv, screen, pairs, loc, initial = self._inputs(56*17+8)
        screen[:] = 25
        screen[::3, :-1] = 0
        screen[1::3, :screen.shape[1]//2] = 0
        for xctype in ('LDA', 'GGA', 'MGGA'):
            values = ao[0] if xctype == 'LDA' else ao
            weights = wv[0] if xctype == 'LDA' else wv[:4] if xctype == 'GGA' else wv
            reference = legacy_xc_vmat(values, weights, 20, screen, pairs, loc,
                                        initial.copy(), xctype=xctype)
            for threads in (1, 4):
                with self.subTest(xctype=xctype, threads=threads), lib.with_omp_threads(threads):
                    actual = numint._xc_vmat(values, weights, 20, screen, pairs, loc,
                                              initial.copy(), xctype=xctype)
                    repeated = numint._xc_vmat(values, weights, 20, screen, pairs, loc,
                                                initial.copy(), xctype=xctype)
                    numpy.testing.assert_allclose(actual, reference, atol=3e-11, rtol=2e-12)
                    numpy.testing.assert_array_equal(actual, repeated)

    def test_legacy_fallbacks(self):
        ao, wv, screen, pairs, loc, initial = self._inputs(120)
        pairs[0, 1] = 0
        pairs[1, 0] = 1
        reference = legacy_vmat(ao, wv, 20, screen, pairs, loc, initial.copy())
        actual = numint._mgga_vmat(ao, wv, 20, screen, pairs, loc, initial.copy())
        numpy.testing.assert_allclose(actual, reference, atol=3e-11, rtol=2e-12)
        noncontiguous = numpy.empty((5, 240))[:, ::2]
        noncontiguous[:] = wv
        reference = legacy_vmat(ao, noncontiguous, 20, None, pairs, loc)
        actual = numint._mgga_vmat(ao, noncontiguous, 20, None, pairs, loc)
        numpy.testing.assert_allclose(actual, reference, atol=3e-11, rtol=2e-12)

    def test_wide_shell(self):
        loc = numpy.array([0, 100, 103], dtype=numpy.int32)
        ao, wv, screen, pairs, loc, initial = self._inputs(120, loc)
        screen[:] = 25
        pairs[:] = 1
        reference = legacy_vmat(ao, wv, 20, screen, pairs, loc)
        actual = numint._mgga_vmat(ao, wv, 20, screen, pairs, loc)
        numpy.testing.assert_allclose(actual, reference, atol=3e-11, rtol=2e-12)

    def test_native_declines_before_output_write(self):
        sentinel = numpy.array([42.])
        ptr = sentinel.ctypes.data_as(ctypes.c_void_p)
        screen = numpy.array([[25]], dtype=numpy.uint8)
        pairs = numpy.ones((1, 1), dtype=numpy.uint8)
        for width, threads, sparse in ((4096, 1, True), (4096, 8, False)):
            loc = numpy.array([0, width], dtype=numpy.int32)
            with lib.with_omp_threads(threads):
                done = numint.libdft.VXCmgga_vmat(
                    ptr, ptr, ptr, ctypes.c_int(width), ctypes.c_int(8),
                    ctypes.c_int(1), ctypes.c_int(20),
                    screen.ctypes.data_as(ctypes.c_void_p) if sparse else None,
                    pairs.ctypes.data_as(ctypes.c_void_p) if sparse else None,
                    loc.ctypes.data_as(ctypes.c_void_p), ctypes.c_ssize_t(width*8))
            self.assertEqual(done, 0)
            numpy.testing.assert_array_equal(sentinel, [42.])

    def test_rks_uks_python_callback(self):
        mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz', verbose=0)
        grids = dft.Grids(mol)
        grids.atom_grid = (10, 14)
        grids.prune = None
        grids.build(with_non0tab=True)
        dm = dft.RKS(mol).get_init_guess()
        ni = numint.NumInt()
        calls = []

        def callback(xc_code, rho, deriv=1, xctype=None, spin=0):
            rho = numpy.asarray(rho)
            calls.append((spin, rho.copy()))
            if xctype == 'LDA':
                density = rho.sum(axis=0) if spin else rho
                vxc = (rho*.02+.01)[:, None, :] if spin else (rho*.02+.01)[None, :]
            else:
                density = rho[:, 0].sum(axis=0) if spin else rho[0]
                vxc = rho*.02+.01
            return density*.03, vxc, None, None

        ni.eval_xc_eff = callback
        for xc in ('LDA,VWN', 'PBE', 'SCAN'):
            for spin in (0, 1):
                matrix = numpy.asarray((dm*.6, dm*.4)) if spin else dm
                method = ni.nr_uks if spin else ni.nr_rks
                with lib.temporary_env(numint, SWITCH_SIZE=0), lib.with_omp_threads(2):
                    calls.clear()
                    actual = method(mol, grids, xc, matrix)
                    actual_calls = list(calls)
                    calls.clear()
                    with mock.patch.object(numint, '_xc_vmat', legacy_xc_vmat):
                        reference = method(mol, grids, xc, matrix)
                for value, expected in zip(actual, reference):
                    numpy.testing.assert_allclose(value, expected, atol=3e-11, rtol=2e-12)
                self.assertEqual(len(calls), len(actual_calls))
                for (s1, rho1), (s2, rho2) in zip(calls, actual_calls):
                    self.assertEqual(s1, s2)
                    numpy.testing.assert_array_equal(rho1, rho2)

    def test_rks_uks_functionals_and_lda_hermi2(self):
        mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz', verbose=0)
        grids = dft.Grids(mol)
        grids.atom_grid = (10, 14)
        grids.prune = None
        grids.build(with_non0tab=True)
        dm = dft.RKS(mol).get_init_guess()
        ni = numint.NumInt()
        for spin in (0, 1):
            matrix = numpy.asarray((dm*.6, dm*.4)) if spin else dm
            method = ni.nr_uks if spin else ni.nr_rks
            with lib.temporary_env(numint, SWITCH_SIZE=0), lib.with_omp_threads(2):
                for xc in ('LDA,VWN', 'PBE', 'B3LYP', 'SCAN'):
                    actual = method(mol, grids, xc, matrix)
                    with mock.patch.object(numint, '_xc_vmat', legacy_xc_vmat):
                        reference = method(mol, grids, xc, matrix)
                    for value, expected in zip(actual, reference):
                        numpy.testing.assert_allclose(value, expected, atol=3e-11, rtol=2e-12)
                with mock.patch.object(numint, '_xc_vmat', side_effect=AssertionError('hermi=2 must use legacy')):
                    method(mol, grids, 'LDA,VWN', matrix, hermi=2)


if __name__ == '__main__':
    unittest.main()
