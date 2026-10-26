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
from unittest import mock
import weakref
import numpy
from pyscf import dft, gto, lib
from pyscf.dft import numint


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0',
                        basis='sto-3g', verbose=0)

    def inputs(self, complex_coeff=False, ngrids=513):
        rng = numpy.random.default_rng(31)
        grids = dft.Grids(self.mol)
        grids.coords = rng.normal(size=(ngrids, 3))
        grids.weights = numpy.ones(ngrids)
        coeff = rng.normal(size=(self.mol.nao, self.mol.nao))
        if complex_coeff:
            coeff = coeff + 1j*rng.normal(size=coeff.shape)
        coeff = numpy.linalg.qr(coeff)[0]
        occ = numpy.linspace(.2, 1.2, self.mol.nao)
        return grids, coeff, occ

    def reference(self, ni, grids, xc, coeff, occ, spin, use_dm):
        xctype = ni._xc_type(xc)
        deriv = 0 if xctype == 'LDA' else 1
        if xctype == 'MGGA' and numint.MGGA_DENSITY_LAPL:
            deriv = 2
        is_rks = not isinstance(coeff, tuple)
        matrices = [(c * o).dot(c.conj().T) for c, o in
                    ([(coeff, occ)] if is_rks else zip(coeff, occ))]
        dm = matrices[0] if is_rks else numpy.asarray(matrices)
        if use_dm:
            make_rho = ni._gen_rho_evaluator(self.mol, dm, 1, False, grids)[0]
        blocks = [[], []]
        for ao, mask, weight, coords in ni.block_loop(
                self.mol, grids, self.mol.nao, deriv, max_memory=.001):
            for s in range(1 if is_rks else 2):
                if use_dm:
                    block = make_rho(s, ao, mask, xctype)
                else:
                    c, o = (coeff, occ) if is_rks else (coeff[s], occ[s])
                    block = ni.eval_rho2(self.mol, ao, c, o, mask, xctype,
                                         numint.MGGA_DENSITY_LAPL)
                blocks[s].append(block)
        if is_rks:
            rho = numpy.hstack(blocks[0])
            if spin:
                rho = numpy.repeat((rho*.5)[None], 2, axis=0)
        else:
            rho = tuple(numpy.hstack(b) for b in blocks)
        vxc, fxc = ni.eval_xc_eff(xc, rho, deriv=2, xctype=xctype, spin=spin)[1:3]
        return dm, (rho, vxc, fxc)

    def test_cache_values(self):
        for complex_coeff in (False, True):
            grids, coeff, occ = self.inputs(complex_coeff)
            for xc in ('LDA,VWN', 'PBE', 'TPSS'):
                for route, spin in (('rks', 0), ('rks', 1), ('uks', 1)):
                    c, o = (coeff, occ) if route == 'rks' else ((coeff, coeff), (occ*.6, occ*.4))
                    for use_dm in (False, True):
                        with self.subTest(complex_coeff=complex_coeff, xc=xc,
                                          route=route, spin=spin, use_dm=use_dm):
                            ni = numint.NumInt()
                            dm, expected = self.reference(ni, grids, xc, c, o, spin, use_dm)
                            if use_dm:
                                actual = ni.cache_xc_kernel1(self.mol, grids, xc, dm,
                                                             spin, max_memory=.001)
                            else:
                                actual = ni.cache_xc_kernel(self.mol, grids, xc, c, o,
                                                            spin, max_memory=.001)
                            self.assertIsInstance(actual[0], tuple if route == 'uks' else numpy.ndarray)
                            for result, ref in zip(actual, expected):
                                numpy.testing.assert_allclose(result, ref, atol=2e-11, rtol=2e-12)

    def test_laplacian_density(self):
        grids, coeff, occ = self.inputs()
        ni = numint.NumInt()
        with lib.temporary_env(numint, MGGA_DENSITY_LAPL=True):
            _, expected = self.reference(ni, grids, 'TPSS', coeff, occ, 0, False)
            actual = ni.cache_xc_kernel(self.mol, grids, 'TPSS', coeff, occ,
                                        max_memory=.001)
        self.assertEqual(actual[0].shape, (6, len(grids.coords)))
        for result, ref in zip(actual, expected):
            numpy.testing.assert_allclose(result, ref, atol=2e-11, rtol=2e-12)

    def test_builtin_xc_is_blocked(self):
        grids, coeff, occ = self.inputs()
        ni = numint.NumInt()
        sizes = []
        ao_refs = []
        block_loop = ni.block_loop
        transform = numint.xc_deriv.transform_xc

        def record_blocks(*args, **kwargs):
            for block in block_loop(*args, **kwargs):
                ao_refs.append(weakref.ref(block[0]))
                yield block

        def record(rho, *args, **kwargs):
            self.assertTrue(all(ref() is None for ref in ao_refs))
            sizes.append(rho.shape[-1])
            return transform(rho, *args, **kwargs)

        ni.block_loop = record_blocks
        with mock.patch.object(numint.xc_deriv, 'transform_xc', side_effect=record):
            ni.cache_xc_kernel(self.mol, grids, 'TPSS', coeff, occ, max_memory=.001)
        self.assertEqual(sizes, [224, 224, 224, 224, 65, 65])

    def test_custom_xc_keeps_whole_grid_callback(self):
        grids, coeff, occ = self.inputs()
        dm = (coeff * occ).dot(coeff.T)
        for use_dm in (False, True):
            ni = numint.NumInt()
            sizes = []
            ao_refs = []
            block_loop = ni.block_loop

            def record_blocks(*args, **kwargs):
                for block in block_loop(*args, **kwargs):
                    ao_refs.append(weakref.ref(block[0]))
                    yield block

            def custom_xc(xc, rho, spin=0, deriv=1, omega=None, **kwargs):
                sizes.append(numpy.asarray(rho).shape[-1])
                self.assertTrue(all(ref() is None for ref in ao_refs))
                return dft.libxc.eval_xc('TPSS', rho, spin, deriv=deriv, omega=omega)

            ni.block_loop = record_blocks
            dft.libxc.define_xc_(ni, custom_xc, xctype='MGGA')
            if use_dm:
                ni.cache_xc_kernel1(self.mol, grids, 'unused', dm, max_memory=.001)
            else:
                ni.cache_xc_kernel(self.mol, grids, 'unused', coeff, occ, max_memory=.001)
            self.assertEqual(sizes, [len(grids.coords)])

    def test_subclass_evaluator_keeps_whole_grid(self):
        grids, coeff, occ = self.inputs()
        for method in ('eval_xc1', 'eval_xc_eff'):
            sizes = []
            original = getattr(numint.NumInt, method)

            def evaluator(ni, xc, rho, *args, **kwargs):
                sizes.append(numpy.asarray(rho).shape[-1])
                return original(ni, xc, rho, *args, **kwargs)

            custom = type('CustomNumInt', (numint.NumInt,), {method: evaluator})()
            custom.cache_xc_kernel(self.mol, grids, 'TPSS', coeff, occ, max_memory=.001)
            self.assertEqual(sizes, [len(grids.coords)])

    def test_density_hook_dtype_promotion(self):
        grids, coeff, occ = self.inputs()
        for spin in (0, 1):
            for use_dm in (False, True):
                ni = numint.NumInt()
                blocks = [[], []]
                calls = []

                def density(s, ao):
                    # RKS changes dtype across blocks; UKS keeps distinct
                    # alpha/beta dtypes, as two independent hstack calls do.
                    dtype = numpy.float64 if s or (not spin and blocks[s]) else numpy.float32
                    block = numpy.full((5, ao.shape[-2]), 1+len(blocks[s])*1e-9, dtype=dtype)
                    blocks[s].append(block)
                    return block

                def rho2(mol, ao, c, o, *args):
                    s = int(spin and len(blocks[0]) > len(blocks[1]))
                    return density(s, ao)

                def evaluator(xc, rho, **kwargs):
                    actual = rho if spin else (rho,)
                    for s, array in enumerate(actual):
                        expected = numpy.hstack(blocks[s])
                        self.assertEqual(array.dtype, expected.dtype)
                        numpy.testing.assert_array_equal(array, expected)
                    calls.append(1)
                    return None, numpy.zeros_like(rho), numpy.zeros((1, len(grids.coords))), None

                ni.eval_rho2 = rho2
                ni.eval_xc_eff = evaluator
                if use_dm:
                    ni._gen_rho_evaluator = lambda *args: (
                        lambda s, ao, *args: density(s, ao), spin+1, self.mol.nao)
                    dm = (coeff * occ).dot(coeff.T)
                    if spin:
                        dm = numpy.asarray((dm, dm))
                    ni.cache_xc_kernel1(self.mol, grids, 'TPSS', dm, spin, max_memory=.001)
                else:
                    c, o = ((coeff, coeff), (occ, occ)) if spin else (coeff, occ)
                    ni.cache_xc_kernel(self.mol, grids, 'TPSS', c, o, spin, max_memory=.001)
                self.assertEqual(calls, [1])

    def test_single_block_releases_ao(self):
        grids, coeff, occ = self.inputs(ngrids=113)
        ni = numint.NumInt()
        ao_refs = []
        block_loop = ni.block_loop

        def record_blocks(*args, **kwargs):
            for block in block_loop(*args, **kwargs):
                ao_refs.append(weakref.ref(block[0]))
                yield block

        original = ni.eval_xc_eff

        def evaluator(*args, **kwargs):
            self.assertTrue(all(ref() is None for ref in ao_refs))
            return original(*args, **kwargs)

        ni.block_loop = record_blocks
        ni.eval_xc_eff = evaluator
        ni.cache_xc_kernel(self.mol, grids, 'TPSS', coeff, occ, max_memory=.001)


if __name__ == '__main__':
    unittest.main()
