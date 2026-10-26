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

import unittest
from unittest import mock
import numpy
from pyscf import dft, gto
from pyscf.dft import numint, xc_deriv
from pyscf.dft.xc_cache import _CompactXCKernel


def dense_cache(ni, *args, **kwargs):
    return ni.cache_xc_kernel(*args, **kwargs)


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0',
                        basis='sto-3g', verbose=0)

    def inputs(self):
        rng = numpy.random.default_rng(38)
        grids = dft.Grids(self.mol)
        grids.coords = rng.normal(size=(281, 3))
        grids.weights = rng.random(281) * .01
        coeff = numpy.linalg.qr(rng.normal(size=(self.mol.nao, self.mol.nao)))[0]
        occ = numpy.linspace(.2, 1.2, self.mol.nao)
        return rng, grids, coeff, occ

    def test_invariant_action(self):
        rng = numpy.random.default_rng(37)
        for xc in ('LDA,VWN', 'PBE', 'TPSS', '.6*TPSS + .2*PBE + .2*LDA_X'):
            xctype = dft.libxc.xc_type(xc)
            for spin in (0, 1):
                nvar = xc_deriv._XC_NVAR[xctype, spin][0]
                rho = rng.random((spin+1, nvar, 43)) + .4
                if nvar > 1:
                    rho[:,1:4] *= .1
                    rho[:,1:4,::7] = 0
                if nvar == 5:
                    rho[:,4] += 1
                if spin == 0:
                    rho = rho[0]
                perturbation = rng.normal(size=rho.shape)
                raw = dft.libxc.eval_xc1(xc, rho, spin=spin, deriv=2)
                original = raw.copy()
                expanded = xc_deriv.transform_xc(rho, raw, xctype, spin, 2)
                if spin:
                    reference = numpy.einsum('axbyg,byg->axg', expanded, perturbation)
                else:
                    reference = numpy.einsum('xyg,yg->xg', expanded, perturbation)
                compact = _CompactXCKernel(rho, raw, xctype, spin)
                for p0, p1 in ((0, 7), (7, 43)):
                    with self.subTest(xc=xc, spin=spin, p0=p0):
                        actual = compact.apply(perturbation[...,p0:p1], p0, p1)
                        numpy.testing.assert_allclose(actual, reference[...,p0:p1],
                                                      atol=2e-12, rtol=2e-12)
                numpy.testing.assert_array_equal(raw, original)

    def test_spin_projection_and_scaling(self):
        rng = numpy.random.default_rng(39)
        rho = rng.random((5, 23)) + .4
        rho[1:4] *= .1
        rho[4] += 1
        rho = numpy.repeat((rho*.5)[None], 2, axis=0)
        raw = dft.libxc.eval_xc1('TPSS', rho, spin=1, deriv=2)
        expanded = xc_deriv.transform_xc(rho, raw, 'MGGA', 1, 2)
        perturbation = rng.normal(size=(5, 23))
        compact = _CompactXCKernel(rho, raw, 'MGGA', 1)
        for singlet in (False, True):
            sign = 1 if singlet else -1
            reference = .5*numpy.einsum('xyg,yg->xg',
                expanded[0,:,0] + sign*expanded[0,:,1], perturbation)
            projected = compact.scaled(.5).projected(singlet)
            self.assertIs(projected.values, compact.values)
            self.assertIs(projected.rho, compact.rho)
            numpy.testing.assert_allclose(projected.apply(perturbation, 0, 23),
                                          reference, atol=2e-12, rtol=2e-12)
        self.assertEqual(compact.scale, 1)
        self.assertIsNone(compact.projection)

    def test_batched_matrix_actions(self):
        rng, grids, coeff, occ = self.inputs()
        ni = numint.NumInt()
        nao = self.mol.nao
        for route in ('rks', 'singlet', 'triplet', 'uks'):
            spin = int(route != 'rks')
            c, o = ((coeff, coeff), (occ*.6, occ*.4)) if route == 'uks' else (coeff, occ)
            _, _, expanded = ni.cache_xc_kernel(self.mol, grids, 'TPSS', c, o,
                                                 spin, max_memory=.001)
            _, vxc, compact = numint._cache_xc_kernel_response(
                ni, self.mol, grids, 'TPSS', c, o, spin, max_memory=.001)
            self.assertIsInstance(compact, _CompactXCKernel)
            self.assertIsNone(vxc)
            if route in ('singlet', 'triplet'):
                expanded *= .5
                compact = compact.scaled(.5)
            shape = (2, 2, nao, nao) if route == 'uks' else (2, nao, nao)
            perturbation = rng.normal(size=shape)
            for hermi in (0, 1):
                dm1 = perturbation if hermi == 0 else perturbation + perturbation.swapaxes(-1, -2)
                kwargs = dict(hermi=hermi, max_memory=.001)
                if route == 'uks':
                    method = ni.nr_uks_fxc
                elif route in ('singlet', 'triplet'):
                    method = ni.nr_rks_fxc_st
                    kwargs['singlet'] = route == 'singlet'
                else:
                    method = ni.nr_rks_fxc
                with self.subTest(route=route, hermi=hermi):
                    reference = method(self.mol, grids, 'TPSS', None, dm1,
                                       fxc=expanded, **kwargs)
                    actual = method(self.mol, grids, 'TPSS', None, dm1,
                                    fxc=compact, **kwargs)
                    numpy.testing.assert_allclose(actual, reference, atol=2e-11, rtol=2e-12)

    def test_response_closures(self):
        rng, grids, coeff, occ = self.inputs()
        perturbation = rng.normal(size=(2, self.mol.nao, self.mol.nao))
        perturbation += perturbation.swapaxes(-1, -2)
        for spin in (0, 1):
            mf = dft.UKS(self.mol) if spin else dft.RKS(self.mol)
            mf.grids = grids
            mf.xc = 'TPSS'
            mf.mo_coeff = numpy.asarray((coeff, coeff)) if spin else coeff
            mf.mo_occ = numpy.asarray((occ*.6, occ*.4)) if spin else occ
            dm1 = numpy.asarray((perturbation*.6, perturbation*.4)) if spin else perturbation
            for singlet in ((None,) if spin else (None, False, True)):
                kwargs = {} if spin else {'singlet': singlet}
                actual = mf.gen_response(hermi=1, with_nlc=False, **kwargs)(dm1)
                with mock.patch.object(numint, '_cache_xc_kernel_response', side_effect=dense_cache):
                    reference = mf.gen_response(hermi=1, with_nlc=False, **kwargs)(dm1)
                numpy.testing.assert_allclose(actual, reference, atol=2e-11, rtol=2e-12)

    def test_response_secondary_grids(self):
        rng, grids, coeff, occ = self.inputs()
        secondary = dft.Grids(self.mol)
        secondary.coords = grids.coords[:137] + .2
        secondary.weights = grids.weights[:137]
        explicit = dft.Grids(self.mol)
        explicit.coords = grids.coords[:181] - .3
        explicit.weights = grids.weights[:181]
        perturbation = rng.normal(size=(self.mol.nao, self.mol.nao))
        perturbation += perturbation.T
        for spin in (0, 1):
            mf = dft.UKS(self.mol) if spin else dft.RKS(self.mol)
            mf.grids = grids
            mf.second_grids = secondary
            mf.xc = 'TPSS'
            mf.mo_coeff = numpy.asarray((coeff, coeff)) if spin else coeff
            mf.mo_occ = numpy.asarray((occ*.6, occ*.4)) if spin else occ
            dm1 = numpy.asarray((perturbation*.6, perturbation*.4)) if spin else perturbation
            for chosen, selected in ((None, secondary), (explicit, explicit)):
                for singlet in ((None,) if spin else (None, False, True)):
                    kwargs = {} if spin else {'singlet': singlet}
                    with self.subTest(spin=spin, explicit=chosen is not None,
                                      singlet=singlet):
                        with mock.patch.object(numint, '_cache_xc_kernel_response',
                                               wraps=numint._cache_xc_kernel_response) as cache:
                            actual = mf.gen_response(grids=chosen, hermi=1,
                                                     with_nlc=False, **kwargs)(dm1)
                        self.assertIs(cache.call_args[0][2], selected)
                        with mock.patch.object(numint, '_cache_xc_kernel_response',
                                               side_effect=dense_cache):
                            reference = mf.gen_response(grids=selected, hermi=1,
                                                        with_nlc=False, **kwargs)(dm1)
                        numpy.testing.assert_allclose(actual, reference,
                                                      atol=2e-11, rtol=2e-12)

    def test_response_density_matrix_fallback(self):
        rng, grids, coeff, occ = self.inputs()
        perturbation = rng.normal(size=(self.mol.nao, self.mol.nao))
        perturbation += perturbation.T
        for spin in (0, 1):
            mf = dft.UKS(self.mol) if spin else dft.RKS(self.mol)
            mf.second_grids = grids
            mf.xc = 'TPSS'
            coeffs = numpy.asarray((coeff, coeff)) if spin else coeff
            occs = numpy.asarray((occ*.6, occ*.4)) if spin else occ
            dm0 = numpy.asarray(mf.make_rdm1(coeffs, occs))
            dm1 = numpy.asarray((perturbation*.6, perturbation*.4)) if spin else perturbation
            for singlet in ((None,) if spin else (None, False, True)):
                kwargs = {} if spin else {'singlet': singlet}
                with self.subTest(spin=spin, singlet=singlet):
                    with mock.patch.object(numint, '_cache_xc_kernel_response',
                                           side_effect=dense_cache):
                        reference = mf.gen_response(coeffs, occs, grids=grids,
                                                    hermi=1, with_nlc=False, **kwargs)(dm1)
                    with mock.patch.object(numint, '_cache_xc_kernel_response') as compact:
                        with mock.patch.object(mf._numint, 'cache_xc_kernel1',
                                               wraps=mf._numint.cache_xc_kernel1) as cache:
                            actual = mf.gen_response(dm0=dm0, hermi=1,
                                                     with_nlc=False, **kwargs)(dm1)
                    compact.assert_not_called()
                    cache.assert_called_once()
                    self.assertIs(cache.call_args[0][1], grids)
                    numpy.testing.assert_allclose(actual, reference,
                                                  atol=2e-11, rtol=2e-12)

    def test_fallbacks_and_public_arrays(self):
        _, grids, coeff, occ = self.inputs()
        for variant in ('complex', 'subclass', 'xc1', 'xc_eff', 'rho', 'lda'):
            ni = type('CustomNumInt', (numint.NumInt,), {})() if variant == 'subclass' else numint.NumInt()
            c = coeff.astype(complex) if variant == 'complex' else coeff
            xc = 'LDA,VWN' if variant == 'lda' else 'TPSS'
            method = {'xc1': 'eval_xc1', 'xc_eff': 'eval_xc_eff', 'rho': 'eval_rho2'}.get(variant)
            if method:
                original = getattr(ni, method)
                setattr(ni, method, lambda *args, **kwargs: original(*args, **kwargs))
            _, vxc, fxc = numint._cache_xc_kernel_response(
                ni, self.mol, grids, xc, c, occ, max_memory=.001)
            self.assertIsInstance(vxc, numpy.ndarray)
            self.assertIsInstance(fxc, numpy.ndarray)
        ni = numint.NumInt()
        rho, vxc, fxc = ni.cache_xc_kernel(self.mol, grids, 'TPSS', coeff, occ, max_memory=.001)
        self.assertTrue(all(isinstance(a, numpy.ndarray) for a in (rho, vxc, fxc)))
        for spin in (0, 1):
            r, v, compact = numint._cache_xc_kernel_response(
                ni, self.mol, grids, 'TPSS', coeff, occ, spin, max_memory=.001)
            count = 15 if spin == 0 else 46
            self.assertEqual(r.nbytes + compact.values.nbytes, count*8*len(grids.coords))

    def test_analytic_cpks(self):
        from pyscf.hessian.rhf import gen_vind
        from pyscf.scf import cphf
        mf = dft.RKS(self.mol)
        mf.xc = 'TPSS'
        mf.grids.level = 0
        mf.conv_tol = 1e-11
        mf.kernel()
        self.assertTrue(mf.converged)
        coeff, occ, energies = mf.mo_coeff, mf.mo_occ, mf.mo_energy
        occupied = coeff[:,occ > 0]
        dipole = self.mol.intor_symmetric('int1e_r', comp=3)
        h1 = numpy.einsum('pi,xpq,qj->xij', coeff, dipole, occupied)
        s1 = numpy.zeros_like(h1)
        compact_action = gen_vind(mf, coeff, occ)
        actual = cphf.solve(compact_action, energies, occ, h1, s1, tol=1e-11)
        with mock.patch.object(numint, '_cache_xc_kernel_response', side_effect=dense_cache):
            reference_action = gen_vind(mf, coeff, occ)
        reference = cphf.solve(reference_action, energies, occ, h1, s1, tol=1e-11)
        for result, expected in zip(actual, reference):
            numpy.testing.assert_allclose(result, expected, atol=3e-10, rtol=3e-10)
        gap = energies[occ == 0,None] - energies[occ > 0]
        residual = actual[0][:,occ == 0]*gap + h1[:,occ == 0] + compact_action(actual[0])[:,occ == 0]
        self.assertLess(abs(residual).max(), 2e-8)

    def test_different_spin_orbital_counts(self):
        _, grids, coeff, occ = self.inputs()
        ni = numint.NumInt()
        occupations = (occ, occ[:3])
        for complex_beta in (False, True):
            beta = coeff[:,:3].astype(complex) if complex_beta else coeff[:,:3]
            coefficients = (coeff, beta)
            reference = ni.cache_xc_kernel(self.mol, grids, 'TPSS', coefficients,
                                           occupations, spin=1, max_memory=.001)
            rho, vxc, fxc = numint._cache_xc_kernel_response(
                ni, self.mol, grids, 'TPSS', coefficients, occupations,
                spin=1, max_memory=.001)
            numpy.testing.assert_allclose(rho, reference[0], atol=2e-12, rtol=2e-12)
            if complex_beta:
                self.assertIsInstance(fxc, numpy.ndarray)
                numpy.testing.assert_allclose(vxc, reference[1], atol=2e-12, rtol=2e-12)
                numpy.testing.assert_allclose(fxc, reference[2], atol=2e-12, rtol=2e-12)
            else:
                self.assertIsInstance(fxc, _CompactXCKernel)
                expanded = xc_deriv.transform_xc(fxc.rho, fxc.values, 'MGGA', 1, 2)
                numpy.testing.assert_allclose(expanded, reference[2], atol=2e-12, rtol=2e-12)


if __name__ == '__main__':
    unittest.main()
