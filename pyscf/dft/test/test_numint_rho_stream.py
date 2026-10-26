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

import ctypes
import unittest
import numpy
from pyscf import gto, lib
from pyscf.dft import numint


def legacy_mgga(mol, ao, coeff, occ, screen=None):
    rho = numpy.zeros((5, ao.shape[1]))
    for sign in (1, -1):
        selected = sign * occ > numint.OCCDROP
        if not numpy.any(selected):
            continue
        weighted = numpy.einsum('ij,j->ij', coeff[:, selected], numpy.sqrt(sign*occ[selected]))
        orbitals = [numint._dot_ao_dm(mol, ao[c], weighted, screen,
                                     (0, mol.nbas), mol.ao_loc_nr()) for c in range(4)]
        rho[0] += sign * numint._contract_rho(orbitals[0], orbitals[0])
        for c in range(1, 4):
            grad, tau = numint._contract_rho_grad_tau(orbitals[0], orbitals[c])
            rho[c] += sign * 2 * grad
            rho[4] += sign * .5 * tau
    return rho


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0',
                        basis='cc-pvdz', verbose=0)

    def _inputs(self, ngrids):
        rng = numpy.random.default_rng(27)
        nao = self.mol.nao
        ao = rng.normal(size=(8, nao, ngrids))[::2].transpose(0, 2, 1)
        coeff = numpy.linalg.qr(rng.normal(size=(nao, nao)))[0]
        occ = numpy.linspace(-.8, 1.7, nao)
        occ[:3] = (0, numint.OCCDROP*.5, -numint.OCCDROP*.5)
        return ao, coeff, occ

    def test_signed_occupations(self):
        for ngrids in (1, 127, 128, 129, 1025):
            ao, coeff, occ = self._inputs(ngrids)
            ao = ao[::-1]
            reference = legacy_mgga(self.mol, ao, coeff, occ)
            orbitals = numpy.einsum('xpi,ij->xpj', ao, coeff)
            dropped = numpy.where(abs(occ) > numint.OCCDROP, occ, 0)
            direct = numpy.empty_like(reference)
            direct[0] = numpy.einsum('pi,i,pi->p', orbitals[0], dropped, orbitals[0])
            direct[1:4] = 2*numpy.einsum('pi,i,xpi->xp', orbitals[0], dropped, orbitals[1:4])
            direct[4] = .5*numpy.einsum('xpi,i,xpi->p', orbitals[1:4], dropped, orbitals[1:4])
            for threads in (1, 4):
                with self.subTest(ngrids=ngrids, threads=threads), lib.with_omp_threads(threads):
                    actual = numint.eval_rho2(self.mol, ao, coeff, occ,
                                              xctype='MGGA', with_lapl=False)
                    numpy.testing.assert_allclose(actual, reference, atol=2e-12, rtol=2e-12)
                    numpy.testing.assert_allclose(actual, direct, atol=2e-12, rtol=2e-12)

    def test_zero_occupations(self):
        ao, coeff, occ = self._inputs(129)
        actual = numint.eval_rho2(self.mol, ao, coeff, numpy.zeros_like(occ),
                                  xctype='MGGA', with_lapl=False)
        numpy.testing.assert_array_equal(actual, numpy.zeros((5, 129)))

    def test_exact_zero_compaction(self):
        mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0; '
                         'O 4 0 0; H 4 0 1; H 4 1 0',
                    basis='cc-pvdz', verbose=0)
        rng = numpy.random.default_rng(29)
        coeff = rng.normal(size=(mol.nao, mol.nao))
        coeff[2] *= 1e30
        occ = numpy.linspace(-.8, 1.7, mol.nao)
        for ngrids in (1, 55, 56, 111, 112, 113, 225):
            ao = rng.normal(size=(8, mol.nao, ngrids))[::2].transpose(0, 2, 1)
            ao = ao[::-1]
            ao[:, :, numpy.arange(mol.nao) % 4 != 0] = 0
            ao[1, -1, 1] = .25  # A derivative alone keeps this AO column.
            ao[:, :, 2] = 1e-30  # Small nonzero values must never be screened.
            ao[:, :, 3] = -0.
            orbitals = numpy.einsum('xpi,ij->xpj', ao, coeff)
            reference = numpy.empty((5, ngrids))
            reference[0] = numpy.einsum('pi,i,pi->p', orbitals[0], occ, orbitals[0])
            reference[1:4] = 2*numpy.einsum('pi,i,xpi->xp', orbitals[0], occ,
                                            orbitals[1:4])
            reference[4] = .5*numpy.einsum('xpi,i,xpi->p', orbitals[1:4], occ,
                                          orbitals[1:4])
            for threads in (1, 2):
                with self.subTest(ngrids=ngrids, threads=threads), lib.with_omp_threads(threads):
                    actual = numint.eval_rho2(mol, ao, coeff, occ,
                                              xctype='MGGA', with_lapl=False)
                    numpy.testing.assert_allclose(actual, reference, atol=2e-11,
                                                  rtol=2e-12)

    def test_nonfinite_coefficients_on_zero_ao(self):
        ao = numpy.zeros((4, 48, 113)).transpose(0, 2, 1)
        sign = numpy.ones(32)
        rho = numpy.empty((5, 113))
        for value in (numpy.nan, numpy.inf, -numpy.inf):
            coeff = numpy.ones((48, 32))
            coeff[3, 2] = value
            done = numint.libdft.VXCeval_rho_mgga_mo(
                rho.ctypes.data_as(ctypes.c_void_p),
                ao.ctypes.data_as(ctypes.c_void_p),
                coeff.ctypes.data_as(ctypes.c_void_p),
                sign.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(48),
                ctypes.c_int(32), ctypes.c_int(113),
                ctypes.c_ssize_t(ao.strides[0] // ao.itemsize))
            self.assertEqual(done, 1)
            self.assertTrue(numpy.isnan(rho).all())

    def test_screened_and_layout_fallback(self):
        ao, coeff, occ = self._inputs(129)
        screen = numpy.ones((3, self.mol.nbas), dtype=numpy.uint8)
        with lib.temporary_env(numint, SWITCH_SIZE=0):
            actual = numint.eval_rho2(self.mol, ao, coeff, occ, screen,
                                      'MGGA', with_lapl=False)
            reference = legacy_mgga(self.mol, ao, coeff, occ, screen)
        numpy.testing.assert_allclose(actual, reference, atol=2e-12, rtol=2e-12)
        ao = numpy.asarray(ao, order='C')
        actual = numint.eval_rho2(self.mol, ao, coeff, occ, xctype='MGGA', with_lapl=False)
        reference = legacy_mgga(self.mol, ao, coeff, occ)
        numpy.testing.assert_allclose(actual, reference, atol=2e-12, rtol=2e-12)

    def test_scratch_limit(self):
        # The byte limit is checked before reading inputs or allocating scratch.
        sentinel = numpy.array([42.])
        ptr = sentinel.ctypes.data_as(ctypes.c_void_p)
        with lib.with_omp_threads(8):
            done = numint.libdft.VXCeval_rho_mgga_mo(
                ptr, ptr, ptr, ptr, ctypes.c_int(4096), ctypes.c_int(8192),
                ctypes.c_int(1), ctypes.c_ssize_t(4096))
        self.assertEqual(done, 0)
        numpy.testing.assert_array_equal(sentinel, [42.])


if __name__ == '__main__':
    unittest.main()
