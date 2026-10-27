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
import numpy
from pyscf import gto
from pyscf.dft import numint


class KnownValues(unittest.TestCase):
    def check_mgga(self, complex_coeff):
        mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0',
                    basis='sto-3g', verbose=0)
        rng = numpy.random.default_rng(18)
        coeff = rng.standard_normal((mol.nao, mol.nao))
        if complex_coeff:
            coeff = coeff + 1j*rng.standard_normal(coeff.shape)
        coeff = numpy.linalg.qr(coeff)[0]
        occ = numpy.linspace(-.8, 1.7, mol.nao)
        occ[mol.nao//2] = 0
        dm = (coeff * occ).dot(coeff.conj().T)
        ao2 = numint.eval_ao(mol, rng.standard_normal((113, 3)), deriv=2)
        for deriv, with_lapl in ((1, False), (1, True), (2, True)):
            with self.subTest(deriv=deriv, with_lapl=with_lapl):
                ao = ao2 if deriv == 2 else ao2[:4]
                orbitals = numpy.einsum('xpi,ij->xpj', ao, coeff)
                ref = numpy.zeros((6 if with_lapl else 5, ao.shape[1]))
                ref[0] = numpy.einsum('pi,i,pi->p', orbitals[0].conj(),
                                     occ, orbitals[0]).real
                ref[1:4] = 2*numpy.einsum('pi,i,xpi->xp', orbitals[0].conj(),
                                          occ, orbitals[1:4]).real
                ref[-1] = .5*numpy.einsum('xpi,i,xpi->p', orbitals[1:4].conj(),
                                          occ, orbitals[1:4]).real
                if deriv == 2:
                    lapl = orbitals[4] + orbitals[7] + orbitals[9]
                    ref[4] = 2*numpy.einsum('pi,i,pi->p', orbitals[0].conj(),
                                            occ, lapl).real + 4*ref[-1]
                rho = numint.eval_rho2(mol, ao, coeff, occ, xctype='MGGA',
                                      with_lapl=with_lapl)
                numpy.testing.assert_allclose(rho, ref, atol=2e-12, rtol=2e-12)
                if deriv == 2 or not with_lapl:
                    rho_dm = numint.eval_rho(mol, ao, dm, xctype='MGGA',
                                            hermi=1, with_lapl=with_lapl)
                    numpy.testing.assert_allclose(rho, rho_dm, atol=2e-12,
                                                  rtol=2e-12)

    def test_real_mgga_mixed_occupations(self):
        self.check_mgga(False)

    def test_complex_mgga_mixed_occupations(self):
        self.check_mgga(True)

    def test_noncontiguous_contraction(self):
        rng = numpy.random.default_rng(19)
        bra = rng.standard_normal((115, 10))[:, ::2]
        ket = rng.standard_normal((115, 10))[:, ::2]
        grad, tau = numint._contract_rho_grad_tau(bra, ket)
        numpy.testing.assert_allclose(grad, numpy.einsum('pi,pi->p', bra, ket))
        numpy.testing.assert_allclose(tau, numpy.einsum('pi,pi->p', ket, ket))


if __name__ == '__main__':
    unittest.main()
