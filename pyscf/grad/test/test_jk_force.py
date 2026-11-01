#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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

'''Coulomb/exchange gradient forces contracted per shell quartet
(grad.rhf.get_jk_force) must reproduce the get_jk matrix contraction.'''

import unittest
from unittest import mock
import numpy
from pyscf import gto, scf, dft, lib
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import rks as rks_grad

ATOL = 1e-10


def setUpModule():
    global mol, mol_cation
    mol = gto.M(atom='''
        O  0.00  0.00  0.00
        H  0.00  0.76  0.59
        H  0.00 -0.76  0.59
        O  3.00  0.10  0.00
        H  3.70  0.00  0.50
        H  3.00  0.80 -0.40''', basis='def2-svp', verbose=0)
    mol_cation = mol.copy()
    mol_cation.charge, mol_cation.spin = 1, 1
    mol_cation.build()

def tearDownModule():
    global mol, mol_cation
    del mol, mol_cation


def _contract(mol, v, dm):
    return numpy.array([2 * numpy.einsum('xij,ij->x', v[:,p0:p1], dm[p0:p1])
                        for p0, p1 in mol.aoslice_by_atom()[:,2:]])

def _grad(mf, enabled):
    with lib.temporary_env(rks_grad, XC_FORCE=enabled, JK_FORCE=enabled):
        return mf.nuc_grad_method().kernel()

def _dft(mol, xc, cls=dft.RKS):
    mf = cls(mol, xc=xc)
    mf.grids.level = 2
    mf.conv_tol = 1e-11
    return mf.run()


class KnownValues(unittest.TestCase):
    def test_get_jk_force(self):
        dm = scf.RHF(mol).run().make_rdm1()
        vj, vk = rhf_grad.get_jk(mol, dm)
        for jf, kf in ((1., 0.), (1., .5), (1., .125), (0., .3), (.7, -.2)):
            ref = _contract(mol, jf * vj - kf * vk, dm)
            de = rhf_grad.get_jk_force(mol, dm, jf, kf)
            self.assertAlmostEqual(abs(de - ref).max(), 0, delta=ATOL, msg=(jf, kf))

    def test_spin_densities_range_separation_and_cart(self):
        da, db = scf.UHF(mol_cation).run().make_rdm1()
        vj, vk = rhf_grad.get_jk(mol_cation, numpy.array([da, db]))
        ref = (_contract(mol_cation, vj[0] + vj[1] - .3 * vk[0], da)
               + _contract(mol_cation, vj[0] + vj[1] - .3 * vk[1], db))
        de = rhf_grad.get_jk_force(mol_cation, da + db, 1., .3, dm_k=[da, db])
        self.assertAlmostEqual(abs(de - ref).max(), 0, delta=ATOL)

        dm = da + db
        for omega in (.3, -.3):
            with mol_cation.with_range_coulomb(omega):
                vk = rhf_grad.get_k(mol_cation, dm)
                de = rhf_grad.get_jk_force(mol_cation, dm, 0., .4)
            self.assertAlmostEqual(abs(de - _contract(mol_cation, -.4 * vk, dm)).max(), 0,
                                   delta=ATOL, msg=omega)

        molc = mol.copy()
        molc.cart = True
        molc.build()
        dm = scf.RHF(molc).run().make_rdm1()
        vj, vk = rhf_grad.get_jk(molc, dm)
        de = rhf_grad.get_jk_force(molc, dm, 1., .5)
        self.assertAlmostEqual(abs(de - _contract(molc, vj - .5 * vk, dm)).max(), 0,
                               delta=ATOL)

    def test_rks_uks_gradients_unchanged(self):
        for xc in ('PBE', 'B3LYP', 'wB97X', 'HF'):
            mf = _dft(mol, xc)
            self.assertAlmostEqual(abs(_grad(mf, True) - _grad(mf, False)).max(), 0,
                                   delta=ATOL, msg=xc)
        for xc in ('PBE', 'B3LYP', 'CAM-B3LYP'):
            mf = _dft(mol_cation, xc, dft.UKS)
            self.assertAlmostEqual(abs(_grad(mf, True) - _grad(mf, False)).max(), 0,
                                   delta=ATOL, msg=xc)

    def test_jk_force_used_only_through_grad_elec(self):
        mf = _dft(mol, 'B3LYP')
        g = mf.nuc_grad_method()
        with mock.patch.object(rhf_grad, 'get_jk_force',
                               wraps=rhf_grad.get_jk_force) as force:
            g.kernel()
            self.assertEqual(force.call_count, 1)
            dm = mf.make_rdm1()
            vhf = g.get_veff(mol, dm)       # direct call: full matrix
            self.assertEqual(force.call_count, 1)
        vj, vk = g.get_jk(mol, dm)
        vxc = rks_grad.get_vxc(mf._numint, mol, mf.grids, mf.xc, dm)[1]
        hyb = mf._numint.hybrid_coeff(mf.xc)
        self.assertAlmostEqual(abs(vhf - (vxc + vj - .5 * hyb * vk)).max(), 0, delta=1e-12)

    def test_overridden_jk_keeps_matrix_path(self):
        # Density fitting replaces get_jk; the 4-center force must not be used
        mf = dft.RKS(mol, xc='B3LYP').density_fit()
        mf.grids.level = 2
        mf.run()
        with mock.patch.object(rhf_grad, 'get_jk_force') as force:
            mf.nuc_grad_method().kernel()
            force.assert_not_called()
        mf = _dft(mol, 'B3LYP')
        g = mf.nuc_grad_method()
        g.get_jk = lambda *args, **kwargs: rhf_grad.Gradients.get_jk(g, *args, **kwargs)
        with mock.patch.object(rhf_grad, 'get_jk_force') as force:
            g.kernel()
            force.assert_not_called()


if __name__ == "__main__":
    print("Tests for J/K gradient forces contracted per shell quartet")
    unittest.main()
