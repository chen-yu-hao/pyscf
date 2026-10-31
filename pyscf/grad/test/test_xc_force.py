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

'''XC forces contracted on the grid (grad.rks.get_xc_force and the grad_elec
path) must reproduce the get_vxc matrix contraction.'''

import unittest
from unittest import mock
import numpy
from pyscf import gto, dft, lib
from pyscf.dft import numint
from pyscf.grad import rks as rks_grad

ATOL = 1e-11


def setUpModule():
    global mol, mol_min
    mol = gto.M(atom='''
        O  0.00  0.00  0.00
        H  0.00  0.76  0.59
        H  0.00 -0.76  0.59
        O  3.00  0.10  0.00
        H  3.70  0.00  0.50
        H  3.00  0.80 -0.40''', basis='def2-svp', verbose=0)
    # Minimal basis: more occupied orbitals than nao/2, AO density path
    mol_min = gto.M(atom='O 0 0 0; H 0 .76 .59; H 0 -.76 .59',
                    basis='sto-3g', verbose=0)

def tearDownModule():
    global mol, mol_min
    del mol, mol_min


def _scf(mol, xc, **kwargs):
    mf = dft.RKS(mol, xc=xc)
    mf.grids.level = 2
    mf.conv_tol = 1e-11
    for k, v in kwargs.items():
        setattr(mf, k, v)
    return mf.run()

def _reference_force(mf, dm):
    ni, mol = mf._numint, mf.mol
    vxc = rks_grad.get_vxc(ni, mol, mf.grids, mf.xc, dm)[1]
    return numpy.array([2 * numpy.einsum('xij,ij->x', vxc[:,p0:p1], dm[p0:p1])
                        for p0, p1 in mol.aoslice_by_atom()[:,2:]])

def _grad(mf, enabled, jk=None):
    jk = enabled if jk is None else jk
    with lib.temporary_env(rks_grad, XC_FORCE=enabled, JK_FORCE=jk):
        return mf.nuc_grad_method().kernel()


class KnownValues(unittest.TestCase):
    def test_get_xc_force_matches_matrix_contraction(self):
        for xc in ('LDA', 'PBE', 'TPSS', 'MN15'):
            mf = _scf(mol, xc)
            dm = mf.make_rdm1()
            ref = _reference_force(mf, dm)
            tagged = lib.tag_array(dm, mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ)
            for d in (dm, tagged):   # AO density path and orbital path
                f = rks_grad.get_xc_force(mf._numint, mol, mf.grids, xc, d)
                self.assertAlmostEqual(abs(f - ref).max(), 0, delta=ATOL, msg=xc)

    def test_full_gradient_unchanged(self):
        for xc in ('LDA', 'B3LYP', 'M06L', 'wB97X', 'HF'):
            mf = _scf(mol, xc)
            ref = _grad(mf, False)
            self.assertAlmostEqual(abs(_grad(mf, True) - ref).max(), 0,
                                   delta=ATOL, msg=xc)

    def test_ao_density_path(self):
        # 2*nocc > nao selects the AO density factorization
        mf = _scf(mol_min, 'PBE')
        nocc = numpy.count_nonzero(mf.mo_occ)
        self.assertGreater(2 * nocc, mol_min.nao)
        c, d = rks_grad._xc_force_density(
            lib.tag_array(mf.make_rdm1(), mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ),
            mol_min.nao)
        self.assertIsNone(c)
        self.assertAlmostEqual(abs(_grad(mf, True) - _grad(mf, False)).max(), 0,
                               delta=ATOL)

    def test_screened_products(self):
        # Exercise the shell-screened ao.dot(dm) branch used for nao >= SWITCH_SIZE
        for xc in ('PBE', 'MN15'):
            mf = _scf(mol, xc)
            ref = _grad(mf, False)
            with mock.patch.object(numint, 'SWITCH_SIZE', 0):
                new = _grad(mf, True)
            self.assertAlmostEqual(abs(new - ref).max(), 0, delta=ATOL, msg=xc)

    def test_fractional_occupation(self):
        mf = _scf(mol, 'PBE')
        mo_occ = mf.mo_occ.copy()
        nocc = numpy.count_nonzero(mo_occ)
        mo_occ[nocc-1] = 1.3
        mo_occ[nocc] = .7
        dm = mf.make_rdm1(mf.mo_coeff, mo_occ)
        tagged = lib.tag_array(dm, mo_coeff=mf.mo_coeff, mo_occ=mo_occ)
        f = rks_grad.get_xc_force(mf._numint, mol, mf.grids, mf.xc, tagged)
        self.assertAlmostEqual(abs(f - _reference_force(mf, dm)).max(), 0, delta=ATOL)

    def test_direct_get_veff_keeps_full_matrix(self):
        mf = _scf(mol, 'PBE')
        g = mf.nuc_grad_method()
        dm = mf.make_rdm1()
        vhf = g.get_veff(mol, dm)
        vxc = rks_grad.get_vxc(mf._numint, mol, mf.grids, mf.xc, dm)[1]
        vj = g.get_j(mol, dm)
        self.assertAlmostEqual(abs(vhf - (vxc + vj)).max(), 0, delta=1e-12)
        self.assertIsNone(g._force_terms)

    def test_grid_response_and_nlc_keep_matrix_path(self):
        mf = _scf(mol, 'PBE')
        g = mf.nuc_grad_method()
        g.grid_response = True
        with mock.patch.object(rks_grad, 'get_xc_force') as force:
            g.kernel()
            force.assert_not_called()
        mf = _scf(mol, 'wB97M_V')
        mf.nlcgrids.level = 1
        ref = _grad(mf, False)
        self.assertAlmostEqual(abs(_grad(mf, True) - ref).max(), 0, delta=ATOL)

    def test_custom_density_evaluator_keeps_matrix_path(self):
        mf = _scf(mol, 'PBE')
        mf._numint.eval_rho2 = numint.eval_rho2   # instance override
        with mock.patch.object(rks_grad, 'get_xc_force') as force:
            mf.nuc_grad_method().kernel()
            force.assert_not_called()

    def test_scanner_and_subclass(self):
        mf = _scf(mol, 'PBE')
        scanner = mf.nuc_grad_method().as_scanner()
        mol1 = mol.set_geom_(mol.atom_coords() + .01, unit='Bohr', inplace=False)
        with mock.patch.object(rks_grad, 'get_xc_force',
                               wraps=rks_grad.get_xc_force) as force:
            g1 = scanner(mol1)[1]
            self.assertEqual(force.call_count, 1)
        # Same converged scanner state through the matrix path
        with lib.temporary_env(rks_grad, XC_FORCE=False):
            g0 = scanner.kernel()
        self.assertAlmostEqual(abs(g1 - g0).max(), 0, delta=ATOL)
        # DFT+U chains get_veff and extra_force through the RKS class
        from pyscf.dft import rkspu
        mf = rkspu.RKSpU(mol, xc='PBE', U_idx=['0 O 2p'], U_val=[3.])
        mf.grids.level = 2
        mf.conv_tol = 1e-11
        mf.run()
        self.assertAlmostEqual(abs(_grad(mf, True) - _grad(mf, False)).max(), 0,
                               delta=ATOL)


    def test_uks(self):
        from pyscf.grad import uks as uks_grad
        mol1 = mol.copy()
        mol1.charge, mol1.spin = 1, 1
        mol1.build()
        for xc in ('LDA', 'PBE', 'B3LYP', 'MN15'):
            mf = dft.UKS(mol1, xc=xc)
            mf.grids.level = 2
            mf.conv_tol = 1e-11
            mf.run()
            dm = mf.make_rdm1()
            vxc = uks_grad.get_vxc(mf._numint, mol1, mf.grids, xc, dm)[1]
            ref = numpy.array([2 * numpy.einsum('sxij,sij->x', vxc[:,:,p0:p1], dm[:,p0:p1])
                               for p0, p1 in mol1.aoslice_by_atom()[:,2:]])
            tagged = lib.tag_array(dm, mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ)
            for d in (dm, tagged):
                f = uks_grad.get_xc_force(mf._numint, mol1, mf.grids, xc, d)
                self.assertAlmostEqual(abs(f - ref).max(), 0, delta=ATOL, msg=xc)
            self.assertAlmostEqual(abs(_grad(mf, True) - _grad(mf, False)).max(), 0,
                                   delta=ATOL, msg=xc)

    def test_uks_empty_spin_and_roks(self):
        # One-electron spin channel empty: AO density path for beta
        h = gto.M(atom='H 0 0 0; H 0 0 0.9', charge=1, spin=1,
                  basis='def2-svp', verbose=0)
        mf = dft.UKS(h, xc='PBE')
        mf.grids.level = 2
        mf.conv_tol = 1e-11
        mf.run()
        self.assertEqual(numpy.count_nonzero(mf.mo_occ[1]), 0)
        self.assertAlmostEqual(abs(_grad(mf, True) - _grad(mf, False)).max(), 0,
                               delta=ATOL)
        # ROKS keeps the UHF grad_elec and the matrix path
        mf = dft.ROKS(h, xc='PBE').run()
        from pyscf.grad import uks as uks_grad
        with mock.patch.object(uks_grad, 'get_xc_force') as force:
            mf.nuc_grad_method().kernel()
            force.assert_not_called()


if __name__ == "__main__":
    print("Tests for XC forces contracted on the grid")
    unittest.main()
