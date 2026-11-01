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

'''Fused tile-by-tile RKS XC integration against the block-loop reference.'''

import os
import unittest
from unittest import mock
import numpy
from pyscf import gto, dft, lib
from pyscf.dft import numint


def _reference(ni, mol, grids, xc, dm, hermi=1):
    with mock.patch.dict(os.environ, {'PYSCF_DFT_FUSED': '0'}):
        return ni.nr_rks(mol, grids, xc, dm, hermi=hermi)

def _fused(ni, mol, grids, xc, dm, hermi=1, tile=None):
    env = {'PYSCF_DFT_FUSED': '1'}
    if tile is not None:
        env['PYSCF_DFT_FUSED_TILE'] = str(tile)
    with mock.patch.dict(os.environ, env):
        out = numint._nr_rks_fused(ni, mol, grids, xc, dm, hermi, 2000)
        assert out is not None
        # The public entry point must take the same route.
        out1 = ni.nr_rks(mol, grids, xc, dm, hermi=hermi)
    for a, b in zip(out, out1):
        assert numpy.array_equal(a, b)
    return out


def setUpModule():
    global mol, mol_cart, mol_g, mf, dm_mo, dm_ao, grids
    mol = gto.M(atom='''O 0 0 0; H 0 .757 .587; H 0 -.757 .587;
                        O 2.9 .1 -.2; H 3.4 .8 .3; H 3.4 -.6 .4''',
                basis='def2-tzvp', verbose=0)
    mol_cart = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='cc-pvtz', cart=True,
                     verbose=0)
    mol_g = gto.M(atom='Ne 0 0 0; H 0 0 1.0', verbose=0, spin=1, charge=0,
                  basis={'Ne': 'cc-pvdz',
                         'H': [[0, [1.2, 1.]], [4, [1.1, 1.]], [3, [.9, 1.]]]})
    grids = dft.Grids(mol)
    grids.level = 2
    grids.build(with_non0tab=True)
    mf = dft.RKS(mol, xc='PBE')
    mf.grids = grids
    mf.conv_tol = 1e-10
    mf.kernel()
    dm_mo = mf.make_rdm1()
    dm_ao = numpy.asarray(mf.get_init_guess())

def tearDownModule():
    global mol, mol_cart, mol_g, mf, dm_mo, dm_ao, grids
    del mol, mol_cart, mol_g, mf, dm_mo, dm_ao, grids


class KnownValues(unittest.TestCase):
    def _check(self, m, g, xc, dm, hermi=1, tile=None, atol=1e-11):
        ni = numint.NumInt()
        n0, e0, v0 = _reference(ni, m, g, xc, dm, hermi)
        n1, e1, v1 = _fused(ni, m, g, xc, dm, hermi, tile)
        self.assertAlmostEqual(abs(numpy.asarray(n1) - n0).max(), 0, 10)
        self.assertAlmostEqual(abs(numpy.asarray(e1) - e0).max(), 0, 10)
        self.assertEqual(numpy.shape(v1), numpy.shape(v0))
        self.assertTrue(abs(v1 - v0).max() < atol, abs(v1 - v0).max())
        self.assertTrue(abs(v1 - v1.swapaxes(-1, -2)).max() == 0)

    def test_functionals_mo_density(self):
        for xc in ('LDA,VWN', 'PBE', 'B3LYP', 'M06-L', 'MN15', 'SCAN', 'wB97X'):
            with self.subTest(xc=xc):
                self._check(mol, grids, xc, dm_mo)

    def test_functionals_ao_density(self):
        for xc in ('LDA,VWN', 'PBE', 'MN15'):
            with self.subTest(xc=xc):
                self._check(mol, grids, xc, dm_ao)

    def test_mo_route_on_every_tile(self):
        # Many occupied orbitals: the density-matrix route is chosen instead.
        mo = mf.mo_coeff
        occ = numpy.linspace(2., .05, mo.shape[1])
        occ[3] = -.2  # negative occupations are kept with their sign
        dm = lib.tag_array(numpy.einsum('pi,i,qi->pq', mo, occ, mo),
                           mo_coeff=mo, mo_occ=occ)
        for xc in ('LDA,VWN', 'PBE', 'MN15'):
            with self.subTest(xc=xc):
                self._check(mol, grids, xc, dm)
        # Few occupied orbitals: orbital route on every tile.
        occ = numpy.zeros(mo.shape[1])
        occ[:3] = [2, 1.5, .25]
        dm = lib.tag_array(numpy.einsum('pi,i,qi->pq', mo, occ, mo),
                           mo_coeff=mo, mo_occ=occ)
        for xc in ('LDA,VWN', 'PBE', 'MN15'):
            with self.subTest(xc=xc, few=True):
                self._check(mol, grids, xc, dm)

    def test_multiple_dms_and_hermi0(self):
        numpy.random.seed(2)
        nao = mol.nao
        a = numpy.random.random((nao, nao)) * .01
        dms = numpy.asarray([dm_ao, dm_mo + a])
        for xc in ('PBE', 'MN15'):
            with self.subTest(xc=xc):
                self._check(mol, grids, xc, dms, hermi=0)
                self._check(mol, grids, xc, [dm_ao, dm_mo], hermi=1)

    def test_tile_sizes(self):
        for tile in (1, 3, 8):
            with self.subTest(tile=tile):
                self._check(mol, grids, 'MN15', dm_mo, tile=tile)

    def test_threads(self):
        ni = numint.NumInt()
        ref = _fused(ni, mol, grids, 'MN15', dm_mo)
        with lib.with_omp_threads(1):
            out = _fused(ni, mol, grids, 'MN15', dm_mo)
        self.assertAlmostEqual(abs(out[1] - ref[1]), 0, 11)
        self.assertTrue(abs(out[2] - ref[2]).max() < 1e-12)

    def test_blas_threads_restored(self):
        try:
            from threadpoolctl import threadpool_info, threadpool_limits
        except ImportError:
            self.skipTest('threadpoolctl not available')
        def blas_threads():
            return sorted((i['filepath'], i['num_threads']) for i in threadpool_info()
                          if i['user_api'] == 'blas')
        ni = numint.NumInt()
        ref = _fused(ni, mol, grids, 'PBE', dm_mo)
        with threadpool_limits(limits=2, user_api='blas'):
            before = blas_threads()
            out = _fused(ni, mol, grids, 'PBE', dm_mo)
            self.assertEqual(blas_threads(), before)
        self.assertAlmostEqual(abs(out[2] - ref[2]).max(), 0, 13)

    def test_cartesian_f_functions(self):
        g = dft.Grids(mol_cart)
        g.level = 1
        g.build(with_non0tab=True)
        dm = dft.RKS(mol_cart).get_init_guess()
        for xc in ('LDA,VWN', 'PBE', 'TPSS'):
            with self.subTest(xc=xc):
                self._check(mol_cart, g, xc, dm)

    def test_g_functions_ragged_grid(self):
        g = dft.Grids(mol_g)
        g.level = 1
        g.alignment = 0
        g.build(with_non0tab=True)
        self.assertNotEqual(g.weights.size % numint.BLKSIZE, 0)
        dm = numpy.asarray(dft.UKS(mol_g).get_init_guess()).sum(axis=0)
        for xc in ('LDA,VWN', 'PBE', 'MN15'):
            with self.subTest(xc=xc):
                self._check(mol_g, g, xc, dm)

    def test_without_screening_table(self):
        g = dft.Grids(mol)
        g.coords = grids.coords
        g.weights = grids.weights
        g.non0tab = None
        self._check(mol, g, 'PBE', dm_mo)

    def test_scf_energy(self):
        for xc in ('PBE', 'MN15'):
            with self.subTest(xc=xc):
                energies = []
                for flag in ('0', '1'):
                    with mock.patch.dict(os.environ, {'PYSCF_DFT_FUSED': flag}):
                        m = dft.RKS(mol, xc=xc)
                        m.grids = grids
                        m.conv_tol = 1e-11
                        energies.append(m.kernel())
                self.assertAlmostEqual(energies[0], energies[1], 9)

    def test_scope(self):
        class MyNumInt(numint.NumInt):
            pass
        ni = MyNumInt()
        self.assertIsNone(numint._nr_rks_fused(ni, mol, grids, 'PBE', dm_mo, 1, 2000))
        ni = numint.NumInt()
        self.assertIsNone(numint._nr_rks_fused(ni, mol, grids, 'HF', dm_mo, 1, 2000))
        self.assertIsNone(numint._nr_rks_fused(ni, mol, grids, 'PBE', dm_mo, 2, 2000))
        ni.eval_ao = lambda *args, **kwargs: numint.eval_ao(*args, **kwargs)
        self.assertIsNone(numint._nr_rks_fused(ni, mol, grids, 'PBE', dm_mo, 1, 2000))
        with mock.patch.dict(os.environ, {'PYSCF_DFT_FUSED': '0'}):
            self.assertIsNone(numint._nr_rks_fused(numint.NumInt(), mol, grids,
                                                   'PBE', dm_mo, 1, 2000))


if __name__ == "__main__":
    print("Full Tests for fused RKS XC integration")
    unittest.main()
