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

import unittest
import numpy
from pyscf import gto
from pyscf import dft
from pyscf.dft import numint


def setUpModule():
    global mol, mol1
    mol = gto.M(atom='O 0 0 0; H 0.96 0 0; H -0.24 0.93 0', basis='6-31g*',
                verbose=0)
    mol1 = gto.M(atom='O 0 0 0; H 1.02 0 0; H -0.30 0.95 0', basis='6-31g*',
                 verbose=0)

def tearDownModule():
    global mol, mol1
    del mol, mol1


class KnownValues(unittest.TestCase):
    def test_rks(self):
        for xc in ('lda,vwn', 'pbe', 'b3lyp', 'tpss'):
            ref = dft.RKS(mol, xc=xc)
            ref.grids.level = 1
            ref.kernel()
            mf = dft.RKS(mol, xc=xc)
            mf.grids.level = 1
            mf._numint = numint.CachedNumInt()
            mf.kernel()
            self.assertTrue(mf.converged)
            self.assertAlmostEqual(mf.e_tot, ref.e_tot, 12)
            self.assertEqual(mf.cycles, ref.cycles)
            self.assertAlmostEqual(abs(mf.make_rdm1() - ref.make_rdm1()).max(), 0, 10)
            self.assertTrue(mf._numint.cache_size() > 0)

    def test_uks(self):
        molc = mol.copy()
        molc.charge = 1
        molc.spin = 1
        molc.build(False, False)
        ref = dft.UKS(molc, xc='pbe')
        ref.grids.level = 1
        ref.kernel()
        mf = dft.UKS(molc, xc='pbe')
        mf.grids.level = 1
        mf._numint = numint.CachedNumInt()
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 12)

    def test_cache_reuse(self):
        grids = dft.gen_grid.Grids(mol)
        grids.level = 1
        grids.build(with_non0tab=True)
        dm = dft.RKS(mol).get_init_guess()
        ni0 = numint.NumInt()
        ni = numint.CachedNumInt()
        ncalls = [0]
        eval_ao = ni.eval_ao
        def counting_eval_ao(*args, **kwargs):
            ncalls[0] += 1
            return eval_ao(*args, **kwargs)
        ni.eval_ao = counting_eval_ao

        ref = ni0.nr_rks(mol, grids, 'pbe', dm)
        out = ni.nr_rks(mol, grids, 'pbe', dm)
        self.assertEqual(ncalls[0], 1)
        out1 = ni.nr_rks(mol, grids, 'pbe', dm)
        self.assertEqual(ncalls[0], 1)
        for a, b, c in zip(ref, out, out1):
            self.assertAlmostEqual(abs(a - b).max(), 0, 12)
            self.assertAlmostEqual(abs(a - c).max(), 0, 12)
        nbytes = 4 * grids.coords.shape[0] * mol.nao * 8
        self.assertAlmostEqual(ni.cache_size(), nbytes / 1e6, 9)

        # AO values are served from the cache of the AO derivatives
        rho_ref = ni0.get_rho(mol, dm, grids)
        rho = ni.get_rho(mol, dm, grids)
        self.assertEqual(ncalls[0], 1)
        self.assertAlmostEqual(abs(rho - rho_ref).max(), 0, 12)
        ref = ni0.nr_rks(mol, grids, 'lda,vwn', dm)
        out = ni.nr_rks(mol, grids, 'lda,vwn', dm)
        self.assertEqual(ncalls[0], 1)
        self.assertAlmostEqual(abs(ref[2] - out[2]).max(), 0, 12)

        # response functions
        dm1 = numpy.random.default_rng(1).normal(size=dm.shape)
        dm1 = dm1 + dm1.T
        ref = ni0.nr_rks_fxc(mol, grids, 'pbe', dm, dm1, hermi=1)
        out = ni.nr_rks_fxc(mol, grids, 'pbe', dm, dm1, hermi=1)
        self.assertEqual(ncalls[0], 1)
        self.assertAlmostEqual(abs(ref - out).max(), 0, 11)

        # The cache follows the grids
        grids.prune_by_density_(rho, 1e-5)
        ref = ni0.nr_rks(mol, grids, 'pbe', dm)
        out = ni.nr_rks(mol, grids, 'pbe', dm)
        self.assertEqual(ncalls[0], 2)
        self.assertAlmostEqual(abs(ref[2] - out[2]).max(), 0, 12)
        self.assertEqual(list(ni._ao_cache), [1])
        ni.clear_cache()
        self.assertEqual(ni.cache_size(), 0)

    def test_memory_limit(self):
        grids = dft.gen_grid.Grids(mol)
        grids.level = 2
        grids.build(with_non0tab=True)
        dm = dft.RKS(mol).get_init_guess()
        ref = numint.NumInt().nr_rks(mol, grids, 'pbe', dm)
        blksize = (grids.coords.shape[0] // 4) // numint.BLKSIZE * numint.BLKSIZE
        nblock_bytes = 4 * blksize * mol.nao * 8
        # two of the blocks fit into the cache
        nr_rks = numint.nr_rks
        class _NI(numint.CachedNumInt):
            def block_loop(self, mol, grids, nao=None, deriv=0, max_memory=2000,
                           non0tab=None, blksize_=None, buf=None):
                return numint.CachedNumInt.block_loop(
                    self, mol, grids, nao, deriv, max_memory, non0tab, blksize, buf)
        ni = _NI(max_cache_memory=2.5 * nblock_bytes / 1e6)
        for _ in range(2):
            out = nr_rks(ni, mol, grids, 'pbe', dm)
            self.assertAlmostEqual(abs(ref[2] - out[2]).max(), 0, 12)
            self.assertAlmostEqual(ref[1], out[1], 12)
        self.assertEqual(len(ni._ao_cache[1]['blocks']), 2)
        self.assertTrue(ni.cache_size() <= 2.5 * nblock_bytes / 1e6)

    def test_reset_geometry(self):
        mf = dft.RKS(mol, xc='pbe')
        mf.grids.level = 1
        mf._numint = numint.CachedNumInt()
        mf.kernel()
        mf.reset(mol1)
        e1 = mf.kernel()
        ref = dft.RKS(mol1, xc='pbe')
        ref.grids.level = 1
        self.assertAlmostEqual(e1, ref.kernel(), 9)
        entry = mf._numint._ao_cache[1]
        self.assertTrue(entry['mol'] is mol1)

        # The AO values of a different Mole object are not cached
        ni = mf._numint
        dm = mf.make_rdm1()
        size = ni.cache_size()
        out = ni.nr_rks(mol1.copy(), mf.grids, 'pbe', dm)
        self.assertEqual(ni.cache_size(), size)
        self.assertAlmostEqual(out[1], ni.nr_rks(mol1, mf.grids, 'pbe', dm)[1], 12)


if __name__ == "__main__":
    print("Test CachedNumInt")
    unittest.main()
