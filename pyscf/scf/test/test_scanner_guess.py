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

'''Initial guess of SCF scanners for a new geometry.'''

import unittest
import numpy
from pyscf import gto, scf, dft
from pyscf.scf import hf


def setUpModule():
    global mol, moved
    mol = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587',
                basis='def2-svp', verbose=0)
    moved = mol.set_geom_(mol.atom_coords() + numpy.array([[0, 0, 0],
                                                           [0, .05, .03],
                                                           [.02, -.04, 0]]),
                          unit='Bohr', inplace=False)

def tearDownModule():
    global mol, moved
    del mol, moved


class KnownValues(unittest.TestCase):
    def test_projected_density_rhf(self):
        mf = scf.RHF(mol).run()
        dm = hf._guess_from_last_geometry(mf, moved)
        s = moved.intor('int1e_ovlp')
        self.assertAlmostEqual(numpy.einsum('ij,ji', dm, s), mol.nelectron, 10)
        self.assertAlmostEqual(abs(dm.dot(s).dot(dm) - 2 * dm).max(), 0, 10)
        # The unprojected density is off in the new metric
        self.assertGreater(abs(numpy.einsum('ij,ji', mf.make_rdm1(), s) - mol.nelectron), 1e-4)

    def test_projected_density_open_shell(self):
        cation = mol.copy()
        cation.charge, cation.spin = 1, 1
        cation.build()
        moved1 = cation.set_geom_(moved.atom_coords(), unit='Bohr', inplace=False)
        s = moved1.intor('int1e_ovlp')
        for cls in (scf.UHF, scf.ROHF):
            mf = cls(cation).run()
            dm = hf._guess_from_last_geometry(mf, moved1)
            self.assertEqual(dm.shape, (2, cation.nao, cation.nao))
            for d, n in zip(dm, cation.nelec):
                self.assertAlmostEqual(numpy.einsum('ij,ji', d, s), n, 10)
                self.assertAlmostEqual(abs(d.dot(s).dot(d) - d).max(), 0, 10)
        mf = scf.GHF(cation).run()
        dm = hf._guess_from_last_geometry(mf, moved1)
        s2 = mf.get_ovlp(moved1)
        self.assertAlmostEqual(numpy.einsum('ij,ji', dm, s2).real, cation.nelectron, 10)

    def test_scanner_converges_to_same_solution(self):
        for mf in (dft.RKS(mol, xc='PBE'), scf.UHF(mol)):
            scanner = mf.as_scanner()
            scanner(mol)
            e = scanner(moved)
            ref = mf.__class__(moved)
            if hasattr(mf, 'xc'):
                ref.xc = mf.xc
            self.assertAlmostEqual(e, ref.kernel(), 8)

    def test_periodic_keeps_previous_density(self):
        from pyscf.pbc import gto as pgto, scf as pscf
        cell = pgto.M(a=numpy.eye(3) * 4, atom='He 0 0 0', basis='gth-szv',
                      pseudo='gth-pade', verbose=0)
        mf = pscf.RHF(cell).run()
        dm = hf._guess_from_last_geometry(mf, cell)
        self.assertAlmostEqual(abs(dm - mf.make_rdm1()).max(), 0, 14)


if __name__ == "__main__":
    print("Tests for the scanner initial guess")
    unittest.main()
