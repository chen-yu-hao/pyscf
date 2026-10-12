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

"""UCCSDT integral transformation values, temporary lifetimes, and fallback."""

import copy
import unittest
import weakref

import numpy as np

from pyscf import ao2mo, cc, gto, lib, scf
from pyscf.cc import uccsdt


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='Be 0 0 0; H 0 0 1.342', basis='sto-3g',
                        spin=1, verbose=0)
        with lib.with_omp_threads(1):
            cls.mf = scf.UHF(cls.mol).run(conv_tol=1e-12)
        assert cls.mf.converged

    def test_frozen_unequal_spin_integrals(self):
        # Freeze the Be core in each spin and one alpha virtual orbital.
        mycc = cc.UCCSDT(self.mf, frozen=[[0, 5], [0]])
        cached = mycc.ao2mo()

        def transform(coefficients):
            if isinstance(coefficients, np.ndarray):
                coefficients = (coefficients,) * 4
            return ao2mo.general(self.mf._eri, coefficients, compact=False)

        custom = uccsdt._make_eris_incore_ucc(mycc, ao2mofn=transform)
        moa, mob = cached.mo_coeff
        self.assertNotEqual(moa.shape[1], mob.shape[1])
        self.assertNotEqual(*mycc.nocc)
        # Full AO integrals give an independent reference for both the packed
        # cached-AO and custom-callback transformation paths.
        ao_eri = self.mol.intor('int2e', aosym='s1')
        for name, coefficients in (('pppp', (moa,) * 4),
                                   ('PPPP', (mob,) * 4),
                                   ('pPpP', (moa, moa, mob, mob))):
            shape = tuple(c.shape[1] for c in coefficients)
            expected = ao2mo.general(ao_eri, coefficients, compact=False)
            expected = expected.reshape(shape).transpose(0, 2, 1, 3)
            for eris in (cached, custom):
                with self.subTest(block=name, custom=eris is custom):
                    actual = getattr(eris, name)
                    self.assertTrue(actual.flags.c_contiguous)
                    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_transform_temporaries_released_between_spin_blocks(self):
        mycc = cc.UCCSDT(self.mf, frozen=[[0, 5], [0]])
        transformed = []

        def transform(coefficients):
            # At most the final physicists' blocks should survive the next
            # transformation, not any previous chemists' scratch tensor.
            self.assertTrue(all(reference() is None for reference in transformed))
            if isinstance(coefficients, np.ndarray):
                coefficients = (coefficients,) * 4
            array = np.array(ao2mo.general(self.mf._eri, coefficients, compact=False),
                             copy=True)
            transformed.append(weakref.ref(array))
            return array

        eris = uccsdt._make_eris_incore_ucc(mycc, ao2mofn=transform)
        self.assertEqual(len(transformed), 3)
        self.assertTrue(all(reference() is None for reference in transformed))
        self.assertGreater(np.linalg.norm(eris.pPpP), 0.)

    def test_missing_ao_integrals_rejected_for_multiple_electrons(self):
        mf = copy.copy(self.mf)
        mf._eri = None
        mycc = cc.UCCSDT(mf)
        with self.assertRaisesRegex(NotImplementedError, 'AO integrals|AO2MO'):
            mycc.ao2mo()

    def test_single_electron_without_ao_integrals(self):
        mol = gto.M(atom='H 0 0 0', basis='6-31g', spin=1, verbose=0)
        mf = scf.UHF(mol).run(conv_tol=1e-12)
        self.assertTrue(mf.converged)
        mf._eri = None
        mycc = cc.UCCSDT(mf)
        eris = mycc.ao2mo()
        for name in ('pppp', 'PPPP', 'pPpP'):
            np.testing.assert_array_equal(getattr(eris, name), 0.)
        mycc.kernel(eris=eris)
        self.assertTrue(mycc.converged)
        self.assertAlmostEqual(mycc.e_corr, 0., 12)


if __name__ == '__main__':
    unittest.main()
