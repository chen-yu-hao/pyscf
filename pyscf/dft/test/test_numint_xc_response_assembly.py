#!/usr/bin/env python
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
from pyscf import dft, gto, lib
from pyscf.dft import numint


def reference_vmat(ao, wv, nbins, screen_index, pair_mask, ao_loc, out=None,
                   xctype='MGGA', hermi=1):
    ngrids, nao = ao.shape[-2:]
    if out is None:
        out = numpy.zeros((nao, nao))
    shells = numpy.repeat(numpy.arange(len(ao_loc)-1), numpy.diff(ao_loc))
    sparse = screen_index is not None and pair_mask is not None and ngrids % 8 == 0
    for start in range(0, ngrids, numint.BLKSIZE):
        stop = min(start+numint.BLKSIZE, ngrids)
        phi = ao[start:stop] if xctype == 'LDA' else ao[0, start:stop]
        weights = wv.ravel()[start:stop] if xctype == 'LDA' else wv[0, start:stop]
        value = numpy.einsum('gi,g,gj->ij', phi, weights, phi)
        if xctype != 'LDA':
            for c in range(1, 4):
                grad = ao[c, start:stop]
                cross = numpy.einsum('gi,g,gj->ij', phi, wv[c, start:stop], grad)
                value += cross + cross.T
                if xctype == 'MGGA':
                    value += .5*numpy.einsum('gi,g,gj->ij', grad, wv[4, start:stop], grad)
        if sparse:
            bins = screen_index[start//numint.BLKSIZE].astype(int)
            allowed = ((bins[:, None] > 0) & (bins[None, :] > 0)
                       & (bins[:, None]+bins[None, :] >= nbins) & (pair_mask != 0))
            value *= allowed[shells[:, None], shells[None, :]]
        out += value
    return out


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz', verbose=0)
        cls.grids = dft.Grids(cls.mol)
        cls.grids.atom_grid = (10, 14)
        cls.grids.prune = None
        cls.grids.build(with_non0tab=True)
        cls.dm = dft.RKS(cls.mol).get_init_guess()

    def test_response_against_independent_matrix_formula(self):
        rng = numpy.random.default_rng(31)
        ni = numint.NumInt()
        for spin in (0, 1):
            dm0 = numpy.asarray((self.dm*.6, self.dm*.4)) if spin else self.dm
            method = ni.nr_uks_fxc if spin else ni.nr_rks_fxc
            for xc in ('LDA,VWN', 'PBE', 'SCAN'):
                fxc = ni.cache_xc_kernel1(self.mol, self.grids, xc, dm0, spin=spin)[2]
                frozen = fxc.copy()
                for hermi in (0, 1):
                    dms = rng.normal(size=(2, self.mol.nao, self.mol.nao)) * .01
                    if hermi:
                        dms += dms.transpose(0, 2, 1)
                    if spin:
                        dms = numpy.asarray((dms*.6, dms[::-1]*.4))
                    with self.subTest(spin=spin, xc=xc, hermi=hermi), \
                            lib.temporary_env(numint, SWITCH_SIZE=0), lib.with_omp_threads(2):
                        actual = method(self.mol, self.grids, xc, dm0, dms,
                                        hermi=hermi, fxc=fxc, max_memory=.15)
                        with mock.patch.object(numint, '_xc_vmat', reference_vmat):
                            reference = method(self.mol, self.grids, xc, dm0, dms,
                                               hermi=hermi, fxc=fxc, max_memory=.15)
                    numpy.testing.assert_allclose(actual, reference, atol=3e-11, rtol=2e-11)
                    numpy.testing.assert_array_equal(fxc, frozen)

    def test_lda_hermi2_retains_legacy(self):
        ni = numint.NumInt()
        original = numint._dot_ao_ao_sparse
        seen = []

        def legacy_call(*args, **kwargs):
            seen.append(args[7])
            return original(*args, **kwargs)

        for spin in (0, 1):
            dm0 = numpy.asarray((self.dm*.6, self.dm*.4)) if spin else self.dm
            method = ni.nr_uks_fxc if spin else ni.nr_rks_fxc
            fxc = ni.cache_xc_kernel1(self.mol, self.grids, 'LDA,VWN', dm0, spin=spin)[2]
            seen.clear()
            with lib.temporary_env(numint, SWITCH_SIZE=0), lib.with_omp_threads(2), \
                    mock.patch.object(numint, '_xc_vmat', side_effect=AssertionError('legacy required')), \
                    mock.patch.object(numint, '_dot_ao_ao_sparse', legacy_call):
                method(self.mol, self.grids, 'LDA,VWN', dm0, dm0, hermi=2, fxc=fxc)
            self.assertTrue(seen)
            self.assertTrue(all(hermi == 2 for hermi in seen))


if __name__ == '__main__':
    unittest.main()
