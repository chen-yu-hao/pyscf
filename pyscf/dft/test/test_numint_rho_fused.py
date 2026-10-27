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

from types import SimpleNamespace
import unittest
import numpy
from pyscf import gto, lib
from pyscf.dft import numint


def legacy_mgga(mol, ao, dm, screen, pairs, hermi):
    nbins = numint.NBINS * 2 - int(numint.NBINS * numpy.log(1e-13) / numpy.log(1e-15))
    loc = mol.ao_loc_nr()
    c0 = numint._dot_ao_dm_sparse(ao[0], dm, nbins, screen, pairs, loc)
    rho = numpy.zeros((5, ao.shape[1]))
    rho[0] = numint._contract_rho_sparse(ao[0], c0, screen, loc)
    for c in range(1, 4):
        ci = numint._dot_ao_dm_sparse(ao[c], dm, nbins, screen, pairs, loc)
        rho[4] += numint._contract_rho_sparse(ao[c], ci, screen, loc)
        rho[c] = numint._contract_rho_sparse(ao[c], c0, screen, loc)
        if hermi:
            rho[c] *= 2
        else:
            rho[c] += numint._contract_rho_sparse(ao[0], ci, screen, loc)
    rho[4] *= .5
    return rho


class KnownValues(unittest.TestCase):
    def test_lda_gga_screened(self):
        for ngrids in (8, 56, 120, 224):
            mol, ao, dm, screen, pairs = self._inputs(ngrids)
            loc = mol.ao_loc_nr()
            nbins = numint.NBINS * 2 - int(numint.NBINS * numpy.log(1e-13) / numpy.log(1e-15))
            for hermi in (0, 1):
                matrix = dm if hermi == 0 else dm + dm.T
                c0 = numint._dot_ao_dm_sparse(ao[0], matrix, nbins, screen, pairs, loc)
                ref = [numint._contract_rho_sparse(ao[c], c0, screen, loc)
                       for c in range(4)]
                if hermi:
                    ref[1:] = [x * 2 for x in ref[1:]]
                else:
                    ct = numint._dot_ao_dm_sparse(ao[0], lib.transpose(matrix), nbins, screen, pairs, loc)
                    ref[1:] = [ref[c] + numint._contract_rho_sparse(ao[c], ct, screen, loc)
                               for c in range(1, 4)]
                for xctype in ('LDA', 'GGA'):
                    for threads in (1, 4):
                        with self.subTest(ngrids=ngrids, hermi=hermi, xctype=xctype, threads=threads), lib.with_omp_threads(threads):
                            actual = numint.eval_rho1(mol, ao[0] if xctype == 'LDA' else ao,
                                                      matrix, screen, xctype, hermi,
                                                      cutoff=1e-13, ao_cutoff=1e-15,
                                                      pair_mask=pairs)
                            numpy.testing.assert_allclose(actual, ref[0] if xctype == 'LDA' else ref,
                                                          atol=2e-11, rtol=1e-12)

    def _inputs(self, ngrids, ao_loc=None):
        rng = numpy.random.default_rng(25)
        if ao_loc is None:
            ao_loc = numpy.asarray([0] + list(numpy.cumsum([1, 3, 5] * 15)),
                                    dtype=numpy.int32)
        mol = SimpleNamespace(ao_loc_nr=lambda: ao_loc)
        nbas = len(ao_loc) - 1
        nao = ao_loc[-1]
        ao = rng.normal(size=(8, nao, ngrids))[::2].transpose(0, 2, 1)
        dm = rng.normal(size=(nao, nao))
        screen = rng.choice([0, 40, 80, 120],
                            size=((ngrids+55)//56, nbas)).astype(numpy.uint8)
        # Deliberately nonsymmetric: both DM and pair-mask orientation matter.
        pairs = rng.integers(0, 2, size=(nbas, nbas), dtype=numpy.uint8)
        return mol, ao, dm, screen, pairs

    def test_screened_components(self):
        for ngrids in (8, 56, 120, 224):
            mol, ao, dm, screen, pairs = self._inputs(ngrids)
            for hermi in (0, 1):
                matrix = dm if hermi == 0 else dm + dm.T
                reference = legacy_mgga(mol, ao, matrix, screen, pairs, hermi)
                for threads in (1, 4):
                    with self.subTest(ngrids=ngrids, hermi=hermi, threads=threads), lib.with_omp_threads(threads):
                        actual = numint.eval_rho1(mol, ao, matrix, screen,
                                                  'MGGA', hermi, False,
                                                  cutoff=1e-13, ao_cutoff=1e-15,
                                                  pair_mask=pairs)
                        numpy.testing.assert_allclose(actual, reference,
                                                      atol=2e-11, rtol=1e-12)

    def test_large_shell_and_component_stride(self):
        mol, ao, dm, screen, pairs = self._inputs(120, numpy.array([0, 100, 103], dtype=numpy.int32))
        ao = ao[::-1]
        dm = numpy.asarray(dm, order='F')
        screen[:] = 120
        pairs[:] = 1
        reference = legacy_mgga(mol, ao, dm, screen, pairs, 0)
        actual = numint.eval_rho1(mol, ao, dm, screen, 'MGGA', 0, False,
                                  cutoff=1e-13, ao_cutoff=1e-15, pair_mask=pairs)
        numpy.testing.assert_allclose(actual, reference, atol=2e-11, rtol=1e-12)

    def test_physical_density(self):
        mol = gto.M(atom=[('H', (0, 0, 2*i)) for i in range(8)],
                    basis='cc-pvdz', verbose=0)
        rng = numpy.random.default_rng(26)
        coords = rng.normal(size=(120, 3)) * 3
        screen = numint.make_mask(mol, coords, cutoff=1e-15)
        ao = numint.eval_ao(mol, coords, deriv=1, non0tab=screen, cutoff=1e-15)
        coeff = numpy.linalg.qr(rng.normal(size=(mol.nao, mol.nao)))[0]
        dm = (coeff[:, :4] * 2).dot(coeff[:, :4].T)
        actual = numint.eval_rho1(mol, ao, dm, screen, 'MGGA', 1, False,
                                  cutoff=1e-13, ao_cutoff=1e-15)
        reference = numint.eval_rho(mol, ao, dm, xctype='MGGA', hermi=1, with_lapl=False)
        numpy.testing.assert_allclose(actual, reference, atol=2e-11, rtol=1e-11)

    def test_legacy_fallbacks(self):
        for ngrids, with_lapl in ((113, False), (120, True)):
            mol, ao, dm, screen, pairs = self._inputs(ngrids)
            reference = legacy_mgga(mol, ao, dm, screen, pairs, 0)
            actual = numint.eval_rho1(mol, ao, dm, screen, 'MGGA', 0, with_lapl,
                                      cutoff=1e-13, ao_cutoff=1e-15, pair_mask=pairs)
            if with_lapl:
                actual = actual[[0, 1, 2, 3, 5]]
            numpy.testing.assert_allclose(actual, reference, atol=2e-11, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
