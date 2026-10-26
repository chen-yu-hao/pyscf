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
import math
import unittest
import numpy
from pyscf import gto


class KnownValues(unittest.TestCase):
    def test_scalar_contraction_reference(self):
        # Buffer dimensions follow gto/grid_ao_drv.h.
        blksize, nprimax = 56, 40
        rng = numpy.random.default_rng(31)
        coords = rng.normal(size=(3, blksize)) * 3
        for nprim, nctr in ((1, 1), (3, 1), (8, 2), (40, 8)):
            alpha = numpy.geomspace(.005, 50, nprim)
            coeff = rng.normal(size=(nctr, nprim))
            for ngrids in (1, 39, blksize):
                with self.subTest(nprim=nprim, nctr=nctr, ngrids=ngrids):
                    out = numpy.full((2*nprimax, blksize), numpy.nan)
                    ref = out.copy()
                    ref[:nctr] = 0
                    ref[nprimax:nprimax+nctr] = 0
                    fac = .28209479177387814
                    for j in range(nprim):
                        for i in range(ngrids):
                            x, y, z = map(float, coords[:, i])
                            rr = x*x + y*y + z*z
                            exponent = float(alpha[j])
                            value = math.exp(-exponent*rr) * fac
                            for k in range(nctr):
                                coefficient = float(coeff[k, j])
                                ref[k, i] += value * coefficient
                                ref[nprimax+k, i] += value * (-2.*exponent*coefficient)
                    gto.moleintor.libcgto.GTOcontract_exp1(
                        out.ctypes.data_as(ctypes.c_void_p),
                        coords.ctypes.data_as(ctypes.c_void_p),
                        alpha.ctypes.data_as(ctypes.c_void_p),
                        coeff.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(0), ctypes.c_int(nprim), ctypes.c_int(nctr),
                        ctypes.c_size_t(ngrids), ctypes.c_double(fac))
                    numpy.testing.assert_array_equal(out, ref)

    def test_ao_derivatives(self):
        mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0',
                    basis='aug-cc-pvtz', verbose=0)
        coords = numpy.random.default_rng(32).normal(size=(113, 3))
        for representation in ('sph', 'cart', 'spinor'):
            with self.subTest(representation=representation):
                ao1 = mol.eval_gto('GTOval_' + representation + '_deriv1', coords)
                ao2 = mol.eval_gto('GTOval_' + representation + '_deriv2', coords)
                ref = ao2[:, :4] if representation == 'spinor' else ao2[:4]
                numpy.testing.assert_allclose(ao1, ref, atol=2e-12, rtol=2e-12)


if __name__ == '__main__':
    unittest.main()
