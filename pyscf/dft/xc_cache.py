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

'''Private invariant-derivative cache for repeated semilocal XC responses.'''

import numpy
from pyscf.dft import xc_deriv


class _CompactXCKernel:
    '''Apply the density Hessian without expanding its Cartesian components.

    ``values`` uses the canonical order returned by LibXC eval_xc1: energy,
    first invariant derivatives, and the upper triangle of their Hessian.
    Scaling and spin projection share these arrays and never modify them.
    '''

    def __init__(self, rho, values, xctype, spin, scale=1., projection=None):
        self.rho = numpy.asarray(rho, order='C')
        self.values = values
        self.xctype = xctype
        self.spin = spin
        self.scale = scale
        self.projection = projection

    def scaled(self, factor):
        return _CompactXCKernel(self.rho, self.values, self.xctype, self.spin,
                                self.scale * factor, self.projection)

    def projected(self, singlet):
        assert self.spin == 1
        return _CompactXCKernel(self.rho, self.values, self.xctype, self.spin,
                                self.scale, 1 if singlet else -1)

    def apply(self, rho1, p0, p1):
        '''Return raw, unweighted Cartesian XC potential derivatives.'''
        rho1 = numpy.asarray(rho1)
        if self.projection is not None:
            rho1 = numpy.asarray((rho1, self.projection * rho1))
        nvar, xlen = xc_deriv._XC_NVAR[self.xctype, self.spin]
        ng = p1 - p0
        rho1 = rho1.reshape(self.spin+1, nvar, ng)
        rho0 = self.rho[...,p0:p1].reshape(self.spin+1, nvar, ng)
        values = self.values[:,p0:p1]
        delta = numpy.empty((xlen, ng), dtype=numpy.result_type(rho1, values))
        if self.spin == 0:
            delta[0] = rho1[0,0]
            if nvar > 1:
                delta[1] = 2*numpy.einsum('ig,ig->g', rho0[0,1:4], rho1[0,1:4])
            if nvar == 5:
                delta[2] = rho1[0,4]
        else:
            delta[:2] = rho1[:,0]
            if nvar > 1:
                ga, gb = rho0[:,1:4]
                da, db = rho1[:,1:4]
                delta[2] = 2*numpy.einsum('ig,ig->g', ga, da)
                delta[3] = (numpy.einsum('ig,ig->g', ga, db) +
                            numpy.einsum('ig,ig->g', gb, da))
                delta[4] = 2*numpy.einsum('ig,ig->g', gb, db)
            if nvar == 5:
                delta[5:] = rho1[:,4]

        # Contract the packed symmetric invariant Hessian with J * rho1.
        q = numpy.zeros_like(delta, dtype=numpy.result_type(delta, values))
        k = 1 + xlen
        for i in range(xlen):
            for j in range(i, xlen):
                q[i] += values[k] * delta[j]
                if i != j:
                    q[j] += values[k] * delta[i]
                k += 1

        result = numpy.empty((self.spin+1, nvar, ng), dtype=q.dtype)
        if self.spin == 0:
            result[0,0] = q[0]
            if nvar > 1:
                # The second term is the Hessian of sigma = grad(rho)^2.
                result[0,1:4] = 2*rho0[0,1:4]*q[1] + 2*values[2]*rho1[0,1:4]
            if nvar == 5:
                result[0,4] = q[2]
        else:
            result[:,0] = q[:2]
            if nvar > 1:
                result[0,1:4] = (2*ga*q[2] + gb*q[3] +
                                  2*values[3]*da + values[4]*db)
                result[1,1:4] = (2*gb*q[4] + ga*q[3] +
                                  2*values[5]*db + values[4]*da)
            if nvar == 5:
                result[:,4] = q[5:]
        if self.spin == 0 or self.projection is not None:
            result = result[0]
        if self.scale != 1:
            result *= self.scale
        return result
