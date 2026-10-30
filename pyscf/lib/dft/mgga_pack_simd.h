/* Copyright 2026 The PySCF Developers. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/* Pack MGGA channels without a functional-type branch in the grid loop. */
static void VXCpack_mgga_simd(double *restrict left, double *restrict right,
                            const double *ao, const double *wv,
                            const int *active_ao, int nactive, int ng,
                            int ngrids, int offset, ptrdiff_t stride)
{
    const double *w0=wv+offset, *wx=w0+ngrids, *wy=wx+ngrids;
    const double *wz=wy+ngrids, *wt=wz+ngrids;
    const size_t k=(size_t)4*ng;
    for(int i=0;i<nactive;i++) {
        const double *a=ao+(size_t)active_ao[i]*ngrids+offset;
        const double *x=a+stride,*y=x+stride,*z=y+stride;
        double *restrict l0=left+i*k,*restrict lx=l0+ng;
        double *restrict ly=lx+ng,*restrict lz=ly+ng;
        double *restrict r0=right+i*k,*restrict rx=r0+ng;
        double *restrict ry=rx+ng,*restrict rz=ry+ng;
#pragma omp simd
        for(int g=0;g<ng;g++) {
            const double tau=.5*wt[g];
            l0[g]=a[g];lx[g]=x[g];ly[g]=y[g];lz[g]=z[g];
            r0[g]=w0[g]*a[g]+wx[g]*x[g]+wy[g]*y[g]+wz[g]*z[g];
            rx[g]=wx[g]*a[g]+tau*x[g];
            ry[g]=wy[g]*a[g]+tau*y[g];
            rz[g]=wz[g]*a[g]+tau*z[g];
        }
    }
}
