/* Copyright 2014-2026 The PySCF Developers. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
 *
 * Per-AO contraction for the XC part of molecular nuclear gradients.
 *
 * The XC gradient needs only f[x,i] = sum_j V^x_ij D_ij, never the
 * 3 x nao x nao matrix V^x itself.  With P = phi D, F = aow D and, for
 * meta-GGA, Q_j = (d_j phi) D precomputed by the caller,
 *
 *   G   = F + wv0 P                             (F = 0 for LDA)
 *   R_j = wv_j P + wtau Q_j                     (GGA / meta-GGA only)
 *   f[x,i] += sum_g  d_x phi_i G_i + sum_j d_x d_j phi_i R_j,i
 *
 * All (grid, AO) arrays are Fortran ordered, element (g,i) at [i*ngrids+g],
 * like the output of eval_ao; AO derivative components are ao_stride apart:
 * phi, d_x, d_y, d_z, then xx, xy, xz, yy, yz, zz when deriv >= 2.
 * Each thread owns disjoint AO indices, so no reduction is required.
 * A non-NULL screen_index (nblocks x nbas, BLKSIZE-point blocks, as in
 * grids.non0tab) skips shells whose AO values were not evaluated.
 */

#include <stdint.h>
#include <stddef.h>
#include "config.h"
#include "gto/grid_ao_drv.h"
#include "np_helper/np_helper.h"

/* d_x d_j phi among ao[4:10] = xx, xy, xz, yy, yz, zz */
static const int D2[3][3] = {{4, 5, 6}, {5, 7, 8}, {6, 8, 9}};

void VXCxc_grad_contract(double *f, double *ao, ptrdiff_t ao_stride,
                         double *P, double *F, double *Q, double *wv,
                         int xctype, int nao, int ngrids,
                         uint8_t *screen_index, int *ao_shell, int nbas)
{
        /* xctype: 0 = LDA, 1 = GGA, 2 = meta-GGA (tau without laplacian) */
        const double *w0 = wv;
        const double *w1 = wv + ngrids;
        const double *w2 = wv + ngrids * 2;
        const double *w3 = wv + ngrids * 3;
        const double *wt = wv + ngrids * 4;
        const int nblk = (ngrids + BLKSIZE - 1) / BLKSIZE;
        const size_t nn = (size_t)nao * ngrids;
#pragma omp parallel for schedule(dynamic, 4)
        for (int i = 0; i < nao; i++) {
                const size_t off = (size_t)i * ngrids;
                const double *p = P + off;
                const double *a1 = ao + ao_stride + off;
                const double *a2 = ao + ao_stride * 2 + off;
                const double *a3 = ao + ao_stride * 3 + off;
                double ax = 0, ay = 0, az = 0;
                int b, g, g0, g1;
                if (xctype == 0) {
                        for (b = 0; b < nblk; b++) {
                                if (screen_index != NULL &&
                                    !screen_index[(size_t)b*nbas+ao_shell[i]]) {
                                        continue;
                                }
                                g0 = b * BLKSIZE;
                                g1 = MIN(g0 + BLKSIZE, ngrids);
#pragma omp simd reduction(+:ax,ay,az)
                                for (g = g0; g < g1; g++) {
                                        const double G = w0[g] * p[g];
                                        ax += a1[g] * G;
                                        ay += a2[g] * G;
                                        az += a3[g] * G;
                                }
                        }
                } else {
                        const double *fF = F + off;
                        const double *dxx = ao + ao_stride * D2[0][0] + off;
                        const double *dxy = ao + ao_stride * D2[0][1] + off;
                        const double *dxz = ao + ao_stride * D2[0][2] + off;
                        const double *dyy = ao + ao_stride * D2[1][1] + off;
                        const double *dyz = ao + ao_stride * D2[1][2] + off;
                        const double *dzz = ao + ao_stride * D2[2][2] + off;
                        const int mgga = xctype == 2;
                        const double *q1 = mgga ? Q + off : NULL;
                        const double *q2 = mgga ? Q + nn + off : NULL;
                        const double *q3 = mgga ? Q + nn * 2 + off : NULL;
                        for (b = 0; b < nblk; b++) {
                                if (screen_index != NULL &&
                                    !screen_index[(size_t)b*nbas+ao_shell[i]]) {
                                        continue;
                                }
                                g0 = b * BLKSIZE;
                                g1 = MIN(g0 + BLKSIZE, ngrids);
                                if (mgga) {
#pragma omp simd reduction(+:ax,ay,az)
                                for (g = g0; g < g1; g++) {
                                        const double pg = p[g];
                                        const double G = fF[g] + w0[g] * pg;
                                        const double r1 = w1[g] * pg + wt[g] * q1[g];
                                        const double r2 = w2[g] * pg + wt[g] * q2[g];
                                        const double r3 = w3[g] * pg + wt[g] * q3[g];
                                        ax += a1[g]*G + dxx[g]*r1 + dxy[g]*r2 + dxz[g]*r3;
                                        ay += a2[g]*G + dxy[g]*r1 + dyy[g]*r2 + dyz[g]*r3;
                                        az += a3[g]*G + dxz[g]*r1 + dyz[g]*r2 + dzz[g]*r3;
                                }
                                } else {
#pragma omp simd reduction(+:ax,ay,az)
                                for (g = g0; g < g1; g++) {
                                        const double pg = p[g];
                                        const double G = fF[g] + w0[g] * pg;
                                        const double r1 = w1[g] * pg;
                                        const double r2 = w2[g] * pg;
                                        const double r3 = w3[g] * pg;
                                        ax += a1[g]*G + dxx[g]*r1 + dxy[g]*r2 + dxz[g]*r3;
                                        ay += a2[g]*G + dxy[g]*r1 + dyy[g]*r2 + dyz[g]*r3;
                                        az += a3[g]*G + dxz[g]*r1 + dyz[g]*r2 + dzz[g]*r3;
                                }
                                }
                        }
                }
                f[i] += ax;
                f[nao + i] += ay;
                f[nao * 2 + i] += az;
        }
}
