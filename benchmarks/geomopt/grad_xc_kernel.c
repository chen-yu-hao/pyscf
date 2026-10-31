/* Fused per-AO contraction for RKS XC nuclear gradients (GGA / MGGA).
 *
 * All 2D arrays are (ngrids x nao) in Fortran order, i.e. element (g, i) at
 * [i*ngrids + g], matching PySCF eval_ao output; ao components are separated
 * by ao_stride doubles.  ao[0..9] = phi, d_x, d_y, d_z, xx, xy, xz, yy, yz, zz.
 *
 *   G   = F + wv0 * P
 *   R_j = wv_j * P (+ wtau * Q_j for MGGA)
 *   f[x, i] += sum_g  d_x phi_i G_i  +  sum_j  d_x d_j phi_i R_j,i
 *
 * Threads own disjoint AO indices: no private matrices and no reduction.
 * Optional screen (nblk x nbas, 56-point blocks) skips negligible AO/blocks.
 */
#include <stddef.h>
#include <stdint.h>

#define BLK 56

static const int D2[3][3] = {{4, 5, 6}, {5, 7, 8}, {6, 8, 9}};

void grad_xc_contract(double *f, const double *ao, ptrdiff_t ao_stride,
                      const double *P, const double *F, const double *Q,
                      const double *wv, int nvar, int nao, int ngrids,
                      const uint8_t *screen, const int *ao_shell, int nbas)
{
        const int mgga = Q != NULL;
        const double *w0 = wv, *w1 = wv + ngrids, *w2 = wv + 2*ngrids,
                     *w3 = wv + 3*ngrids, *wt = mgga ? wv + 4*ngrids : NULL;
        const int nblk = (ngrids + BLK - 1) / BLK;
#pragma omp parallel for schedule(dynamic, 4)
        for (int i = 0; i < nao; i++) {
                const size_t off = (size_t)i * ngrids;
                const double *p = P + off, *fF = F + off;
                const double *d[10];
                for (int c = 0; c < 10; c++) d[c] = ao + c*ao_stride + off;
                const double *q0 = mgga ? Q + off : NULL;
                const double *q1 = mgga ? Q + (size_t)nao*ngrids + off : NULL;
                const double *q2 = mgga ? Q + 2*(size_t)nao*ngrids + off : NULL;
                double ax = 0, ay = 0, az = 0;
                for (int b = 0; b < nblk; b++) {
                        if (screen && !screen[(size_t)b*nbas + ao_shell[i]]) continue;
                        const int g0 = b*BLK;
                        const int g1 = g0 + BLK < ngrids ? g0 + BLK : ngrids;
#pragma omp simd reduction(+:ax,ay,az)
                        for (int g = g0; g < g1; g++) {
                                const double pg = p[g];
                                const double G = fF[g] + w0[g]*pg;
                                double r1 = w1[g]*pg, r2 = w2[g]*pg, r3 = w3[g]*pg;
                                if (mgga) {
                                        r1 += wt[g]*q0[g];
                                        r2 += wt[g]*q1[g];
                                        r3 += wt[g]*q2[g];
                                }
                                ax += d[1][g]*G + d[4][g]*r1 + d[5][g]*r2 + d[6][g]*r3;
                                ay += d[2][g]*G + d[5][g]*r1 + d[7][g]*r2 + d[8][g]*r3;
                                az += d[3][g]*G + d[6][g]*r1 + d[8][g]*r2 + d[9][g]*r3;
                        }
                }
                f[i] += ax;
                f[nao + i] += ay;
                f[2*nao + i] += az;
        }
}
