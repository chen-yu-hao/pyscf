/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
  
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
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
//#include <omp.h>
#include "config.h"
#include "np_helper/np_helper.h"
#include "vhf/fblas.h"

/*
 * a * v1 + b * v2.transpose(0,2,1,3)
 */
void CCmake_0213(double *out, double *v1, double *v2, int count, int m,
                 double a, double b)
{
#pragma omp parallel default(none) \
        shared(count, m, out, v1, v2, a, b)
{
        int i, j, k, l;
        size_t d2 = m * m;
        size_t d1 = m * m * m;
        double *pv1, *pv2, *pout;
#pragma omp for schedule (static)
        for (i = 0; i < count; i++) {
                for (j = 0; j < m; j++) {
                for (k = 0; k < m; k++) {
                        pout = out + d1*i + d2*j + m*k;
                        pv1  = v1  + d1*i + d2*j + m*k;
                        pv2  = v2  + d1*i + d2*k + m*j;
                        for (l = 0; l < m; l++) {
                                pout[l] = pv1[l] * a + pv2[l] * b;
                        }
        } } }
}
}

/*
 * out = v1 + v2.transpose(0,2,1)
 */
void CCsum021(double *out, double *v1, double *v2, int count, int m)
{
#pragma omp parallel default(none) \
        shared(count, m, out, v1, v2)
{
        int i, j, k, n;
        size_t mm = m * m;
        double *pout, *pv1, *pv2;
#pragma omp for schedule (static)
        for (i = 0; i < count; i++) {
                pout = out + mm * i;
                pv1  = v1  + mm * i;
                pv2  = v2  + mm * i;
                for (n = 0, j = 0; j < m; j++) {
                for (k = 0; k < m; k++, n++) {
                        pout[n] = pv1[n] + pv2[k*m+j];
                } }
        }
}
}

/*
 * g2 = a * v1 + b * v2.transpose(0,2,1)
 */
void CCmake_021(double *out, double *v1, double *v2, int count, int m,
                double a, double b)
{
        if (a == 1 && b == 1) {
                return CCsum021(out, v1, v2, count, m);
        }

#pragma omp parallel default(none) \
        shared(count, m, out, v1, v2, a, b)
{
        int i, j, k, n;
        size_t mm = m * m;
        double *pout, *pv1, *pv2;
#pragma omp for schedule (static)
        for (i = 0; i < count; i++) {
                pout = out + mm * i;
                pv1  = v1  + mm * i;
                pv2  = v2  + mm * i;
                for (n = 0, j = 0; j < m; j++) {
                for (k = 0; k < m; k++, n++) {
                        pout[n] = pv1[n] * a + pv2[k*m+j] * b;
                } }
        }
}
}

/*
 * if matrix B is symmetric for the contraction A_ij B_ij,
 * Tr(AB) ~ A_ii B_ii + (A_ij + A_ji) B_ij where i > j
 * This function extract the A_ii and the lower triangular part of A_ij + A_ji
 */
void CCprecontract(double *out, double *in, int count, int m, double diagfac)
{
#pragma omp parallel default(none) \
        shared(count, m, in, out, diagfac)
{
        int i, j, k, n;
        size_t mm = m * m;
        size_t m2 = m * (m+1) / 2;
        double *pout, *pin;
#pragma omp for schedule (static)
        for (i = 0; i < count; i++) {
                pout = out + m2 * i;
                pin  = in  + mm * i;
                for (n = 0, j = 0; j < m; j++) {
                        for (k = 0; k < j; k++, n++) {
                                pout[n] = pin[j*m+k] + pin[k*m+j];
                        }
                        pout[n] = pin[j*m+j] * diagfac;
                        n++;
                }
        }
}
}

/*
 * if i1 == j1:
 *     eri = unpack_tril(eri, axis=0)
 * unpack_tril(eri).reshape(i1-i0,j1-j0,nao,nao).transpose(0,2,1,3)
 */
void CCload_eri(double *out, double *eri, int *orbs_slice, int nao)
{
        int i0 = orbs_slice[0];
        int i1 = orbs_slice[1];
        int j0 = orbs_slice[2];
        int j1 = orbs_slice[3];
        size_t ni = i1 - i0;
        size_t nj = j1 - j0;
        size_t nn = nj * nao;
        size_t nao_pair = nao * (nao + 1) / 2;

#pragma omp parallel default(none) \
        shared(out, eri, i1, j1, ni, nj, nn, nao, nao_pair)
{
        int i, j, k, l, ij;
        double *pout;
        double *buf = malloc(sizeof(double) * nao*nao);
#pragma omp for schedule (static)
        for (ij = 0; ij < ni*nj; ij++) {
                i = ij / nj;
                j = ij % nj;
                NPdunpack_tril(nao, eri+ij*nao_pair, buf, 1);
                pout = out + (i*nn+j)*nao;
                for (k = 0; k < nao; k++) {
                for (l = 0; l < nao; l++) {
                        pout[k*nn+l] = buf[k*nao+l];
                } }
        }
        free(buf);
}
}

/* Write a contiguous (j,l) output row directly from pair-packed integrals. */
static void load_s4_row(double *dst, const double *packed, size_t i, size_t k,
                        int j0, int j1, size_t nb, size_t npair, size_t offset)
{
    for (size_t j = j0; j < (size_t)j1; ++j) {
        const size_t high = i > j ? i : j;
        const size_t low = i > j ? j : i;
        const double *src = packed + (high * (high + 1) / 2 + low - offset) * npair;
        memcpy(dst, src + k * (k + 1) / 2, (k + 1) * sizeof(double));
        size_t kl = (k + 1) * (k + 2) / 2 + k;
        for (size_t l = k + 1; l < nb; ++l) {
            dst[l] = src[kl];
            kl += l + 1;
        }
        dst += nb;
    }
}

/* packed begins at first-pair row i0*(i0+1)/2. The two pair dimensions
 * may differ for mixed-spin integrals. No intermediate unpacked array. */
void CCload_eri_s4(double *out, const double *packed, int i0, int i1,
                   int j0, int j1, int nvirb)
{
    const size_t ni = i1 - i0;
    const size_t nb = nvirb;
    const size_t nj = j1 - j0;
    const size_t npair = nb * (nb + 1) / 2;
    const size_t offset = (size_t)i0 * (i0 + 1) / 2;
#pragma omp parallel for schedule(static)
    for (size_t ik = 0; ik < ni * nb; ++ik) {
        load_s4_row(out + ik * nj * nb, packed, ik / nb + i0, ik % nb,
                    j0, j1, nb, npair, offset);
    }
}

/* Status 1: unsupported dimensions or insufficient caller-owned scratch.
 * On every nonzero return, output and all inputs remain untouched. */
int CCcontract_s4vvvv(double *out, const double *packed, const double *t2,
                      int nx, int na, int nb, int tile,
                      double *scratch, size_t capacity_bytes)
{
    if (nx < 0 || na < 0 || nb < 0 || tile <= 0) return 1;
    if (!nx || !na || !nb) return 0;
    if (tile > na) tile = na;
    if ((size_t)na > INT_MAX / (size_t)nb) return 1;
    const int nvir2 = na * nb;
    const size_t width = (size_t)tile * nb;
    if (width > SIZE_MAX / width || (size_t)nx > SIZE_MAX / nvir2) return 1;
    const size_t panel_size = width * width;
    const size_t result_size = (size_t)nx * nvir2;
    if (panel_size > SIZE_MAX - result_size) return 1;
    const size_t stride = panel_size + result_size;
    if (stride > SIZE_MAX / sizeof(double)) return 1;
    const size_t stride_bytes = stride * sizeof(double);
    const size_t npair = (size_t)nb * ((size_t)nb + 1) / 2;
    const size_t ntiles = ((size_t)na + tile - 1) / tile;
    const size_t ntasks = ntiles * (ntiles + 1) / 2;
    int status = 0;
    int threads = 0;

#pragma omp parallel shared(scratch, status, threads)
    {
#pragma omp single
        {
            threads = omp_get_num_threads();
            if ((size_t)threads > SIZE_MAX / stride_bytes) {
                status = 1;
            } else if (!scratch || (size_t)threads * stride_bytes > capacity_bytes) {
                status = 1;
            }
        }
        if (!status) {
            double *panel = scratch + (size_t)omp_get_thread_num() * stride;
            double *h = panel + panel_size;
            memset(h, 0, result_size * sizeof(double));
#pragma omp for schedule(static, 1)
            for (size_t task = 0; task < ntasks; ++task) {
                size_t it = (size_t)((sqrt(8.0 * task + 1) - 1) * 0.5);
                while ((it + 1) * (it + 2) / 2 <= task) ++it;
                while (it * (it + 1) / 2 > task) --it;
                const size_t jt = task - it * (it + 1) / 2;
                const int i0 = it * tile;
                const int j0 = jt * tile;
                const int ni = na - i0 < tile ? na - i0 : tile;
                const int nj = na - j0 < tile ? na - j0 : tile;
                const int m = nj * nb;
                const int kdim = ni * nb;

                for (int ii = 0; ii < ni; ++ii) {
                    for (int k = 0; k < nb; ++k) {
                        load_s4_row(panel + ((size_t)ii * nb + k) * m,
                                    packed, i0 + ii, k, j0, j0 + nj, nb,
                                    npair, 0);
                    }
                }

                const double one = 1;
                const char normal = 'N', transpose = 'T';
                /* Row-major H_j += X_i panel, expressed as Fortran GEMM. */
                dgemm_(&normal, &normal, &m, &nx, &kdim, &one,
                       panel, &m, t2 + (size_t)i0 * nb, &nvir2,
                       &one, h + (size_t)j0 * nb, &nvir2);
                if (i0 != j0) {
                    dgemm_(&transpose, &normal, &kdim, &nx, &m, &one,
                           panel, &m, t2 + (size_t)j0 * nb, &nvir2,
                           &one, h + (size_t)i0 * nb, &nvir2);
                }
            }
            /* omp-for barrier above completes every private accumulator. */
#pragma omp for schedule(static)
            for (size_t p = 0; p < result_size; ++p) {
                double sum = scratch[panel_size + p];
                for (int worker = 1; worker < threads; ++worker)
                    sum += scratch[(size_t)worker * stride + panel_size + p];
                out[p] = sum;
            }
        }
    }
    return status;
}

/*
 * eri put virtual orbital first
 * [ v         ]
 * [ v .       ]
 * [ v . .     ]
 * [ o . . .   ]
 * [ o . . . . ]
 */
void CCsd_sort_inplace(double *eri, int nocc, int nvir, int count)
{
#pragma omp parallel default(none) \
        shared(eri, nocc, nvir, count)
{
        int ic, i, j, ij;
        size_t nmo = nocc + nvir;
        size_t nmo_pair = nmo * (nmo+1) / 2;
        size_t nocc_pair = nocc * (nocc+1) /2;
        size_t nvir_pair = nvir * (nvir+1) /2;
        double *peri, *pout;
        double *buf = malloc(sizeof(double) * nocc*nvir);
#pragma omp for schedule (static)
        for (ic = 0; ic < count; ic++) {
                peri = eri + ic*nmo_pair + nvir_pair;
                for (i = 0; i < nocc; i++, peri+=nvir+i) {
                        for (j = 0; j < nvir; j++) {
                                buf[i*nvir+j] = peri[j];
                        }
                }
                pout = eri + ic*nmo_pair + nvir_pair;
                peri = eri + ic*nmo_pair + nvir_pair + nvir;
                for (ij = 0, i = 0; i < nocc; i++, peri+=nvir+i) {
                        for (j = 0; j <= i; j++, ij++) {
                                pout[ij] = peri[j];
                        }
                }
                pout = eri + ic*nmo_pair + nvir_pair + nocc_pair;
                NPdcopy(pout, buf, nocc*nvir);
        }
        free(buf);
}
}

