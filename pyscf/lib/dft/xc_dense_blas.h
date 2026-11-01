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
#ifndef PYSCF_XC_DENSE_BLAS_H
#define PYSCF_XC_DENSE_BLAS_H

/* LDA/GGA: Q=.5*wrho*A + sum_c wgrad_c*dA_c, V=A^T Q+Q^T A.
 * Use the original AO leading dimension, packing Q only, with DSYR2K.
 * MGGA remains in the tiled driver; it must not use this routine.
 * The symmetric output uses the same lower-triangle convention as before.
 * Negative weights and negative derivative strides are valid. No new cutoff.
 * Return 0 without touching out when the workspace cannot be allocated.
 */
static int VXCdense_blas(double *out, const double *ao, const double *wv,
                         int nao, int ngrids, int ncomp, ptrdiff_t ao_stride,
                         size_t max_bytes)
{
        if (nao <= 0 || ngrids <= 0) return 1;
        if (ncomp != 1 && ncomp != 2) return 0;
        const int tile = 896;
        const int nblock = (ngrids-1)/tile + 1;
        const size_t n = (size_t)nao;
        if (n > SIZE_MAX/n || n*n > SIZE_MAX/sizeof(double)) return 0;
        const size_t matrix_size = n*n;
        if (n > (SIZE_MAX/sizeof(double)-matrix_size)/tile) return 0;
        const size_t thread_size = matrix_size + tile*n;
        const size_t thread_bytes = thread_size*sizeof(double);
        if (thread_bytes > max_bytes) return 0;
        int nthreads = 1;
#ifdef _OPENMP
        nthreads = omp_get_max_threads();
#endif
        nthreads = MIN(nthreads, nblock);
        nthreads = MIN((size_t)nthreads, max_bytes/thread_bytes);
        double *scratch = malloc(thread_bytes*nthreads);
        if (scratch == NULL) return 0;
#pragma omp parallel num_threads(nthreads)
{
        const char T = 'T', U = 'U';
        const double one = 1.;
        const int tid = omp_get_thread_num(), team = omp_get_num_threads();
        double *partial = scratch + thread_size*tid;
        double *right = partial + matrix_size;
        for (size_t ij = 0; ij < matrix_size; ij++) partial[ij] = 0.;
        const int begin = (size_t)nblock*tid/team;
        const int end = (size_t)nblock*(tid+1)/team;
        for (int b = begin; b < end; b++) {
                const int start = b*tile;
                const int ng = MIN(tile, ngrids-start);
                for (int i = 0; i < nao; i++) {
                        const double *a0 = ao + (size_t)i*ngrids + start;
                        double *q = right + (size_t)i*ng;
                        if (ncomp == 1) {
#pragma GCC ivdep
                                for (int g = 0; g < ng; g++) {
                                        q[g] = .5*wv[start+g]*a0[g];
                                }
                        } else {
                                const double *ax = a0 + ao_stride;
                                const double *ay = ax + ao_stride;
                                const double *az = ay + ao_stride;
#pragma GCC ivdep
                                for (int g = 0; g < ng; g++) {
                                        q[g] = .5*wv[start+g]*a0[g]
                                            + wv[(size_t)ngrids+start+g]*ax[g]
                                            + wv[(size_t)2*ngrids+start+g]*ay[g]
                                            + wv[(size_t)3*ngrids+start+g]*az[g];
                                }
                        }
                }
                dsyr2k_(&U, &T, &nao, &ng, &one, ao+start, &ngrids,
                        right, &ng, &one, partial, &nao);
        }
#pragma omp barrier
#pragma omp for schedule(static)
        for (size_t ij = 0; ij < matrix_size; ij++) {
                const size_t i = ij/nao, j = ij%nao;
                const size_t src = i < j ? j*nao+i : ij;
                double sum = 0.;
                for (int t = 0; t < team; t++) sum += scratch[thread_size*t+src];
                out[ij] += sum;
        }
}
        free(scratch);
        return 1;
}
#endif
