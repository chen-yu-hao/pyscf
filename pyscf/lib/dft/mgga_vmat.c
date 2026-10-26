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

#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include "config.h"
#include "np_helper/np_helper.h"
#include "vhf/fblas.h"

#define GRID_BLOCK 56
#define GRID_TILE 224
#define AO_TILE 64

static int grid_cut(const uint64_t *prefix, int nblock, int thread, int nthread)
{
        if (thread == 0) return 0;
        if (thread == nthread) return nblock;
        uint64_t total = prefix[nblock];
        uint64_t target = (total/nthread)*thread + (total%nthread)*thread/nthread;
        int lo = 0, hi = nblock;
        while (lo < hi) {
                int mid = lo + (hi-lo)/2;
                if (prefix[mid] < target) lo = mid+1;
                else hi = mid;
        }
        return lo;
}

/* Add a symmetric XC potential for raw (unscaled) grid weights.
 * ncomp is 1 (LDA), 2 (GGA), or 4 (MGGA).
 * Return zero before modifying out if the caller should use its old path. */
int VXCxc_vmat(double *out, double *ao, double *wv,
                 int nao, int ngrids, int nbas, int nbins, int ncomp,
                 uint8_t *screen_index, uint8_t *pair_mask, int *ao_loc,
                 ptrdiff_t ao_stride)
{
        const int sparse = screen_index != NULL;
        if (sparse) {
                for (int i = 0; i < nbas; i++) {
                for (int j = 0; j < i; j++) {
                        if (pair_mask[i*nbas+j] != pair_mask[j*nbas+i]) {
                                return 0;
                        }
                } }
        }
        const int grid_tile = sparse ? GRID_BLOCK : GRID_TILE;
        const int nblock = (ngrids + grid_tile - 1) / grid_tile;
        const size_t pack_size = (size_t)ncomp * grid_tile * nao;
        const size_t matrix_size = (size_t)nao * nao;
        const size_t thread_size = 2 * pack_size + matrix_size + AO_TILE*AO_TILE;
        const size_t thread_bytes = thread_size*sizeof(double) + (size_t)2*nao*sizeof(int);
        int nthreads = 1;
#ifdef _OPENMP
        nthreads = omp_get_max_threads();
#endif
        const int with_plan = sparse && nthreads > 1;
        const size_t plan_bytes = with_plan ? ((size_t)nblock+1)*sizeof(uint64_t) + nblock : 0;
        if (thread_bytes * nthreads + plan_bytes > 128*1024*1024) {
                return 0;
        }
        double *scratch = malloc(thread_size * nthreads * sizeof(double) + plan_bytes);
        if (scratch == NULL) {
                return 0;
        }
        uint64_t *prefix = with_plan ? (uint64_t *)(scratch + thread_size*nthreads) : NULL;
        uint8_t *max_bins = with_plan ? (uint8_t *)(prefix+nblock+1) : NULL;
        if (with_plan) {
                prefix[0] = 0;
                for (int b = 0; b < nblock; b++) {
                        uint8_t *screen = screen_index + (size_t)b*nbas;
                        int maximum = 0;
                        for (int s = 0; s < nbas; s++) maximum = MAX(maximum, screen[s]);
                        max_bins[b] = maximum;
                        uint64_t nactive = 0;
                        for (int s = 0; s < nbas; s++) {
                                if (screen[s] && screen[s]+maximum >= nbins) {
                                        nactive += ao_loc[s+1] - ao_loc[s];
                                }
                        }
                        uint64_t remainder = nactive % AO_TILE;
                        uint64_t diagonal = (nactive/AO_TILE)*AO_TILE*AO_TILE + remainder*remainder;
                        int ng = MIN(grid_tile, ngrids-b*grid_tile);
                        prefix[b+1] = prefix[b] + ng*(nactive*nactive+diagonal);
                }
        }
#pragma omp parallel
{
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        const int team_size = omp_get_num_threads();
        const int thread_id = omp_get_thread_num();
        double *partial = scratch + thread_size * thread_id;
        double *left = partial + matrix_size;
        double *right = left + pack_size;
        double *product = right + pack_size;
        int active_ao[nao], active_shell[nao];
        int block, ig0, ng, k, nactive, max_bin, ib, jb, ni, nj, i, j, s, g, t;
        int begin = (size_t)nblock*thread_id/team_size;
        int end = (size_t)nblock*(thread_id+1)/team_size;
        if (with_plan && prefix[nblock] != 0) {
                begin = grid_cut(prefix, nblock, thread_id, team_size);
                end = grid_cut(prefix, nblock, thread_id+1, team_size);
        }
        size_t ij, src;
        uint8_t *screen;
        double q, tau_weight, sum;
        double *a0, *ax, *ay, *az, *pl, *pr;
        for (ij = 0; ij < matrix_size; ij++) {
                partial[ij] = 0;
        }
        for (block = begin; block < end; block++) {
                ig0 = block * grid_tile;
                ng = MIN(grid_tile, ngrids - ig0);
                k = ncomp * ng;
                screen = sparse ? screen_index + (ig0/GRID_BLOCK)*nbas : NULL;
                max_bin = 0;
                if (sparse) {
                        if (with_plan) max_bin = max_bins[block];
                        else {
                                for (s = 0; s < nbas; s++) max_bin = MAX(max_bin, screen[s]);
                        }
                        if (2*max_bin < nbins) continue;
                }
                nactive = 0;
                for (s = 0; s < nbas; s++) {
                        if (sparse && (screen[s] == 0 || screen[s]+max_bin < nbins)) {
                                continue;
                        }
                        for (i = ao_loc[s]; i < ao_loc[s+1]; i++) {
                                active_ao[nactive] = i;
                                active_shell[nactive++] = s;
                        }
                }
                /* Pack each AO once: one weighted channel for LDA, the two
                 * cross channels for GGA, or the full four-channel MGGA form. */
                for (i = 0; i < nactive; i++) {
                        a0 = ao + (size_t)active_ao[i]*ngrids + ig0;
                        ax = a0 + ao_stride;
                        ay = ax + ao_stride;
                        az = ay + ao_stride;
                        pl = left + i*k;
                        pr = right + i*k;
                        for (g = 0; g < ng; g++) {
                                if (ncomp == 1) {
                                        pl[g] = a0[g];
                                        pr[g] = wv[ig0+g]*a0[g];
                                        continue;
                                }
                                if (ncomp == 2) {
                                        q = .5*wv[ig0+g]*a0[g]
                                          + wv[ngrids+ig0+g]*ax[g]
                                          + wv[(size_t)2*ngrids+ig0+g]*ay[g]
                                          + wv[(size_t)3*ngrids+ig0+g]*az[g];
                                        pl[g] = a0[g];
                                        pl[ng+g] = q;
                                        pr[g] = q;
                                        pr[ng+g] = a0[g];
                                        continue;
                                }
                                pl[g] = a0[g];
                                pl[ng+g] = ax[g];
                                pl[2*ng+g] = ay[g];
                                pl[3*ng+g] = az[g];
                                tau_weight = .5*wv[(size_t)4*ngrids+ig0+g];
                                pr[g] = wv[ig0+g]*a0[g]
                                      + wv[ngrids+ig0+g]*ax[g]
                                      + wv[(size_t)2*ngrids+ig0+g]*ay[g]
                                      + wv[(size_t)3*ngrids+ig0+g]*az[g];
                                pr[ng+g] = wv[ngrids+ig0+g]*a0[g] + tau_weight*ax[g];
                                pr[2*ng+g] = wv[(size_t)2*ngrids+ig0+g]*a0[g] + tau_weight*ay[g];
                                pr[3*ng+g] = wv[(size_t)3*ngrids+ig0+g]*a0[g] + tau_weight*az[g];
                        }
                }
                for (ib = 0; ib < nactive; ib += AO_TILE) {
                        ni = MIN(AO_TILE, nactive-ib);
                        for (jb = 0; jb <= ib; jb += AO_TILE) {
                                nj = MIN(AO_TILE, nactive-jb);
                                dgemm_(&TRANS_T, &TRANS_N, &nj, &ni, &k,
                                       &D1, right+(size_t)jb*k, &k,
                                       left+(size_t)ib*k, &k, &D0, product, &nj);
                                for (i = 0; i < ni; i++) {
                                for (j = 0; j < nj && jb+j <= ib+i; j++) {
                                        if (!sparse ||
                                            (pair_mask[active_shell[ib+i]*nbas+active_shell[jb+j]] &&
                                             screen[active_shell[ib+i]] + screen[active_shell[jb+j]] >= nbins)) {
                                                partial[(size_t)active_ao[ib+i]*nao+active_ao[jb+j]] += product[i*nj+j];
                                        }
                                } }
                        }
                }
        }
#pragma omp barrier
#pragma omp for schedule(static)
        for (ij = 0; ij < matrix_size; ij++) {
                src = ij / nao < ij % nao ? (ij % nao)*nao + ij / nao : ij;
                sum = 0;
                for (t = 0; t < team_size; t++) {
                        sum += scratch[thread_size*t+src];
                }
                out[ij] += sum;
        }
}
        free(scratch);
        return 1;
}

int VXCmgga_vmat(double *out, double *ao, double *wv,
                 int nao, int ngrids, int nbas, int nbins,
                 uint8_t *screen_index, uint8_t *pair_mask, int *ao_loc,
                 ptrdiff_t ao_stride)
{
        return VXCxc_vmat(out, ao, wv, nao, ngrids, nbas, nbins, 4,
                          screen_index, pair_mask, ao_loc, ao_stride);
}
