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
 *
 * Fused spin-restricted XC integration for molecular SCF.
 *
 * The grid is processed in tiles of a few consecutive BLKSIZE blocks. For
 * each tile a thread
 *   1. collects the shells with nonzero screen index in any block of the
 *      tile (the "active" shells) and evaluates only their AO values into a
 *      compact, cache-resident buffer, using the same per-block evaluators and
 *      zero-fill rules as eval_ao;
 *   2. builds rho (and its gradient and tau) from either the occupied
 *      orbitals or the AO density matrix, whichever needs fewer operations
 *      for the tile;
 *   3. evaluates the functional with Libxc on the tile;
 *   4. accumulates the lower triangle of the XC matrix of the active AOs
 *      into a thread-private matrix.
 * The thread-private matrices are reduced once per call. Nothing scales
 * with the full grid except the input coordinates and weights.
 */

#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "config.h"
#include "gto/grid_ao_drv.h"
#include "np_helper/np_helper.h"
#include "vhf/fblas.h"

#define XC_LDA          0
#define XC_GGA          1
#define XC_MGGA         2
#define AO_TILE         64

typedef void (*FPtr_xc_tile)(int nfn, void *fn_obj, const double *fac,
                             int spin, int deriv, int nvar, int np, int outlen,
                             double *rho_u, double *output, double *work);

void GTOeval_sph_iter(FPtr_eval feval, FPtr_exp fexp, double fac,
                      size_t nao, size_t ngrids, size_t bgrids,
                      int param[], int *shls_slice, int *ao_loc, double *buf,
                      double *ao, double *coord, uint8_t *non0table,
                      int *atm, int natm, int *bas, int nbas, double *env);
void GTOeval_cart_iter(FPtr_eval feval, FPtr_exp fexp, double fac,
                       size_t nao, size_t ngrids, size_t bgrids,
                       int param[], int *shls_slice, int *ao_loc, double *buf,
                       double *ao, double *coord, uint8_t *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env);
void GTOshell_eval_grid_cart(double *gto, double *ri, double *exps,
                             double *coord, double *alpha, double *coeff,
                             double *env, int l, int np, int nc,
                             size_t nao, size_t ngrids, size_t blksize);
void GTOshell_eval_grid_cart_deriv1(double *gto, double *ri, double *exps,
                                    double *coord, double *alpha, double *coeff,
                                    double *env, int l, int np, int nc,
                                    size_t nao, size_t ngrids, size_t blksize);

typedef struct {
        size_t coords;
        size_t ao;
        size_t gto_buf;
        size_t dens;
        size_t coeff;
        size_t rho;
        size_t xc_out;
        size_t xc_work;
        size_t wv;
        size_t right;
        size_t prod;
        size_t partial;
        size_t total;
} FusedLayout;

static size_t _round8(size_t n)
{
        return (n + 7) & ~(size_t)7;
}

static FusedLayout _layout(int xctype, int nao, int nset, int max_nact,
                           int tile_size, int nocc, int with_mo)
{
        const size_t ng = tile_size;
        const size_t na = max_nact;
        const int ncomp = xctype == XC_LDA ? 1 : 4;
        const int nright = xctype == XC_MGGA ? 4 : 1;
        const int nao_dens = xctype == XC_MGGA ? 4 : 1;
        size_t dens_ao = na * na + nao_dens * ng * na;
        size_t dens_mo = with_mo ? (size_t)ncomp * ng * nocc : 0;
        FusedLayout s;
        s.coords = _round8(3 * ng);
        s.ao = _round8(ncomp * na * ng);
        s.gto_buf = _round8(BLKSIZE * (3 + NPRIMAX*2 + (size_t)NCTR_CART*ncomp + 8) + 8);
        s.dens = _round8(MAX(dens_ao, dens_mo));
        s.coeff = with_mo ? _round8(na * nocc) : 0;
        s.rho = _round8(5 * ng);
        s.xc_out = _round8(4 * ng);
        s.xc_work = _round8(11 * ng);
        s.wv = _round8(5 * ng);
        s.right = _round8(nright * na * ng);
        s.prod = _round8(AO_TILE * AO_TILE);
        s.partial = (size_t)nset * nao * (nao + 1) / 2;
        s.total = s.coords + s.ao + s.gto_buf + s.dens + s.coeff + s.rho +
                s.xc_out + s.xc_work + s.wv + s.right + s.prod + s.partial;
        return s;
}

/* Active shells of a tile: nonzero screen index in any of its blocks. */
static int _tile_shells(int *shells, uint8_t *non0tab, int blk0, int nblk_t,
                        int nbas)
{
        int nshell = 0;
        for (int s = 0; s < nbas; s++) {
                for (int b = 0; b < nblk_t; b++) {
                        if (non0tab[(size_t)(blk0+b)*nbas+s]) {
                                shells[nshell++] = s;
                                break;
                        }
                }
        }
        return nshell;
}

/* Evaluate AO values of the active shells into ao[comp,nact,ng].
 * Consecutive active shells on one atom are evaluated by one call. */
static void _eval_tile_ao(double *ao, double *coords_t, double *gto_buf,
                          int *shells, int nshell, int *aoff, int nact,
                          int ng, int ib0, int nblk_t, uint8_t *non0tab,
                          int deriv, int cart, int *ao_loc,
                          int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[2] = {1, deriv == 0 ? 1 : 4};
        FPtr_eval feval = deriv == 0 ? GTOshell_eval_grid_cart :
                GTOshell_eval_grid_cart_deriv1;
        FPtr_exp fexp = deriv == 0 ? GTOcontract_exp0 : GTOcontract_exp1;
        int i0, i1, b, bgrids;
        int shls_slice[2];
        for (i0 = 0; i0 < nshell; i0 = i1) {
                const int atm_id = bas[shells[i0]*BAS_SLOTS+ATOM_OF];
                for (i1 = i0 + 1; i1 < nshell; i1++) {
                        if (shells[i1] != shells[i1-1] + 1 ||
                            bas[shells[i1]*BAS_SLOTS+ATOM_OF] != atm_id) {
                                break;
                        }
                }
                shls_slice[0] = shells[i0];
                shls_slice[1] = shells[i1-1] + 1;
                for (b = 0; b < nblk_t; b++) {
                        bgrids = MIN(BLKSIZE, ng - b*BLKSIZE);
                        double *pao = ao + (size_t)aoff[i0]*ng + b*BLKSIZE;
                        double *pcoord = coords_t + b*BLKSIZE;
                        uint8_t *mask = non0tab + (size_t)(ib0+b)*nbas;
                        if (cart) {
                                GTOeval_cart_iter(feval, fexp, 1., nact, ng, bgrids,
                                                  param, shls_slice, ao_loc, gto_buf,
                                                  pao, pcoord, mask,
                                                  atm, natm, bas, nbas, env);
                        } else {
                                GTOeval_sph_iter(feval, fexp, 1., nact, ng, bgrids,
                                                 param, shls_slice, ao_loc, gto_buf,
                                                 pao, pcoord, mask,
                                                 atm, natm, bas, nbas, env);
                        }
                }
        }
}

/* rho[nvar,ng] from occupied orbitals: coeff_t[nocc,nact] (column-major
 * nact x nocc) holds sqrt(|occ|)*C of the active AOs, sign the sign of occ. */
static void _rho_mo(double *rho, double *ao, double *coeff_t, double *sign,
                    double *psi, int xctype, int nact, int nocc, int ng)
{
        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        const int ncomp = xctype == XC_LDA ? 1 : 4;
        const size_t ao_comp = (size_t)nact * ng;
        const size_t psi_comp = (size_t)nocc * ng;
        int c, k, g;
        for (c = 0; c < ncomp; c++) {
                dgemm_(&TRANS_N, &TRANS_N, &ng, &nocc, &nact,
                       &D1, ao + c*ao_comp, &ng, coeff_t, &nact,
                       &D0, psi + c*psi_comp, &ng);
        }
        double *rho0 = rho;
        for (g = 0; g < ng; g++) rho0[g] = 0;
        for (k = 0; k < nocc; k++) {
                const double s = sign[k];
                const double *p0 = psi + k*ng;
#pragma GCC ivdep
                for (g = 0; g < ng; g++) rho0[g] += s * p0[g] * p0[g];
        }
        if (xctype == XC_LDA) {
                return;
        }
        for (c = 1; c < 4; c++) {
                double *grad = rho + c*ng;
                for (g = 0; g < ng; g++) grad[g] = 0;
                for (k = 0; k < nocc; k++) {
                        const double s2 = 2 * sign[k];
                        const double *p0 = psi + k*ng;
                        const double *pc = psi + c*psi_comp + k*ng;
#pragma GCC ivdep
                        for (g = 0; g < ng; g++) grad[g] += s2 * p0[g] * pc[g];
                }
        }
        if (xctype == XC_MGGA) {
                double *tau = rho + 4*ng;
                for (g = 0; g < ng; g++) tau[g] = 0;
                for (c = 1; c < 4; c++) {
                for (k = 0; k < nocc; k++) {
                        const double s = .5 * sign[k];
                        const double *pc = psi + c*psi_comp + k*ng;
#pragma GCC ivdep
                        for (g = 0; g < ng; g++) tau[g] += s * pc[g] * pc[g];
                } }
        }
}

/* rho[nvar,ng] from a symmetric AO density matrix restricted to active AOs.
 * dm_t is the column-major (symmetric) nact x nact submatrix. */
static void _rho_ao(double *rho, double *ao, double *dm_t, double *cbuf,
                    int xctype, int nact, int ng)
{
        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        const size_t ao_comp = (size_t)nact * ng;
        int a, c, g;
        double *c0 = cbuf;
        dgemm_(&TRANS_N, &TRANS_N, &ng, &nact, &nact,
               &D1, ao, &ng, dm_t, &nact, &D0, c0, &ng);
        double *rho0 = rho;
        for (g = 0; g < ng; g++) rho0[g] = 0;
        for (a = 0; a < nact; a++) {
                const double *pa = ao + (size_t)a*ng;
                const double *pc = c0 + (size_t)a*ng;
#pragma GCC ivdep
                for (g = 0; g < ng; g++) rho0[g] += pa[g] * pc[g];
        }
        if (xctype == XC_LDA) {
                return;
        }
        for (c = 1; c < 4; c++) {
                double *grad = rho + c*ng;
                const double *aoc = ao + c*ao_comp;
                for (g = 0; g < ng; g++) grad[g] = 0;
                for (a = 0; a < nact; a++) {
                        const double *pa = aoc + (size_t)a*ng;
                        const double *pc = c0 + (size_t)a*ng;
#pragma GCC ivdep
                        for (g = 0; g < ng; g++) grad[g] += 2 * pa[g] * pc[g];
                }
        }
        if (xctype == XC_MGGA) {
                double *tau = rho + 4*ng;
                for (g = 0; g < ng; g++) tau[g] = 0;
                for (c = 1; c < 4; c++) {
                        const double *aoc = ao + c*ao_comp;
                        double *cc = cbuf + (size_t)c*ao_comp;
                        dgemm_(&TRANS_N, &TRANS_N, &ng, &nact, &nact,
                               &D1, aoc, &ng, dm_t, &nact, &D0, cc, &ng);
                        for (a = 0; a < nact; a++) {
                                const double *pa = aoc + (size_t)a*ng;
                                const double *pc = cc + (size_t)a*ng;
#pragma GCC ivdep
                                for (g = 0; g < ng; g++) tau[g] += .5 * pa[g] * pc[g];
                        }
                }
        }
}

/* Lower triangle of the XC matrix of the active AOs, added to partial, a
 * packed lower triangle (row r starts at r*(r+1)/2). active_ao is ascending,
 * so every element formed below has row >= column.
 *   LDA : V_ij = sum_g a0_i R0_j,                 R0 = w0 a0
 *   GGA : V_ij = sum_g a0_i R0_j + R0_i a0_j,     R0 = .5 w0 a0 + sum_c wc ac
 *   MGGA: V_ij = sum_g a0_i R0_j + sum_c ac_i Rc_j,
 *         R0 = w0 a0 + sum_c wc ac,  Rc = wc a0 + .5 w4 ac
 * Each is symmetric, so only the lower triangle is formed. */
static void _vmat_tile(double *partial, double *ao, double *wv, double *right,
                       double *prod, int *active_ao, int xctype,
                       int nact, int ng)
{
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        const size_t ao_comp = (size_t)nact * ng;
        const double *w0 = wv;
        const double *wx = wv + ng;
        const double *wy = wv + ng*2;
        const double *wz = wv + ng*3;
        const double *w4 = wv + ng*4;
        int a, c, g, ib, jb, ni, nj, i, j;

        for (a = 0; a < nact; a++) {
                const double *a0 = ao + (size_t)a*ng;
                double *r0 = right + (size_t)a*ng;
                if (xctype == XC_LDA) {
#pragma GCC ivdep
                        for (g = 0; g < ng; g++) r0[g] = w0[g] * a0[g];
                        continue;
                }
                const double *ax = a0 + ao_comp;
                const double *ay = ax + ao_comp;
                const double *az = ay + ao_comp;
                if (xctype == XC_GGA) {
#pragma GCC ivdep
                        for (g = 0; g < ng; g++) {
                                r0[g] = .5*w0[g]*a0[g] + wx[g]*ax[g]
                                      + wy[g]*ay[g] + wz[g]*az[g];
                        }
                        continue;
                }
                double *rx = r0 + ao_comp;
                double *ry = rx + ao_comp;
                double *rz = ry + ao_comp;
#pragma GCC ivdep
                for (g = 0; g < ng; g++) {
                        const double t = .5 * w4[g];
                        r0[g] = w0[g]*a0[g] + wx[g]*ax[g] + wy[g]*ay[g] + wz[g]*az[g];
                        rx[g] = wx[g]*a0[g] + t*ax[g];
                        ry[g] = wy[g]*a0[g] + t*ay[g];
                        rz[g] = wz[g]*a0[g] + t*az[g];
                }
        }

        for (ib = 0; ib < nact; ib += AO_TILE) {
                ni = MIN(AO_TILE, nact - ib);
                for (jb = 0; jb <= ib; jb += AO_TILE) {
                        nj = MIN(AO_TILE, nact - jb);
                        /* prod[i*nj+j] = V[ib+i, jb+j] */
                        dgemm_(&TRANS_T, &TRANS_N, &nj, &ni, &ng,
                               &D1, right + (size_t)jb*ng, &ng,
                               ao + (size_t)ib*ng, &ng, &D0, prod, &nj);
                        if (xctype == XC_GGA) {
                                dgemm_(&TRANS_T, &TRANS_N, &nj, &ni, &ng,
                                       &D1, ao + (size_t)jb*ng, &ng,
                                       right + (size_t)ib*ng, &ng, &D1, prod, &nj);
                        } else if (xctype == XC_MGGA) {
                                for (c = 1; c < 4; c++) {
                                        dgemm_(&TRANS_T, &TRANS_N, &nj, &ni, &ng,
                                               &D1, right + c*ao_comp + (size_t)jb*ng, &ng,
                                               ao + c*ao_comp + (size_t)ib*ng, &ng,
                                               &D1, prod, &nj);
                                }
                        }
                        for (i = 0; i < ni; i++) {
                                const int stop = jb == ib ? i + 1 : nj;
                                const size_t row = active_ao[ib+i];
                                double *dst = partial + row*(row+1)/2;
                                const double *src = prod + (size_t)i*nj;
                                const int *col = active_ao + jb;
                                for (j = 0; j < stop; j++) {
                                        dst[col[j]] += src[j];
                                }
                        }
                }
        }
}

/* A pthreads OpenBLAS called from inside an OpenMP region runs each small
 * GEMM on its own thread pool and oversubscribes the cores (4x slower on 4
 * cores). Ask it to run single-threaded while this kernel executes and
 * restore the previous setting afterwards. OpenMP builds of OpenBLAS detect
 * nested parallelism themselves; other BLAS libraries are not touched. */
extern int openblas_get_parallel(void) __attribute__((weak));
extern int openblas_get_num_threads(void) __attribute__((weak));
extern void openblas_set_num_threads(int) __attribute__((weak));

static int _blas_serial_begin(int nthreads)
{
        if (nthreads > 1 && openblas_get_parallel != NULL &&
            openblas_get_num_threads != NULL && openblas_set_num_threads != NULL &&
            openblas_get_parallel() == 1) {
                const int saved = openblas_get_num_threads();
                if (saved > 1) {
                        openblas_set_num_threads(1);
                        return saved;
                }
        }
        return 0;
}

static void _blas_serial_end(int saved)
{
        if (saved > 0) {
                openblas_set_num_threads(saved);
        }
}

/* First tile of part `part` out of `nparts` with equal estimated cost. */
static int _partition(const double *prefix, int ntile, int part, int nparts)
{
        if (part <= 0) return 0;
        if (part >= nparts) return ntile;
        const double target = prefix[ntile] * part / nparts;
        int lo = 0, hi = ntile;
        while (lo < hi) {
                const int mid = lo + (hi - lo) / 2;
                if (prefix[mid] < target) lo = mid + 1;
                else hi = mid;
        }
        return lo;
}

/*
 * vmat[nset,nao,nao] += V_xc; nelec_exc[nset,2] += (N, E_xc).
 *
 * dms[nset,nao,nao] are symmetric AO density matrices. If mo_coeff is not
 * NULL (nset must be 1) it holds sqrt(|occ|)*C[nao,nocc] for the nonzero
 * occupations and mo_sign their signs; each tile then picks the cheaper of the
 * orbital and the density-matrix route.
 * coords[ngrids,3], weights[ngrids], non0tab[nblk,nbas] as in grids.
 * Returns 0 without modifying outputs if the workspace cannot be allocated.
 */
int VXCnr_rks_fused(double *vmat, double *nelec_exc, int xctype, int nset,
                    double *dms, double *mo_coeff, double *mo_sign, int nocc,
                    double *coords, double *weights, int ngrids,
                    uint8_t *non0tab, int tile_blocks, double max_memory_mb,
                    FPtr_xc_tile fxc, int nfn, void *fn_obj, double *fn_fac,
                    int cart, int *ao_loc, int *atm, int natm,
                    int *bas, int nbas, double *env)
{
        const int nao = ao_loc[nbas];
        const int nblk = (ngrids + BLKSIZE - 1) / BLKSIZE;
        if (nblk == 0 || nao == 0) {
                return 1;
        }
        tile_blocks = MAX(1, tile_blocks);
        const int ntile = (nblk + tile_blocks - 1) / tile_blocks;
        const int tile_size = tile_blocks * BLKSIZE;
        const int with_mo = mo_coeff != NULL && nocc > 0;
        const int deriv = xctype == XC_LDA ? 0 : 1;
        const int nvar = xctype == XC_LDA ? 1 : (xctype == XC_GGA ? 4 : 5);
        const int outlen = xctype == XC_LDA ? 2 : (xctype == XC_GGA ? 3 : 4);
        const int ncomp = xctype == XC_LDA ? 1 : 4;
        if (with_mo && nset != 1) {
                return 0;
        }

        /* Active-set size of every tile. It sizes the workspace and gives a
         * cost estimate for a deterministic static partition of the tiles. */
        int *tile_nact = malloc(sizeof(int) * ntile);
        double *prefix = malloc(sizeof(double) * (ntile + 1));
        if (tile_nact == NULL || prefix == NULL) {
                free(tile_nact);
                free(prefix);
                return 0;
        }
        int scan_failed = 0;
#pragma omp parallel
{
        int *shells = malloc(sizeof(int) * nbas);
        if (shells == NULL) {
#pragma omp atomic write
                scan_failed = 1;
        }
#pragma omp for schedule(static)
        for (int t = 0; t < ntile; t++) {
                if (shells == NULL) continue;
                const int blk0 = t * tile_blocks;
                const int nblk_t = MIN(tile_blocks, nblk - blk0);
                const int nshell = _tile_shells(shells, non0tab, blk0, nblk_t, nbas);
                int na = 0;
                for (int i = 0; i < nshell; i++) {
                        na += ao_loc[shells[i]+1] - ao_loc[shells[i]];
                }
                tile_nact[t] = na;
        }
        free(shells);
}
        int max_nact = 0;
        prefix[0] = 0;
        for (int t = 0; t < ntile && !scan_failed; t++) {
                const double na = tile_nact[t];
                const int ng = MIN(tile_size, ngrids - t*tile_size);
                max_nact = MAX(max_nact, tile_nact[t]);
                /* AO evaluation ~ na, density ~ na*(na or nocc), matrix ~ na^2 */
                prefix[t+1] = prefix[t] + (na > 0 ? ng * (na * na + 64. * na + 256.) : 0);
        }
        free(tile_nact);
        if (scan_failed) {
                free(prefix);
                return 0;
        }
        if (max_nact == 0) {
                free(prefix);
                return 1;
        }

        const FusedLayout lay = _layout(xctype, nao, nset, max_nact, tile_size,
                                        nocc, with_mo);
        const size_t thread_bytes = lay.total * sizeof(double) +
                sizeof(int) * ((size_t)3 * nbas + 2 * max_nact + 8);
        int nthreads = 1;
#ifdef _OPENMP
        nthreads = omp_get_max_threads();
#endif
        nthreads = MIN(nthreads, ntile);
        const double budget = max_memory_mb * 1e6;
        if (budget > 0) {
                int fit = (int)MIN((double)nthreads, budget / (double)thread_bytes);
                if (fit < 1) {
                        free(prefix);
                        return 0;
                }
                nthreads = fit;
        }
        double **bufs = calloc(nthreads, sizeof(double *));
        if (bufs == NULL) {
                free(prefix);
                return 0;
        }
        int failed = 0;
        int nteam = nthreads;
        const size_t partial_off = lay.total - lay.partial;
        double *acc = calloc((size_t)nthreads * nset * 2, sizeof(double));
        if (acc == NULL) {
                free(bufs);
                free(prefix);
                return 0;
        }

        const int blas_saved = _blas_serial_begin(nthreads);
#pragma omp parallel num_threads(nthreads)
{
        const int tid = omp_get_thread_num();
#pragma omp single
        nteam = omp_get_num_threads();
        /* Allocate (and first touch) the workspace on the executing thread. */
        double *buf = malloc(thread_bytes);
        bufs[tid] = buf;
        if (buf == NULL) {
#pragma omp atomic write
                failed = 1;
        }
#pragma omp barrier
        if (!failed) {
        double *coords_t = buf;
        double *ao = coords_t + lay.coords;
        double *gto_buf = ao + lay.ao;
        double *dens = gto_buf + lay.gto_buf;
        double *coeff_t = dens + lay.dens;
        double *rho = coeff_t + lay.coeff;
        double *xc_out = rho + lay.rho;
        double *xc_work = xc_out + lay.xc_out;
        double *wv = xc_work + lay.xc_work;
        double *right = wv + lay.wv;
        double *prod = right + lay.right;
        double *partial = buf + partial_off;
        int *shells = (int *)(partial + lay.partial);
        int *aoff = shells + nbas;
        int *active_ao = aoff + nbas;
        double *my_acc = acc + (size_t)tid * nset * 2;
        memset(partial, 0, sizeof(double) * lay.partial);
        const size_t nao2 = (size_t)nao * nao;
        const size_t npair = (size_t)nao * (nao + 1) / 2;
        int t, i, a, g, k, iset;

        const int t0 = _partition(prefix, ntile, tid, nteam);
        const int t1 = _partition(prefix, ntile, tid + 1, nteam);
        for (t = t0; t < t1; t++) {
                const int blk0 = t * tile_blocks;
                const int nblk_t = MIN(tile_blocks, nblk - blk0);
                const int ig0 = blk0 * BLKSIZE;
                const int ng = MIN(tile_size, ngrids - ig0);
                const int nshell = _tile_shells(shells, non0tab, blk0, nblk_t, nbas);
                int nact = 0;
                for (i = 0; i < nshell; i++) {
                        const int s = shells[i];
                        aoff[i] = nact;
                        for (a = ao_loc[s]; a < ao_loc[s+1]; a++) {
                                active_ao[nact++] = a;
                        }
                }
                if (nact == 0) {
                        continue;
                }
                for (g = 0; g < ng; g++) {
                        coords_t[g]        = coords[(size_t)(ig0+g)*3  ];
                        coords_t[ng+g]     = coords[(size_t)(ig0+g)*3+1];
                        coords_t[ng*2+g]   = coords[(size_t)(ig0+g)*3+2];
                }
                _eval_tile_ao(ao, coords_t, gto_buf, shells, nshell, aoff, nact,
                              ng, blk0, nblk_t, non0tab, deriv, cart, ao_loc,
                              atm, natm, bas, nbas, env);

                const double *w = weights + ig0;
                for (iset = 0; iset < nset; iset++) {
                        /* Cost of the orbital route ~ ncomp*nocc, of the
                         * density-matrix route ~ (1 or 4)*nact per point. */
                        const int use_mo = with_mo &&
                                ncomp * nocc < (xctype == XC_MGGA ? 4 : 1) * nact;
                        if (use_mo) {
                                for (a = 0; a < nact; a++) {
                                        const double *row = mo_coeff + (size_t)active_ao[a]*nocc;
                                        for (k = 0; k < nocc; k++) {
                                                coeff_t[(size_t)k*nact+a] = row[k];
                                        }
                                }
                                _rho_mo(rho, ao, coeff_t, mo_sign, dens,
                                        xctype, nact, nocc, ng);
                        } else {
                                const double *dm = dms + iset * nao2;
                                double *dm_t = dens;
                                for (i = 0; i < nact; i++) {
                                        const double *row = dm + (size_t)active_ao[i]*nao;
                                        for (a = 0; a < nact; a++) {
                                                dm_t[(size_t)i*nact+a] = row[active_ao[a]];
                                        }
                                }
                                _rho_ao(rho, ao, dm_t, dens + (size_t)nact*nact,
                                        xctype, nact, ng);
                        }

                        memset(xc_out, 0, sizeof(double) * outlen * ng);
                        (*fxc)(nfn, fn_obj, fn_fac, 0, 1, nvar, ng, outlen,
                               rho, xc_out, xc_work);

                        double ne = 0, ex = 0;
                        for (g = 0; g < ng; g++) {
                                const double den = rho[g] * w[g];
                                ne += den;
                                ex += den * xc_out[g];
                        }
                        my_acc[iset*2  ] += ne;
                        my_acc[iset*2+1] += ex;

                        const double *vrho = xc_out + ng;
                        for (g = 0; g < ng; g++) wv[g] = w[g] * vrho[g];
                        if (xctype != XC_LDA) {
                                const double *vsigma = xc_out + ng*2;
                                for (i = 1; i < 4; i++) {
                                        const double *grad = rho + i*ng;
                                        for (g = 0; g < ng; g++) {
                                                wv[i*ng+g] = 2 * w[g] * vsigma[g] * grad[g];
                                        }
                                }
                        }
                        if (xctype == XC_MGGA) {
                                const double *vtau = xc_out + ng*3;
                                for (g = 0; g < ng; g++) wv[4*ng+g] = w[g] * vtau[g];
                        }
                        _vmat_tile(partial + iset * npair, ao, wv, right, prod,
                                   active_ao, xctype, nact, ng);
                }
        }

        /* Reduce the lower triangles of all threads and mirror them. */
#pragma omp barrier
#pragma omp for schedule(static)
        for (size_t ij = 0; ij < nset * nao2; ij++) {
                const size_t iset0 = ij / nao2;
                const size_t r = ij % nao2;
                const size_t row = r / nao;
                const size_t col = r % nao;
                const size_t src = iset0 * npair +
                        (row >= col ? row*(row+1)/2+col : col*(col+1)/2+row);
                double sum = 0;
                for (int tt = 0; tt < nteam; tt++) {
                        sum += bufs[tt][partial_off + src];
                }
                vmat[ij] += sum;
        }
        }
}
        _blas_serial_end(blas_saved);
        if (!failed) {
                for (int tt = 0; tt < nteam; tt++) {
                for (int iset = 0; iset < nset; iset++) {
                        nelec_exc[iset*2  ] += acc[(size_t)tt*nset*2+iset*2  ];
                        nelec_exc[iset*2+1] += acc[(size_t)tt*nset*2+iset*2+1];
                } }
        }
        for (int tt = 0; tt < nthreads; tt++) {
                free(bufs[tt]);
        }
        free(bufs);
        free(acc);
        free(prefix);
        return !failed;
}
