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
 * Coulomb/exchange contribution to molecular nuclear gradients, contracted
 * with the density for every shell quartet instead of building the
 * 3 x nao x nao J/K derivative matrices of grad.rhf.get_jk.
 *
 *   E = 1/2 sum_{ijkl} (ij|kl) G_ijkl
 *   G_ijkl = j_fac Dj_ij Dj_kl - k_fac sum_s (Ds_ik Ds_jl + Ds_il Ds_jk)
 *
 * G has the 8-fold permutational symmetry of (ij|kl), so only unique shell
 * quartets (i>=j, k>=l, ij>=kl) are visited, with degeneracy factor g.
 * Derivatives come from int2e_ip1 with the derivative center first. By
 * translational invariance only the centers off one atom are evaluated;
 * one-center quartets are skipped.
 * Each thread keeps natm x 3 accumulators; there are no matrix reductions.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "config.h"
#include "cint.h"
#include "gto/gto.h"
#include "np_helper/np_helper.h"

/* sum_x[c] = sum_{abcd} buf[c][...] * G[...] for the three components,
 * where buf is in libcint order for the permuted quartet and the index map
 * gives the offset of each G element (i fastest) in that order. */
static void contract3(double *s, const double *buf, const double *G,
                      const int *idx, int n)
{
        double s0 = 0, s1 = 0, s2 = 0;
        const double *b0 = buf;
        const double *b1 = buf + n;
        const double *b2 = buf + n * 2;
        for (int p = 0; p < n; p++) {
                const double g = G[p];
                const int q = idx[p];
                s0 += b0[q] * g;
                s1 += b1[q] * g;
                s2 += b2[q] * g;
        }
        s[0] = s0;
        s[1] = s1;
        s[2] = s2;
}

void CVHFgrad_jk_force(double *de, double *dmj, double *dmk, int nsetk,
                       double j_fac, double k_fac,
                       int (*intor)(), CINTOpt *cintopt,
                       double *q_cond, double *dm_cond, double cutoff,
                       int *ao_loc, int *atm, int natm,
                       int *bas, int nbas, double *env)
{
        const size_t nao = ao_loc[nbas];
        const size_t Nbas = nbas;
        const double *qd = q_cond;            /* derivative bound, (d_i j|d_i j) */
        const double *qr = q_cond + Nbas*Nbas; /* Schwarz bound (ij|ij) */
        const double *dcj = dm_cond;           /* shell max |Dj| */
        const double *dck = dm_cond + Nbas*Nbas; /* shell max_s |Ds| */
        int shls_slice[] = {0, nbas, 0, nbas, 0, nbas, 0, nbas};
        const size_t cache_size = GTOmax_cache_size(intor, shls_slice, 4,
                                                    atm, natm, bas, nbas, env);
        int dmax = 0;
        for (int i = 0; i < nbas; i++) {
                dmax = MAX(dmax, ao_loc[i+1] - ao_loc[i]);
        }
        const size_t d4 = (size_t)dmax * dmax * dmax * dmax;
        const size_t npair = Nbas * (Nbas + 1) / 2;
        const int has_j = j_fac != 0 && dmj != NULL;
        const int has_k = k_fac != 0 && dmk != NULL && nsetk > 0;

#pragma omp parallel
{
        double *de_priv = calloc((size_t)natm * 3, sizeof(double));
        double *buf = malloc(sizeof(double) * (d4 * 3 + d4 + cache_size));
        double *G = buf + d4 * 3;
        double *cache = G + d4;
        int *idx = malloc(sizeof(int) * d4 * 3);
        int *idxj = idx;
        int *idxk = idx + d4;
        int *idxl = idx + d4 * 2;
        int shls[4];
#pragma omp for schedule(dynamic, 1)
        for (size_t ij = 0; ij < npair; ij++) {
                /* Largest pairs first: they own the most kl quartets */
                const size_t r = npair - 1 - ij;
                int i = (int)((sqrt(8.0 * r + 1) - 1) / 2);
                while ((size_t)i * (i + 1) / 2 > r) i--;
                while ((size_t)(i + 1) * (i + 2) / 2 <= r) i++;
                const int j = (int)(r - (size_t)i * (i + 1) / 2);
                const double qdij = MAX(qd[i*Nbas+j], qd[j*Nbas+i]);
                const double qij = qr[i*Nbas+j];
                const int i0 = ao_loc[i], di = ao_loc[i+1] - i0;
                const int j0 = ao_loc[j], dj = ao_loc[j+1] - j0;
                const int ia = bas[ATOM_OF+i*BAS_SLOTS];
                const int ja = bas[ATOM_OF+j*BAS_SLOTS];
                for (int k = 0; k <= i; k++) {
                const int lmax = k == i ? j : k;
                for (int l = 0; l <= lmax; l++) {
                        const double qdkl = MAX(qd[k*Nbas+l], qd[l*Nbas+k]);
                        const double qkl = qr[k*Nbas+l];
                        const double bound = MAX(qdij * qkl, qij * qdkl);
                        if (bound < cutoff) {
                                continue;
                        }
                        double dm_max = 0;
                        if (has_j) {
                                dm_max += fabs(j_fac) * dcj[i*Nbas+j] * dcj[k*Nbas+l];
                        }
                        if (has_k) {
                                dm_max += fabs(k_fac) * nsetk *
                                        (dck[i*Nbas+k] * dck[j*Nbas+l] +
                                         dck[i*Nbas+l] * dck[j*Nbas+k]);
                        }
                        if (bound * dm_max < cutoff) {
                                continue;
                        }
                        const int ka = bas[ATOM_OF+k*BAS_SLOTS];
                        const int la = bas[ATOM_OF+l*BAS_SLOTS];
                        if (ia == ja && ia == ka && ia == la) {
                                continue;  /* one-center quartet: no net force */
                        }
                        /* Translational invariance holds per atom: only the
                         * centers off one atom are differentiated, and that
                         * atom receives minus their sum. Drop the atom whose
                         * centers are the most expensive to differentiate. */
                        const int at[4] = {ia, ja, ka, la};
                        double wd[4];
                        int c, c2;
                        for (c = 0; c < 4; c++) {
                                const int sh = c == 0 ? i : (c == 1 ? j : (c == 2 ? k : l));
                                const int ang = bas[ANG_OF+sh*BAS_SLOTS] + 1;
                                wd[c] = (ang + 1) * (ang + 2);
                        }
                        int drop = at[0];
                        double drop_cost = -1;
                        for (c = 0; c < 4; c++) {
                                double cost = 0;
                                for (c2 = 0; c2 < 4; c2++) {
                                        if (at[c2] == at[c]) cost += wd[c2];
                                }
                                if (cost > drop_cost) {
                                        drop_cost = cost;
                                        drop = at[c];
                                }
                        }
                        int need[4];
                        for (c = 0; c < 4; c++) {
                                need[c] = at[c] != drop;
                        }

                        const int k0 = ao_loc[k], dk = ao_loc[k+1] - k0;
                        const int l0 = ao_loc[l], dl = ao_loc[l+1] - l0;
                        const int n = di * dj * dk * dl;
                        /* G in (i,j,k,l) order, i fastest, and the offsets of
                         * each element in the (j,i,k,l), (k,l,i,j), (l,k,i,j)
                         * integral layouts (derivative center first) */
                        int m, p, q, t, u;
                        for (u = 0; u < dl; u++) {
                        for (t = 0; t < dk; t++) {
                        for (q = 0; q < dj; q++) {
                        for (m = 0; m < di; m++) {
                                const size_t im = i0 + m, jq = j0 + q;
                                const size_t kt = k0 + t, lu = l0 + u;
                                double g = 0;
                                if (has_j) {
                                        g = j_fac * dmj[im*nao+jq] * dmj[kt*nao+lu];
                                }
                                if (has_k) {
                                        double gk = 0;
                                        for (int s = 0; s < nsetk; s++) {
                                                const double *D = dmk + s * nao * nao;
                                                gk += D[im*nao+kt] * D[jq*nao+lu]
                                                    + D[im*nao+lu] * D[jq*nao+kt];
                                        }
                                        g -= k_fac * gk;
                                }
                                p = m + di * (q + dj * (t + dk * u));
                                G[p] = g;
                                if (need[1]) idxj[p] = q + dj * (m + di * (t + dk * u));
                                if (need[2]) idxk[p] = t + dk * (u + dl * (m + di * q));
                                if (need[3]) idxl[p] = u + dl * (t + dk * (m + di * q));
                        } } } }

                        double sc[4][3] = {{0}};
                        int nonzero = 0;
                        for (c = 0; c < 4; c++) {
                                if (!need[c]) {
                                        continue;
                                }
                                const int *map = NULL;
                                switch (c) {
                                case 0: shls[0] = i; shls[1] = j; shls[2] = k; shls[3] = l; break;
                                case 1: shls[0] = j; shls[1] = i; shls[2] = k; shls[3] = l;
                                        map = idxj; break;
                                case 2: shls[0] = k; shls[1] = l; shls[2] = i; shls[3] = j;
                                        map = idxk; break;
                                default: shls[0] = l; shls[1] = k; shls[2] = i; shls[3] = j;
                                        map = idxl;
                                }
                                if (!(*intor)(buf, NULL, shls, atm, natm, bas, nbas, env,
                                              cintopt, cache)) {
                                        continue;
                                }
                                nonzero = 1;
                                if (map == NULL) {
                                        double s0 = 0, s1 = 0, s2 = 0;
                                        for (p = 0; p < n; p++) {
                                                s0 += buf[p] * G[p];
                                                s1 += buf[n+p] * G[p];
                                                s2 += buf[n*2+p] * G[p];
                                        }
                                        sc[c][0] = s0; sc[c][1] = s1; sc[c][2] = s2;
                                } else {
                                        contract3(sc[c], buf, G, map, n);
                                }
                        }
                        if (!nonzero) {
                                continue;
                        }
                        double fac = .5;  /* E = 1/2 sum (ij|kl) G */
                        fac *= 8;
                        if (i == j) fac *= .5;
                        if (k == l) fac *= .5;
                        if (i == k && j == l) fac *= .5;
                        /* nuclear derivative = -(electronic gradient on that center) */
                        for (int x = 0; x < 3; x++) {
                                double total = 0;
                                for (c = 0; c < 4; c++) {
                                        if (need[c]) {
                                                de_priv[at[c]*3+x] -= fac * sc[c][x];
                                                total += sc[c][x];
                                        }
                                }
                                de_priv[drop*3+x] += fac * total;
                        }
                } }
        }
#pragma omp critical
        for (int a = 0; a < natm * 3; a++) {
                de[a] += de_priv[a];
        }
        free(de_priv);
        free(buf);
        free(idx);
}
}
