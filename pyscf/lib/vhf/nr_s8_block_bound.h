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
#ifndef PYSCF_NR_S8_BLOCK_BOUND_H
#define PYSCF_NR_S8_BLOCK_BOUND_H
#include <math.h>
#include <stddef.h>

/* Bounds contain maxima, not sums: with nonnegative operands the same
 * floating-point products bound every original shell-screen product.
 * Invalid/nonfinite inputs decline the optimization, never discard work.
 * No retained state: changed delta-densities always use fresh bounds.
 */
static double CVHFblock_max(const double *a, size_t n, const int *rows,
                           const int *cols)
{
        double m = 0.;
        for (int i = rows[0]; i < rows[1]; i++) {
                for (int j = cols[0]; j < cols[1]; j++) {
                        const double v = a[(size_t)i*n+j];
                        if (!isfinite(v) || v < 0.) return -1.;
                        if (v > m) m = v;
                }
        }
        return m;
}

/* mode = 1 (J), 2 (K), 3 (J+K). Return 1 only if every leaf would fail
 * CVHFnrs8[_vj/_vk]_prescreen at its unchanged direct_scf_cutoff.
 * Multiplication order intentionally matches each original predicate.
 */
static int CVHFs8_block_reject(const double *q, const double *dm, size_t n,
                              double cutoff, int mode, const int *is,
                              const int *js, const int *ks, const int *ls)
{
        if (!q || !dm || !isfinite(cutoff) || cutoff < 0. ||
            mode < 1 || mode > 3) return 0;
        const int *ranges[4] = {is, js, ks, ls};
        for (int a = 0; a < 4; a++) {
                if (ranges[a][0] < 0 || ranges[a][1] <= ranges[a][0] ||
                    (size_t)ranges[a][1] > n) return 0;
        }
        const double a = CVHFblock_max(q, n, is, js);
        const double b = CVHFblock_max(q, n, ks, ls);
        if (a < 0. || b < 0.) return 0;
        const double bound = a*b;
        if (!isfinite(bound)) return 0;
        if (bound <= cutoff) return 1;
        if (mode & 1) {
                const double dji = CVHFblock_max(dm, n, js, is);
                const double dlk = CVHFblock_max(dm, n, ls, ks);
                if (dji < 0. || dlk < 0.) return 0;
                if (mode == 1) {
                        if (4*bound*dji > cutoff || 4*bound*dlk > cutoff) return 0;
                } else {
                        if (4*dji*bound > cutoff || 4*dlk*bound > cutoff) return 0;
                }
        }
        if (mode & 2) {
                const int *r[4] = {js, js, is, is};
                const int *c[4] = {ks, ls, ks, ls};
                for (int k = 0; k < 4; k++) {
                        const double d = CVHFblock_max(dm, n, r[k], c[k]);
                        if (d < 0. || d*bound > cutoff) return 0;
                }
        }
        return 1;
}
#endif
