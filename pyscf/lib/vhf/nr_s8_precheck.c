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
#include "cint.h"
#include "optimizer.h"
#include "nr_s8_block_bound.h"

/* Profiling only: access/reset after all worker threads have joined. */
static unsigned long long bound_counts[2];
void CVHFround3_bound_stats(unsigned long long *out, int reset)
{
        out[0] = bound_counts[0];
        out[1] = bound_counts[1];
        if (reset) bound_counts[0] = bound_counts[1] = 0;
}

static int reject(CVHFOpt *opt, int mode, int *is, int *js, int *ks, int *ls)
{
        const char *flag = getenv("PYSCF_JK_BLOCK_BOUND");
        if (!opt || !flag || flag[0] != '1' || flag[1] != '\0') return 0;
        const int rejected = CVHFs8_block_reject(opt->q_cond, opt->dm_cond, opt->nbas,
                                  opt->direct_scf_cutoff, mode, is, js, ks, ls);
        const char *stats = getenv("PYSCF_JK_BLOCK_BOUND_STATS");
        if (stats && stats[0] == '1' && stats[1] == '\0') {
#pragma omp atomic update
                bound_counts[0]++;
                if (rejected) {
#pragma omp atomic update
                        bound_counts[1]++;
                }
        }
        return rejected;
}

/* Only nr_direct.c is compiled with the block-screen aliases in CMake.
 * The legacy optimizer symbols and all leaf screens remain unchanged.
 * Unsupported/custom callbacks never reach these wrappers.
 */
int CVHFnrs8_precheck_block(CVHFOpt *opt, int *is, int *js, int *ks, int *ls)
{
        return reject(opt, 3, is, js, ks, ls) ? 0 :
                CVHFnrs8_prescreen_block(opt, is, js, ks, ls);
}
int CVHFnrs8_vj_precheck_block(CVHFOpt *opt, int *is, int *js, int *ks, int *ls)
{
        return reject(opt, 1, is, js, ks, ls) ? 0 :
                CVHFnrs8_vj_prescreen_block(opt, is, js, ks, ls);
}
int CVHFnrs8_vk_precheck_block(CVHFOpt *opt, int *is, int *js, int *ks, int *ls)
{
        return reject(opt, 2, is, js, ks, ls) ? 0 :
                CVHFnrs8_vk_prescreen_block(opt, is, js, ks, ls);
}
