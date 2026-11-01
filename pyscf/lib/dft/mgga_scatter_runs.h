/* Copyright 2026 The PySCF Developers. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0.
 */
#ifndef PYSCF_MGGA_SCATTER_RUNS_H
#define PYSCF_MGGA_SCATTER_RUNS_H

/* Scatter a lower product tile, preserving each output element's additions.
 * An entire shell row shares the pair/bin decision. Compute its accepted
 * column runs once, not once per AO. Adjacent accepted runs can be joined only
 * when BOTH product columns and destination AO indices are consecutive.
 * No grid, active AO, mask, BLAS call, reduction order, or cutoff is changed.
 * Local storage is two AO_TILE-byte arrays, included in the workspace budget.
 */
static void VXCmgga_scatter_runs(double *partial, const double *product,
                                 const int *active_ao, const int *active_shell,
                                 const uint8_t *screen, const uint8_t *pair_mask,
                                 const int *ao_loc, int nao, int nbas, int nbins,
                                 int ib, int ni, int jb, int nj)
{
        uint8_t starts[AO_TILE], ends[AO_TILE];
        for (int i = 0; i < ni; ) {
                const int ish = active_shell[ib+i];
                const int iend = MIN(ni, i+ao_loc[ish+1]-active_ao[ib+i]);
                const int stop_max = MIN(nj, ib+iend-jb);
                int nruns = 0;
                for (int j = 0; j < stop_max; ) {
                        const int jsh = active_shell[jb+j];
                        const int end = MIN(stop_max, j+ao_loc[jsh+1]-active_ao[jb+j]);
                        if (pair_mask[(size_t)ish*nbas+jsh] &&
                            screen[ish]+screen[jsh] >= nbins) {
                                if (nruns > 0 && ends[nruns-1] == j &&
                                    active_ao[jb+j] == active_ao[jb+j-1]+1) {
                                        ends[nruns-1] = (uint8_t)end;
                                } else {
                                        starts[nruns] = (uint8_t)j;
                                        ends[nruns++] = (uint8_t)end;
                                }
                        }
                        j = end;
                }
                for (int row = i; row < iend; row++) {
                        const int stop = MIN(nj, ib+row-jb+1);
                        double *dst = partial + (size_t)active_ao[ib+row]*nao;
                        const double *src = product + (size_t)row*nj;
                        for (int r = 0; r < nruns && starts[r] < stop; r++) {
                                const int begin = starts[r];
                                const int end = MIN((int)ends[r], stop);
                                const int ao_j = active_ao[jb+begin]-begin;
#pragma GCC ivdep
                                for (int j = begin; j < end; j++) {
                                        dst[ao_j+j] += src[j];
                                }
                        }
                }
                i = iend;
        }
}
#endif
