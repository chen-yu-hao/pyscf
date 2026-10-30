/* Copyright 2026 The PySCF Developers. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0.
 */
#ifndef PYSCF_MGGA_COALESCE_H
#define PYSCF_MGGA_COALESCE_H
#include <string.h>

/* Exact equivalence of the EXISTING shell-selection and pair/bin predicates.
 * No cutoff is introduced or relaxed. Equal byte rows are a cheap special case.
 * Different bins can also be merged if they select exactly the same shells and
 * pair contributions. The caller already checked pair_mask symmetry.
 */
static int VXCmgga_same_screen(const uint8_t *a, const uint8_t *b,
                               const uint8_t *pair_mask, int nbas, int nbins)
{
        if (memcmp(a, b, (size_t)nbas) == 0) return 1;
        int am = 0, bm = 0;
        for (int s = 0; s < nbas; s++) {
                am = MAX(am, a[s]);
                bm = MAX(bm, b[s]);
        }
        int amin = 256, bmin = 256;
        for (int i = 0; i < nbas; i++) {
                const int ai = a[i] != 0 && a[i] + am >= nbins;
                const int bi = b[i] != 0 && b[i] + bm >= nbins;
                if (ai != bi) return 0;
                if (ai) { amin = MIN(amin, a[i]); bmin = MIN(bmin, b[i]); }
        }
        /* Fast proof when every active bin pair passes the original threshold. */
        if (2*amin >= nbins && 2*bmin >= nbins) return 1;
        for (int i = 0; i < nbas; i++) {
                if (a[i] == 0 || a[i] + am < nbins) continue;
                for (int j = 0; j <= i; j++) {
                        if (a[j] != 0 && a[j] + am >= nbins &&
                            pair_mask[(size_t)i*nbas+j] &&
                            ((a[i] + a[j] >= nbins) !=
                             (b[i] + b[j] >= nbins))) return 0;
                }
        }
        return 1;
}

static int VXCmgga_merge_width(const uint8_t *screen,
                               const uint8_t *pairs, int nbas, int nbins,
                               int block, int end, int limit)
{
        int width = 1;
        const uint8_t *first = screen + (size_t)block*nbas;
        while (width < limit && width < end-block &&
               VXCmgga_same_screen(first, first+(size_t)width*nbas,
                                   pairs, nbas, nbins)) width++;
        return width;
}
#endif
