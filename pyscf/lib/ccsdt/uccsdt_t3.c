/* Copyright 2026 The PySCF Developers. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 */

#include <stdint.h>

static int64_t pair_index(int64_t i, int64_t j, int64_t n, int *sign)
{
    if (i == j) { *sign = 0; return 0; }
    if (i > j) { int64_t x = i; i = j; j = x; *sign = -*sign; }
    return i * (2*n-i-1)/2 + j-i-1;
}

static int64_t triple_index(int64_t i, int64_t j, int64_t k,
                            int64_t n, int *sign)
{
    if (i == j || j == k || i == k) { *sign = 0; return 0; }
    if (i > j) { int64_t x = i; i = j; j = x; *sign = -*sign; }
    if (j > k) { int64_t x = j; j = k; k = x; *sign = -*sign; }
    if (i > j) { int64_t x = i; i = j; j = x; *sign = -*sign; }
    return n*(n-1)*(n-2)/6 - (n-i)*(n-i-1)*(n-i-2)/6
           + (n-i-1)*(n-i-2)/2 - (n-j)*(n-j-1)/2 + k-j-1;
}

/* Every requested tile is tightly packed, including either minority-spin
 * dimension. Unlike the legacy unpacker, no full minority slab is required. */
void UCCSDT_t3_gather(double *restrict out, const double *restrict packed,
                      int mixed, const int64_t *restrict dims,
                      const int64_t *restrict start, const int64_t *restrict size,
                      const int64_t *restrict packed_stride)
{
    const int64_t no = dims[0], nv = dims[mixed ? 2 : 3];
    const int64_t outer = size[0]*size[1];
#pragma omp parallel for schedule(static)
    for (int64_t ij = 0; ij < outer; ++ij) {
        const int64_t i = start[0] + ij/size[1];
        const int64_t j = start[1] + ij%size[1];
        double *dst = out + ij*size[2]*size[3]*size[4]*size[5];
        if (mixed) {
            int so = 1;
            const int64_t oi = pair_index(i, j, no, &so);
            for (int64_t a = start[2]; a < start[2]+size[2]; ++a)
                for (int64_t b = start[3]; b < start[3]+size[3]; ++b) {
                    int s = so;
                    const int64_t vi = pair_index(a, b, nv, &s);
                    for (int64_t k = start[4]; k < start[4]+size[4]; ++k)
                        for (int64_t c = start[5]; c < start[5]+size[5]; ++c)
                            *dst++ = s ? s*packed[oi*packed_stride[0]+vi*packed_stride[1]
                                            +k*packed_stride[2]+c*packed_stride[3]] : 0.0;
                }
        } else {
            for (int64_t k = start[2]; k < start[2]+size[2]; ++k) {
                int so = 1;
                const int64_t oi = triple_index(i, j, k, no, &so);
                for (int64_t a = start[3]; a < start[3]+size[3]; ++a)
                    for (int64_t b = start[4]; b < start[4]+size[4]; ++b)
                        for (int64_t c = start[5]; c < start[5]+size[5]; ++c) {
                            int s = so;
                            const int64_t vi = triple_index(a, b, c, nv, &s);
                            *dst++ = s ? s*packed[oi*packed_stride[0]+vi*packed_stride[1]] : 0.0;
                        }
            }
        }
    }
}

static const int perm3[6][3] = {{0,1,2}, {0,2,1}, {1,0,2},
                               {1,2,0}, {2,0,1}, {2,1,0}};
static const int sign3[6] = {1,-1,-1,1,1,-1};

/* Sum A(occupied) A(virtual) of an unantisymmetrized residual tile directly
 * into unique storage. Each thread owns complete packed rows; tiles may
 * intersect only some permutations, without atomics or double counting. */
void UCCSDT_t3_pack(double *restrict packed, const double *restrict raw,
                    int mixed, const int64_t *restrict dims,
                    const int64_t *restrict start, const int64_t *restrict size,
                    const int64_t *restrict unused)
{
    (void)unused;
    const int64_t no = dims[0], nv = dims[mixed ? 2 : 3];
    const int64_t nvp = mixed ? nv*(nv-1)/2 : nv*(nv-1)*(nv-2)/6;
    int64_t stride[6];
    stride[5] = 1;
    for (int p = 4; p >= 0; --p) stride[p] = stride[p+1]*size[p+1];
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < no-1; ++i) {
        for (int64_t j = i+1; j < no; ++j) {
            if (mixed) {
                int dummy = 1;
                const int64_t oi = pair_index(i, j, no, &dummy);
                for (int64_t a = 0; a < nv-1; ++a)
                    for (int64_t b = a+1; b < nv; ++b) {
                        const int64_t vi = pair_index(a, b, nv, &dummy);
                        for (int po = 0; po < 2; ++po) {
                            const int64_t u = (po ? j : i) - start[0];
                            const int64_t v = (po ? i : j) - start[1];
                            if (u < 0 || u >= size[0] || v < 0 || v >= size[1]) continue;
                            for (int pv = 0; pv < 2; ++pv) {
                                const int64_t x = (pv ? b : a) - start[2];
                                const int64_t y = (pv ? a : b) - start[3];
                                if (x < 0 || x >= size[2] || y < 0 || y >= size[3]) continue;
                                const double s = po == pv ? 1.0 : -1.0;
                                const double *src = raw+u*stride[0]+v*stride[1]+x*stride[2]+y*stride[3];
                                for (int64_t k = 0; k < size[4]; ++k) {
                                    double *dst = packed+((oi*nvp+vi)*dims[4]+k+start[4])*dims[5]+start[5];
                                    for (int64_t c = 0; c < size[5]; ++c)
                                        dst[c] += s*src[k*size[5]+c];
                                }
                            }
                        }
                    }
            } else {
                for (int64_t k = j+1; k < no; ++k) {
                    const int64_t occ[3] = {i,j,k};
                    int dummy = 1;
                    const int64_t oi = triple_index(i,j,k,no,&dummy);
                    for (int64_t a = 0; a < nv-2; ++a)
                        for (int64_t b = a+1; b < nv-1; ++b)
                            for (int64_t c = b+1; c < nv; ++c) {
                                const int64_t vir[3] = {a,b,c};
                                const int64_t vi = triple_index(a,b,c,nv,&dummy);
                                double sum = 0.0;
                                for (int po = 0; po < 6; ++po) {
                                    int64_t offset_o = 0;
                                    int valid = 1;
                                    for (int p = 0; p < 3; ++p) {
                                        const int64_t x = occ[perm3[po][p]]-start[p];
                                        if (x < 0 || x >= size[p]) { valid = 0; break; }
                                        offset_o += x*stride[p];
                                    }
                                    if (!valid) continue;
                                    for (int pv = 0; pv < 6; ++pv) {
                                        int64_t offset = offset_o;
                                        valid = 1;
                                        for (int p = 0; p < 3; ++p) {
                                            const int64_t x = vir[perm3[pv][p]]-start[p+3];
                                            if (x < 0 || x >= size[p+3]) { valid = 0; break; }
                                            offset += x*stride[p+3];
                                        }
                                        if (valid) sum += sign3[po]*sign3[pv]*raw[offset];
                                    }
                                }
                                packed[oi*nvp+vi] += sum;
                            }
                }
            }
        }
    }
}
