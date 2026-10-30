/* Copyright 2026 The PySCF Developers. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0.
 * Optional in-memory semi-direct ERI reuse. No integral approximation.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include "cint.h"
#include "nr_direct.h"

typedef int (*ERIIntor)(double *, int *, int *, int *, int,
                        int *, int, double *, CINTOpt *, double *);
extern int int2e_sph(double *, int *, int *, int *, int,
                     int *, int, double *, CINTOpt *, double *);

typedef struct {
        _Atomic unsigned state; /* 0 never used, 1 publishing, 2 immutable, 3 reusable tombstone */
        uint64_t key;
        size_t offset, count;
        int value;
} ERIEntry;

typedef struct {
        int dimension, angular;
        unsigned nprim;
} ERIShellInfo;

typedef struct ERICache {
        struct ERICache *next;
        CINTOpt *owner;
        size_t refs, slots, limit, bytes, arena_size;
        int natm, nbas, nenv;
        unsigned min_work;
        int fast_lookup;
        ERIShellInfo *shells;
        int *atm, *bas, *ao_loc;
        const int *source_atm, *source_bas;
        const double *source_env;
        double *env;
        ERIEntry *entries;
        unsigned char *arena;
        _Atomic size_t used;
        _Atomic size_t filled;
        _Atomic int diagnostics;
        _Atomic uint64_t stats[4]; /* considered, hit, miss, cost bypass */
        /* Diagnostic only: probes, empty stops, reserve failures, tombstone reuse. */
        _Atomic uint64_t lookup_stats[4];
} ERICache;

static ERICache *registry;
static atomic_flag registry_mutex = ATOMIC_FLAG_INIT;
static void lock_registry(void)
{
        while (atomic_flag_test_and_set_explicit(&registry_mutex, memory_order_acquire)) {}
}
static void unlock_registry(void)
{
        atomic_flag_clear_explicit(&registry_mutex, memory_order_release);
}
static void destroy_cache(ERICache *c)
{
        free(c->atm); free(c->bas); free(c->ao_loc); free(c->env);
        free(c->entries); free(c->arena); free(c->shells); free(c);
}
static void drop_reference(ERICache *c)
{
        lock_registry();
        int last = --c->refs == 0;
        unlock_registry();
        if (last) destroy_cache(c);
}

/* Retained allocation includes index, molecule snapshot and data arena, not RSS.
 * A cache belongs to one live CINTOpt. The caller retains that optimizer until
 * release. Keys are ordered shell quartets; no tensor permutation is assumed.
 */
static void *create_cache(void *owner, int natm, int nbas, int nenv,
                          const int *atm, const int *bas, const double *env,
                          const int *ao_loc, size_t max_bytes, unsigned min_work, int fast_lookup)
{
        if (!owner || natm <= 0 || natm > 1000000 || nbas <= 0 || nbas > 65535 ||
            nenv <= 0 || max_bytes < 65536 || !atm || !bas || !env || !ao_loc) return NULL;
        size_t snapshot = sizeof(ERICache) + (size_t)natm*ATM_SLOTS*sizeof(int) +
                (size_t)nbas*BAS_SLOTS*sizeof(int) + (size_t)nenv*sizeof(double) +
                ((size_t)nbas+1)*sizeof(int) +
                (fast_lookup ? (size_t)nbas*sizeof(ERIShellInfo) : 0);
        if (snapshot >= max_bytes || max_bytes-snapshot < 65536) return NULL;
        size_t slots = 64;
        while (slots < (1U<<20) && slots <= (max_bytes-snapshot)/512) slots *= 2;
        size_t index_bytes = slots*sizeof(ERIEntry);
        if (index_bytes > (max_bytes-snapshot)/2) return NULL;
        ERICache *c = calloc(1, sizeof(*c));
        if (!c) return NULL;
        c->owner = owner; c->refs = 1; c->slots = slots;
        c->source_atm = atm; c->source_bas = bas; c->source_env = env;
        c->limit = c->bytes = max_bytes;
        c->arena_size = max_bytes-snapshot-index_bytes;
        c->natm = natm; c->nbas = nbas; c->nenv = nenv; c->min_work = min_work;
        c->fast_lookup = fast_lookup;
        c->atm = malloc((size_t)natm*ATM_SLOTS*sizeof(int));
        c->bas = malloc((size_t)nbas*BAS_SLOTS*sizeof(int));
        c->ao_loc = malloc(((size_t)nbas+1)*sizeof(int));
        c->env = malloc((size_t)nenv*sizeof(double));
        c->entries = calloc(slots, sizeof(ERIEntry));
        c->arena = malloc(c->arena_size);
        if (fast_lookup) c->shells = malloc((size_t)nbas*sizeof(ERIShellInfo));
        if (!c->atm || !c->bas || !c->ao_loc || !c->env || !c->entries || !c->arena || (fast_lookup && !c->shells)) {
                destroy_cache(c); return NULL;
        }
        memcpy(c->atm, atm, (size_t)natm*ATM_SLOTS*sizeof(int));
        memcpy(c->bas, bas, (size_t)nbas*BAS_SLOTS*sizeof(int));
        memcpy(c->env, env, (size_t)nenv*sizeof(double));
        memcpy(c->ao_loc, ao_loc, ((size_t)nbas+1)*sizeof(int));
        if (fast_lookup) {
                for (int i = 0; i < nbas; i++) {
                        c->shells[i].dimension = ao_loc[i+1] - ao_loc[i];
                        c->shells[i].angular = bas[(size_t)i*BAS_SLOTS+ANG_OF];
                        c->shells[i].nprim = (unsigned)bas[(size_t)i*BAS_SLOTS+NPRIM_OF];
                }
        }
        atomic_init(&c->used, 0); atomic_init(&c->filled, 0);
        atomic_init(&c->diagnostics, 0);
        for (int i = 0; i < 4; i++) {
                atomic_init(&c->stats[i], 0);
                atomic_init(&c->lookup_stats[i], 0);
        }
        for (size_t i = 0; i < slots; i++) atomic_init(&c->entries[i].state, 0);
        lock_registry();
        for (ERICache *p = registry; p; p = p->next) {
                if (p->owner == owner) { unlock_registry(); destroy_cache(c); return NULL; }
        }
        c->next = registry; registry = c;
        unlock_registry();
        return c;
}

/* Preserve the original constructor/ABI. The new policy is explicitly opt-in. */
void *CVHFeri_cache_create(void *owner, int natm, int nbas, int nenv,
                          const int *atm, const int *bas, const double *env,
                          const int *ao_loc, size_t max_bytes, unsigned min_work)
{
        return create_cache(owner, natm, nbas, nenv, atm, bas, env, ao_loc,
                            max_bytes, min_work, 0);
}
void *CVHFeri_cache_create_fast(void *owner, int natm, int nbas, int nenv,
                               const int *atm, const int *bas, const double *env,
                               const int *ao_loc, size_t max_bytes, unsigned min_work)
{
        return create_cache(owner, natm, nbas, nenv, atm, bas, env, ao_loc,
                            max_bytes, min_work, 1);
}
/* Read only between driver calls, with the handle kept alive by its scope. */
void CVHFeri_cache_lookup_stats(void *ptr, uint64_t *out)
{
        ERICache *c = ptr;
        for (int i = 0; i < 4; i++)
                out[i] = atomic_load_explicit(&c->lookup_stats[i], memory_order_relaxed);
        out[4] = c->fast_lookup;
}

void CVHFeri_cache_release(void *ptr)
{
        if (!ptr) return;
        lock_registry();
        ERICache **p = &registry;
        while (*p && *p != ptr) p = &(*p)->next;
        ERICache *c = *p;
        if (c) *p = c->next;
        unlock_registry();
        if (c) drop_reference(c); /* active calls retain their own references */
}

/* Read between calls; diagnostics are OFF during normal SCF timings. */
void CVHFeri_cache_stats(void *ptr, uint64_t *out, int diagnostics, int reset)
{
        ERICache *c = ptr;
        for (int i = 0; i < 4; i++) {
                out[i] = atomic_load_explicit(&c->stats[i], memory_order_relaxed);
                if (reset) {
                        atomic_store_explicit(&c->stats[i], 0, memory_order_relaxed);
                        atomic_store_explicit(&c->lookup_stats[i], 0, memory_order_relaxed);
                }
        }
        out[4] = atomic_load_explicit(&c->used, memory_order_relaxed);
        out[5] = c->bytes; out[6] = c->limit;
        out[7] = atomic_load_explicit(&c->filled, memory_order_relaxed);
        if (diagnostics >= 0) atomic_store_explicit(&c->diagnostics, diagnostics != 0,
                                                    memory_order_relaxed);
}

static uint64_t hash_key(uint64_t x)
{
        x ^= x >> 30; x *= UINT64_C(0xbf58476d1ce4e5b9);
        x ^= x >> 27; x *= UINT64_C(0x94d049bb133111eb);
        return x ^ (x >> 31);
}
static void count(ERICache *c, int index, int enabled)
{
        if (enabled) atomic_fetch_add_explicit(&c->stats[index], 1, memory_order_relaxed);
}
static size_t reserve_bytes(ERICache *c, size_t n)
{
        size_t old = atomic_load_explicit(&c->used, memory_order_relaxed);
        do {
                if (n > c->arena_size-old) return SIZE_MAX;
        } while (!atomic_compare_exchange_weak_explicit(&c->used, &old, old+n,
                                                        memory_order_relaxed, memory_order_relaxed));
        return old;
}

/* This callback only replaces int2e_sph. Cache-size queries may pass opt=NULL. */
static int cached_int2e(double *out, int *dims, int *shls, int *atm, int natm,
                         int *bas, int nbas, double *env, CINTOpt *opt, double *work)
{
        if (!opt) return int2e_sph(out, dims, shls, atm, natm, bas, nbas, env, NULL, work);
        ERICache *c = (ERICache *)opt;
        if (!out || dims) return int2e_sph(out, dims, shls, atm, natm, bas, nbas, env, c->owner, work);
        int diag = atomic_load_explicit(&c->diagnostics, memory_order_relaxed);
        count(c, 0, diag);
        size_t elements = 1;
        uint64_t key = 0;
        unsigned work_est = 1;
        int angular = 0;
        if (c->fast_lookup) {
                /* Metadata is density independent and is invalidated by the
                 * existing whole-input identity/content guard in the driver.
                 * Cheap integrals avoid key and tensor-size work altogether. */
                for (int i = 0; i < 4; i++) {
                        int s = shls[i];
                        if (s < 0 || s >= c->nbas)
                                return int2e_sph(out, dims, shls, atm, natm, bas, nbas, env, c->owner, work);
                        const ERIShellInfo *m = c->shells+s;
                        if (work_est < c->min_work) {
                                if (m->nprim && work_est > UINT32_MAX/m->nprim) work_est = UINT32_MAX;
                                else work_est *= m->nprim;
                        }
                        angular += m->angular;
                }
                if (work_est < c->min_work && angular < 4) {
                        count(c, 3, diag);
                        return int2e_sph(out, dims, shls, atm, natm, bas, nbas, env, c->owner, work);
                }
                for (int i = 0; i < 4; i++) {
                        int s = shls[i], d = c->shells[s].dimension;
                        if (d <= 0 || elements > SIZE_MAX/(size_t)d)
                                return int2e_sph(out, dims, shls, atm, natm, bas, nbas, env, c->owner, work);
                        elements *= d;
                        key = (key<<16) | (unsigned)s;
                }
        } else {
        for (int i = 0; i < 4; i++) {
                int s = shls[i];
                if (s < 0 || s >= c->nbas) return int2e_sph(out, dims, shls, atm, natm, bas, nbas, env, c->owner, work);
                key = (key<<16) | (unsigned)s;
                int d = c->ao_loc[s+1]-c->ao_loc[s];
                if (d <= 0 || elements > SIZE_MAX/(size_t)d) return int2e_sph(out, dims, shls, atm, natm, bas, nbas, env, c->owner, work);
                elements *= d;
                unsigned nprim = bas[(size_t)s*BAS_SLOTS+NPRIM_OF];
                if (work_est < c->min_work) {
                        if (nprim && work_est > UINT32_MAX/nprim) work_est = UINT32_MAX;
                        else work_est *= nprim;
                }
                angular += bas[(size_t)s*BAS_SLOTS+ANG_OF];
        }
        }
        if ((work_est < c->min_work && angular < 4) || elements > SIZE_MAX/sizeof(double)) {
                count(c, 3, diag);
                return int2e_sph(out, dims, shls, atm, natm, bas, nbas, env, c->owner, work);
        }
        size_t start = hash_key(key) & (c->slots-1), empty = SIZE_MAX;
        unsigned claim_state = 0;
        /* In the fast policy a used slot NEVER returns to state zero. A truly
         * never-used slot proves the key is absent from the rest of this probe
         * sequence. A failed arena reservation must leave a tombstone: resetting
         * it to zero could hide an entry published beyond it by another thread.
         * Tombstones may be reused, but do not terminate the lookup. */
        for (size_t p = 0; p < 8; p++) {
                size_t i = (start+p) & (c->slots-1);
                ERIEntry *e = c->entries+i;
                unsigned state = atomic_load_explicit(&e->state, memory_order_acquire);
                if (diag) atomic_fetch_add_explicit(&c->lookup_stats[0], 1, memory_order_relaxed);
                if (state == 2 && e->key == key && e->count == elements) {
                        count(c, 1, diag);
                        if (e->value) memcpy(out, c->arena+e->offset, elements*sizeof(double));
                        return e->value;
                }
                if ((state == 0 || (c->fast_lookup && state == 3)) && empty == SIZE_MAX) {
                        empty = i; claim_state = state;
                }
                if (c->fast_lookup && state == 0) {
                        if (diag) atomic_fetch_add_explicit(&c->lookup_stats[1], 1, memory_order_relaxed);
                        break;
                }
        }
        count(c, 2, diag);
        int value = int2e_sph(out, dims, shls, atm, natm, bas, nbas, env, c->owner, work);
        if (empty == SIZE_MAX || value < 0) return value;
        size_t bytes = value ? elements*sizeof(double) : 0;
        if (bytes > c->arena_size-atomic_load_explicit(&c->used, memory_order_relaxed)) return value;
        ERIEntry *e = c->entries+empty;
        unsigned state = claim_state;
        if (!atomic_compare_exchange_strong_explicit(&e->state, &state, 1,
                                                     memory_order_acq_rel, memory_order_relaxed)) return value;
        size_t offset = reserve_bytes(c, bytes);
        if (offset == SIZE_MAX) {
                if (diag) atomic_fetch_add_explicit(&c->lookup_stats[2], 1, memory_order_relaxed);
                atomic_store_explicit(&e->state, c->fast_lookup ? 3 : 0, memory_order_release);
                return value;
        }
        if (diag && claim_state == 3)
                atomic_fetch_add_explicit(&c->lookup_stats[3], 1, memory_order_relaxed);
        e->key = key; e->count = elements; e->offset = offset; e->value = value;
        if (value) memcpy(c->arena+offset, out, bytes);
        atomic_fetch_add_explicit(&c->filled, 1, memory_order_relaxed);
        atomic_store_explicit(&e->state, 2, memory_order_release);
        return value;
}

/* CMake renames only the driver in nr_direct.c. Existing block/leaf screening,
 * contraction, scheduling and output reduction execute unchanged beneath us.
 * Unregistered optimizers and unsupported operators follow the original driver.
 */
void CVHFnr_direct_drv_uncached(int (*)(), void (*)(), JKOperator **,
                               double **, double **, int, int, int *, int *,
                               CINTOpt *, CVHFOpt *, int *, int, int *, int, double *);
void CVHFnr_direct_drv(int (*intor)(), void (*fdot)(), JKOperator **jkop,
                       double **dms, double **vjk, int n_dm, int ncomp,
                       int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                       CVHFOpt *vhfopt, int *atm, int natm, int *bas, int nbas, double *env)
{
        ERICache *c = NULL;
        if (ncomp == 1 && intor == (int (*)())int2e_sph && cintopt) {
                lock_registry();
                for (ERICache *p = registry; p; p = p->next) {
                        if (p->owner == cintopt && p->natm == natm && p->nbas == nbas &&
                            p->source_atm == atm && p->source_bas == bas && p->source_env == env &&
                            memcmp(p->atm, atm, (size_t)natm*ATM_SLOTS*sizeof(int)) == 0 &&
                            memcmp(p->bas, bas, (size_t)nbas*BAS_SLOTS*sizeof(int)) == 0 &&
                            memcmp(p->ao_loc, ao_loc, ((size_t)nbas+1)*sizeof(int)) == 0 &&
                            memcmp(p->env, env, (size_t)p->nenv*sizeof(double)) == 0) {
                                c = p; c->refs++; break;
                        }
                }
                unlock_registry();
        }
        CVHFnr_direct_drv_uncached(c ? (int (*)())cached_int2e : intor, fdot, jkop,
                dms, vjk, n_dm, ncomp, shls_slice, ao_loc,
                c ? (CINTOpt *)c : cintopt, vhfopt, atm, natm, bas, nbas, env);
        if (c) drop_reference(c);
}
