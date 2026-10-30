# Round 3: MN15 MGGA packing and direct J/K prechecks

Experimental, default OFF. Stacked on round 2 (PR #2), not a merge into DFT.

## Numerical work

`PYSCF_DFT_MGGA_PACK_SIMD=1` specializes the four-channel packing loop and uses
OpenMP SIMD. It keeps the same formulas, double precision, grid/AO tiles,
weights, masks, reduction and workspace budget. It applies to dense AND
screened MGGA calls. The older path remains selectable in the same binary.

`PYSCF_JK_BLOCK_BOUND=1` places a conservative maximum-based bound before the
existing density-dependent block screens. A block is rejected only when every
original shell predicate would reject it at the SAME cutoff. Otherwise the
original block predicate and original leaf predicates still run. There is no
persistent cache or new approximation; current and delta density bounds are
recomputed. CMake aliases only the three block-screen calls in nr_direct.c,
leaving optimizer.c and all legacy symbols intact. Custom callbacks do not use
these wrappers. `PYSCF_JK_BLOCK_BOUND_STATS=1` enables atomic profile-only counts;
read/reset those counts only after all worker threads have joined.

## Local checks actually executed before publication

945 standalone C input/thread combinations plus two asymmetric-mask fallback
checks passed. Against the round-two C kernel, the SIMD path was bitwise equal
in all these cases. Maximum independent NumPy-formula difference: 1.99e-13.
18,045 independent block-bound / original-leaf checks passed, 11,171 random
blocks rejected and no false rejections observed. These are not full PySCF
molecular regressions. A signed-Gram MGGA redesign was also tried but rejected:
it regressed on one tested size and is NOT included.

## Real molecular gate

The baseline is round-two source 5060bd524fa5e15ad0408fc80ddb3bc3d5cc6130,
with PYSCF_DFT_DENSE_BLAS=1 on BOTH sides; do not compare the candidate against
round-one defaults and credit round-two gains again. Both use NUMA off and
PYSCF_DFT_VMAT_TILE_SCREEN=0. Compilation uses matched dependencies and O2,
no-fast-math. No grid, basis, integral or SCF threshold changes.

The paired harness independently tests MGGA only, JK only, and both enabled.
It preserves the existing energy <=1e-8 Eh and XC atol=1e-9/rtol=1e-10 gates,
and also verifies direct J and K matrices on the SAME reference density at
atol=1e-9/rtol=1e-10, outside the SCF timer.

Acceptance: 18 standard combined cases (PBE/MN15, n=1/2/4, 1/2/4 threads),
3 MN15 MGGA-only cases (n=1/2/4, 2 threads), and 2 MN15 JK-only cases (n=4/8,
2 threads). Each has 3 ordinary fresh-process timings and a separate profile
per variant: 23 cases / 184 records, excluding the early two-case smoke.
No merge or default enablement until native regressions, all arrays, repeated
SCF timing and profile dispatch are independently audited. Larger n=8 means
472 AO for these geometries, not a many-socket scalability test.

The C dispatch counts cover profile SCF plus post-SCF XC verification. The
J/K block counters are sampled before the extra J/K verification, thus cover
profile SCF only. Ordinary timings have neither instrumented wrappers nor
atomic J/K counters. New filters must show real end-to-end gains, not merely
high rejection counts; they may cost more than they save on dense systems.
