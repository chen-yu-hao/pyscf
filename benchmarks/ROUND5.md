# Round 5: exact-screen MGGA grid coalescing (experimental, default OFF)

Parent: round-three branch, calculation baseline b5b4f1c711b63c1dbb4fb409e979f53852601eca.
The full round-three archive 41058156813 was independently reopened: 23 cases,
184 converged records, unchanged energy/XC/J/K gates passed. Its MGGA-only SCF
speedups at n=1/2/4, t=2 were 1.0654/1.0580/1.0329x. JK-only n=4/8 were
1.0015/1.0002x, not persuasive gains. Round four remains a separate experiment.

## Algorithm
`PYSCF_DFT_MGGA_COALESCE=1` joins at most four adjacent 56-point sparse MGGA
blocks, only if the existing active-shell and pair/bin predicates are identical.
Identical byte rows are a fast path, not a requirement. The per-call pair mask
is unchanged. Existing grid order, grid points, weights, cutoff, basis and SCF
convergence criteria are untouched. Each group stays inside the original thread
partition. Packing grows at most to 224 points; the original 256-MiB workspace
cap still applies, with original-size packing when a larger pack cannot fit.
Dense LDA/GGA/MGGA and sparse LDA/GGA retain their original paths.

This changes floating-point summation grouping, not the mathematical terms.
Bitwise identity is NOT promised. Keep energy <=1e-8 Eh and common-density
Ne/Exc/Vxc/J/K atol=1e-9, rtol=1e-10. Do not relax old regression assertions.

## Inspiration and scope
Public work on GauXC/NWChemEx motivates reducing small matrix-operation overhead
and respecting local sparsity (Williams-Young et al., 2020,
doi:10.3389/fchem.2020.581058). This is an independent CPU implementation, not a
GauXC integration, copied third-party kernel, or GPU performance claim.

## Evidence and limits
Local standalone C: 945 inherited input/thread cases, two asymmetric fallbacks,
and 1,155 additional coalescing/boundary cases passed. Extended cases include
signed/zero weights, changed/equivalent masks, negative derivative strides,
ragged tiles and 1/2/4 threads. Maximum new-suite C A/B difference 3.41e-13.
Local microbenchmarks were noisy and include regressions; no molecular speedup
is inferred. The opt-in remains OFF until matched real SCF measurements justify
it. Native PySCF tests and molecular acceptance must run on Actions.

## Controlled experiment
Both variants: portable libcint, dense BLAS ON, round-three SIMD packing ON,
NUMA OFF, JK precheck OFF, round-one tile screening OFF, AO cache OFF, BLAS=1.
Only the candidate enables coalescing. Use the same O2/no-fast-math/no-FP-contract
main build flags. Integral dependencies use their own recorded Release flags.

Formal suite: MN15 n=4/8 at 1/2/4 threads (6 cases), MN15 n=2,t=2 dense control,
and PBE n=4,t=2 GGA control. Total 8 cases /64 records; per variant three fresh,
uninstrumented ordinary SCFs and one separate profile. Early MN15 n=4,t=2
smoke (4 records) is not included. All use def2-TZVPP, grid level 6.

C diagnostics are OFF in ordinary SCF and profile timings. Counts are collected
only in the subsequent common-density XC check, including entirely screened
batches. They establish actual grouping, NOT whole-SCF speedup. Profile C-call
counts, in contrast, cover only the profiled SCF. Stage time comparisons are
unbiased by these C counters, although ordinary Python profiling overhead remains.

Keep every PR draft; do not auto-merge or replace DFT. Preserve raw timings,
array files, build provenance and adverse performance cases.
