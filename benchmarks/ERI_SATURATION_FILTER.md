# Saturated ERI cache membership-filter experiment

This is a new opt-in experiment, not a default change or a release. Keep all PRs draft. Do not restart PR7's full SCF run or PR8's completed lookup replay.

## Evidence motivating the work
PR8 run 41104440996 / artifact 13001996286 completed. All 36 J/K-only records were independently reopened and passed the unchanged same-density and uncached-reference matrix gates. Candidate fast lookup saved about 0.6-1.3% versus its same-source control, with overlapping sample ranges in some cases. There is no new SCF energy or convergence claim. In the eight-water/64-MiB post-timing diagnostic, fast lookup still visited 27,234,607 index slots and missed 6,780,192 admitted requests. These diagnostics are not SCF hit rates.

## Implementation
`cached_eri(mf, max_mb=64, fast_lookup=True, saturation_filter=True)` enables a compact one-word blocked membership filter. Three bits are atomically ORed before publishing an immutable entry. No deletion clears filter bits. Possible hits still need the exact ordered key and element count. A negative result bypasses index probes only when the tensor would not fit in the remaining arena; the original integral is recomputed, never discarded. Zero integrals retain index admission without a second integral call. Concurrent publication not yet observed can at worst cause redundant recomputation, never an omitted contribution. Cold filling pays for filter publication, and filter storage is taken from the SAME budget, reducing payload capacity. Default `saturation_filter=False` allocates no filter. It is not a physical ERI screening approximation.

This is an independent implementation of standard membership filtering, not copied Parquet code or the Parquet wire format. Concept background: https://parquet.apache.org/docs/file-format/bloomfilter/ . Semi-direct recomputation background: https://nwchemgit.github.io/Density-Functional-Theory-for-Molecules.html . Neither project's benchmark figures predict this experiment's performance.

## Executed locally before publication
Legacy, fast and filtered policies each passed 18 synthetic-native configurations at 1/2/4 threads, normally and under UBSan. Seven targeted checks cover full-arena retained hits, negative bypass, forced false-positive lookup, zero-integral admission, reset without clearing membership, concurrent atomic publication and reference callback equality. 65,536 concurrent membership keys and 131,072 callback comparisons passed. Six variable-payload configurations add 1,572,864 cold/warm callback comparisons per normal/UBSan run. They contain both a no-benefit index-full case and data-saturated cases with roughly 82% fewer index probes. Those are operation counts, NOT molecular timings. An initial test assumed every tail was smaller than the smallest tensor; the actual hash admission disproved that assumption. It was corrected to check the true byte bound; no numerical gate was relaxed. Python, YAML and shell syntax checks passed. Native PySCF/molecular validation is delegated to the pinned workflow and is not preclaimed.

## Fixed comparison
Candidate computation: ac42c7bc809378284aa5b7853c9def634c8b83f1.
Parent computation: c10ab2ce464f60a37e74924e770f4110913369f6.
Same portable libcint/Libxc, same main strict-FP build, NUMA off, BLAS one thread, two OpenMP threads. No Qcint, native ISA, AO cache or grid coalescing. Reuse the accepted fifth-round density artifact and verify its SHA256.

Two cases (four/eight waters, def2-TZVPP reference, 64 MiB) x three variants x three fresh processes = 18 records. Variants: parent fast lookup, new source/filter off, new source/filter on. Each executes D,0.1D,0.01D and includes cold context/cache initialization. Diagnostics and independent uncached J/K verification are outside all timers. All J/K arrays must be finite and pass atol=1e-9,rtol=1e-10, with byte-identical fixed_dm and matching source/configuration/library provenance. Recompute medians and report regressions, noise, memory and filter activation, even if bypasses are zero. This is NOT full SCF, and energy<=1e-8 Eh plus Ne/Exc/Vxc/J/K gates remain required before promotion.
