# Title: Preserve grid grouping order with stable integer keys

Branch: `DFT-grid`. Head: `3721140b263c8a974fe9ef97e7ccdfe97be7d775`. Direct base: `44313dfb457b93358918c391fc66285fa44153ef`.

Replace multi-column unique grouping with stable sorting of bounded mixed-radix keys. The point order is identical, including ties; integer overflow falls back to the original grouping.

Default grids, cutoffs, FP64 precision and public interfaces are preserved.

10 grid tests passed.

Measurement: arg_group_grids only on the same shuffled 439,040-point fixture; this is not a whole-SCF speedup. Three fresh process repeats, one warmup and three timed calls per process, OMP as listed and BLAS one; values are medians of process medians. Shared-host observations do not establish universal acceleration.

| Case | Functional | Threads | Parent / s | This head / s | Speed ratio |
|---|---|---:|---:|---:|---:|
| water20-rks | grid grouping | 1 | 0.672588 | 0.061314 | 10.970× |

Every measured cell passed its full-array checks; parent and candidate timing ranges are retained in the evidence CSV. No slower cell from this retained scope was removed.

Reproducible scripts, all sample ranges, source identities and numerical checks: [evidence](https://github.com/chen-yu-hao/pyscf/tree/DFT-speed-evidence/benchmarks/dft-speed-selection).

OpenAI Codex substantially assisted implementation, tests, benchmark automation, code review, and documentation. A human must understand and review the code, edit the factual draft in their own voice, and submit the PR under AI_POLICY.md.
