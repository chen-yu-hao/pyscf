# Title: Improve sparse AO-pair contraction work partitioning

Branch: `DFT-sparse-tiles`. Head: `38d820b97fa81cba1fa4e8a23d41b547cffc1bac`. Direct base: `44313dfb457b93358918c391fc66285fa44153ef`.

Use smaller output tiles and finer dynamic scheduling while retaining shell, bin and pair screening. Allocate scratch using actual shell partition sizes so wide shells remain supported.

Default grids, cutoffs, FP64 precision and public interfaces are preserved.

2 focused sparse-contraction tests passed.

Measurement: NumInt.nr_rks/nr_uks with the same frozen DM and grid; includes AO/density/XC/matrix assembly and excludes grid construction, J/K and SCF. Three fresh process repeats, one warmup and three timed calls per process, OMP as listed and BLAS one; values are medians of process medians. Shared-host observations do not establish universal acceleration.

| Case | Functional | Threads | Parent / s | This head / s | Speed ratio |
|---|---|---:|---:|---:|---:|
| water12-rks | LDA,VWN | 8 | 0.385259 | 0.264725 | 1.455× |
| water12-rks | PBE | 8 | 0.663644 | 0.474381 | 1.399× |
| water12-rks | MN15 | 8 | 1.682907 | 1.090203 | 1.544× |
| water11oh-uks | LDA,VWN | 8 | 0.702763 | 0.466487 | 1.507× |
| water11oh-uks | PBE | 8 | 1.138968 | 0.719364 | 1.583× |
| water11oh-uks | MN15 | 8 | 3.045557 | 1.935390 | 1.574× |
| water20-rks | MN15 | 24 | 7.635249 | 6.353869 | 1.202× |
| water19oh-uks | MN15 | 24 | 14.443783 | 11.726608 | 1.232× |

Every measured cell passed its full-array checks; parent and candidate timing ranges are retained in the evidence CSV. No slower cell from this retained scope was removed.

Reproducible scripts, all sample ranges, source identities and numerical checks: [evidence](https://github.com/chen-yu-hao/pyscf/tree/DFT-speed-evidence/benchmarks/dft-speed-selection).

OpenAI Codex substantially assisted implementation, tests, benchmark automation, code review, and documentation. A human must understand and review the code, edit the factual draft in their own voice, and submit the PR under AI_POLICY.md.
