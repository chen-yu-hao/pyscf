# Title: Fuse screened density-matrix contractions for molecular DFT

Branch: `DFT-density-dm`. Head: `a5d33360cd0fd9c16e48fed7e752690b98a78b61`. Direct base: `44313dfb457b93358918c391fc66285fa44153ef`.

Form density, gradient and kinetic-energy density from screened AO/DM blocks without repeated full AO-DM intermediates. Restrict each block sparsity decision to its own mask rows. The MO evaluator and nr_numint.c are unchanged from upstream; this PR contains no MO streaming optimization.

Default grids, cutoffs, FP64 precision and public interfaces are preserved.

32 tests passed: the complete test_numint module plus the fused-DM regressions. Source and native build were verified against this exact commit.

Measurement: NumInt.nr_rks/nr_uks on frozen core-Hamiltonian projector DMs (not converged SCF densities), identical grid coordinates/weights, LDA/VWN/PBE/MN15; no J/K or SCF in the timed interval. Three fresh process repeats, one warmup and three timed calls per process, OMP as listed and BLAS one; values are medians of process medians. Shared-host observations do not establish universal acceleration.

| Case | Functional | Threads | Parent / s | This head / s | Speed ratio |
|---|---|---:|---:|---:|---:|
| water12-rks | LDA,VWN | 8 | 0.498747 | 0.327228 | 1.524× |
| water12-rks | PBE | 8 | 0.912036 | 0.720445 | 1.266× |
| water12-rks | MN15 | 8 | 2.204895 | 1.418032 | 1.555× |
| water11oh-uks | LDA,VWN | 8 | 0.907806 | 0.571748 | 1.588× |
| water11oh-uks | PBE | 8 | 1.564873 | 1.248410 | 1.253× |
| water11oh-uks | MN15 | 8 | 4.013210 | 2.498919 | 1.606× |
| water20-rks | MN15 | 24 | 7.131991 | 3.764156 | 1.895× |
| water19oh-uks | MN15 | 24 | 13.625819 | 7.146858 | 1.907× |

Every measured cell passed its full-array checks; parent and candidate timing ranges are retained in the evidence CSV. No slower cell from this retained scope was removed.

Reproducible scripts, all sample ranges, source identities and numerical checks: [evidence](https://github.com/chen-yu-hao/pyscf/tree/DFT-speed-evidence/benchmarks/dft-speed-selection).

OpenAI Codex substantially assisted implementation, tests, benchmark automation, code review, and documentation. A human must understand and review the code, edit the factual draft in their own voice, and submit the PR under AI_POLICY.md.
