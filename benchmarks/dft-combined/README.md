# Three combined DFT optimizations: measurements of the submitted code

Code branch: [DFT-combined](https://github.com/chen-yu-hao/pyscf/tree/DFT-combined).

Baseline: `44313dfb457b93358918c391fc66285fa44153ef` (upstream master snapshot). Combined: `a8cbddc874d7846dedfdc482944fc299d76f2409` (the pushed and measured commit).

The branch combines stable integer grid grouping, smaller sparse AO-pair output tiles, and sparse DM density contractions. It contains three commits and changes seven files (461 insertions, 7 deletions). There are no MO-streaming, LibXC, AO-derivative, potential-kernel or response-cache changes.

**Acceptance:** 44/44 numerical unit tests; 48/48 XC processes, 6/6 sorting processes, and 48/48 grid-build/SCF processes accepted. All 102 measured processes passed native-library identity, source commit, import location and thread-count checks. DFT lint passed with the upstream CI rules and recursive test-directory exclusions.

## Test conditions

Local shared host, Intel Xeon Platinum 8176. Performance processes ran serially, on physical CPUs 5–12 at 8 threads and 5–27 plus 3 at 24 threads; CPU 4 and its SMT sibling 60 were excluded after the utilization check. One BLAS thread, passive OpenMP wait policy, GCC 11.3.1, CMake 4.3.1, LibXC 7.0.0, Release `-O3 -DNDEBUG -msse3 -fopenmp`, no fast-math.

Every comparison has three independent fresh processes per version with alternating version order. XC, sorting and grid-build processes have one excluded warmup and three timed calls; the table uses the median of the three process medians. SCF has one calculation per process. Ratios are baseline/combined; values below 1 indicate a slowdown. Min/max ranges and RSS are retained in the acceptance JSON and CSV. Three repeats are descriptive observations, not confidence intervals or a universal performance guarantee.

## Fixed-DM XC-only integration

`nr_rks`/`nr_uks` includes AO evaluation, density, LibXC and potential assembly; it excludes grid construction, J/K and SCF. The same frozen coordinates, weights, screening indices and core-Hamiltonian-projector DMs are used by both versions. These are not converged SCF densities. Level-2 pruned grids have 263424/258056 points for the 288/283-AO cases and 439040/433672 points for the 1160/1146-AO cases.

| Case | Functional | Threads | Baseline (s) | Combined (s) | Speedup | RSS base/new (MiB) |
|---|---|---:|---:|---:|---:|---:|
| water12-rks | LDA,VWN | 8 | 0.5519 | 0.2371 | 2.328x | 403.7/253.9 |
| water12-rks | PBE | 8 | 0.9435 | 0.5123 | 1.842x | 948.5/823.1 |
| water12-rks | MN15 | 8 | 2.2828 | 0.8599 | 2.655x | 1230.4/831.9 |
| water11oh-uks | LDA,VWN | 8 | 0.9518 | 0.3909 | 2.435x | 407.0/260.1 |
| water11oh-uks | PBE | 8 | 1.5586 | 0.7603 | 2.050x | 949.7/828.9 |
| water11oh-uks | MN15 | 8 | 4.1717 | 1.3704 | 3.044x | 1238.8/845.1 |
| water20-rks | MN15 | 24 | 6.9516 | 2.4932 | 2.788x | 1670.5/1149.8 |
| water19oh-uks | MN15 | 24 | 14.2166 | 5.2252 | 2.721x | 1712.8/1223.8 |

Largest absolute baseline/combined differences: electrons 1.421e-14, Exc 2.842e-14 Ha, Vxc 1.044e-14. Gates: 1e-8 electrons, 1e-8 Ha and 1e-9 Vxc.

## Grid sorting and complete grid construction

Sorting is `arg_group_grids` on a fixed permutation of 439040 grid points at one thread; output indices match exactly. Grid construction includes `Grids` creation and `build(with_non0tab=True)` for water20/cc-pVTZ, level 2. All generated coordinates, weights and screening indices exactly match the frozen baseline fixture.

| Boundary | Threads | Baseline (s) | Combined (s) | Speedup |
|---|---:|---:|---:|---:|
| Sorting only | 1 | 0.8222 | 0.0922 | 8.917x |
| Complete grid build | 8 | 1.5971 | 0.7481 | 2.135x |
| Complete grid build | 24 | 1.2251 | 0.4795 | 2.555x |

## Complete SCF

Water4 RKS/cc-pVDZ (96 AOs) and O2 UKS/cc-pVTZ (60 AOs), eight threads. Level-3 nwchem-pruned grids, no density-based grid removal, grid cutoff 1e-15, numerical-integration cutoff 1e-13, conventional direct J/K with tolerance 1e-13, no density fitting, Hückel initial guess, no symmetry, 1024 MB memory setting, DIIS space 12, maximum 100 cycles. Stock PySCF convergence uses energy tolerance 1e-10 Ha and orbital-gradient tolerance 1e-6 including the extra cycle. Normal automatic DM/MO route selection is retained.

Kernel timing covers `mf.kernel()` including grid setup and J/K. Process timing covers parent subprocess start-to-exit, including imports, normal output and the common provenance/validation harness overhead; it is not mixed with kernel timing.

| Case | XC | Kernel base/new (s) | Kernel speedup | Process base/new (s) | Process speedup | Cycles base/new | RSS base/new (MiB) |
|---|---|---:|---:|---:|---:|---|---:|
| water4 | LDA,VWN | 4.635/4.444 | 1.043x | 6.027/5.877 | 1.026x | [8, 8, 8]/[8, 8, 8] | 234.8/235.6 |
| water4 | PBE | 5.447/5.347 | 1.019x | 6.879/6.779 | 1.015x | [8, 8, 8]/[8, 8, 8] | 405.4/398.7 |
| water4 | MN15 | 8.653/8.598 | 1.006x | 10.136/10.136 | 1.000x | [9, 9, 9]/[9, 9, 9] | 442.6/428.5 |
| o2 | LDA,VWN | 1.561/1.591 | 0.981x | 3.070/3.020 | 1.016x | [8, 8, 8]/[8, 8, 8] | 150.3/131.6 |
| o2 | PBE | 1.734/1.703 | 1.018x | 3.271/3.121 | 1.048x | [8, 8, 8]/[8, 8, 8] | 176.5/176.6 |
| o2 | MN15 | 2.453/2.379 | 1.031x | 4.023/3.772 | 1.066x | [8, 8, 8]/[8, 8, 8] | 192.0/208.0 |

Maximum SCF energy difference: 5.116e-13 Ha; maximum DM element difference: 4.392e-11. Gates are 1e-8 Ha and 1e-6 respectively. Grid coordinates, weights and masks match exactly. Every run is converged, with overlap electron counts checked to 1e-9.

## Interpretation and scope

The independently measured historical speedups must not be multiplied. The new tables measure their actual combination. Sorting gains apply only to sorting; fixed-DM XC gains cannot be substituted for normal SCF speedups because SCF can use the MO route, includes J/K and diagonalization, and may require different iteration counts. Large-system results here cover XC-only integration; the complete-SCF cases are small. Peak RSS is a Linux process high-water mark including inputs and warmup, not a phase-local allocation measurement. Shared-host noise remains possible despite core affinity.

These three changes make no response-cache memory claim; TDDFT/CPHF are outside this combined-branch benchmark.

## Artifacts and reproduction

`comparison.csv` contains every timing boundary; `results.json` includes accepted comparisons, ranges and numerical errors. `protocols/`, `builds/`, `machine.json`, `tests.json`, `tests.log`, `lint-ci.json` and per-run JSON files retain provenance. See `REPRODUCE.md` for commands. Raw numerical output arrays remain in the local measurement directories and can be regenerated by these scripts.

OpenAI Codex substantially assisted the code combination, testing, benchmark automation and documentation. A human author must understand and review the kernels and personally submit any upstream PR under `AI_POLICY.md`. No upstream PR was created.
