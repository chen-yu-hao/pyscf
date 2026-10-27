# Full implementation retained as a cumulative PR stack

**90/90 full-SCF processes, 405/405 cross-version numerical pairs, and 18/18 CPKS processes accepted.**

Full branch: [DFT-full-stack](https://github.com/chen-yu-hao/pyscf/tree/DFT-full-stack), `ee16ee33a279cdcc6e0f45dcfaee3ce0bdcf17eb`. Its complete tracked tree is identical to the original full aggregate `72e5b26c91964acbf3186c5348f77a0f7faf44dd` (tree `47ea3d3ec8b07a1e0dbe4b3aaba63e1575fe4b1f`). All five core C files also exactly match the earlier AO64 full implementation. No optimization family was removed based on an isolated benchmark.

## Full-SCF endpoint

MN15; the same frozen Bohr geometries, basis functions, 99x590 unpruned grids, screening thresholds, FP64 arithmetic, conventional direct J/K and strict convergence checks are used for every stage. Each cell has three fresh processes; timings below are medians of parent subprocess start-to-exit. No historical timing is substituted. CPU14–21 were selected after other load appeared on CPU5–12; OMP8/BLAS1, one physical core per thread, serial execution, shared Xeon Platinum 8176 host.

| System | Baseline seconds | Full seconds | Full speedup | Baseline/full RSS MiB | Cycles baseline/full |
|---|---:|---:|---:|---:|---|
| water4 RKS/cc-pvdz | 18.9096 | 13.4468 | 1.406x | 502.8/386.8 | [11, 11, 11]/[11, 11, 11] |
| o2 UKS/cc-pvtz | 4.1754 | 3.7744 | 1.106x | 374.6/350.3 | [10, 10, 10]/[10, 10, 10] |
| water12 RKS/cc-pvdz | 153.0203 | 59.6632 | 2.565x | 1438.1/867.6 | [13, 13, 13]/[13, 13, 13] |

## Incremental full-SCF gains

Every row compares one stage with its immediate cumulative parent. Ratios below1 indicate an observed slowdown. Min/max ranges, all samples, cycles and RSS are preserved in increments.csv and measurements.csv. These are order-dependent merge increments, not isolated or leave-one-out contributions; they must not be replaced by kernel speedups or multiplied across different campaigns. Overlapping ranges do not establish a stable speed difference. Three repeats are descriptive, not confidence intervals.

| Stage | Change | Water4 RKS increment | O2 UKS increment | Water12 RKS increment | Water12 cumulative |
|---|---|---:|---:|---:|---:|
| 01-grid | Preserve molecular grid grouping with stable integer keys | 1.083x | 1.037x | 1.022x | 1.022x |
| 02-libxc | Pack and evaluate each LibXC grid slice in one parallel region | 0.997x | 1.000x | 1.022x | 1.044x |
| 03-ao | Reuse primitive exponentials in first AO derivatives | 1.009x | 1.012x | 0.994x | 1.038x |
| 04-density-dm | Fuse screened density-matrix contractions for numerical integration | 1.024x | 0.988x | 1.021x | 1.059x |
| 05-density-mo | Stream MO density, gradient and tau contractions | 1.033x | 1.025x | 1.112x | 1.178x |
| 06-sparse-tiles | Use finer sparse AO-pair output tasks with actual-width scratch | 1.090x | 0.988x | 1.620x | 1.909x |
| 07-potential | Fuse molecular XC potential and kinetic-density matrix assembly | 1.115x | 1.096x | 1.276x | 2.436x |
| 08-cache-blocks | Build XC caches without overlapping AO and derivative buffers | 1.011x | 0.961x | 0.973x | 2.370x |
| 09-response | Store invariant XC derivatives for internal response actions | 0.993x | 1.000x | 1.082x | 2.565x |

All retained steps remain in the final implementation, including weak or negative observations. The cache and compact-response steps target response work and storage; their ordinary ground-state SCF measurements are recorded without claiming a ground-state acceleration.

## Dependencies and PR preparation

The branches form a cumulative review sequence. The hard source dependencies are actual-width sparse scratch before fused potential assembly, and blocked cache construction before compact response. Other branch ordering is the chosen measurement/merge order, not a mathematical prerequisite. The density proposal is split into separate DM and MO steps.

| Branch | Direct parent | Head | Review diff |
|---|---|---|---|
| [DFT-stack-01-grid](https://github.com/chen-yu-hao/pyscf/tree/DFT-stack-01-grid) | 00-base | `8ee7cc3ea223d1b6e46913306b4a522c04746e98` | [diff](https://github.com/chen-yu-hao/pyscf/compare/44313dfb457b93358918c391fc66285fa44153ef...DFT-stack-01-grid) |
| [DFT-stack-02-libxc](https://github.com/chen-yu-hao/pyscf/tree/DFT-stack-02-libxc) | 01-grid | `a52b05e3afa58345ba95a17a07b782d2d5a5a090` | [diff](https://github.com/chen-yu-hao/pyscf/compare/DFT-stack-01-grid...DFT-stack-02-libxc) |
| [DFT-stack-03-ao](https://github.com/chen-yu-hao/pyscf/tree/DFT-stack-03-ao) | 02-libxc | `87505fe4200e3115ce7f38e3c53a4c15f46ccc2d` | [diff](https://github.com/chen-yu-hao/pyscf/compare/DFT-stack-02-libxc...DFT-stack-03-ao) |
| [DFT-stack-04-density-dm](https://github.com/chen-yu-hao/pyscf/tree/DFT-stack-04-density-dm) | 03-ao | `343beca98c837ae2b6edb28e39f74e88b40a8971` | [diff](https://github.com/chen-yu-hao/pyscf/compare/DFT-stack-03-ao...DFT-stack-04-density-dm) |
| [DFT-stack-05-density-mo](https://github.com/chen-yu-hao/pyscf/tree/DFT-stack-05-density-mo) | 04-density-dm | `35eb265e4017fffab2d015bf2449e5b790c38e46` | [diff](https://github.com/chen-yu-hao/pyscf/compare/DFT-stack-04-density-dm...DFT-stack-05-density-mo) |
| [DFT-stack-06-sparse-tiles](https://github.com/chen-yu-hao/pyscf/tree/DFT-stack-06-sparse-tiles) | 05-density-mo | `9c18323670177880b3925e53fc7dbb6e230b46af` | [diff](https://github.com/chen-yu-hao/pyscf/compare/DFT-stack-05-density-mo...DFT-stack-06-sparse-tiles) |
| [DFT-stack-07-potential](https://github.com/chen-yu-hao/pyscf/tree/DFT-stack-07-potential) | 06-sparse-tiles | `560f88db00b6f9d37b3d66140e0c3cc43d9ec39d` | [diff](https://github.com/chen-yu-hao/pyscf/compare/DFT-stack-06-sparse-tiles...DFT-stack-07-potential) |
| [DFT-stack-08-cache-blocks](https://github.com/chen-yu-hao/pyscf/tree/DFT-stack-08-cache-blocks) | 07-potential | `f50e82b1501fcda646bf35d49d9eed1f58d09a1e` | [diff](https://github.com/chen-yu-hao/pyscf/compare/DFT-stack-07-potential...DFT-stack-08-cache-blocks) |
| [DFT-stack-09-response](https://github.com/chen-yu-hao/pyscf/tree/DFT-stack-09-response) | 08-cache-blocks | `ee16ee33a279cdcc6e0f45dcfaee3ce0bdcf17eb` | [diff](https://github.com/chen-yu-hao/pyscf/compare/DFT-stack-08-cache-blocks...DFT-stack-09-response) |

## Direct-parent DM API coverage

48 additional fresh processes compare03-ao with04-density-dm using identical frozen bare DMs, coordinates, weights and screening masks. This covers LDA/VWN, PBE and MN15 on288/283-AO RKS/UKS cases at8threads plus MN15 on1160/1146-AO cases at16threads. These are core-Hamiltonian projector densities, not converged SCF densities. Timing covers nr_rks/nr_uks only, with one excluded warmup and three timed calls per process. The reported statistic is the median of three process medians. All numerical outputs, source/native identities and effective threads are accepted. The16-core affinity was selected from free physical cores after the SCF/CPKS campaigns; these jobs did not run concurrently.

| Fixed-DM case | XC | Threads | Parent seconds | Candidate seconds | XC speedup |
|---|---|---:|---:|---:|---:|
| water12-rks | LDA,VWN | 8 | 0.5697 | 0.3778 | 1.508x |
| water12-rks | PBE | 8 | 0.9226 | 0.7408 | 1.245x |
| water12-rks | MN15 | 8 | 2.2872 | 1.4904 | 1.535x |
| water11oh-uks | LDA,VWN | 8 | 0.9489 | 0.6230 | 1.523x |
| water11oh-uks | PBE | 8 | 1.5628 | 1.2043 | 1.298x |
| water11oh-uks | MN15 | 8 | 4.1428 | 2.6227 | 1.580x |
| water20-rks | MN15 | 16 | 7.7174 | 4.3462 | 1.776x |
| water19oh-uks | MN15 | 16 | 14.6682 | 8.1658 | 1.796x |

## Numerical acceptance and reproduction

Largest cross-version energy difference: 2.956e-12 Ha; DM RMS 2.188e-10; maximum DM element 3.358e-09; UKS spin-square difference 2.043e-14. Gates are1e-8 Ha,1e-9,1e-7 and1e-8 respectively. All ordered grid coordinates, weights and point counts match; every run must pass the same strict final convergence and overlap electron counts. Source commits, native files, imported library origins and effective thread counts were checked.

Stage-specific suites together invoke109 tests with one existing skip; the full combined suite independently invokes the same109 tests with one existing skip. These overlapping counts must not be added as218 distinct tests. All CI-style lint commands pass. Build flags are `-O3 -DNDEBUG -msse3 -fopenmp`, GCC11.3.1, CMake4.3.1, LibXC7.0.0, without fast-math.

The fullSCF worker is byte-identical to the historical worker. In a fresh evidence directory, build the source stages with build.py, run test.py, then run_scf.py. finish.py accepts all90 SCFs, runs18 response jobs, validates response native identity, renders this report and cleans registered scratch. Two converged reference fixtures are generated once from00-base with response/response_benchmark.py prepare --source SOURCE --waters 4 --spin 0 (or --spin 1) --output FIXTURE.npz, with PYTHONPATH=SOURCE; use the matching interpreter and current idle cores recorded in protocol.json. Paths are recorded execution provenance and should be adapted for a new machine; preserve the adapted protocol. Raw density arrays and response fixtures are retained locally in /home/lyl/soft/pyscf-stack-bench-20261027 and can be regenerated.

[CPKS time/memory report](response/runs/REPORT.md), [SCF increments](increments.csv), [all measured samples](measurements.csv), [PR fact notes](drafts/INDEX.md), [kernel review guide](REVIEW_GUIDE.md). No upstream PR has been created. [AI_POLICY.md](https://github.com/pyscf/pyscf/blob/master/AI_POLICY.md) states: "A human must review any generated code and submit a pull request". OpenAI Codex substantially assisted implementation, testing, benchmark automation and documentation.

Additional DM API inputs, all timings, min/max ranges and RSS are in [dm/runs/accepted/acceptance.json](dm/runs/accepted/acceptance.json). Numerical gates are1e-8 electrons,1e-8 Ha in Exc and1e-9 in Vxc. Existing raw density fixtures are reused in place; the scripts can regenerate them from00-base.
