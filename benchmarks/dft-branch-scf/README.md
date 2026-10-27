# Complete SCF benchmark of the GitHub DFT branch

**18/18 fresh-process runs and 9/9 cross-version numerical pairs passed.** All timings were measured anew in this campaign.

- Candidate: [DFT](https://github.com/chen-yu-hao/pyscf/tree/DFT), `72e5b26c91964acbf3186c5348f77a0f7faf44dd`.
- Upstream baseline: `44313dfb457b93358918c391fc66285fa44153ef`.

This measures the GitHub branch itself. The earlier AO64 filesystem snapshot is a distinct implementation and is not the candidate in this table. Both checkouts have clean tracked sources and verified native builds. Compiler GCC 11.3.1, CMake 4.3.1, LibXC 7.0.0, standard Release `-O3 -DNDEBUG -msse3 -fopenmp`, without fast-math.

## Common protocol

MN15, the same fixed Bohr geometries and spherical cc-pVDZ water-cluster / cc-pVTZ O2 bases as the historical full-SCF table. Each atom requests 99x590 points, no pruning and no density-based point removal; grid cutoff 1e-10, numerical-integration cutoff 1e-8. Conventional direct J/K with direct_scf_tol=1e-13, no density fitting, no symmetry, Hückel guess, 2000 MB memory setting and DIIS space 12. The input worker is byte-identical to the historical worker.

Strict final convergence checks density RMS <1e-9, maximum density change <1e-7, energy change <1e-7 Ha and orbital gradient <3e-5, including the extra cycle. One additional polishing SCF is included in time if the first strict check fails. Normal DM/MO dispatch is retained.

Local shared Xeon Platinum 8176, physical CPUs 1,3,5,7,9,12,13,22 on socket 0, OMP8/BLAS1. The second socket was heavily occupied; selected cores and their SMT siblings had low sampled utilization. Both versions run serially on the same mask, alternating order across cases and replicates. OpenMP passive wait and NUMPY_MADVISE_HUGEPAGE=0.

## Complete-process timings

Primary timing is parent subprocess start-to-exit, including imports, the entire SCF, normal output and result checks. Each value is the median of three independent processes; brackets show observed min/max ranges, not confidence intervals. Speedup is upstream/DFT.

| System | Upstream (s) | DFT branch (s) | Speedup |
|---|---:|---:|---:|
| water4 RKS | 18.7574 [18.7074–18.9549] | 13.1418 [13.0926–13.4441] | 1.427x |
| o2 UKS | 4.2744 [4.2741–4.2747] | 3.6738 [3.6722–3.7239] | 1.163x |
| water12 RKS | 152.8644 [150.5577–158.2031] | 61.4464 [59.6595–61.4539] | 2.488x |

## Kernel time and memory

`mf.kernel()` includes grid setup, numerical integration, J/K, diagonalization and any polishing pass. Its time is kept separate from process time. Peak RSS is the Linux process high-water mark through SCF and checks, not a phase-local allocation.

| System | Version | Kernel median [min/max] (s) | Peak RSS median [min/max] (MiB) | Cycles | Polishing cycles | Points |
|---|---|---:|---:|---|---|---:|
| water4 | upstream | 18.1959 [18.1245/18.3713] | 502.8 [502.8/503.7] | [11, 11, 11] | [0, 0, 0] | 700920 |
| water4 | DFT | 12.6340 [12.5308/12.8813] | 388.8 [388.7/389.6] | [11, 11, 11] | [0, 0, 0] | 700920 |
| o2 | upstream | 3.7339 [3.7217/3.8026] | 374.4 [373.9/375.2] | [10, 10, 10] | [0, 0, 0] | 116824 |
| o2 | DFT | 3.1960 [3.1781/3.2600] | 348.5 [348.5/348.6] | [10, 10, 10] | [0, 0, 0] | 116824 |
| water12 | upstream | 152.1727 [149.8887/157.5081] | 1438.2 [1437.4/1440.6] | [13, 13, 13] | [0, 0, 0] | 2102760 |
| water12 | DFT | 60.7605 [59.0047/60.8350] | 868.7 [864.7/870.1] | [13, 13, 13] | [0, 0, 0] | 2102760 |

## Acceptance and scope

All baseline/candidate pairs have exactly matching ordered grid coordinates, weights and point counts: water4 700920, O2 116824, water12 2102760. Every run passes strict convergence, finite FP64 DM, overlap electron count and OMP8/BLAS1 checks.

Maximum energy difference: 1.364e-12 Ha; DM RMS: 2.188e-10; maximum DM element: 3.358e-09; spin-square difference: 1.066e-14. Gates are 1e-8 Ha, 1e-9, 1e-7 and 1e-8 respectively. Repeat energy ranges must be below 1e-9 Ha.

Source and native identities are frozen before measurement and checked again after all runs; mapped libraries must originate from the selected checkout. CPU and sibling counters are retained per run. This is a shared host: three repeats and core affinity do not establish exclusive execution or a population confidence interval. These results cover the stated three MN15 workloads and eight-thread setting.

## Artifacts and reproduction

Use versions.json for exact source commits, source/native hashes and runtime paths. provenance/ records original build manifests and the aggregate build configuration. Use the same interpreter/dependencies for both builds. Copy the benchmark directory to a fresh namespace, choose eight idle physical cores, and update cpu_mask and smt_siblings_monitored in protocol.json if needed. Create tmp/ and run `python run.py`, `python accept.py`, then `python report.py`. The driver requires a fresh runs/ directory. finish.py is the original serial acceptance/cleanup controller.

measurements.csv contains all medians, ranges, memory and cycles; comparison.csv contains process and kernel ratios. acceptance.json retains every numerical pair. Raw density arrays and worker logs remain locally in `/home/lyl/soft/pyscf-dft-scf-20261027/runs`; the public text archive omits binaries. Registered temporary scratch is removed only after acceptance.

OpenAI Codex substantially assisted benchmark setup, validation and reporting. No upstream pull request was created.
