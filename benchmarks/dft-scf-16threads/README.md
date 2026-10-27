# Three-version complete SCF comparison at 16 threads

**27/27 fresh-process runs and 27/27 pairwise numerical comparisons passed.** Every number in the primary comparison was measured in this same campaign.

| Version | Tested commit |
|---|---|
| Upstream | `44313dfb457b93358918c391fc66285fa44153ef` |
| DFT | `72e5b26c91964acbf3186c5348f77a0f7faf44dd` |
| DFT-combined | `a8cbddc874d7846dedfdc482944fc299d76f2409` |

The two candidates are the actual GitHub branches. Each clean checkout uses its own verified native libraries. GCC 11.3.1, CMake 4.3.1, LibXC 7.0.0, standard Release `-O3 -DNDEBUG -msse3 -fopenmp`, no fast-math.

## Protocol

MN15, fixed Bohr geometries; water4/water12 RKS with spherical cc-pVDZ and triplet O2 UKS with spherical cc-pVTZ. Each atom uses an unpruned 99x590 grid, with no density-based point removal. AO cutoff 1e-10, integration cutoff 1e-8, conventional direct J/K with direct_scf_tol=1e-13, no density fitting, no symmetry, Hückel initial guess, 2000 MB memory setting and DIIS space 12.

Strict final convergence: density RMS <1e-9, maximum density change <1e-7, energy change <1e-7 Ha and orbital gradient <3e-5, including the extra cycle. At most one additional polishing SCF is included in timing if the first strict check fails. The worker is identical to the prior 8-thread worker; its --cores argument is 16.

Local shared Xeon Platinum 8176. Sixteen OpenMP threads bind to physical CPUs 32–47 on socket 1; SMT siblings 88–103 are monitored. BLAS remains single-threaded. Busy CPUs 14–21 on the other socket were avoided. Processes run serially with version order rotated by case and replicate. Three fresh processes per case/version; passive OpenMP wait and NUMPY_MADVISE_HUGEPAGE=0.

## Complete-process timing

Parent subprocess start-to-exit includes imports, the entire SCF, normal output and result checks. Values are medians of three processes; brackets are observed min/max ranges, not confidence intervals. Speedups use the same-batch upstream baseline.

| System | Upstream (s) | DFT (s) | DFT-combined (s) | Upstream/DFT | Upstream/combined |
|---|---:|---:|---:|---:|---:|
| water4 RKS | 19.7651 [19.6113–19.8637] | 10.9448 [10.9412–11.2412] | 16.8090 [16.6031–16.8530] | 1.806x | 1.176x |
| o2 UKS | 4.2262 [4.1763–4.2757] | 3.3741 [3.3734–3.4244] | 4.0263 [4.0248–4.1262] | 1.253x | 1.050x |
| water12 RKS | 142.3573 [123.3143–144.1112] | 42.9045 [40.1493–46.0617] | 74.4809 [69.9229–79.1418] | 3.318x | 1.911x |

## Kernel time, memory and iterations

`mf.kernel()` includes grid setup, integration, J/K, diagonalization and polishing. Its timing is separate from whole-process timing. Peak RSS is the Linux process high-water mark through SCF and checks, not a phase-local allocation.

| System | Version | Kernel median [min/max] (s) | Peak RSS median [min/max] (MiB) | Cycles | Polishing | Points |
|---|---|---:|---:|---|---|---:|
| water4 | upstream | 19.2698 [19.0545/19.3533] | 489.3 [489.2/489.7] | [11, 11, 11] | [0, 0, 0] | 700920 |
| water4 | DFT | 10.4494 [10.3693/10.6616] | 389.3 [389.2/390.3] | [11, 11, 11] | [0, 0, 0] | 700920 |
| water4 | DFT-combined | 16.2704 [16.0895/16.3136] | 480.1 [479.6/480.9] | [11, 11, 11] | [0, 0, 0] | 700920 |
| o2 | upstream | 3.7766 [3.7461/3.7770] | 354.9 [354.2/356.3] | [10, 10, 10] | [0, 0, 0] | 116824 |
| o2 | DFT | 2.8968 [2.8826/2.9240] | 326.9 [325.7/328.3] | [10, 10, 10] | [0, 0, 0] | 116824 |
| o2 | DFT-combined | 3.5784 [3.5444/3.6464] | 356.1 [355.7/356.3] | [10, 10, 10] | [0, 0, 0] | 116824 |
| water12 | upstream | 141.7000 [122.6422/143.4131] | 1418.6 [1417.4/1418.8] | [13, 13, 13] | [0, 0, 0] | 2102760 |
| water12 | DFT | 42.2161 [39.5481/45.3830] | 864.5 [863.6/866.9] | [13, 13, 13] | [0, 0, 0] | 2102760 |
| water12 | DFT-combined | 73.7845 [69.2941/78.4309] | 1083.6 [1083.1/1084.3] | [13, 13, 13] | [0, 0, 0] | 2102760 |

## Acceptance

Every pair of versions is compared at each replicate. All 27 comparisons require exactly matching ordered grid coordinate/weight digests and point counts: water4 700920, O2 116824, water12 2102760. All runs pass strict convergence, finite FP64 DM, overlap electron count and effective OMP16/BLAS1 checks.

Maximum cross-version energy difference: 1.819e-12 Ha; DM RMS: 2.188e-10; maximum DM element: 3.358e-09; spin-square difference: 1.865e-14. The respective gates are 1e-8 Ha, 1e-9, 1e-7 and 1e-8. Repeat energy ranges must be below 1e-9 Ha. Source commits, tracked source hashes and native library identities are frozen before timing and checked again after the campaign.

The host is shared. Core affinity and three repeats do not establish exclusive execution or population confidence intervals. All regressions remain in the tables. These results cover the three stated MN15 cases. Earlier 8-thread results used different CPU masks and batches: [DFT 8-thread report](../dft-branch-scf) and [combined 8-thread report](../dft-scf-matched). Cross-batch 8/16 time ratios are descriptive references rather than controlled parallel-scaling measurements.

## Artifacts and reproduction

versions.json binds all three checkouts to exact commits and binary identities; provenance/ contains build records. measurements.csv retains medians, ranges, RSS and cycles; comparison.csv contains all process/kernel ratios. acceptance.json retains every numerical pair, and runs/ contains each result and the ledger.

Use matching builds/dependencies and the same interpreter for all versions. Copy this benchmark to a fresh directory, choose sixteen idle physical cores on one socket, update cpu_mask and smt_siblings_monitored in protocol.json, and create tmp/. Run `python run.py`, `python accept.py`, then `python report.py`. The driver requires a fresh runs/ namespace. finish.py is the serial acceptance/cleanup controller. Raw density arrays and worker stdout remain locally in `/home/lyl/soft/pyscf-scf-16t-20261027/runs`; public evidence omits these binaries and compiled libraries. Registered temporary scratch is removed after acceptance.

OpenAI Codex substantially assisted benchmark preparation, validation and reporting. No upstream pull request was created.
