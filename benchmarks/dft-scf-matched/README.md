# Matched-protocol complete SCF comparison

**All 36 fresh-process runs and all 54 cross-version numerical pairs passed.** These are newly measured full-SCF results using the protocol of the historical three-system table.

| Version | Identity |
|---|---|
| Legacy original | Frozen PySCF 2.14.0 runtime used by the historical benchmark |
| Current upstream | `44313dfb457b93358918c391fc66285fa44153ef` |
| Three changes | `DFT-combined`, `a8cbddc874d7846dedfdc482944fc299d76f2409` |
| Full optimized | Frozen `runtime-ao64-recovered`: round3 plus AO_TILE=64 and stable grid sorting |

The four versions use the same interpreter and input script. `pyscf_case.py` is byte-identical to the historical worker. The two current source checkouts share a baseline and matching build configuration; their ratio isolates the three selected changes. The two legacy packages reproduce the historical software comparison. Package/build and baseline differences also contribute when comparing a current checkout with a legacy package.

## Identical scientific protocol

MN15, fixed Bohr geometries and spherical basis functions, 99x590 atom grids without pruning or density-based grid removal; AO cutoff 1e-10 and numerical-integration cutoff 1e-8. Conventional direct J/K, no density fitting, direct_scf_tol=1e-13, no symmetry, Hückel guess, 2000 MB memory setting and DIIS space 12. Strict final convergence checks density RMS <1e-9, maximum density change <1e-7, energy change <1e-7 Ha and orbital gradient <3e-5, including the extra cycle. One additional polishing SCF is included in time if the first strict check fails. Natural DM/MO dispatch is retained.

The test runs locally on the shared Xeon Platinum 8176 host, serially, using physical CPUs 5–12 with OMP8/BLAS1. CPU4 and sibling60 were avoided after the utilization check. Three independent processes per case/version; version order rotates with case and replicate. OpenMP passive wait and NUMPY_MADVISE_HUGEPAGE=0 match the historical environment. No performance result was copied from an earlier experiment.

## Whole-process timing

Primary timing is parent subprocess start-to-exit, including imports, complete SCF, normal output and output checks. The statistic is the median of three processes; brackets are observed min/max ranges, not confidence intervals. Speed ratios are baseline/candidate, with values below 1 indicating a slowdown.

| System | Legacy original (s) | Upstream (s) | Three changes (s) | Full optimized (s) | Upstream/three | Legacy/full |
|---|---:|---:|---:|---:|---:|---:|
| water4 RKS | 22.2670 [21.4670–22.4658] | 22.9699 [22.7188–24.1719] | 20.7629 [20.3654–21.4651] | 13.9483 [13.5457–14.4995] | 1.106x | 1.596x |
| o2 UKS | 4.9268 [4.7258–5.2773] | 4.4264 [4.4258–5.2273] | 4.7760 [4.4768–5.2268] | 4.0749 [3.9748–4.3756] | 0.927x | 1.209x |
| water12 RKS | 167.7585 [167.7083–169.4597] | 165.2489 [157.6873–169.1591] | 104.6631 [100.8954–108.8690] | 64.6676 [64.6155–68.2754] | 1.579x | 2.594x |

The historical complete-package acceleration is reproduced under a common protocol. The three selected changes recover part of the full-SCF benefit, especially for water12, while the O2 timing ranges overlap and show no demonstrated benefit. The complete package remains faster than the three-change package on these systems. This comparison measures their combined behavior; it does not assign the remaining benefit to individual excluded kernels.

## Kernel timing, memory and iteration counts

Kernel time covers the entire `mf.kernel()`, including grid setup, numerical integration, J/K, diagonalization and any polishing pass. It is reported separately from whole-process time. Peak RSS is the Linux process high-water mark through the SCF and result checks, not a phase-local allocation.

| System | Version | Kernel median [min/max] (s) | Peak RSS median [min/max] (MiB) | Cycles | Polishing cycles | Grid points |
|---|---|---:|---:|---|---|---:|
| water4 | Legacy original 2.14.0 | 21.4312 [20.6249/21.6572] | 506.0 [496.1/507.3] | [11, 11, 11] | [0, 0, 0] | 700920 |
| water4 | Upstream 44313dfb4 | 22.3898 [22.1299/23.6266] | 498.8 [491.9/500.6] | [11, 11, 11] | [0, 0, 0] | 700920 |
| water4 | Three changes a8cbddc87 | 20.2010 [19.8269/20.9082] | 495.0 [493.1/496.9] | [11, 11, 11] | [0, 0, 0] | 700920 |
| water4 | Full AO64 snapshot | 13.3929 [13.0001/13.9121] | 384.0 [383.9/384.8] | [11, 11, 11] | [0, 0, 0] | 700920 |
| o2 | Legacy original 2.14.0 | 4.2025 [3.9506/4.5249] | 365.4 [364.1/370.9] | [10, 10, 10] | [0, 0, 0] | 116824 |
| o2 | Upstream 44313dfb4 | 3.9786 [3.8984/4.6876] | 354.0 [352.9/355.5] | [10, 10, 10] | [0, 0, 0] | 116824 |
| o2 | Three changes a8cbddc87 | 4.3132 [3.9514/4.7198] | 356.3 [356.2/356.7] | [10, 10, 10] | [0, 0, 0] | 116824 |
| o2 | Full AO64 snapshot | 3.5795 [3.4826/3.8808] | 330.8 [330.4/331.0] | [10, 10, 10] | [0, 0, 0] | 116824 |
| water12 | Legacy original 2.14.0 | 166.7759 [166.7214/168.5049] | 1426.6 [1425.1/1438.8] | [13, 13, 13] | [0, 0, 0] | 2102760 |
| water12 | Upstream 44313dfb4 | 164.5464 [157.0415/168.3986] | 1417.4 [1417.0/1437.0] | [13, 13, 13] | [0, 0, 0] | 2102760 |
| water12 | Three changes a8cbddc87 | 103.9832 [100.2189/108.0864] | 1080.1 [1079.3/1094.8] | [13, 13, 13] | [0, 0, 0] | 2102760 |
| water12 | Full AO64 snapshot | 63.9387 [63.9229/67.5796] | 853.1 [851.0/868.0] | [13, 13, 13] | [0, 0, 0] | 2102760 |

## Scientific and runtime acceptance

Every pair of versions is checked at each replicate, giving 54 comparisons. All versions have exactly matching ordered grid coordinate/weight digests and actual point counts: water4 700920, O2 116824 and water12 2102760.

Largest cross-version energy difference: **2.046e-12 Ha**; DM RMS: **2.188e-10**; maximum DM element: **3.358e-09**; UKS spin-square difference: **1.155e-14**. The respective gates are 1e-8 Ha, 1e-9, 1e-7 and 1e-8. Electron counts, strict final convergence and finite FP64 density matrices are checked for every run. Repeat energy ranges must remain below 1e-9 Ha.

Source identities and native library hashes are frozen in versions.json and checked again after the campaign. Every worker records the imported package, mapped native libraries and effective OpenMP/BLAS thread counts. The two current checkouts must retain their recorded clean git commits. CPU accounting and SMT-sibling utilization are retained per run; affinity on a shared host does not establish exclusive execution.

## Reproduction and provenance

Use the source/runtime paths in versions.json with matching dependencies, choose eight currently idle physical cores, and update the CPU mask and sibling list in a fresh copy of protocol.json if necessary. Run `python run.py`, `python accept.py`, then `python report.py` with `/home/lyl/soft/conda-envs/pyscf-new/bin/python` or the equivalent shared interpreter. The driver requires a fresh runs/ namespace and a writable tmp/ directory. The input worker uses protocol.json from its own directory. Binary density matrices and temporary files are retained locally under `/home/lyl/soft/pyscf-scf-matched-20261027` until acceptance; temporary scratch is removed only after all runs are accepted.

measurements.csv contains every version/system timing range, memory and cycle count; comparison.csv contains all reported ratios. acceptance.json includes all numerical pairs. runs/ contains individual results and the execution ledger; provenance/ retains the earlier build identities. Full density arrays remain in the local runs/ directory and are regenerated by the worker.

OpenAI Codex substantially assisted benchmark setup, automation, validation and reporting. No upstream pull request was created.
