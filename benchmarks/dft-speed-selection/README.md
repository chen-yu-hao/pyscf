# Measured DFT speed improvements selected for review

The recommended code contributions are **DFT-grid**, **DFT-sparse-tiles**, and **DFT-density-dm**. These are independent branches based on upstream `44313dfb457b93358918c391fc66285fa44153ef`. The last branch removes all MO optimizations from the earlier density proposal. The actual new head `a5d33360cd0fd9c16e48fed7e752690b98a78b61` was rebuilt and remeasured.

| Contribution | Measured speed ratio | Timing boundary |
|---|---:|---|
| Stable grid keys | 10.970x | Grid grouping only, 439040 points |
| Sparse AO-pair tiles | 1.202–1.583x across 8 cells | `nr_rks`/`nr_uks`, fixed DM |
| Sparse DM density contractions | 1.253–1.907x across 8 cells | `nr_rks`/`nr_uks`, fixed DM |

These ratios belong to different independent changes and must not be multiplied. All retained cells passed full output-array comparisons; each candidate's observed time range is below its parent's range. Three repeats on a shared host are descriptive observations, not confidence intervals or a claim of universal acceleration.

The DM-only new commit passed 32 numerical tests and all 48 fresh-process comparison runs. MO evaluator functions and `nr_numint.c` are byte-identical to upstream. No MO streaming kernel, LibXC parallel-region fusion, AO derivative modification, potential-assembly kernel, or response-cache compression is included in this DM-only branch.

The remaining proposals are deferred: the original MO path regressed in an UKS case; LibXC and AO gains were mixed or small; potential assembly regressed on the large cases relative to its direct sparse-tiles parent; cache changes chiefly reduce memory rather than full response time. Negative evidence remains archived locally and the original research branches remain available. The selection removes code paths, not unfavorable samples from a retained path.

## Evidence

`results/*.csv` contains every selected cell's medians, min/max ranges, peak RSS, source commits and maximum array errors. `results/*.json` records the original acceptance checksum and selected comparison rows. `builds.json` binds native files and C sources to the tested commits. `selection.json` records the inclusion/exclusion decisions. `drafts/` contains factual preparation notes for a human author, not automatically submitted PR messages.

The numerical gates for XC are absolute 1e-8 electrons, 1e-8 Ha in Exc and 1e-9 in Vxc; grid grouping indices must match exactly. The new DM-only matrix contains LDA/VWN, PBE and MN15 on RKS/UKS at 8 threads, plus MN15 at 24 threads on 1160/1146-AO systems. The medium systems have 288/283 AOs. Both checkouts reuse identical frozen coordinates, weights and core-Hamiltonian-projector density matrices, which are not converged SCF densities. This isolates numerical integration and excludes grid construction, J/K and SCF iterations.

All tests used GCC 11.3.1, CMake 4.3.1, LibXC 7.0.0, standard Release `-O3 -DNDEBUG -msse3 -fopenmp`, no fast-math, one BLAS thread and fixed physical cores. Native mappings and thread counts were verified. Peak XC RSS is the process high-water mark, including input loading and warmup.

## Reproduction

Build the base and each tested branch with matching dependencies and Release flags using the repository build instructions. Run these Linux benchmark scripts with Python 3.9+ and the same interpreter/dependencies for both checkouts. Choose idle physical cores on your machine; the recorded experiments used CPUs 0–23 on one socket. Clear unrelated `PYTHONPATH`/`LD_LIBRARY_PATH` and set `PYSCF_EXT_PATH=/dev/null`.

Generate the frozen fixtures once from the base (repeat for `water12-rks`, `water11oh-uks`, `water20-rks`, `water19oh-uks`):

```bash
python xc_benchmark.py --source /path/to/base   --commit 44313dfb457b93358918c391fc66285fa44153ef   --make-fixture --case water12-rks --fixture fixtures/water12-rks.npz   --threads 1 --affinity 0
```

Run the eight-cell DM comparison, using a fresh output directory:

```bash
python run_xc_campaign.py --protocol protocols/xc-component.json   --versions baseline=/path/to/base density-dm=/path/to/density-dm   --python /path/to/python --affinity 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23   --output dm-results
python accept_xc_campaign.py dm-results/batch.json --baseline baseline --output dm-results/accepted
```

Use `tiles=/path/to/tiles` for the sparse-tiles comparison. For grid grouping use `run_micro_campaign.py --protocol protocols/micro-grid.json` with `baseline=/path/to/base grid=/path/to/grid` and the same interpreter/affinity/output arguments. Each cell has three fresh processes; each process performs one excluded warmup and three timed calls. The summaries use the ordinary median of the three process medians.

Public protocol metadata corrects redundant expected-count fields to eight cells/48 processes; the measured cells and operational settings are unchanged. Raw output arrays and full machine-local logs are retained locally rather than committed here; rerunning the scripts produces all arrays and acceptance checks.

OpenAI Codex substantially assisted implementation, tests, benchmark automation, code review, and documentation. The human author must understand, verify and edit the material and personally submit any PR according to `AI_POLICY.md`. See `REVIEW_GUIDE.md` for the equations and implementation boundaries.
