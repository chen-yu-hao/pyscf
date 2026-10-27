# Build XC caches without overlapping AO and derivative buffers

Internal factual notes: human review, understanding and rewriting remain required before submission.

AO buffers and large XC derivative arrays can be live at the same time during cache construction. Finish density construction before creating blocked derivative outputs. Public return types remain full arrays; custom and overridden callbacks retain their whole-grid invocation semantics.

Head: `f50e82b1501fcda646bf35d49d9eed1f58d09a1e`. Direct parent: `560f88db00b6f9d37b3d66140e0c3cc43d9ec39d`. [Review diff](https://github.com/chen-yu-hao/pyscf/compare/DFT-stack-07-potential...DFT-stack-08-cache-blocks).

Full-SCF changes below are measured after all preceding stack changes, on the submitted code. They include grid setup, numerical integration, conventional J/K, diagonalization, convergence and process startup. Ratios are parent/candidate; lower than1 means slower.

| MN15 system | Parent seconds [min/max] | Candidate seconds [min/max] | Increment | Observation |
|---|---:|---:|---:|---|
| water4 | 13.4970 [13.3487/13.5475] | 13.3468 [13.2974/13.4974] | 1.011x | observed ranges overlap |
| o2 | 3.6250 [3.6241/3.6736] | 3.7740 [3.6245/3.7743] | 0.961x | observed ranges overlap |
| water12 | 62.8139 [59.0066/63.0102] | 64.5627 [59.3573/65.3567] | 0.973x | observed ranges overlap |

The stage-specific numerical suite ran 7 tests with 0 skips. The full combined head independently ran109 tests:108 passed and one pre-existing backend skip. All90 SCF runs and405 cross-version pairs passed identical-grid, convergence, electron-count, energy, DM and spin checks. Raw ranges and RSS are in the evidence tables.

Changed files:

- pyscf/dft/numint.py
- pyscf/dft/test/test_numint_cache_blocks.py

Response validation uses separate frozen converged TPSS/cc-pVDZ fixtures: water4 RKS and water3+OH UKS, level3 grids,8threads, three Cartesian-field RHSs. The timed interval includes cache construction plus the complete CPKS solve; it excludes SCF and checks the unpreconditioned residual and polarizability tensor.

| Response case | Parent/candidate total seconds | Parent/candidate sampled peak MiB | Parent/candidate cache bytes |
|---|---:|---:|---:|
| water4-rks | 12.6489/12.5529 | 610.4/605.5 | [37741760]/[37741760] |
| oh-water3-uks | 27.2893/27.2220 | 660.0/674.9 | [119984640]/[119984640] |

Cache payload bytes are not whole-process RSS. Sampled RSS at5ms can miss brief transients.

Conventional Coulomb ERI storage is recorded per run and matches across compared stages; see response/runs/coulomb-storage.json.

OpenAI Codex substantially assisted implementation, tests, benchmark automation and documentation. The human author must review and understand the kernels and personally submit under [AI_POLICY.md](https://github.com/pyscf/pyscf/blob/master/AI_POLICY.md).
