# Store invariant XC derivatives for internal response actions

Internal factual notes: human review, understanding and rewriting remain required before submission.

Expanded Cartesian XC Hessians increase persistent response-cache storage. Store invariant derivatives and apply the exact chain rule, including sigma geometric Hessian terms, when building response weights. Keep public cache APIs and unsupported-input fallbacks; retain the parent fused AO assembly when integrating the compact response path.

Head: `ee16ee33a279cdcc6e0f45dcfaee3ce0bdcf17eb`. Direct parent: `f50e82b1501fcda646bf35d49d9eed1f58d09a1e`. [Review diff](https://github.com/chen-yu-hao/pyscf/compare/DFT-stack-08-cache-blocks...DFT-stack-09-response).

Full-SCF changes below are measured after all preceding stack changes, on the submitted code. They include grid setup, numerical integration, conventional J/K, diagonalization, convergence and process startup. Ratios are parent/candidate; lower than1 means slower.

| MN15 system | Parent seconds [min/max] | Candidate seconds [min/max] | Increment | Observation |
|---|---:|---:|---:|---|
| water4 | 13.3468 [13.2974/13.4974] | 13.4468 [13.1957/13.4481] | 0.993x | observed ranges overlap |
| o2 | 3.7740 [3.6245/3.7743] | 3.7744 [3.7244/3.7747] | 1.000x | observed ranges overlap |
| water12 | 64.5627 [59.3573/65.3567] | 59.6632 [59.3643/62.3568] | 1.082x | observed ranges overlap |

The stage-specific numerical suite ran 9 tests with 0 skips. The full combined head independently ran109 tests:108 passed and one pre-existing backend skip. All90 SCF runs and405 cross-version pairs passed identical-grid, convergence, electron-count, energy, DM and spin checks. Raw ranges and RSS are in the evidence tables.

Changed files:

- pyscf/dft/numint.py
- pyscf/dft/test/test_xc_cache.py
- pyscf/dft/xc_cache.py
- pyscf/scf/_response_functions.py

Response validation uses separate frozen converged TPSS/cc-pVDZ fixtures: water4 RKS and water3+OH UKS, level3 grids,8threads, three Cartesian-field RHSs. The timed interval includes cache construction plus the complete CPKS solve; it excludes SCF and checks the unpreconditioned residual and polarizability tensor.

| Response case | Parent/candidate total seconds | Parent/candidate sampled peak MiB | Parent/candidate cache bytes |
|---|---:|---:|---:|
| water4-rks | 12.5529/12.6182 | 605.5/584.3 | [37741760]/[16175040] |
| oh-water3-uks | 27.2220/27.2283 | 674.9/604.1 | [119984640]/[45994112] |

Cache payload bytes are not whole-process RSS. Sampled RSS at5ms can miss brief transients.

Conventional Coulomb ERI storage is recorded per run and matches across compared stages; see response/runs/coulomb-storage.json.

OpenAI Codex substantially assisted implementation, tests, benchmark automation and documentation. The human author must review and understand the kernels and personally submit under [AI_POLICY.md](https://github.com/pyscf/pyscf/blob/master/AI_POLICY.md).
