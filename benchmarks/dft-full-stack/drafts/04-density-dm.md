# Fuse screened density-matrix contractions for numerical integration

Internal factual notes: human review, understanding and rewriting remain required before submission.

Screened density evaluation creates repeated AO-by-DM intermediates. Compute density, gradients and tau in local screened blocks, reusing the relevant AO components and masks. Keep unsupported dtype, layout, Laplacian and nonsymmetric-gradient cases on their existing paths. Bound the dispatch mask to the current grid block rather than its unprocessed tail.

Head: `343beca98c837ae2b6edb28e39f74e88b40a8971`. Direct parent: `87505fe4200e3115ce7f38e3c53a4c15f46ccc2d`. [Review diff](https://github.com/chen-yu-hao/pyscf/compare/DFT-stack-03-ao...DFT-stack-04-density-dm).

Full-SCF changes below are measured after all preceding stack changes, on the submitted code. They include grid setup, numerical integration, conventional J/K, diagonalization, convergence and process startup. Ratios are parent/candidate; lower than1 means slower.

| MN15 system | Parent seconds [min/max] | Candidate seconds [min/max] | Increment | Observation |
|---|---:|---:|---:|---|
| water4 | 17.3550 [17.2572/17.5061] | 16.9548 [16.8564/17.0073] | 1.024x | candidate observed range is faster |
| o2 | 3.9753 [3.9246/4.0246] | 4.0239 [3.8765/4.0248] | 0.988x | observed ranges overlap |
| water12 | 147.4830 [140.4623/149.6927] | 144.4715 [142.5155/147.0325] | 1.021x | observed ranges overlap |

The stage-specific numerical suite ran 32 tests with 0 skips. The full combined head independently ran109 tests:108 passed and one pre-existing backend skip. All90 SCF runs and405 cross-version pairs passed identical-grid, convergence, electron-count, energy, DM and spin checks. Raw ranges and RSS are in the evidence tables.

Changed files:

- pyscf/dft/numint.py
- pyscf/dft/test/test_numint.py
- pyscf/dft/test/test_numint_rho_fused.py
- pyscf/lib/dft/nr_numint_sparse.c

The direct DM API also has48 accepted fresh processes comparing this head with03-ao. Unlike normal SCF, frozen bare-DM calls select the density-matrix evaluator directly. These XC-only timings exclude grid construction, J/K and SCF and do not substitute for the full-SCF increments above. The frozen core-Hamiltonian-projector DMs are not converged SCF densities.

| Fixed-DM case | XC | Threads | Parent/candidate XC seconds | Speedup | Parent/candidate RSS MiB |
|---|---|---:|---:|---:|---:|
| water12-rks | LDA,VWN | 8 | 0.5697/0.3778 | 1.508x | 404.5/255.8 |
| water12-rks | PBE | 8 | 0.9226/0.7408 | 1.245x | 950.8/827.5 |
| water12-rks | MN15 | 8 | 2.2872/1.4904 | 1.535x | 1231.3/835.1 |
| water11oh-uks | LDA,VWN | 8 | 0.9489/0.6230 | 1.523x | 407.6/262.0 |
| water11oh-uks | PBE | 8 | 1.5628/1.2043 | 1.298x | 951.9/834.8 |
| water11oh-uks | MN15 | 8 | 4.1428/2.6227 | 1.580x | 1240.3/849.1 |
| water20-rks | MN15 | 16 | 7.7174/4.3462 | 1.776x | 1656.6/1151.8 |
| water19oh-uks | MN15 | 16 | 14.6682/8.1658 | 1.796x | 1710.8/1225.3 |

OpenAI Codex substantially assisted implementation, tests, benchmark automation and documentation. The human author must review and understand the kernels and personally submit under [AI_POLICY.md](https://github.com/pyscf/pyscf/blob/master/AI_POLICY.md).
