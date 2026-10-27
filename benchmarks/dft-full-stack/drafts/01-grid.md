# Preserve molecular grid grouping with stable integer keys

Internal factual notes: human review, understanding and rewriting remain required before submission.

Grid grouping currently performs a three-column unique operation before stable sorting. Encode the box IDs as mixed-radix int64 keys and sort them stably, preserving the exact existing index order. Check integer range before NumPy arithmetic and retain the original fallback.

Head: `8ee7cc3ea223d1b6e46913306b4a522c04746e98`. Direct parent: `44313dfb457b93358918c391fc66285fa44153ef`. [Review diff](https://github.com/chen-yu-hao/pyscf/compare/44313dfb457b93358918c391fc66285fa44153ef...DFT-stack-01-grid).

Full-SCF changes below are measured after all preceding stack changes, on the submitted code. They include grid setup, numerical integration, conventional J/K, diagonalization, convergence and process startup. Ratios are parent/candidate; lower than1 means slower.

| MN15 system | Parent seconds [min/max] | Candidate seconds [min/max] | Increment | Observation |
|---|---:|---:|---:|---|
| water4 | 18.9096 [18.7597/21.3145] | 17.4580 [17.3566/19.3107] | 1.083x | observed ranges overlap |
| o2 | 4.1754 [4.1247/4.2253] | 4.0245 [3.9750/4.1747] | 1.037x | observed ranges overlap |
| water12 | 153.0203 [150.9909/157.0691] | 149.7574 [147.8751/152.1700] | 1.022x | observed ranges overlap |

The stage-specific numerical suite ran 10 tests with 0 skips. The full combined head independently ran109 tests:108 passed and one pre-existing backend skip. All90 SCF runs and405 cross-version pairs passed identical-grid, convergence, electron-count, energy, DM and spin checks. Raw ranges and RSS are in the evidence tables.

Changed files:

- pyscf/dft/gen_grid.py
- pyscf/dft/test/test_grids.py

OpenAI Codex substantially assisted implementation, tests, benchmark automation and documentation. The human author must review and understand the kernels and personally submit under [AI_POLICY.md](https://github.com/pyscf/pyscf/blob/master/AI_POLICY.md).
