# Pack and evaluate each LibXC grid slice in one parallel region

Internal factual notes: human review, understanding and rewriting remain required before submission.

Repeated packing, evaluation and merging regions add synchronization to LibXC calls. Give each thread ownership of one grid slice through packing, component evaluation and output assembly. Retain the functional-component accumulation order at each point, including empty thread slices.

Head: `a52b05e3afa58345ba95a17a07b782d2d5a5a090`. Direct parent: `8ee7cc3ea223d1b6e46913306b4a522c04746e98`. [Review diff](https://github.com/chen-yu-hao/pyscf/compare/DFT-stack-01-grid...DFT-stack-02-libxc).

Full-SCF changes below are measured after all preceding stack changes, on the submitted code. They include grid setup, numerical integration, conventional J/K, diagonalization, convergence and process startup. Ratios are parent/candidate; lower than1 means slower.

| MN15 system | Parent seconds [min/max] | Candidate seconds [min/max] | Increment | Observation |
|---|---:|---:|---:|---|
| water4 | 17.4580 [17.3566/19.3107] | 17.5049 [17.3561/17.5077] | 0.997x | observed ranges overlap |
| o2 | 4.0245 [3.9750/4.1747] | 4.0240 [3.9746/4.0244] | 1.000x | observed ranges overlap |
| water12 | 149.7574 [147.8751/152.1700] | 146.5262 [140.7435/149.8654] | 1.022x | observed ranges overlap |

The stage-specific numerical suite ran 20 tests with 1 skips. The full combined head independently ran109 tests:108 passed and one pre-existing backend skip. All90 SCF runs and405 cross-version pairs passed identical-grid, convergence, electron-count, energy, DM and spin checks. Raw ranges and RSS are in the evidence tables.

Changed files:

- pyscf/dft/test/test_libxc.py
- pyscf/lib/dft/libxc_itrf.c

OpenAI Codex substantially assisted implementation, tests, benchmark automation and documentation. The human author must review and understand the kernels and personally submit under [AI_POLICY.md](https://github.com/pyscf/pyscf/blob/master/AI_POLICY.md).
