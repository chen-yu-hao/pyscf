# Use finer sparse AO-pair output tasks with actual-width scratch

Internal factual notes: human review, understanding and rewriting remain required before submission.

Coarse output tiles limit sparse AO-pair task granularity. Target 32-AO tiles while keeping each shell intact and each output AO pair owned by one task. Allocate scratch using actual partition width so shells wider than the target remain supported.

Head: `9c18323670177880b3925e53fc7dbb6e230b46af`. Direct parent: `35eb265e4017fffab2d015bf2449e5b790c38e46`. [Review diff](https://github.com/chen-yu-hao/pyscf/compare/DFT-stack-05-density-mo...DFT-stack-06-sparse-tiles).

Full-SCF changes below are measured after all preceding stack changes, on the submitted code. They include grid setup, numerical integration, conventional J/K, diagonalization, convergence and process startup. Ratios are parent/candidate; lower than1 means slower.

| MN15 system | Parent seconds [min/max] | Candidate seconds [min/max] | Increment | Observation |
|---|---:|---:|---:|---|
| water4 | 16.4058 [16.3547/16.6544] | 15.0516 [15.0001/15.1505] | 1.090x | candidate observed range is faster |
| o2 | 3.9250 [3.9248/3.9750] | 3.9747 [3.9245/4.0235] | 0.988x | observed ranges overlap |
| water12 | 129.8687 [128.4149/132.2278] | 80.1561 [75.8033/80.3817] | 1.620x | candidate observed range is faster |

The stage-specific numerical suite ran 2 tests with 0 skips. The full combined head independently ran109 tests:108 passed and one pre-existing backend skip. All90 SCF runs and405 cross-version pairs passed identical-grid, convergence, electron-count, energy, DM and spin checks. Raw ranges and RSS are in the evidence tables.

Changed files:

- pyscf/dft/test/test_numint_pair_tiles.py
- pyscf/lib/dft/nr_numint_sparse.c

OpenAI Codex substantially assisted implementation, tests, benchmark automation and documentation. The human author must review and understand the kernels and personally submit under [AI_POLICY.md](https://github.com/pyscf/pyscf/blob/master/AI_POLICY.md).
