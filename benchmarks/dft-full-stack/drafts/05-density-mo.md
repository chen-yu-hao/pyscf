# Stream MO density, gradient and tau contractions

Internal factual notes: human review, understanding and rewriting remain required before submission.

Normal SCF can evaluate density from occupied MOs rather than a bare density matrix. Form orbitals in short grid blocks and immediately contract density, gradient and tau. Compress AO columns only when their value and all first derivatives are exactly zero; retain signed occupations, nonfinite propagation, resource checks and existing fallbacks.

Head: `35eb265e4017fffab2d015bf2449e5b790c38e46`. Direct parent: `343beca98c837ae2b6edb28e39f74e88b40a8971`. [Review diff](https://github.com/chen-yu-hao/pyscf/compare/DFT-stack-04-density-dm...DFT-stack-05-density-mo).

Full-SCF changes below are measured after all preceding stack changes, on the submitted code. They include grid setup, numerical integration, conventional J/K, diagonalization, convergence and process startup. Ratios are parent/candidate; lower than1 means slower.

| MN15 system | Parent seconds [min/max] | Candidate seconds [min/max] | Increment | Observation |
|---|---:|---:|---:|---|
| water4 | 16.9548 [16.8564/17.0073] | 16.4058 [16.3547/16.6544] | 1.033x | candidate observed range is faster |
| o2 | 4.0239 [3.8765/4.0248] | 3.9250 [3.9248/3.9750] | 1.025x | observed ranges overlap |
| water12 | 144.4715 [142.5155/147.0325] | 129.8687 [128.4149/132.2278] | 1.112x | candidate observed range is faster |

The stage-specific numerical suite ran 9 tests with 0 skips. The full combined head independently ran109 tests:108 passed and one pre-existing backend skip. All90 SCF runs and405 cross-version pairs passed identical-grid, convergence, electron-count, energy, DM and spin checks. Raw ranges and RSS are in the evidence tables.

Changed files:

- pyscf/dft/numint.py
- pyscf/dft/test/test_numint_rho.py
- pyscf/dft/test/test_numint_rho_stream.py
- pyscf/lib/dft/nr_numint.c

OpenAI Codex substantially assisted implementation, tests, benchmark automation and documentation. The human author must review and understand the kernels and personally submit under [AI_POLICY.md](https://github.com/pyscf/pyscf/blob/master/AI_POLICY.md).
