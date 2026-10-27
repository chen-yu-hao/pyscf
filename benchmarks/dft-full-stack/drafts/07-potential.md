# Fuse molecular XC potential and kinetic-density matrix assembly

Internal factual notes: human review, understanding and rewriting remain required before submission.

Potential assembly repeatedly scales AO arrays and contracts separate channels. Pack the LDA/GGA/MGGA channels together and assemble the potential with a 64-AO C kernel. Retain signed weights, Hermitian factors, partial blocks and decline-before-output-write fallbacks. This step requires the actual-width sparse scratch correction in the preceding tiles step.

Head: `560f88db00b6f9d37b3d66140e0c3cc43d9ec39d`. Direct parent: `9c18323670177880b3925e53fc7dbb6e230b46af`. [Review diff](https://github.com/chen-yu-hao/pyscf/compare/DFT-stack-06-sparse-tiles...DFT-stack-07-potential).

Full-SCF changes below are measured after all preceding stack changes, on the submitted code. They include grid setup, numerical integration, conventional J/K, diagonalization, convergence and process startup. Ratios are parent/candidate; lower than1 means slower.

| MN15 system | Parent seconds [min/max] | Candidate seconds [min/max] | Increment | Observation |
|---|---:|---:|---:|---|
| water4 | 15.0516 [15.0001/15.1505] | 13.4970 [13.3487/13.5475] | 1.115x | candidate observed range is faster |
| o2 | 3.9747 [3.9245/4.0235] | 3.6250 [3.6241/3.6736] | 1.096x | candidate observed range is faster |
| water12 | 80.1561 [75.8033/80.3817] | 62.8139 [59.0066/63.0102] | 1.276x | candidate observed range is faster |

The stage-specific numerical suite ran 18 tests with 0 skips. The full combined head independently ran109 tests:108 passed and one pre-existing backend skip. All90 SCF runs and405 cross-version pairs passed identical-grid, convergence, electron-count, energy, DM and spin checks. Raw ranges and RSS are in the evidence tables.

Changed files:

- pyscf/dft/numint.py
- pyscf/dft/test/test_numint_mgga_vmat.py
- pyscf/dft/test/test_numint_tau.py
- pyscf/dft/test/test_numint_xc_response_assembly.py
- pyscf/lib/dft/CMakeLists.txt
- pyscf/lib/dft/mgga_vmat.c
- pyscf/lib/dft/nr_numint_sparse.c

OpenAI Codex substantially assisted implementation, tests, benchmark automation and documentation. The human author must review and understand the kernels and personally submit under [AI_POLICY.md](https://github.com/pyscf/pyscf/blob/master/AI_POLICY.md).
