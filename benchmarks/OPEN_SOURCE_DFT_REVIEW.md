# Open-source DFT review and round-four decisions

Review date: 2026-10-30. This is a selected review of 28 major open-source programs/projects, not an exhaustive census, an audit of every source file, or a benchmark of all those programs. Official manuals, project/author pages and selected source interfaces were inspected. A project's published speedup is not a measured speedup for this PySCF fork.

## Individually reviewed projects

| Project | Public design worth studying | Transfer decision for this fork | Primary source |
|---|---|---|---|
| PySCF | Existing libcint/Qcint, native builds, DF/SGX and C/OpenMP paths | Reuse existing backend interfaces; do not call existing functionality a new implementation | https://pyscf.org/user/install.html |
| GPU4PySCF | GPU workflows and host/device object conversion | Device residency is useful; CPU CI cannot establish GPU gains | https://pyscf.org/user/gpu.html |
| Psi4 | VBase collocation cache, blocked Gaussian grid processing | Independently implement bounded AO reuse; no Psi4 source copied | https://psicode.org/psi4manual/master/api/psi4.core.VBase.html |
| NWChem | Gaussian DFT, explicit grid/screening and DIRECT/SEMIDIRECT/NOIO controls | Memory and integral-reuse policies must be separate from accuracy thresholds | https://nwchemgit.github.io/Density-Functional-Theory-for-Molecules.html |
| NWChemEx | Modular high-performance electronic-structure infrastructure; GauXC ecosystem | Separate grid tasks, collocation and backend operations; no runtime imported here | https://github.com/NWChemEx/NWChemEx |
| ExaChem | TAMM/Libint/GauXC component-based computation | Task/tensor blocking is a longer-term option, not an immediate speed claim | https://github.com/ExaChem/exachem |
| MPQC | TiledArray, MADWorld and Libint ecosystem | Study task-sized contractions and scheduling; avoid fine-grained Python dispatch | https://mpqc.org/ |
| VeloxChem | Python control with C++ integral and DFT kernels | Preserve native hot paths and avoid nested parallel oversubscription | https://veloxchem.org/docs/source-struct/ |
| OpenMolcas | Threshold-controlled Cholesky ERIs and RI-related capabilities | A separate accuracy-controlled algorithm experiment, not unchanged exact-J/K timing | https://molcas.gitlab.io/OpenMolcas/sphinx/introduction.html |
| Ergo | Gaussian all-electron DFT, FMM, hierarchical sparse algebra, purification | Valuable for large/local systems; small clusters need crossover measurements | https://ergoscf.org/ |
| CP2K | GPW/GAPW, DBCSR and ADMM | Study blocked sparse operations; changing representation or ADMM requires a separate comparison | https://manual.cp2k.org/trunk/methods/dft/index.html |
| Quantum ESPRESSO | Plane-wave workflows and adaptively compressed exchange (ACE) | Exchange subspace reuse merits research; not a drop-in arbitrary-density AO K interface | https://www.quantum-espresso.org/Doc/INPUT_PW.html |
| ABINIT | k-point/band/FFT parallel layouts and autoparal | Select layout by work, memory and communication, not thread count alone | https://docs.abinit.org/variables/paral/ |
| GPAW | Real-space, plane-wave and LCAO representations | Backend selection matters; keep the current Gaussian molecular representation fixed | https://gpaw.readthedocs.io/documentation/basic.html |
| SIESTA | Local numerical orbitals, sparse H/S and alternative solvers including PEXSI | Local index maps and sparse structures are useful; solver changes are separate | https://docs.siesta-project.org/projects/siesta/en/5.4/reference/siesta.html |
| OpenMX | Numerical atomic orbitals and Krylov/local-system methods | Explicitly validate truncation and crossover size before adopting linear scaling | https://openmx-square.org/openmx_man4.0/s27_3_krylov.html |
| ABACUS | Plane-wave and LCAO workflows; RI/communication components | Separate basis, communication and optional approximations | https://abacus.deepmodeling.com/en/stable/index.html |
| CONQUEST | Diagonalization and linear-scaling density-matrix minimization | O(N) is not automatically faster at a few hundred AOs | https://conquest.readthedocs.io/en/latest/groundstate.html |
| DFTK | Composable SCF mixing and physical preconditioning | Separate iteration-count improvements from per-iteration kernel improvements | https://docs.dftk.org/stable/guide/self_consistent_field/ |
| JDFTx | Operator-oriented C++ and CPU/GPU memory design | Keep memory ownership explicit and reduce backend transfers | https://jdftx.org/ |
| Octopus | Real-space domain/state parallelism and GPU workflows | Task locality and data residency transfer as design principles, not as a discretization swap | https://www.octopus-code.org/documentation/main/tutorial/hpc/gpu/ |
| SPARC | Real-space finite-difference/spectral methods | Study local operator application; not equivalent to changing this Gaussian basis | https://github.com/SPARC-X/SPARC |
| DFT-FE | Adaptive high-order finite elements and CPU/GPU scaling | Adaptive work allocation is instructive, but fixed-grid performance must remain fixed-grid | https://sites.google.com/umich.edu/dftfe |
| BigDFT | Daubechies wavelets and local/linear-scaling representations | Multiresolution is a different numerical route with its own accuracy controls | https://bigdft.org/ |
| MRChem | Adaptive multiwavelets and explicit precision controls | Adopt explicit error accounting, not an unvalidated representation change | https://mrchem.readthedocs.io/en/latest/users/user_inp.html |
| Elk | All-electron FP-LAPW solids | Symmetry/structured eigensolvers are relevant to periodic work, not the dominant small-water hotspot | https://elk.sourceforge.io/ |
| exciting | All-electron LAPW+lo | Different nuclear/interstitial representation; defaults cannot be timed as equivalent grids | https://www.exciting-code.org/exciting |
| FLEUR | LAPW with documented MPI/OpenMP decomposition | Match independent tasks and parallel layers to hardware | https://www.flapw.de/master/documentation/Calculations/parallelizationSchemes/ |

The research depth is broad documentation review, with focused source/interface inspection for this fork, Psi4's cache interface and Qcint/GauXC project metadata. These 28 projects were NOT compiled and benchmarked against each other. NWChemEx is included as infrastructure, not misrepresented as a fully equivalent standalone driver.

## Low-level components and attribution

- **Qcint**: https://github.com/sunqm/qcint/tree/v6.2.0 . Its README states same libcint APIs and x86 SSE/AVX optimization, and identifies the library as GPL. Published performance ranges are not used as results for this fork.
- **GauXC**: https://github.com/wavefunction91/GauXC . Modular distributed Gaussian XC/K integration on CPU and accelerators, with ExchCXX, IntegratorXX and Gau2Grid dependencies. No GauXC backend has been integrated in this change.
- **Gau2Grid**: https://psicode.org/psi4manual/1.10.x/gau2grid.html . A possible future collocation backend; ordering, normalization and derivative layouts need independent tests.
- **Libint**: https://evaleev.github.io/libint/ . Integral library/code generator; replacing the integral ABI is a separate engineering project.
- **Libxc**: https://libxc.gitlab.io/ . Retain the present functional definitions/derivative conventions, especially MN15 tau/spin conventions.
- **DBCSR / LIBXS / LIBXSMM**: https://cp2k.github.io/dbcsr/develop/page/2-user-guide/1-installation/index.html . Current host small-matrix backend documentation describes LIBXS with optional LIBXSMM; shape grouping, batches and locality are ideas for later contraction kernels, not implemented integrations here.
- **BLAS**: Keep OpenBLAS at one thread under outer OpenMP for this experiment. MKL/BLIS changes require another matched test, not an assertion of universal superiority.

## What round four actually changes

`pyscf/dft/ao_cache.py` implements a new opt-in context manager. Only density-independent AO collocation is retained. Keys include molecule array contents/cart flag, coordinate and mask contents, derivative order, shell slice and cutoff. Cache entries are independent, read-only copies; hits copy into writable output buffers in the native layout. Molecule mutation invalidates the cache. A scan-resistant admission policy keeps retained blocks when the sequential grid exceeds capacity, rather than continuously evicting potential hits.

The retained-AO budget is in MiB and additionally capped at half the current PySCF memory headroom. This is NOT a whole-process RSS cap: native workspaces, output copies, grids and ERIs remain additional allocations. Record hit/miss/admission/peak-byte statistics, RSS and the in-core/direct ERI choice. A change in memory-induced ERI routing must be disclosed when interpreting performance. Context exit restores the evaluator and releases arrays even on exceptions. Only standard molecular NumInt/Mole with explicit NUMA off is supported; custom AO callbacks, PBC and NUMA-worker caching are not claimed.

This is an independent implementation of the publicly documented cache idea, not copied LGPL source. Psi4 interface/source reference: https://github.com/psi4/psi4/blob/1ea3b9552909a365ea66cc4a10f4b4ca0e522db3/psi4/src/psi4/libfock/v.cc . Optional Qcint remains an external dependency, not copied into Apache-licensed core. Any optional linked distribution must still respect its dependency licenses.

## Pinned acceptance, not assumed speedup

Candidate calculation source: `5318ea91cd597d8cd94812cc18067f5a0cf5d8b7`.
Baseline: validated round two `5060bd524fa5e15ad0408fc80ddb3bc3d5cc6130`.
Workflow commit: `33d9c3044be2f0bcded11500fd33037a92428ed9`.
Run: https://github.com/chen-yu-hao/pyscf/actions/runs/41065766924 . Documentation commits do not change the pinned calculation.

Three independent comparisons:
1. portable libcint vs portable libcint plus bounded AO cache;
2. portable libcint vs native-ISA libcint, no cache;
3. native-ISA libcint vs native-ISA Qcint, no cache.

PySCF's existing `BUILD_MARCH_NATIVE`/`USE_QCINT` options are reused. The native option is a build-policy bundle, not proof that a particular AVX instruction caused any gain. Native binaries may not be portable to another CPU. Main-project flags explicitly disable fast-math and FMA contraction; dependency projects use their own release policies, which are recorded rather than incorrectly claimed identical to the parent flags. Separate integral builds avoid accidentally testing two checkouts against the same selected integral backend.

Both sides enable round-two dense BLAS and disable round-one tile and round-three MGGA/J-K experiments. NUMA off, BLAS one thread. No changes to grids, block policy, basis, screening thresholds, initial guess or SCF convergence parameters.

Cache/native each have 18 formal cases: PBE/MN15, n=1/2/4 waters, 1/2/4 OpenMP threads. Qcint has MN15 n=4/8, 2 threads. All use def2-TZVPP, grid6. Each variant has three ordinary fresh-process SCFs and one independent profile: 38 cases /304 formal records total, excluding separate early smoke. Compare fixed-density Ne/Exc/Vxc/J/K arrays at atol=1e-9,rtol=1e-10 and total energy <=1e-8 Eh. Require convergence, finite arrays, matching grids, expected source/build/options and complete record counts. Report ordinary median timings, all raw samples, profile boundaries and memory effects; no speedup is inferred from cache hits or compiler flags.

Locally executed before CI: nine standalone synthetic-evaluator cache tests passed; Python compilation and all workflow shell syntax checks passed. These do not replace native chemistry regression or the molecular matrix. At publication the new run exists and the first two ablations are building; no round-four molecular speedup is accepted yet. Keep this PR draft, keep original DFT untouched and do not auto-merge.

## Future candidates, not silently bundled

Local AO compaction and task/batched contractions (Psi4/GauXC/DBCSR ideas) are the next C-kernel candidates after these isolated results. RI/Cholesky/ADMM, seminumerical exchange, FMM/truncation and changed discretizations need their own accuracy experiments. ACE preserves operator action on its construction subspace but is not an unconditional full-K replacement for arbitrary AO densities. SCF preconditioning must be evaluated separately from per-step cost. Current CPU tests establish neither GPU nor multi-NUMA gains.
