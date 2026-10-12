# Conventional UCCSDT[Q] and UCCSDT(Q)

The correction uses converged UCCSDT amplitudes with canonical UHF orbitals
and real `float64` arrays. It implements energy corrections only; it does not
solve quadruple amplitudes iteratively or provide gradients.

```python
mycc = cc.UCCSDT(mf, frozen=frozen, compact_tamps=True)
eris = mycc.ao2mo()
mycc.kernel(eris=eris)
e_q_bracket, e_q_paren = mycc.ccsdt_q(eris=eris)
e_total = mycc.e_tot + e_q_paren
```

The two outputs are the `[Q]` correction and the complete `(Q)` correction,
in hartree. The latter **already contains `[Q]`**. The additional term is
`e_q_paren - e_q_bracket`; adding both returned values to `e_tot` would count
`[Q]` twice. The call leaves `e_tot`, `e_corr`, amplitudes, and integrals
unchanged. [67-uccsdt_q.py](67-uccsdt_q.py) is a small open-shell example.

## Input conventions

`tamps` defaults to `mycc.tamps` and is `[t1, t2, t3]`. Explicit amplitudes
must use the representation selected on the CC object. The spin blocks use
the existing UCCSDT order, including `bba`, rather than `abb`:

| Tensor | Full index order | Compact index order |
| --- | --- | --- |
| `t1a`, `t1b` | `ia`, `IA` | unchanged |
| `t2aa`, `t2ab`, `t2bb` | `ijab`, `iaJB`, `IJAB` | unchanged |
| `t3aaa`, `t3bbb` | `ijkabc`, `IJKABC` | `(ijk),(abc)` |
| `t3aab` | `ijabKC` | `(ij),(ab),K,C` |
| `t3bba` | `IJABkc` | `(IJ),(AB),k,c` |

Lowercase and uppercase indices denote alpha and beta orbitals. Parentheses
in the compact column enumerate strictly increasing same-spin combinations.
Their order and permutation signs are those of the UCCSDT conversion
functions. Both `compact_tamps=True` and `False` use the same correction API.

`eris` defaults to `mycc.ao2mo(mycc.mo_coeff)`. Supply the **original, bare**
UCCSDT ERI object; T1-dressed iteration intermediates are not inputs to this
correction. Passing the same bare `eris` to both calls as above avoids a
second AO-to-MO transformation. This reuse is explicit: the solver does not
retain an extra integral cache. Its physicists' integrals are

\[
V_{pqrs}=\langle pq|rs\rangle=(pr|qs),\qquad
v_{pqrs}=\langle pq\Vert rs\rangle=V_{pqrs}-V_{pqsr}.
\]

The exchange subtraction above refers to complete spin-orbital integrals.
The spatial `pPpP` block cannot simply be subtracted from its last-two-axis
transpose: direct and exchange contributions have different spin-selection
rules. `pppp`, `PPPP`, and `pPpP` are all bare, not antisymmetrized.

Existing UCCSDT frozen-orbital settings apply, including integer counts,
shared index lists, spin-specific lists, and frozen virtual orbitals. Fock
matrices, amplitudes, integral blocks, and `eris.mo_energy` must refer to the
same active spaces. Alpha and beta active spaces may have different sizes.

The denominator is

\[
D_{ijkl}^{abcd}=\epsilon_i+\epsilon_j+\epsilon_k+\epsilon_l
                -\epsilon_a-\epsilon_b-\epsilon_c-\epsilon_d,
\]

using each index's spin component of the active `eris.mo_energy`. Iterative
`level_shift` is not included. Following the restricted implementation,
components with `abs(D) <= 1e-15` contribute zero and generate a warning;
such a warning warrants inspection of the reference spectrum.

Complex or noncanonical inputs are rejected. The canonical check requires
the active alpha and beta Fock off-diagonal elements to be at most
`1e-7` hartree. Fock matrices and orbital energies must be finite.
Native inputs may have positive, negative, or transposed strides but must
be aligned `float64` arrays; unaligned raw-buffer views raise `ValueError`
instead of causing an implicit copy of a complete ERI or T3 array.
No orbital rotation is performed. T1 enters the UCCSDT
solution but does not enter the correction explicitly.

## Operator and contraction specification

Let `o` and `v` denote occupied and virtual **spin orbitals**, respectively.
With the reference-normal-ordered two-electron operator \(V_N\) and

\[
T_n=\frac{1}{(n!)^2}\sum t_{i_1\ldots i_n}^{a_1\ldots a_n}
 a^\dagger_{a_1}\cdots a^\dagger_{a_n}a_{i_n}\cdots a_{i_1},
\]

the specification used by the reference implementations is

\[
W_4=P_4\left([V_N,T_3]+\tfrac12[[V_N,T_2],T_2]\right)|0\rangle,
\qquad q_4=D_4^{-1}W_4,
\]
\[
E_{[Q]}=\langle0|T_2^\dagger V_N|q_4\rangle,\qquad
E_{\mathrm{extra}}=\langle0|T_3^\dagger V_N|q_4\rangle,\qquad
E_{(Q)}=E_{[Q]}+E_{\mathrm{extra}}.
\]

For reproducibility, the table below specifies seeds in the output order
`ijklabcd`. Apply independent **unnormalized** signed sums over all 24
occupied and all 24 virtual permutations to each seed. Internal `m,n` are
occupied and `e,f` virtual indices. The table uses antisymmetrized `v`.

| Contribution | Seed | Coefficient before antisymmetrization |
| --- | --- | --- |
| `voov` | `v[amie] t2[mkbc] t2[jled]` | `-1/4` |
| `oooo` | `v[mnij] t2[mlad] t2[nkbc]` | `1/16` |
| `vvvv` | `v[abef] t2[jkfc] t2[iled]` | `1/16` |
| `vvov` | `v[cdke] t3[ijlabe]` | `1/24` |
| `vooo` | `v[cmkl] t3[ijmabd]` | `-1/24` |
| Left moment `L2` | `v[abij] t2[klcd]` | `1/16` |

`W4` is the sum of the first five rows. For real integrals, `L3` is the sum
of the `vvov` and `vooo` rows. Their equivalence to the operator expressions
is checked separately using fermionic creation and annihilation operators.
The coefficients account for 144, 36, 36, 24, 24, and 36 distinct signed
placements, respectively; the repeated placements in the 576-term full
antisymmetrization supply the remaining multiplicity. The negative signs
follow the displayed operator ordering.

A full spin-orbital tensor contraction of `L2` or `L3` with `q4` is divided
by `(4!)**2 = 576`. Summing full indices separately within the
`aaaa/aaab/aabb/abbb/bbbb` spin blocks instead gives weights
`1/576, 1/36, 1/16, 1/36, 1/576`. Enumerating each occupied and virtual
same-spin combination only once gives weight one. These are alternative
counting conventions and must not be applied together. The restricted
implementation's spin-summed projection and `/12` factor do not apply to
these unrestricted tensors.

The production kernel uses a third, equivalent arrangement: occupied
same-spin combinations are unique, while virtual indices are summed in
full tiles. Its weights are consequently `1/24, 1/6, 1/4, 1/6, 1/24` for
`aaaa/aaab/aabb/abbb/bbbb`. Each surviving contraction retains its seed
name for comparison with the simple tensor reference.

Only nondecreasing virtual tile starts are visited within each spin. If a
tile start repeats `c` times, that group contributes a divisor `c!` to the
tile orbit size. The task energy is multiplied by `na! nb! / prod(c!)`,
while coordinates inside each repeated tile are still summed in full.
Tiles with too few orbitals for their repeated same-spin axes have zero
moments and are omitted. This preserves the original sum exactly and
avoids materializing every virtual combination in low-memory runs.

## Native backend, threads, and memory

The default correction backend is the dedicated C implementation in
`pyscf/lib/ccsdt/uccsdt_q.c`, built into `libccsdt`. Rebuild the library after
updating the source. The previous Python contraction implementation remains
available as the private `_kernel_python` reference for development checks.
The public API and returned energy corrections are unchanged.

Set the OpenMP thread count through the existing PySCF interface:

```python
from pyscf import lib
lib.num_threads(4)
e_q_bracket, e_q_paren = mycc.ccsdt_q()
```

Alternatively set `OMP_NUM_THREADS=4` before starting Python. The native
kernel parallelizes occupied-combination/virtual-tile tasks. Each worker
uses private work buffers, register-held matrix outputs, and SIMD loops;
it does not call BLAS, so the correction does not create nested BLAS teams.
SCF, UCCSDT, and the Python reference may separately use BLAS threads.
`mycc.einsum_backend` selects contractions in the Python reference; the
native correction uses its own contraction implementation.

Task energies are accumulated in task order, independently of scheduling.
Changing the number of threads or memory budget can change the tile
partition, so numerical agreement is checked within tolerance rather than
promising bitwise equality across all settings. Small calculations may have
too few tasks to use every requested thread and may not benefit from
additional threads.

The automatic virtual tile is selected with a cost/traffic heuristic after
the memory check. Spaces below 24 virtual orbitals use one tile; larger
spaces target roughly one quarter of the virtual range, with a minimum tile
of 12, and are reduced further when the native workspace budget requires it.
This avoids treating the largest memory-fitting tile as automatically fastest.
Pass `--block-size` to the profiling tools when comparing a fixed tile.

The production correction accumulates energy from spin blocks and orbital
blocks without retaining a complete T4 tensor. Compact inputs are accessed
locally rather than expanded into complete T3 tensors; full inputs are read
using their array strides without copying complete T3 tensors. Each native
worker has seven scratch buffers. The native workspace query measures their
required storage, and the Python wrapper budgets for the maximum possible
worker count before selecting a virtual tile size. C allocates for the
actual team inside `omp master`, with a barrier before workers use it;
allocation and release therefore occur on the same calling thread. Tiles are reduced to fit
the estimated available space, with an error if even the smallest tile
does not fit. Reducing `lib.num_threads()` can reduce scratch memory.
Python bookkeeping and allocator overhead are not exact tensor-memory
quantities. The
pre-existing in-memory ERIs and amplitudes also consume memory;
`max_memory` is not a hard process-wide memory limit.

This implementation runs on CPU RAM with OpenMP, and allocates no GPU
memory. It has no CUDA/CuPy correction backend. The native correction reads
NumPy arrays (including memory-mapped arrays) by pointer and does not write
T4 or scratch tensors to disk. A memory map can still fault pages from disk;
it does not turn the contraction into an efficient out-of-core algorithm.
HDF5 datasets must be loaded or explicitly staged by the caller.

The parent UCCSDT solver still stores complete spatial ERIs. Its in-core
AO2MO finishes and releases one chemists' spin block before transforming
the next, reducing temporary overlap while retaining contiguous
physicists' ERIs. With no cached AO integrals or density fitting, automatic
multi-electron AO2MO raises `NotImplementedError`; the zero-integral path is
reserved for a one-electron molecule. Density fitting changes the integral
approximation and must be selected explicitly.

Disk-backed DIIS belongs to the iterative solver (`incore_complete=False`),
not this correction. Put `PYSCF_TMPDIR` and an explicit `diis_file`, if used,
on suitable local scratch before running the solver. The profiler records
physical I/O counters and minor/major page faults during correction only;
its synthetic resident inputs do not measure disk-DIIS or AO2MO throughput.

## Independent validation

The test helpers provide two independent descriptions:

- A straightforward spin-orbital tensor implementation forms full small
  T4 tensors and exposes the five right moments and two left moments.
- A determinant implementation applies fermionic operators with explicit
  signs and reference normal ordering, independently of tensor permutation
  helpers. It uses
  \(V_N=V_{\rm bare}-\sum_{pq}w_{pq}a_p^\dagger a_q+E_2 I\), with
  \(w_{pq}=\sum_i\langle pi\Vert qi\rangle\) and
  \(E_2=\tfrac12\sum_{ij}\langle ij\Vert ij\rangle\).

Acceptance compares each contraction, both energy outputs, and the extra
term; a total-energy comparison alone can hide cancelling errors. Tests
also cover the RHF limit, genuine open-shell references, compact/full
storage, freezing, spin exchange, empty excitation spaces, and unchanged
inputs. An independent determinant calculation is an algebraic reference,
not an external MRCC or CFOUR numerical benchmark.

## Numerical checks

The initial **Python-backend** validation on upstream commit `2ab96f44be3e` passed 18 new
tests, 29 existing UCCSDT/full-storage regression tests, and the two existing
RCCSDT(Q) tests. Run the new tests from the repository root with:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest \
  pyscf.cc.test.test_uccsdt_q pyscf.cc.test.test_uccsdt_q_reference
```

The new tests include 60 production-versus-reference contraction checks:
six contraction classes, five spin sectors, and both amplitude formats.
They also reject nonfinite Fock matrices and orbital energies, which must
not be mistaken for vanishing denominators.

| System | `[Q]` correction / Eh | Complete `(Q)` correction / Eh |
| --- | ---: | ---: |
| BeH/STO-3G, doublet | `2.5486782537324673e-8` | `-4.584336210071545e-8` |
| BH2/STO-3G, doublet, frozen B 1s | `-1.613660873008072e-6` | `-3.3604280109780666e-6` |
| H2O/cc-pVDZ, RHF amplitudes converted to UHF | `-4.4374833993992153e-4` | `-4.917163847324406e-4` |

The first two rows were obtained independently from the determinant oracle;
their coordinates and convergence settings are fixed in the tests. The
water calculation uses the geometry and amplitudes of `test_rccsdt_q`.
Both full and compact UHF corrections agree with its RCCSDT(Q) benchmark
within `2.2e-13 Eh`. NumPy and PySCF contraction backends were exercised;
the optional pytblis backend was not installed in this validation environment.

For the original Python backend, a synthetic `nocc=(2,2)`, `nvir=(8,8)` compact calculation with virtual
block sizes 8 and 4 gave the same corrections. Additional allocations
tracked by Python's `tracemalloc` peaked at 551792 and 373833 bytes,
respectively, after formula-cache construction. A full spin-orbital T4
would occupy 134217728 bytes. These measurements exclude pre-existing
inputs and untracked native allocations; they are not total process RSS.

Native C/OpenMP validation and timing are recorded separately from those
historical Python results. The current native build passes all 64 tests:
49 common/reference and existing solver regressions, plus 15 dedicated
native tests. Native checks cover all contraction classes, compact/full and
strided inputs, virtual tiling, and actual OpenMP teams of 1, 2, and 4
threads. Run them with:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest \
  pyscf.cc.test.test_uccsdt_q_native
```

The 15 C-core tests also passed with AddressSanitizer and
UndefinedBehaviorSanitizer. Builds with OpenMP enabled and disabled were
both exercised: the native kernel observed a four-thread team in the
enabled build and a one-thread team in the serial build. A four-thread
H2O/cc-pVDZ run returned `(-4.43748339939921e-4, -4.91716384732440e-4)`
hartree, agreeing with the RCCSDT(Q) benchmark within `2.16e-13 Eh`;
correction wall times were 8.96 s for full and 10.51 s for compact storage.

## Reproducible thread benchmark

[68-benchmark_uccsdt_q.py](68-benchmark_uccsdt_q.py) creates fixed-seed,
physically symmetric spin-specific bare ERIs, antisymmetric T2 amplitudes,
and compact T3 amplitudes. These synthetic inputs measure the correction
kernel; they are not converged molecular amplitudes. Its default active
space has `nocc=(3,3)` and `nvir=(12,12)`.

Run from the repository root after building the native library:

```sh
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  taskset -c 0-3 python examples/cc/68-benchmark_uccsdt_q.py \
  --output /tmp/uccsdt-q-threads.json
```

Choose CPU IDs permitted by your allocation and, for scaling measurements,
distinct physical cores. `taskset` is optional; the script records the actual
CPU affinity and does not change it. `--nvir 16` increases the work per task;
`--nocc` changes the occupied count per spin.

The script warms and times the Python reference with one thread, then warms
the C kernel and reports medians of three repetitions at 1, 2, and 4 threads.
The JSON includes each sample, CPU and wall time, their ratio, numerical
error against the Python reference, speedup over one C thread, software
versions, Git commit and dirty status, and the loaded native-library path
and SHA256, plus SHA256 hashes of the C source, Python wrapper, and benchmark
script. Every timed correction must satisfy `atol=1e-11` and `rtol=1e-10`.
CPU/wall ratios help distinguish concurrent execution from elapsed-time
speedup; both depend on CPU allocation and workload size.

On an Intel Xeon Platinum 8176, with affinity fixed to four distinct physical
cores (CPU IDs 0-3), Python 3.11.15, and NumPy 1.26.4, the native build with
SHA256 `934116cc8aec29fe3bce5c7593047a1ce4ee519712624eacae039759589ada92`
gave the following measured wall times. C results are medians of three
repetitions; each Python reference time is one run after warm-up.

| Virtual orbitals per spin | Python, 1 thread / s | C, 1 thread / s | C, 2 threads / s | C, 4 threads / s | C 4-thread speedup over C 1-thread |
| --- | ---: | ---: | ---: | ---: | ---: |
| 12 | 1.625 | 0.424 | 0.208 | 0.114 | 3.72x |
| 16 | 3.604 | 1.240 | 0.645 | 0.333 | 3.72x |

The four-thread CPU/wall ratios were 3.86 and 3.81, consistent with
concurrent native work. Maximum errors against the Python corrections were
`4.07e-20 Eh` and `2.31e-19 Eh`, respectively. One C thread was about
3.83x and 2.91x faster than the one-thread Python reference; four C threads
were about 14.26x and 10.83x faster than that reference. These measured
results do not establish performance for larger molecular calculations.

## Subsequent efficiency and memory audit

The timing table above records the first C implementation. A subsequent
same-host audit froze that library and compared it with row-wise compact
decoding, register-held matrix outputs, symmetry-reduced virtual tiles,
and actual-team workspace allocation. Fifteen configurations used fresh
processes, one cold call and three warm calls, CPU affinity 0-7, and
`OMP_PROC_BIND=false`. The latter avoids an observed affinity collapse
when two OpenMP runtimes are loaded in this environment.

| Per-spin size / representation | Threads | Previous C / s | Optimized C / s | Previous peak RSS / MB | Optimized peak RSS / MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| o3/v12 compact | 8 | 0.1141 | 0.0458 | 101.8 | 101.7 |
| o3/v24 compact | 8 | 1.3408 | 0.5261 | 127.3 | 127.4 |
| o2/v48 compact | 8 | 1.9227 | 0.9769 | 466.0 | 284.4 |
| o3/v24 compact, forced tile4 | 4 | 11.5839 | 2.1953 | 114.5 | 114.2 |

Times are warm medians for synthetic symmetry-valid inputs with ERIs
already resident. RSS includes native allocation, sampled every 2 ms;
the sampled peak can miss short transients. The o2/v48 eight-thread
scratch bound decreased from 214.106 to 32.293 MB, distinct from the
39% process-RSS decrease. Other configurations generally retained the
same scratch requirement. Same-input energy differences across the audit
were at most `2.11e-16 Eh`.

Keeping at least two waves of independent tasks was faster and used less
memory than an experimentally tested single-wave rule. Conversely, very
small tiles still cost time: optimized o3/v24 with four threads takes
1.298 s at automatic tile24 versus 2.195 s at tile4, while RSS falls from
178.9 to 114.2 MB. This memory/time tradeoff remains after removing
permutation-equivalent work.

[69-profile_uccsdt_q.py](69-profile_uccsdt_q.py) reproduces the timing/RSS
protocol, including raw per-call samples, actual C teams, workspace queries,
input-resident memory, preparation time, and source/library hashes.

## References and literature comparison

Y. J. Bomble, J. F. Stanton, M. Kallay, and J. Gauss, “Coupled-cluster methods
including noniterative corrections for quadruple excitations,” *J. Chem.
Phys.* **123**, 054101 (2005),
[doi:10.1063/1.1950567](https://doi.org/10.1063/1.1950567).

The implementation follows PySCF's existing RCCSDT(Q) API and unrestricted
amplitude conventions introduced in
[PR #3049](https://github.com/pyscf/pyscf/pull/3049). The restricted
correction is described in [PR #3255](https://github.com/pyscf/pyscf/pull/3255).

The 2005 article's original full text remains unavailable here. The operator
structure has now been checked against Matthews, JCTC 16, 6195-6206 (2020),
[DOI:10.1021/acs.jctc.0c00522](https://doi.org/10.1021/acs.jctc.0c00522),
[arXiv:2005.10178](https://arxiv.org/abs/2005.10178), Eqs. (13), (19), and
(20), for canonical HF references. This supplements the independent
per-contraction spin-orbital and determinant checks.

An external numerical comparison uses Eriksen et al., JCP 144, 194103
(2016), [DOI:10.1063/1.4948781](https://doi.org/10.1063/1.4948781),
[arXiv:1601.06379](https://arxiv.org/abs/1601.06379). Its MRCC UHF/cc-pVDZ
frozen-core C-atom results are compared by
[70-validate_uccsdt_q_literature.py](70-validate_uccsdt_q_literature.py).
Four active electrons allow full CI in the same frozen UHF Hamiltonian to
provide the CCSDTQ limit; the example diagonalizes all 3718 determinants.
It reproduces SI Table S5's `+0.005 kcal/mol` error at printed precision
(`+0.004928021324`). The quadruples-recovery error is `-26.91563010893`
percentage points versus Table S2's `-26.91`: it exceeds the prespecified
half-last-digit window by `0.00063010893` percentage points. Tighter SCF/CC
convergence and the full FCI diagonalization do not remove that difference.
The example intentionally exits nonzero for this unresolved comparison
after saving the complete JSON; internal tests passing does not imply
that every external literature criterion passed. The paper does not provide
enough raw energies or numerical settings to identify the remaining cause.

For performance context, Jin et al.,
[arXiv:2607.00981](https://arxiv.org/abs/2607.00981), discuss RCCSDT(Q)
IJK/ABC traversal, minimum output buffers `2*Nv^4`/`2*No^4`, and scaling
near 90 threads on dual EPYC 9474F processors with pytblis. Those restricted
calculations and hardware differ from this UHF audit. Their minimum output
buffers exclude other intermediates and are not directly comparable to
our complete seven-buffer workspace or total process RSS. No cross-machine
wall-time speedup over that paper is claimed. As also discussed by Matthews
2020, aggressively tiled recomputation can increase cost beyond the nominal
unblocked `O(n^9)` algorithm.
