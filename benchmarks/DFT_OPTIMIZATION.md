# DFT optimization experiment, 2026-10-29

## Status and scope

Baseline: `3a6238ea20a467eabaf6290e0f23de2ea23460ec` on DFT.
This is an experimental branch, not a change to DFT itself.
No complete grid6/def2-TZVPP molecular SCF measurement has been obtained in this
round. The local environment could compile the isolated C kernel but could not
install the full PySCF dependency stack. At the initial status check the GitHub
benchmark branch still had zero workflow runs. Do not confuse the measurements
below with water-cluster SCF timings or confirmed GitHub Actions results.

## Implemented changes

### 1. Exact pre-GEMM tile screening (explicit opt-in)

`pyscf/lib/dft/mgga_vmat.c` now offers
`PYSCF_DFT_VMAT_TILE_SCREEN=1`. The default remains the existing path.
For a sparse tile, it checks shell runs before DGEMM and skips the multiplication
only when every shell pair would already be rejected by the existing pair/bin
masks. There is no new threshold, grid approximation, precision reduction, or
change to the surviving accumulation order. Dense integration is unchanged.

This is NOT enabled by default because some measured cases regress.
The standalone build used the exact baseline file (Git blob
`0571a99dfdf8e4da97b02abd1d8bb1a2f122fd4b`) and the exact candidate file
(`5b7ba97021aff049fecb13b2cf1b9f3b846689d4`).

Local environment: x86-64 Xeon Platinum 8573C virtual machine, five visible CPUs
but a four-CPU cgroup quota; GCC, `-O2 -msse3 -fno-fast-math`, OpenMP, OpenBLAS
0.3.30 LP64 from SciPy with BLAS limited to one thread. Both versions used the
same compiler and BLAS. Minimal header shims only supplied the existing
MIN/MAX macros, OpenMP declarations and BLAS prototype. No full PySCF build was
used for these isolated tests.

Executed: 945 C-kernel input/thread combinations (LDA/GGA/MGGA; 1/2/4 OpenMP
threads; non-packed derivative strides; partial grid tiles; shells crossing AO
tile boundaries; zero, mixed and local masks). Candidate versus baseline was
bitwise identical in all tested cases. Maximum error against independent NumPy
formulas was 1.99e-13 for these synthetic arrays. Two asymmetric-mask fallback
checks also passed. These are not physical energy error bounds.

Synthetic example: 472 AO, 3,584 grid points, seven alternating timing samples,
median milliseconds per call. Allocation/packing is included. These arrays
are not an actual (H2O)8 grid6 calculation.

| OpenMP | Mask pattern | Channel | Baseline ms | Candidate ms | Ratio |
|---|---|---|---:|---:|---:|
| 1 | dense mask | GGA | 97.624 | 95.511 | 1.022 |
| 1 | dense mask | MGGA | 164.437 | 164.830 | 0.998 |
| 1 | local pairs | GGA | 91.740 | 55.730 | 1.646 |
| 1 | local pairs | MGGA | 152.779 | 94.271 | 1.621 |
| 4 | dense mask | GGA | 44.773 | 61.034 | 0.734 |
| 4 | dense mask | MGGA | 74.782 | 82.107 | 0.911 |
| 4 | local pairs | GGA | 40.274 | 24.531 | 1.642 |
| 4 | local pairs | MGGA | 70.645 | 43.910 | 1.609 |
| 4 | mixed screen | GGA | 29.820 | 36.692 | 0.813 |
| 4 | mixed screen | MGGA | 83.934 | 65.959 | 1.273 |

Ratios are baseline/candidate, not end-to-end speedups. The environment is
shared and noisy; no confidence interval or universal scaling claim is made.
The regressions are retained in the report, not discarded as inconvenient.

### 2. Compact NUMA payload and one serialization per evaluation

The worker coordinator uses the new private `_numa_payload` helper. It removes
only exactly-zero-occupation MO columns, preserving fractional, negative and
tiny nonzero occupations. The density matrix is not modified. Nonstandard tags
and the all-zero-occupation case retain their original representation.

It serializes the payload once and sends the same Connection-compatible frame
to every worker, instead of pickling the full payload once per worker. Ordinary
`Connection.recv()` remains the receiver; the socket framing/protocol is unchanged.

Executed: 10 native-independent tests including socket-pair send/recv,
serialization count, no-input-mutation, noncontiguous arrays, fractional/negative
occupations and orbital-density/gradient equivalence. All passed.

Synthetic float64 payload sizes, bytes **per worker**:

| AO / nonzero MOs | Before | After | Reduction |
|---|---:|---:|---:|
| 59 / 5 | 56,423 | 30,500 | 45.94% |
| 236 / 20 | 893,297 | 483,749 | 45.85% |
| 472 / 40 | 3,568,598 | 1,933,908 | 45.81% |
| 944 / 80 | 14,266,006 | 7,734,164 | 45.79% |

This reduces serialization and transport work; it does not prove an equivalent
SCF speedup. Genuine NUMA end-to-end performance still requires suitable hardware.
The existing `test_numa_process.py` exercises actual workers with MO-tagged and
fractionally occupied densities and is included in the pending Actions suite.

## Repaired molecular benchmark

`benchmarks/dft_water_grid6.py` compares two separately compiled source trees.
The Actions configuration pins the old DFT SHA above; both builds share identical
native dependencies and compiler flags. It explicitly builds C libraries with
CMake rather than assuming an editable Python install rebuilt them.

Default matrix: PBE and MN15; n=1,2,4; def2-TZVPP; grid level 6; 1,2,4 OpenMP
threads; BLAS=1; three fresh-process SCFs per version, alternating order. An
additional profiled pair is excluded from speedup medians. The deterministic
4-Angstrom-spaced monomers are performance geometries, not optimized clusters.

The old benchmark replaced `ni.nr_rks` on the instance, which makes the NUMA
compatibility guard reject it. The replacement records XC at the dispatcher and
uses class-level instrumentation with staticmethod preservation, leaving NumInt
instance compatibility checks intact. Profiling records nested timers; do not
sum parent and child timers. Child NUMA subphases are not captured by parent-only
class instrumentation.

Numerical checks require convergence, matching grid hashes, total-energy
agreement within 1e-8 Eh, and Ne/Exc/full Vxc agreement on the **same first-baseline
density**, with atol=1e-9 and rtol=1e-10. Verification runs outside the SCF timer.
Actual source path/SHA, grids, iterations, BLAS pools, affinity, and whether the
SCF used cached in-core ERIs are recorded. Failures keep logs and partial results.
Two mocked harness tests passed locally; these only test pairing/aggregation and
failure detection, not quantum chemistry.

Example after both source trees have been compiled:

```sh
python benchmarks/dft_water_grid6.py \
  --baseline-root /path/to/pyscf-dft-baseline \
  --candidate-root "$PWD" --sizes 1 2 4 --xc PBE MN15 \
  --threads 1 2 4 --repeat 3 --profile --numa off \
  --output benchmark-results
```

Do not promote the tile-screen option or claim whole-SCF improvements until the
paired molecular results show a reproducible net gain. Do not replace the C
kernels with Fortran merely because of language choice; the measured candidates
address avoided work and redundant data movement instead.
