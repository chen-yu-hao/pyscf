# Optional CPU-local numerical integration

Standard molecular RKS now supports automatic detection and direct activation
without importing these examples. See [automatic NUMA integration](../../../pyscf/dft/NUMA.md).
The tools below remain available for explicit experiments and custom orchestration.

These examples accompany the DFT branch's fused XC workspace and shell-run
accumulation improvements. They are explicit opt-in tools, not a replacement
for the default NumInt API. No CPU model, CPU number, NUMA node, installation
directory or fixed thread count is assumed by the worker implementation.

## Native changes

The fused XC potential workspace defaults to 256 MiB. Set
`PYSCF_DFT_VMAT_MAX_MEMORY_MB` to a positive integer to change that limit (the
unit is MiB). When a requested OpenMP team exceeds the budget, only that fused
region reduces its team. J/K and other kernels retain their thread settings.
If even one thread cannot fit, the existing fallback remains available.
Malformed limits use the finite default. Shell runs share screening checks and
accumulate consecutive AO entries; the existing screening thresholds and
summation order are preserved. J/K rejects forbidden symmetry blocks earlier.

## Portable, opt-in grid workers

Requirements: Linux, the active PySCF installation, and `threadpoolctl`.
NUMA memory binding additionally requires `numactl`. Without it, a worker with
no requested memory-node binding still sets CPU affinity before loading its
numerical libraries. An application should retain native integration if its
requested binding is unsupported.

Capture the scheduler's CPU allocation before importing NumPy/PySCF: OpenMP
may pin the main thread to a single CPU after import. `topology.discover()`
respects this allocation, removes SMT duplicates, checks allowed memory nodes
and identifies memoryless nodes. Use `local_grid_partitions(topology, cpus)`
to propose local groups; a single usable memory node returns an empty list,
meaning retain native integration. The allocation must also respect any
scheduler CPU quota that is not represented by CPU affinity.

`PartitionedXC` owns persistent fresh processes, copies grid data into local
memory, and exchanges only density/MO information and final results per call.
It uses the calling Python executable and checks the worker's PySCF path.
Grid shares follow group core counts, with 56-point mask and 8-point kernel
alignment. `phase_threads.limit_xc_threads` independently limits this SCF
object's XC calls and restores the J/K budget, including on exceptions.

For example, after obtaining a bounded allocation and building a standard
real RKS grid:

```python
from xc_process import PartitionedXC

with PartitionedXC(mol, grids, 'MN15', partitions,
                   allowed_cpus=allocation, memory_mb=4000,
                   ni_cutoff=mf._numint.cutoff,
                   log_prefix='run-xc') as pool:
    nelec, exc, vxc = pool.evaluate(dm)
```

Reuse the pool for updated densities. Preserve `mo_coeff` and `mo_occ` tags
when available. A complete SCF can wrap that object's `nr_rks` with
`pool.evaluate(dm)` after its grid has been initialized; keep the pool open
through the SCF and restore the original method on exit. Rebuild the pool
after changing molecule, basis, functional, grid, cutoff or custom XC state.
The prototype supports one fixed molecule/grid and real RKS only. It does not
serialize custom NumInt implementations or customized Libxc parameters.
UKS, geometry optimization and custom XC callbacks should use the native path.

Measure before enabling. `choose_measured` requires three numerical-checked
paired samples, at least 10% savings in each, and expected savings exceeding
initialization cost. Include any calibration cost in that budget. It is a
selection utility, not an automatic calibration service. Measure against the
best existing configuration, including transfer/startup costs and full SCF.
Neither eight threads nor a two-process split is a universal recommendation.

Run the example checks from this directory using the intended environment:

```sh
python -m unittest -v test_topology
python test_xc_process.py
```

## Optional integral backend

The normal libcint backend remains the default. The benchmark candidate also
used QCINT 6.2.0 (`cdda2892fc988d94e66a33e4807901576b3e8391`). PySCF already
provides `USE_QCINT`; this change does not force QCINT or a native ISA on users.
For the measured configuration, QCINT was built separately with these options:

```sh
cmake -S qcint -B build-qcint \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG -fno-fast-math" \
  -DBUILD_MARCH_NATIVE=ON \
  -DWITH_RANGE_COULOMB=1 -DWITH_F12=1 -DWITH_4C1E=1 \
  -DWITH_FORTRAN=0 -DWITH_CINT2_INTERFACE=0 \
  -DMIN_EXPCUTOFF=20 -DKEEP_GOING=1 \
  -DCMAKE_INSTALL_PREFIX="$PWD/pyscf/lib/deps" -DCMAKE_INSTALL_LIBDIR=lib
cmake --build build-qcint --parallel
cmake --install build-qcint
```

That native build is for the build host; build on each target machine or use
explicit compatible target flags for distributable binaries. Keep
`-fno-fast-math`. Then configure PySCF against that dependency with
`BUILD_LIBCINT=OFF`. Use the environment's existing libxc/xcfun and BLAS as
appropriate; the measurements retained PySCF's O2/SSE3 settings. The default
portable libcint route is also supported by the native changes.

## Measured evidence and limits

One EPYC 7551 host, CPU only, MN15, AME418 water and n-hexane. The host has
physical memory on two NUMA nodes. Measurements are exploratory, not promises
for other hardware or functionals.

* Native grid6, n-hexane/def2-QZVPPD, 840 AO: fixed-density XC 20.384 s with
  one local 8-core group versus median 10.696 s with two local 8-core groups
  (one control, three candidate calls). Maximum Vxc element differences are
  below 1e-10.
* Native grid6, n-hexane/def2-TZVPP, 382 AO, the same 24-core allocation:
  complete cold SCF 192.086 -> 124.175 s (1.547x), both 12 iterations;
  energy difference 6.25e-13 Eh. Startup and transfers included.
* Separate common-grid comparison, unpruned 120 radial x 770 angular points
  per atom, 382 AO, same 24 cores and two memory nodes: upstream PySCF
  `44313dfb` 421.416 s, this candidate with QCINT 216.834 s, G16 A.03
  215.046 s. Thus 1.943x versus upstream, 0.992x versus G16: essentially tied
  with G16 in this sample. Upstream's 8/24-thread XC calibration selected 24.
  Both PySCF runs took 12 iterations; G16 took 11. These are cold native
  initial guesses, fixed integral tolerance 1e-13 and density convergence
  criteria; G16 used `NoVarAcc`. Each complete SCF is one sample.

Do not combine grid6 timings with the common-grid timings or multiply speedup
ratios from different experiments. G16/Libxc MN15 has a separately checked
absolute-energy offset (~0.031 kcal/mol for this n-hexane case); the two PySCF
implementations agree within 5.12e-13 Eh. The comparison does not cover every
Gaussian setting, other CPUs, or open shells.

Exact ERI caching and overlapping XC with J/K were tested but are not included:
their gains were insufficient or regressed compared with local grid partitions.
No binary libraries, molecular checkpoints, private job data, or benchmark
artifacts are required by these examples.

AI assistance: Codex assisted with implementation, numerical tests and local
benchmark analysis. The source and test results were reviewed before the
user-authorized branch publication.
