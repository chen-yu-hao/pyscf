# Automatic NUMA integration on the DFT branch

Standard molecular `dft.RKS(mol).kernel()` now checks whether local-memory
grid partitions are useful. No example-directory imports, CPU ids, manual
pool management, or changes to a normal SCF script are needed.

The default is `PYSCF_NUMA=auto`. To compare against native integration:

```sh
PYSCF_NUMA=off python calculation.py
PYSCF_NUMA=auto python calculation.py
```

Use the same scheduler allocation and thread settings for both runs. On Linux,
install `numactl` for memory binding (`apt-get install numactl` on Debian/Ubuntu).
The Python `threadpoolctl` dependency is installed with this branch. No GPU or
QCINT backend is required. Ordinary libcint remains supported.

## What happens automatically

* Capture CPU affinity before PySCF imports NumPy/OpenMP. Detect physical
  cores and local memory nodes. The total worker allocation does not exceed
  the OpenMP budget, known CPU allocation, Slurm CPUs-per-task, or cgroup-v2
  ancestor CPU quotas. MPI multi-rank jobs and v1 CPU-controller containers
  retain native integration until their resource handling is supported.
  An already converged SCF object also retains native integration when rerun.
* Skip small calculations (<200 AO or <100,000 grid points), tasks with fewer
  than eight possible SCF cycles, insufficient memory, fewer than two usable
  local-memory groups, or missing/denied NUMA binding.
* Wait for an occupied-MO-tagged density, then measure the original numerical
  integration. If it takes at least 0.5 seconds, try one equivalent partitioned
  evaluation. Compare electron count, XC energy and every potential-matrix
  element against the original result. Return the original result for this
  calibration call.
* Enable persistent workers only when the measured partition call is at least
  20% faster and six subsequent calls would recover both worker initialization
  and the extra calibration evaluation. This is a local heuristic based on one
  paired sample, not a guarantee against timing noise or early SCF convergence.
* Divide the available workspace hint between groups, reserve worker overhead,
  and also consider host/cgroup-v2 memory headroom. These remain estimates,
  not hard operating-system memory limits.
* Reuse workers for later density matrices. Hash geometry/basis/environment,
  grid coordinates, weights and masks to invalidate stale pools, including
  in-place changes. Changes in functional, cutoffs, thread/memory settings and
  main-thread affinity invalidate the pool too.
* Close workers at normal SCF completion, SCF exception, reset, object cleanup
  or interpreter exit. Worker failures return to the native calculation.

At normal PySCF INFO verbosity, successful activation looks like:

```text
NUMA XC enabled: [8, 8] cores/group; native 10.058s, partition 6.265s
```

These numbers and group sizes depend on the actual machine. A slower candidate
logs that native integration was retained. There is deliberately no force-on
mode that bypasses the numerical/performance checks.

## Current scope

Only the standard real, closed-shell, spherical-basis molecular RKS path with
the built-in Libxc interface is enabled. UKS, symmetry-adapted subclasses,
periodic/response integrations, custom NumInt methods or XC callbacks, custom
registered Libxc functionals, omega overrides and nonlocal correlation retain
their original behavior. In particular, a fitted custom functional is not
silently replaced by its built-in counterpart in a child process.

The optimization does not change grids, screening tolerances or convergence
criteria. It does not force eight threads, a particular NUMA topology, a native
instruction set, or a specific integral backend. Performance is hardware- and
problem-dependent. Compare complete cold SCF walltime including calibration;
kernel speedup alone is not end-to-end speedup.

Applications importing NumPy/OpenMP before PySCF may already have narrowed the
visible allocation; in that case automatic selection can conservatively retain
the native path. Use the scheduler to allocate/bind the intended cores before
launching Python. Do not assume the optimization enlarges a scheduler quota.

## Initial end-to-end check

On one EPYC 7551 host, n-hexane/MN15/def2-TZVPP (382 AO), native grid6 and the
same 24-core allocation: native 186.754 s, automatic 135.302 s (1.380x).
Both cold SCFs converged in 12 iterations; energy difference 2.27e-13 Eh.
The automatic time includes detection, the extra paired evaluation, worker
startup, transfers and final verification. The program selected two local
8-core groups without caller-specified partitions. Each mode was measured once;
this demonstrates the automatic workflow, not a hardware-independent speedup.
