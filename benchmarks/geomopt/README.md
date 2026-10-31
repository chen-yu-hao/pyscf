# Geometry-optimization profile of the DFT performance stack

Date: 2026-10-31. Source under test: `perf/dft-eri-saturation-filter-20261031`
(`d5913330e`), i.e. `DFT` plus rounds 1-3 and the ERI-cache series, with every
opt-in experiment left at its default (off). Nothing in `pyscf/` is changed by
this directory: it contains measurement harnesses, raw results and one
prototype kernel.

Machine: 4-vCPU Intel Xeon @ 2.1 GHz VM, one NUMA node, 15 GB. GCC 13.3,
CMake Release (`-O3`), OpenBLAS 0.3.26 limited to one BLAS thread (as in the
existing water benchmarks), NumPy 2.5.3, SciPy 1.18.1, pyberny 0.7.0. The VM is
noisy; all speedups below are paired measurements in the same process or
back-to-back runs. Single samples unless stated; treat them as indicative.

## Why a geometry-optimization profile

All existing acceptance runs time cold single-point SCF. A geometry step runs a
warm-started SCF **plus** an analytic gradient through the scanner objects in
`pyscf.grad.rhf.SCF_GradScanner` / `pyscf.scf.hf.SCF_Scanner`. Neither the
gradient code (`pyscf/grad/`) nor the scanner path is touched by the stack.

## 1. Where a geometry step spends its time

`prof_geomopt.py`: (H2O)4 from `dft_water_grid6.py`, def2-TZVPP, grid level 6
(nao 236, 529,520 points), 4 OpenMP threads, two scanner steps (cold, then
0.02 A perturbed). Warm step (step 1):

| Phase                    | PBE           | MN15          |
|--------------------------|---------------|---------------|
| Step wall time           | 31.3 s        | 47.9 s        |
| SCF XC (10 calls)        | 11.9 s (38%)  | 18.2 s (38%)  |
| **Gradient XC (1 call)** | 11.7 s (37%)  | 17.7 s (37%)  |
| Gradient J / JK          | 4.9 s (16%)   | 6.1 s (13%)   |
| Grid build               | 0.2 s         | 0.2 s         |
| Rest (in-core ERI, J/K, diag, 1e) | ~2.6 s | ~5.8 s     |

One gradient XC call costs as much as all XC work in the SCF.
Raw data: `results/stock_*.json`.

## 2. NUMA workers never engage inside a geometry optimization

`numa._supported()` requires `type(ks) is RKS` and `not ks.converged`.
`GradScanner.__init__` replaces `base` with `base.as_scanner()`, whose class is
the dynamically built `RKS_Scanner`, so the first test is always false.
`SCF.reset()` does not clear `converged`, so the second is false from step 2
onwards. Every decision logged in `results/stock_*.json` (`numa_decisions`)
is `False` with `type: RKS_Scanner`.

Even with the gate fixed, `RKS.scf()` (finally block) and `KohnShamDFT.reset()`
both close the pool, so every step would restart workers. `bench_pool_startup.py`
on this VM: pool startup 0.5 s, plus a first evaluation 1.2-1.9 s slower than
steady state (first touch and initialization), about 1.5-2 native XC calls per
step before grid-transfer costs on larger systems. (Same run, steady state:
2 x 2-thread workers 1.29-1.41 s versus native 4-thread `nr_rks` 1.57-1.71 s on
this single-socket VM; two samples, indicative only.)

## 3. XC gradient: density-first contraction (prototype, exact)

`pyscf.grad.rks.get_vxc` builds 3 x nao x nao matrices (6 nao^2 GEMMs for GGA,
15 for MGGA) and only afterwards contracts them with D per atom.
`prof_xc_grad.py` shows it is memory-traffic bound, not FLOP bound (PBE, 4 threads):

| Component                                   | Time   | Share |
|---------------------------------------------|--------|-------|
| `_make_dR_dao_w` (12 `_scale_ao` + 9 serial NumPy `+=`) | 3.74 s | 46% |
| `_dot_ao_ao` (nao^2 GEMMs)                  | 2.77 s | 34%   |
| `eval_ao`, deriv 2                          | 0.98 s | 12%   |
| rho + libxc                                 | 0.30 s | 4%    |

Only `f[x, mu] = sum_nu V^x_{mu nu} D_{mu nu}` is needed. With
`P = phi D`, `F = aow D`, `Q_j = (d_j phi) D` (MGGA), written as thin GEMMs
through the occupied MOs, the remainder is one pass over the AO derivatives.
`grad_xc_kernel.c` does that pass in a single OpenMP loop. Threads own disjoint
AO indices, so there are no private matrices and no reduction; it honours
`non0tab`. `xc_grad_contracted.get_xc_grad_fused` is the driver.

`bench_xc_grad.py` (same density, same grid; `results/xc_grad_fused_vs_reference.txt`):

| Case                | Reference | Fused  | Speedup | max abs diff |
|---------------------|-----------|--------|---------|--------------|
| PBE, 4 threads      | 9.03 s    | 2.47 s | 3.66x   | 5.8e-15      |
| MN15, 4 threads     | 13.46 s   | 3.40 s | 3.96x   | 1.5e-14      |
| PBE, 1 thread       | 21.15 s   | 6.91 s | 3.06x   | 1.6e-14      |
| MN15, 1 thread      | 43.77 s   | 11.27 s| 3.89x   | 1.2e-14      |

End to end (`FUSED=1 prof_geomopt.py`, routed through `extra_force`; same SCF
cycles; energies within 3e-13 Eh; on 2-water checks full nuclear gradients are
equal to the 1e-10 logging precision):

| Warm step | Stock  | Fused  | Step speedup |
|-----------|--------|--------|--------------|
| PBE       | 31.3 s | 23.1 s | 1.35x        |
| MN15      | 47.9 s | 35.2 s | 1.36x        |

Prototype limits: RKS only, GGA/MGGA (LDA has the NumPy path only), no grid
response, no NLC. The thin GEMMs use canonical MOs and ignore `non0tab`; for
large sparse systems a screened `phi D` (like `_dot_ao_dm_sparse`) should
replace them.

## 4. Gradient J/K is 4x a Fock build

`bench_jk.py`, same density, 4 threads (`results/jk_gradient_vs_scf.txt`):
direct SCF J 1.06 s, JK 1.38 s; gradient J 4.56 s (4.3x), JK 5.56 s (4.0x).
`grad.rhf.get_jk` uses `int2e_ip1` with `s2kl` symmetry and accumulates
3-component nao x nao matrices. An energy-contracted driver (s8 unique quartets,
three derivative centres plus translational invariance, density-product
screening, per-thread natm x 3 accumulators) needs 3/8 rather than 1/2 of N^4
ip1 evaluations and removes the matrix accumulation. Not prototyped here.

## 5. Fewer SCF cycles per step

`guess_experiment.py`: pyberny optimisation of a distorted glycine, PBE/def2-SVP,
grid 3, 4 threads (`results/guess_*.json`):

| Variant                                   | Steps | SCF cycles | Final geometry vs stock |
|-------------------------------------------|-------|------------|-------------------------|
| Stock scanner                             | 14    | 117        | n/a                     |
| Loewdin-projected occupied MOs as guess   | 14    | 111 (-5%)  | 1.5e-5 Bohr             |
| + tolerance from the last nuclear gradient| 14    | 98 (-16%)  | 2.3e-5 Bohr, dE 8.7e-10 Eh (tight re-SCF) |

Adaptive rule: `conv_tol_grad = clip(0.03 * max|g_prev|, 10**-4.5, 1e-3)`,
`conv_tol = max(conv_tol_grad**2, 1e-9)`, reaching the default 1e-9 near
convergence. In this small case the gradient was 59% of total time
(46.9 s gradient versus 32.8 s SCF).

## Reproduce

Build native libraries for the tree under test, then, from this directory:

```sh
python prof_geomopt.py   <root> 4 PBE def2-tzvpp 6 4          # stock step profile
FUSED=1 python prof_geomopt.py <root> 4 PBE def2-tzvpp 6 4    # fused XC gradient
VARIANT=fused python bench_xc_grad.py <root> 4 MN15 def2-tzvpp 6 4
python prof_xc_grad.py   <root> 4 PBE def2-tzvpp 6 4
python bench_jk.py       <root> 4 def2-tzvpp 4
python guess_experiment.py <root> adaptive PBE def2-svp 3 4   # stock|project|adaptive
python bench_pool_startup.py <root> 4 def2-tzvpp 6            # needs _numa_worker
```

`xc_grad_contracted.py` compiles `grad_xc_kernel.c` with `gcc -O3 -fopenmp`
on first use. `prof_geomopt.py` and `bench_pool_startup.py` need the DFT stack
(`pyscf.dft.numa`); the other scripts also run on upstream master.
