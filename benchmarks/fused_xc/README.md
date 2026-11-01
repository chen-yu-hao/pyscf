# Fused RKS XC integration

`pyscf/lib/dft/nr_numint_fused.c`, entered from `numint.nr_rks`. Default on;
`PYSCF_DFT_FUSED=0` (or `__config__.dft_numint_fused_xc = False`) restores the
block-loop path. `PYSCF_DFT_FUSED_TILE` sets the tile size in 56-point blocks
(default 4).

## What changes

The block-loop path evaluates AOs for blocks of up to 67,200 grid points into a
`(comp, nao, ngrids)` array, then reads that array again for the density and
again for the XC matrix, with Python, Libxc and several OpenMP regions between
the steps. For (H2O)4/def2-TZVPP/grid 6 only ~40% of the AOs are nonzero in a
grid block, yet the AO buffer is written and read in full (zeros included), and
the density and matrix kernels read 56-point slices with a stride of `ngrids`.

The fused kernel runs one OpenMP region per XC evaluation. Each thread takes a
contiguous, cost-balanced range of 224-point tiles and, per tile:

1. evaluates only the shells with a nonzero screen index in the tile into a
   compact buffer (same `GTOeval_*_iter` evaluators and per-block zero-fill
   rules as `eval_ao`);
2. forms rho, its gradient and tau through the occupied orbitals or through the
   density matrix, whichever is cheaper for that tile's active-AO count;
3. calls Libxc on the tile (`LIBXC_eval_xc_serial`, the serial form of
   `LIBXC_eval_xc` using the same cached functional objects);
4. adds the lower triangle of the active-AO XC matrix to a thread-private packed
   triangle (same symmetric 1/2/4-channel products as `VXCxc_vmat`).

The thread-private triangles are reduced once per call. Static partitioning
keeps results bitwise reproducible for a fixed thread count. With a pthreads
OpenBLAS, BLAS runs single-threaded while the kernel executes (and the thread
count is restored), since nested pools oversubscribe the cores.

Scope: `NumInt.nr_rks` with the stock evaluators and Libxc; LDA/GGA/meta-GGA
without laplacian; real densities with `hermi` 0 or 1; any number of density
matrices; spherical or Cartesian basis. Subclasses, replaced `eval_ao`/
`eval_xc_eff`/... methods, XCFun, UKS, response kernels and PBC keep their
existing paths. Note that instrumenting `NumInt.eval_ao` or `eval_xc_eff` (as the
`--profile` mode of `dft_water_grid6.py` does) also selects the old path.

## Numerical check

`pyscf/dft/test/test_numint_fused.py` compares with the block-loop path:
LDA, PBE, B3LYP, M06-L, MN15, SCAN, wB97X; orbital and density-matrix routes,
fractional and negative occupations; several density matrices and `hermi=0`;
tile sizes 1, 3, 4, 8; 1 and 4 threads; Cartesian f and spherical g functions;
grids that are not a multiple of 56 points; missing screening tables; SCF
energies. On (H2O)4/def2-TZVPP/grid 6 at a converged density the differences are
|dExc| <= 3e-14, |dN| <= 9e-14 and max|dV| <= 1.5e-14.

## Measurements

4-vCPU Intel Xeon VM (one NUMA node), GCC Release build, OpenBLAS 0.3.26,
BLAS limited to one thread, def2-TZVPP, grid level 6, direct J/K, conv_tol 1e-8.
Medians of three alternating runs for the XC column; SCF rows are single
alternating runs. The VM is noisy; read ratios below ~1.1x as parity.

XC call on a converged density, (H2O)4 (nao 236, 529,520 points):

| Functional | Threads | Block loop | Fused | Speedup |
|---|---|---|---|---|
| PBE  | 1 | 4.00 s | 1.52 s | 2.64x |
| PBE  | 2 | 2.93 s | 0.89 s | 3.28x |
| PBE  | 4 | 1.27 s | 0.45 s | 2.80x |
| MN15 | 1 | 6.46 s | 2.18 s | 2.97x |
| MN15 | 2 | 3.24 s | 1.18 s | 2.75x |
| MN15 | 4 | 1.62 s | 0.61 s | 2.64x |

Complete cold SCF (same cycle counts and energies to 1e-10 Eh):

| System | Functional | Threads | Block loop | Fused | SCF speedup | XC time |
|---|---|---|---|---|---|---|
| (H2O)2 | PBE  | 4 | 4.86 s | 1.98 s | 2.45x | 4.41 -> 1.58 s |
| (H2O)2 | MN15 | 4 | 6.18 s | 3.00 s | 2.06x | 5.62 -> 2.39 s |
| (H2O)4 | PBE  | 1 | 80.8 s | 35.7 s | 2.26x | 59.7 -> 15.2 s |
| (H2O)4 | PBE  | 4 | 23.3 s | 11.0 s | 2.12x | 17.3 -> 4.8 s |
| (H2O)4 | MN15 | 1 | 84.0 s | 42.5 s | 1.98x | 60.7 -> 19.6 s |
| (H2O)4 | MN15 | 4 | 23.9 s | 13.6 s | 1.76x | 17.2 -> 6.7 s |
| (H2O)8 | PBE  | 4 | 98.8 s | 53.2 s | 1.86x | 60.6 -> 16.2 s |
| (H2O)8 | MN15 | 4 | 94.4 s | 57.2 s | 1.65x | 54.4 -> 19.1 s |

Small systems (grid 3, 4 threads, one XC call) are never slower: water/def2-SVP
1.07-3.8x, water/def2-TZVPP 1.08-2.7x, benzene/def2-SVP 1.5-2.2x. LDA on small
molecules is close to parity because AO exponentials dominate there.

With BLAS left at its default four pthreads, the block-loop XC call for
(H2O)4/PBE/4 threads takes 4.9 s and the fused one 0.49 s (unchanged from the
single-thread BLAS setting).

After this change XC is ~30-50% of SCF time; at 8 waters direct J/K is the
larger part. A `perf` profile of the fused kernel is dominated by DGEMM
(~36-49% of samples), then AO evaluation (~17%) and Libxc (~6%).

Reproduce (after building the native libraries):

```sh
python benchmarks/fused_xc/bench_fused_xc.py xc  "$PWD" 4 MN15 4
python benchmarks/fused_xc/bench_fused_xc.py scf "$PWD" 4 MN15 4 1
```

Not covered yet: UKS (`nr_uks`), response kernels (`nr_rks_fxc`), NLC grids.
