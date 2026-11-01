# Next optimization: bounded semi-direct cache lookup (opt-in)

## Priorities and scope

1. Finish the already-running PR7 full-SCF audit without restarting or altering it.
2. Reduce overhead in its native cache: precompute immutable shell metadata; bypass cheap integrals before constructing keys; stop probing at the first genuinely never-used hash slot.
3. Independently measure uncached-vs-old-cache (PR7) and old-cache-vs-new-lookup (this PR). Do not multiply unrelated runner speedups. Tune admission/capacity only after these measurements.
4. If J/K replay is promising, perform fresh-process full-SCF A/B with the unchanged energy <=1e-8 Eh and common-density Ne/Exc/Vxc/J/K atol=1e-9, rtol=1e-10 gates. Until then this is NOT a new SCF performance result.
5. Later larger algorithm changes (batched native integral/contraction work, sparse exchange task ordering) require separate ablations. RI/DF, ISDF, seminumerical exchange, grid/threshold changes are not silently included.

The resource trade-off follows public NWChem semi-direct and Psi4 direct/PK/out-of-core designs. This native lookup implementation is independent, no third-party code copied.

Primary references:
- https://nwchemgit.github.io/Hartree-Fock-Theory-for-Molecules.html
- https://nwchemgit.github.io/Density-Functional-Theory-for-Molecules.html
- https://psicode.org/psi4manual/1.10.x/scf.html

## Correctness invariant

An entry is immutable after publication. In the new policy a slot that has once been claimed must never again be marked never-used. An arena-reservation failure leaves a reusable tombstone, not an empty slot. Otherwise early-stop lookup could miss a colliding entry published farther along the probe chain. Tombstones do not stop lookup; insertion may reuse them. Existing maximum eight-probe bound remains.

Only the new `fast_lookup=True` policy uses this invariant. Default `cached_eri(...)` and original native constructor retain legacy probing. Metadata allocation is charged to the same budget, not added above it. No integral cutoff, FP64 values, screens or contractions change. The new default-off argument is per context, not a process-global environment toggle.

```python
from pyscf.scf.eri_cache import cached_eri
with cached_eri(mf, max_mb=256, fast_lookup=True):
    energy = mf.kernel()
```

The complete package must be built first. The Python API raises a clear error if new native symbols are absent.

## Evidence actually executed locally

The standalone C harness has synthetic integral callbacks, NOT libcint/PySCF molecules. Legacy and new policy each passed 18 configurations spanning 1/2/4 OpenMP threads, two budgets and three shell counts, including cold/warm/full-cache, zero-valued integrals, input identity/content changes, simultaneous host callers and release. Both also passed under UBSan. Four targeted white-box checks passed normally and under UBSan: collision behind a failed publisher, tombstone reuse, injected reservation failure, and the eighth permitted probe. Test-only failure injection is not present in production source.

Synthetic probe-count reductions range from zero on a fully occupied warm table to large reductions on an almost empty cold table. These are operation counts, not a speedup claim. No full molecular run has been performed locally; this environment lacks the complete PySCF checkout and network access.

## Independent remote gate (no duplicate SCFs)

Pin original calculation bb8e98d0 and the new candidate separately from the workflow commit. Share the same portable libcint/Libxc dependencies. Keep prior experimental flags off; dense BLAS and MGGA SIMD remain on equally.

Reuse the accepted round-five artifact 12992745465 from run41075953719, SHA256 16e60b7cda1a97aff3b30a926512c912c5cf98de37bcc7c9c3d05d25244fe596. Only read its fixed densities; never rerun that SCF matrix.

Four J/K replay cases: n=4/8 water, def2-TZVPP, 2 threads, 64/256 MiB budgets. Three variants: old-source legacy cache, new-source legacy control, new-source fast lookup. Each has three fresh-process repeats, hence 36 records. Each record builds J/K for D, 0.1D and 0.01D, includes cold allocation/fill, and checks every matrix against an independent uncached optimizer. Diagnostics are enabled only after timing. Report sequence time, cold/warm per-call times, spread, RSS, exact library/source/CPU provenance, and negative results.

This is a J/K-only diagnostic. It does not execute SCF, re-evaluate XC, produce a new total energy, or establish SCF convergence/speed. Full-SCF release gates are retained for the subsequent stage, not replaced by matrix replay.
