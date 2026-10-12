# Fully iterative UCCSDTQ

`cc.UCCSDTQ` solves the unrestricted single, double, triple and quadruple
amplitude equations together. Its `e_tot` is the final iterative CCSDTQ total
energy. The existing `ccsdt_q()` method remains the separate noniterative
UCCSDT[Q]/UCCSDT(Q) correction to a converged UCCSDT calculation.

```python
from pyscf import gto, scf, cc, lib

lib.num_threads(4)
mol = gto.M(atom='Be 0 0 0; H 0 0 1.342', basis='sto-3g', spin=1)
mf = scf.UHF(mol).run()
mycc = cc.UCCSDTQ(mf, frozen=0, compact_tamps=True).run(max_cycle=100)
assert mycc.converged
print(mycc.e_tot)
```

See `71-uccsdtq.py` for a complete example comparing compact and full T3/T4
storage, starting both calculations from converged UCCSDT amplitudes.

## Equations and conventions

The solver uses real UHF orbitals and the full coupled-cluster projection
`<Phi_mu|exp(-T) H exp(T)|Phi_0> = 0`, with `T = T1 + T2 + T3 + T4`.
T1 dressing reuses the existing UCCSDT implementation. The complete R4
residual and T4 feedback to R2/R3 use the connected spin-orbital term table
in `pyscf/cc/_uccsdtq_terms.py`. Each seed records its diagram family,
contraction labels and rational coefficient. The table uses the
unnormalized occupied and virtual antisymmetrizers; its coefficients must
not be multiplied by another factorial normalization.

The conventional spin-orbital cluster definition is
`T_n = sum(t_ij...^ab... a_a^+ a_b^+ ... a_j a_i)/(n!)^2`, with bare
antisymmetric integrals `<pq||rs>`. The iteration retains off-diagonal
Fock terms. Orbital energy differences and the iteration `level_shift`
enter the Jacobi preconditioner, not a redefinition of the residual.

Methods and factorization context:

- A. Barman, G. H. Jones, J. M. L. Martin, *Chem. Phys. Lett.* **898**,
  143003 (2026), DOI: [10.1016/j.cplett.2026.143003](https://doi.org/10.1016/j.cplett.2026.143003),
  the UHF/ROHF implementation and performance context supplied with the request.
- S. A. Kucharski and R. J. Bartlett, *J. Chem. Phys.* **97**, 4282 (1992),
  spin-orbital CCSDTQ equations.
- D. A. Matthews and J. F. Stanton, *J. Chem. Phys.* **142**, 064108 (2015),
  DOI: [10.1063/1.4907278](https://doi.org/10.1063/1.4907278), cited by the
  existing PySCF RCCSDTQ implementation.
- I. Shavitt and R. J. Bartlett, *Many-Body Methods in Chemistry and Physics*
  (2009), DOI: [10.1017/CBO9780511596834](https://doi.org/10.1017/CBO9780511596834),
  the existing UCCSDT implementation's equation reference.

## Storage, starting amplitudes and restart

T1 and T2 use the existing full UCCSDT layouts. The mixed doubles block
is `t2ab[i,a,J,B]`. Full triples are ordered `aaa/aab/bba/bbb`, with mixed
blocks `t3aab[i,j,a,b,K,C]` and `t3bba[I,J,A,B,k,c]`.

Compact T3 stores `aaa/bbb` as `(ijk),(abc)` and `aab/bba` as
`(ij),(ab),K,C`, with strictly increasing same-spin combinations. Both
T3 amplitudes and residuals remain packed through the compact iteration;
the full-storage variant retains the full layouts above.

T4 uses five blocks in the order `aaaa/aaab/aabb/bbba/bbbb`. The majority
spin comes first; for `aabb` alpha comes first. This preserves the existing
UCCSDT interleaved occupied/virtual convention.

| Block | Full layout | Compact layout |
| --- | --- | --- |
| aaaa | `ijkl abcd` | `(ijkl) (abcd)` |
| aaab | `ijk abc L D` | `(ijk) (abc) L D` |
| aabb | `ij ab KL CD` | `(ij) (ab) (KL) (CD)` |
| bbba | `IJK ABC l d` | `(IJK) (ABC) l d` |
| bbbb | `IJKL ABCD` | `(IJKL) (ABCD)` |

Parentheses represent strictly increasing same-spin combinations in
lexicographic order. `compact_tamps=True` compresses both T3 and T4.
`mycc.tamps_full2tri(tamps_full)` and `mycc.tamps_tri2full(tamps_compact)`
convert a complete `[t1, t2, t3, t4]` list. The private storage helpers
`_uccsdtq_utils.pack_t4(mycc, t4)` and `unpack_t4(mycc, t4)` convert only
the five T4 blocks. Full T4 expansion checks the remaining `max_memory` budget before
allocating its output.

A converged UCCSDT starting point may be supplied as
`mycc.kernel(tamps=ccsdt.tamps, eris=eris)`. The driver accepts compact or
full T3, converts it to the solver's storage and initializes absent T4 blocks
to zero. Reusing compatible
bare ERIs avoids another AO-to-MO transformation; the source and target
must use the same orbitals and frozen space.
The iteration updates supplied amplitude arrays in place; use
`copy.deepcopy(ccsdt.tamps)` when the source amplitudes must be retained.

T4 stays out of DIIS by default (`do_diis_max_t=False`). Set the option to
`True` to include T4 for small calculations. DIIS vectors contain only
symmetry-unique elements for all participating ranks. Changing this option
changes the DIIS history's vector size, so create a new history when the
participating ranks change. Restoring T1--T3 DIIS preserves an existing T4;
on a new object it initializes T4 to zero.

`mycc.dump_chk()` stores data under `uccsdtq` or `uccsdtq_highm`, preserving
the UCCSDT checkpoint groups. Metadata records the active dimensions and
T4 representation. Checkpoint helpers also accept older compact checkpoints
that stored full T3, converting each rank from its actual shape. Standard
PySCF checkpoint loading can recover a saved
amplitude list:

```python
from pyscf.lib import chkfile

tamps = chkfile.load('uccsdtq.chk', 'uccsdtq/tamps')
mycc.kernel(tamps=tamps)
```

## Contraction backend and working memory

`mycc.quadruples_backend` selects the quadruple contractions:

- `'auto'` (default) keeps direct C/OpenMP evaluation for small spaces and
  chooses the factorized backend for larger spaces.
- `'direct'` forces the transparent scalar C evaluator used as a numerical
  reference. OpenMP divides independent output amplitudes among threads.
- `'blocked'` forces binary BLAS contractions, bounded matrix panels and
  reusable intermediates. It handles the complete R4 and T4 feedback to
  R2/R3 using the same spin-resolved connected equations.

The automatic selection uses
`Q_unique * (oa+ob)^2 * (va+vb)^2 >= 1_000_000 * threads`, where `Q_unique` is the
number of independent T4 amplitudes across all five spin blocks. This is a
conservative heuristic based on the tested crossover, not a hardware-specific
cost model. The selected backend is logged, and either path can be forced:

```python
mycc = cc.UCCSDTQ(mf, compact_tamps=True)
mycc.quadruples_backend = 'blocked'
mycc.max_memory = 4000  # MB, including already resident arrays
mycc.run(max_cycle=100)
```

The blocked path caches reusable two- and four-index intermediates within
its budget. Six-index intermediates are built in occupied batches, used by
all consuming contractions for that batch, and released. With serial native
BLAS and at least four requested threads, each selected occupied axis has
batch width up to three; otherwise the width is one. The panel budget can
reduce these widths, and small six-index intermediates may fit in one batch.
Further virtual tiling bounds matrix panels. Compact T4 is gathered directly
into panels; there is no dense spin-orbital T4 expansion. Insufficient space for a
required reusable four-index intermediate raises `MemoryError`.

Equivalent same-spin virtual permutations share a contraction; the native
scatter combines their signed contributions while visiting unique virtual
tuples. The final gather/GEMM/scatter loop runs together in C, with reusable
per-thread buffers and independent occupied outputs. Structurally zero spin
blocks and batches with repeated fixed indices in an antisymmetric group
are pruned before contraction. A six-index batch is also skipped if none
of its consumers can produce a canonical occupied output. These steps
preserve the full CCSDTQ equations and avoid recomputing symmetry-equivalent
or structurally zero work.

`max_memory` determines the remaining workspace after current resident
memory and newly returned residual arrays. Cache/panel allocations and
symbolic planning have separate accounting. This controls contraction
workspace; it is not a strict process RSS guarantee or a replacement for
checking whether the amplitudes and bare ERIs fit before calculation.

Threading uses `lib.num_threads(n)` or `OMP_NUM_THREADS`. Gather/scatter
operations use OpenMP for sufficiently large panels. Dense gather decodes
outer coordinates once per short row segment and walks the final axis by
its stride, including transposed and reversed views. GEMM uses the linked
BLAS; the native wrapper can divide sufficiently large GEMMs into independent
row blocks when that BLAS is confirmed to be serial, avoiding nested BLAS
thread pools. Small panels remain serial to avoid parallel-launch overhead.
The native thresholds are 8,192 elements for gather/scatter and at least
128 matrix rows plus 1,000,000 FLOPs for the explicit parallel GEMM path.
The fused final loop distributes complete occupied tuples when its estimated
work reaches 100,000 FLOPs and the linked BLAS is serial. It filters
noncanonical same-spin tuples before packing; distinct canonical tuples
own disjoint output orbits. This avoids limiting the worker count to the
length of a single occupied axis. The actual team can still be smaller than
the requested thread count, and sparse valid tuples can limit useful
parallelism. Native runtime thread counts are checked in the benchmarks.

## Memory and validation

`mycc.memory_estimate_log()` reports T1--T4, residuals, bare and dressed ERIs,
and DIIS vectors/errors separately. The sum covers these known components;
it is not a strict process RSS cap and does not include every contraction
workspace or allocator overhead. Compact T4 needs
`C(oa,na) C(va,na) C(ob,nb) C(vb,nb)` entries per spin block. Including T4
in a DIIS history of size `s` adds up to `2*s` unique T4 arrays.

The validation suite checks storage signs and packing independently of
solver convergence, full/compact agreement, spin exchange, and the projected
residuals against a determinant-space similarity transform. For at most four
correlated electrons, converged CCSDTQ must equal the corresponding FCI
energy. Systems with five or more correlated electrons require residual
checks or another CCSDTQ implementation; FCI equality is not the expected
acceptance condition there.

This implementation does not establish the large-system timing or memory
targets in the supplied implementation guide. Comparisons with published
CFOUR/MRCC timings require matched molecule, basis, frozen space, iteration,
thread count and hardware. Point-group storage compression, distributed
memory, and a production-scale performance claim are outside the initial
implementation's validated scope.

`72-benchmark_uccsdtq.py --threads 1 2 4 --repeat 3` runs a small fixed-amplitude
residual test in fresh processes. SCF and integral preparation always use
one thread so the thread comparison starts from identical orbitals and
integrals. JSON output includes repeated timings, residual fingerprints,
thread settings, CPU affinity, source/library hashes, and resident memory
before and after warm-up. Its reported peak is the maximum of the OS high
water marks and sampled RSS; the individual OS readings are also retained.
