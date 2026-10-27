# Reproducing the combined-branch measurements

Use the same Python, NumPy, BLAS, LibXC and compiler configuration for both checkouts. The recorded builds use GCC 11.3.1, LibXC 7.0.0 and Release `-O3 -DNDEBUG -msse3 -fopenmp`, without fast-math. Build `44313dfb457b93358918c391fc66285fa44153ef` as the baseline and `a8cbddc874d7846dedfdc482944fc299d76f2409` as the combined version using the PySCF build instructions. `builds/*.json` records the original commands, unchanged dependency identities and rebuilt `libdft` identity.

These examples run from this benchmark directory. Set `BASE`, `COMBINED` and `PYTHON` to the two source checkouts and one shared interpreter. Select 24 currently idle physical cores on one socket. The recorded affinity order is `5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,3`; its first eight cores are used for eight-thread cases. Do not overlap the campaigns with other benchmarks. All output directories must be new.

```bash
unset PYTHONPATH LD_LIBRARY_PATH
export PYSCF_EXT_PATH=/dev/null OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
BASE=/path/to/base
COMBINED=/path/to/combined
PYTHON=/path/to/python
AFFINITY=5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,3
```

Generate the shared frozen fixtures once from the baseline. Grid level 2 is the fixture generator's default. Geometry, basis, spin, grids and density definitions are in `xc_benchmark.py` and the emitted metadata. Do not generate separate density matrices for the candidate.

```bash
for case in water12-rks water11oh-uks water20-rks water19oh-uks; do
    "$PYTHON" xc_benchmark.py --source "$BASE" \
      --commit 44313dfb457b93358918c391fc66285fa44153ef \
      --make-fixture --case "$case" --fixture "fixtures/$case.npz" \
      --threads 1 --affinity 5
done
```

Run and accept the fixed-DM XC and sorting comparisons:

```bash
"$PYTHON" run_xc_campaign.py --protocol protocols/xc-component.json \
  --versions "baseline=$BASE" "combined=$COMBINED" --python "$PYTHON" \
  --fixtures fixtures --affinity "$AFFINITY" --output xc-v1
"$PYTHON" accept_xc_campaign.py xc-v1/batch.json --baseline baseline --output xc-v1/accepted
"$PYTHON" run_micro_campaign.py --protocol protocols/micro-grid.json \
  --versions "baseline=$BASE" "combined=$COMBINED" --python "$PYTHON" \
  --fixtures fixtures --affinity "$AFFINITY" --output grid-sort-v1
"$PYTHON" accept_xc_campaign.py grid-sort-v1/batch.json --baseline baseline --output grid-sort-v1/accepted
```

Run grid construction and SCF. If the CPU placement changes, first copy and update `affinity_order` in the end-to-end protocol; retain the exact updated protocol with the rerun.

```bash
"$PYTHON" run_end_to_end.py --protocol protocols/end-to-end.json \
  --versions "baseline=$BASE" "combined=$COMBINED" --fixtures fixtures --output end-to-end-v1
"$PYTHON" accept_end_to_end.py end-to-end-v1
```

The recorded native acceptance can be rerun with `python accept_native.py DIRECTORY`. For a new build, create new manifests from that build's actual source commit, library paths and hashes; the archived binary hashes are not expected to match a different compiler or dependency build. Retain the measured native maps and threadpool records, and check that the imported libraries originate from the intended checkout.

Numerical regression tests used four physical cores, four OpenMP threads and one BLAS thread:

```bash
PYTHONPATH="$COMBINED" OMP_NUM_THREADS=4 taskset -c 28-31 "$PYTHON" -m unittest \
  pyscf.dft.test.test_numint pyscf.dft.test.test_numint_rho_fused \
  pyscf.dft.test.test_numint_pair_tiles pyscf.dft.test.test_grids
```

`continue_tests.py` is the original machine-local serial controller, retained as execution provenance. `summarize.py` generates the report only after all scientific and native acceptance files pass. Full numerical arrays are retained locally under `/home/lyl/soft/pyscf-combined-bench-20261027`; public evidence omits binary arrays, compiled libraries, fixtures and temporary files. Running the commands above regenerates the arrays needed for scientific acceptance.
