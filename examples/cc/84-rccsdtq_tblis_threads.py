#!/usr/bin/env python
"""RCCSDTQ with explicit TBLIS/OpenMP threads and serial pthread BLAS.

Requires pytblis (tested with 0.0.16) and threadpoolctl. For example:

  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=false \
      python examples/cc/84-rccsdtq_tblis_threads.py

Use CPU/NUMA affinity appropriate to your machine for scaling measurements.
This small example demonstrates configuration, not large-system efficiency.
"""
import os

import pytblis
from threadpoolctl import threadpool_limits
from pyscf import cc, gto, lib, scf

threads = int(os.environ.get('OMP_NUM_THREADS', '4'))
lib.num_threads(threads)
pytblis.set_num_threads(threads)
assert pytblis.get_num_threads() == threads

# Keep NumPy/SciPy pthread BLAS from competing with the OpenMP tensor backend.
with threadpool_limits(limits=1, user_api='blas'):
    mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                basis='sto-3g', verbose=4)
    mf = scf.RHF(mol).run()
    mycc = cc.RCCSDTQ(mf, frozen=1, compact_tamps=True)
    mycc.set_einsum_backend('pytblis')
    mycc.run()
    print('Converged:', mycc.converged, 'E(RCCSDTQ):', mycc.e_tot)
