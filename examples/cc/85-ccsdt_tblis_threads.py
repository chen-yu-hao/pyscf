#!/usr/bin/env python
"""RCCSDT and UCCSDT with explicit PySCF/TBLIS thread control.

Install the optional contraction backend in the same Python environment:
    python -m pip install pytblis

Run this script in a fresh Python process so the environment settings below
take effect before NumPy, PySCF and TBLIS load their thread runtimes.
The small NH3/STO-3G calculation demonstrates the interface and is not a
benchmark for large molecules. Change THREADS to suit the available CPUs.
"""

import os

THREADS = 4
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
os.environ['OMP_NUM_THREADS'] = str(THREADS)
for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

try:
    import pytblis
except ImportError:
    raise SystemExit('This example requires pytblis: python -m pip install pytblis')

from pyscf import cc, gto, lib, scf

# These runtimes have separate controls. lib.num_threads alone does not set
# TBLIS's team size; BLAS remains at one thread to avoid competing teams.
lib.num_threads(THREADS)
pytblis.set_num_threads(THREADS)
print('PySCF OpenMP threads:', lib.num_threads())
print('TBLIS threads:', pytblis.get_num_threads())
print('BLAS threads: 1; OMP_WAIT_POLICY:', os.environ['OMP_WAIT_POLICY'])

mol = gto.M(atom='''
    N   0.000000   0.000000   0.117300
    H   0.000000   0.937700  -0.273700
    H   0.812100  -0.468850  -0.273700
    H  -0.812100  -0.468850  -0.273700
    ''', basis='sto-3g', unit='Angstrom', verbose=3)
mf = scf.RHF(mol).run(conv_tol=1e-12)
assert mf.converged

# Identical spatial orbitals let the two spin representations be compared.
restricted = cc.RCCSDT(mf, compact_tamps=True)
unrestricted = cc.UCCSDT(mf.to_uhf(), compact_tamps=True)
for name, solver in (('RCCSDT', restricted), ('UCCSDT', unrestricted)):
    solver.einsum_backend = 'pytblis'
    solver.conv_tol = 1e-10
    solver.conv_tol_normt = 1e-8
    solver.run()
    assert solver.converged
    print('%s total energy: %.12f Eh' % (name, solver.e_tot))

assert abs(restricted.e_tot - unrestricted.e_tot) < 1e-8
