"""Compare prototype contracted XC gradient against pyscf.grad.rks.get_vxc.
Usage: bench_xc_grad.py ROOT N XC BASIS LEVEL THREADS
"""
import os, sys, time
root, n, xc, basis, level, threads = sys.argv[1], int(sys.argv[2]), sys.argv[3], \
    sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
sys.path.insert(0, root)
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pyscf import gto, dft, lib
from threadpoolctl import threadpool_limits
from xc_grad_contracted import get_xc_grad_contracted, reference_xc_grad, get_xc_grad_fused
import os
VARIANT = os.environ.get('VARIANT', 'numpy')
if VARIANT == 'fused': get_xc_grad_contracted = get_xc_grad_fused

lib.num_threads(threads)

def water_cluster(n):
    atoms = []
    for i in range(n):
        x, y = 4.0 * (i % 3), 4.0 * (i // 3)
        atoms += [('O', (x, y, 0)), ('H', (x+.757, y, .587)), ('H', (x-.757, y, .587))]
    return atoms

mol = gto.M(atom=water_cluster(n), basis=basis, unit='Angstrom', verbose=0, max_memory=8000)
mf = dft.RKS(mol, xc=xc)
mf.grids.level = level
mf.conv_tol = 1e-10
with threadpool_limits(limits=1, user_api='blas'):
    mf.kernel()
    dm = mf.make_rdm1()
    ni, grids = mf._numint, mf.grids
    mm = 4000
    # warm-up both (libxc init, page faults)
    t = time.perf_counter(); ref = reference_xc_grad(ni, mol, grids, xc, dm, mm); t_ref = time.perf_counter() - t
    t = time.perf_counter(); new = get_xc_grad_contracted(ni, mol, grids, xc, mf.mo_coeff, mf.mo_occ, mm); t_new = time.perf_counter() - t
    t = time.perf_counter(); ref = reference_xc_grad(ni, mol, grids, xc, dm, mm); t_ref2 = time.perf_counter() - t
    t = time.perf_counter(); new = get_xc_grad_contracted(ni, mol, grids, xc, mf.mo_coeff, mf.mo_occ, mm); t_new2 = time.perf_counter() - t
print(f'n={n} xc={xc} basis={basis} level={level} threads={threads} nao={mol.nao} '
      f'ngrids={len(grids.weights)} nocc={int((mf.mo_occ>0).sum())}')
print(f'  max|diff| = {abs(ref-new).max():.3e}   max|grad_xc| = {abs(ref).max():.3e}')
print(f'  reference get_vxc+contract: {t_ref:.2f}s, {t_ref2:.2f}s')
print(f'  {VARIANT:9s} prototype      : {t_new:.2f}s, {t_new2:.2f}s   speedup {min(t_ref,t_ref2)/min(t_new,t_new2):.2f}x')
