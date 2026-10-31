"""Component timings: reference grad.rks.get_vxc vs contracted prototype.
Usage: prof_xc_grad.py ROOT N XC BASIS LEVEL THREADS
"""
import os, sys, time
from collections import defaultdict
root, n, xc, basis, level, threads = sys.argv[1], int(sys.argv[2]), sys.argv[3], \
    sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
sys.path.insert(0, root)
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pyscf import gto, dft, lib
from pyscf.dft import numint
from pyscf.grad import rks as rks_grad
from threadpoolctl import threadpool_limits
import xc_grad_contracted as proto
lib.num_threads(threads)

def water_cluster(n):
    atoms = []
    for i in range(n):
        x, y = 4.0 * (i % 3), 4.0 * (i // 3)
        atoms += [('O', (x, y, 0)), ('H', (x+.757, y, .587)), ('H', (x-.757, y, .587))]
    return atoms

T = defaultdict(float)
import inspect
def wrap(obj, name, label, static=False):
    static = static or (isinstance(obj, type) and
                        isinstance(inspect.getattr_static(obj, name), staticmethod))
    orig = getattr(obj, name)
    def f(*a, **k):
        t = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            T[label] += time.perf_counter() - t
    setattr(obj, name, staticmethod(f) if static else f)

wrap(numint.NumInt, 'eval_ao', 'eval_ao (deriv)', static=True)
for nm in ('eval_rho1', 'eval_rho2'):
    wrap(numint.NumInt, nm, 'eval_rho')
wrap(numint.NumInt, 'eval_xc_eff', 'eval_xc_eff (libxc)')
wrap(numint, '_scale_ao', '_scale_ao')
wrap(rks_grad, '_make_dR_dao_w', 'ref: _make_dR_dao_w (numpy +=)')
wrap(numint, '_dot_ao_ao', 'ref: _dot_ao_ao (nao^2 GEMMs)')
wrap(proto, '_dm_apply', 'proto: thin GEMMs (X D)')
wrap(proto, '_cr', 'proto: rho contractions')

mol = gto.M(atom=water_cluster(n), basis=basis, unit='Angstrom', verbose=0, max_memory=8000)
mf = dft.RKS(mol, xc=xc); mf.grids.level = level; mf.max_cycle = 2
with threadpool_limits(limits=1, user_api='blas'):
    mf.kernel()
    dm = mf.make_rdm1()
    ni, grids = mf._numint, mf.grids
    for name, fn in (('reference', lambda: proto.reference_xc_grad(ni, mol, grids, xc, dm, 4000)),
                     ('prototype', lambda: proto.get_xc_grad_contracted(ni, mol, grids, xc, mf.mo_coeff, mf.mo_occ, 4000))):
        fn()  # warm
        T.clear()
        t = time.perf_counter(); fn(); total = time.perf_counter() - t
        print(f'{name}: total {total:.2f}s  ({xc}, n={n}, nao={mol.nao}, ngrids={len(grids.weights)}, threads={threads})')
        for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
            print(f'    {k:34s} {v:7.2f}s  {100*v/total:5.1f}%')
        print(f'    {"(unattributed)":34s} {total-sum(T.values()):7.2f}s')
