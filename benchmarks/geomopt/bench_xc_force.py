"""Fixed-density timing of grad.rks.get_xc_force against the get_vxc matrix
contraction it replaces, for larger clusters (nao >= 800 uses the
shell-screened products). Usage: bench_xc_force.py ROOT N XC BASIS LEVEL THREADS
"""
import sys, time
root, n, xc, basis, level, threads = sys.argv[1], int(sys.argv[2]), sys.argv[3], \
    sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
sys.path.insert(0, root)
import numpy as np
from pyscf import gto, dft, lib
from pyscf.grad import rks as rks_grad
from threadpoolctl import threadpool_limits
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
mf.max_cycle = 2      # timing only: a non-converged density is fine
with threadpool_limits(limits=1, user_api='blas'):
    mf.kernel()
    dm = mf.make_rdm1()
    tagged = lib.tag_array(dm, mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ)
    ni, grids = mf._numint, mf.grids
    aoslices = mol.aoslice_by_atom()[:, 2:]
    def reference():
        vxc = rks_grad.get_vxc(ni, mol, grids, xc, dm, max_memory=4000)[1]
        return np.array([2 * np.einsum('xij,ij->x', vxc[:, p0:p1], dm[p0:p1])
                         for p0, p1 in aoslices])
    def force():
        return rks_grad.get_xc_force(ni, mol, grids, xc, tagged, max_memory=4000)
    times = {}
    for name, fn in (('reference', reference), ('get_xc_force', force)):
        best = 1e30
        for _ in range(2):
            t = time.perf_counter(); out = fn(); best = min(best, time.perf_counter() - t)
        times[name] = (best, out)
diff = abs(times['reference'][1] - times['get_xc_force'][1]).max()
print(f'n={n} {xc} {basis} grid{level} threads={threads} nao={mol.nao} '
      f'ngrids={len(grids.weights)}: reference {times["reference"][0]:.2f}s, '
      f'get_xc_force {times["get_xc_force"][0]:.2f}s, '
      f'speedup {times["reference"][0]/times["get_xc_force"][0]:.2f}x, max|diff| {diff:.1e}')
