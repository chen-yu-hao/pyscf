"""grad.rhf.get_jk_force against the get_j/get_jk matrices + contraction it
replaces, same density. Usage: bench_jk_force.py ROOT N BASIS THREADS"""
import sys, time
root, n, basis, threads = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
sys.path.insert(0, root)
import numpy as np
from pyscf import gto, scf, lib
from pyscf.grad import rhf as rhf_grad
from threadpoolctl import threadpool_limits
lib.num_threads(threads)

def water_cluster(n):
    atoms = []
    for i in range(n):
        x, y = 4.0 * (i % 3), 4.0 * (i // 3)
        atoms += [('O', (x, y, 0)), ('H', (x+.757, y, .587)), ('H', (x-.757, y, .587))]
    return atoms

mol = gto.M(atom=water_cluster(n), basis=basis, unit='Angstrom', verbose=0)
mf = scf.RHF(mol); mf.max_cycle = 3
with threadpool_limits(limits=1, user_api='blas'):
    mf.kernel()
    dm = mf.make_rdm1()
    s = mol.aoslice_by_atom()[:, 2:]
    contract = lambda v: np.array([2*np.einsum('xij,ij->x', v[:, p0:p1], dm[p0:p1]) for p0, p1 in s])
    def best(fn):
        fn(); t = []
        for _ in range(2):
            t0 = time.perf_counter(); out = fn(); t.append(time.perf_counter() - t0)
        return min(t), out
    tj, rj = best(lambda: contract(rhf_grad.get_j(mol, dm)))
    fj, dj = best(lambda: rhf_grad.get_jk_force(mol, dm, 1., 0.))
    def jk():
        vj, vk = rhf_grad.get_jk(mol, dm)
        return contract(vj - .22 * vk)
    tjk, rjk = best(jk)
    fjk, djk = best(lambda: rhf_grad.get_jk_force(mol, dm, 1., .22))
print(f'n={n} {basis} nao={mol.nao} threads={threads}')
print(f'  J : matrix {tj:6.2f}s  force {fj:6.2f}s  speedup {tj/fj:.2f}x  max|diff| {abs(rj-dj).max():.1e}')
print(f'  JK: matrix {tjk:6.2f}s  force {fjk:6.2f}s  speedup {tjk/fjk:.2f}x  max|diff| {abs(rjk-djk).max():.1e}')
