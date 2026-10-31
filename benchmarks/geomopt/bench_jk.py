"""Direct SCF J / JK build vs. gradient J / JK build on the same density.
Usage: bench_jk.py ROOT N BASIS THREADS
"""
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
mf = scf.RHF(mol)
dm = mf.get_init_guess()
def t(fn, reps=2):
    fn(); best = 1e9
    for _ in range(reps):
        s = time.perf_counter(); fn(); best = min(best, time.perf_counter() - s)
    return best
with threadpool_limits(limits=1, user_api='blas'):
    tj = t(lambda: scf.hf.get_jk(mol, dm, with_k=False))
    tjk = t(lambda: scf.hf.get_jk(mol, dm))
    gj = t(lambda: rhf_grad.get_j(mol, dm))
    gjk = t(lambda: rhf_grad.get_jk(mol, dm))
print(f'n={n} {basis} nao={mol.nao} threads={threads}')
print(f'  direct SCF   J: {tj:6.2f}s   JK: {tjk:6.2f}s')
print(f'  gradient     J: {gj:6.2f}s   JK: {gjk:6.2f}s   ratio J {gj/tj:.1f}x  JK {gjk/tjk:.1f}x')
