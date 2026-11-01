"""Startup cost of the persistent XC worker pool, which geometry optimization
would re-pay every step because RKS.scf() and reset() close it.
Two 2-CPU groups without memory binding (this VM has one NUMA node).
Usage: bench_pool_startup.py ROOT N BASIS LEVEL
"""
import os, sys, tempfile, time
root, n, basis, level = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
sys.path.insert(0, root)
import numpy as np
from pyscf import gto, dft, lib
from pyscf.dft._numa_worker import PartitionedXC
from threadpoolctl import threadpool_limits

def water_cluster(n):
    atoms = []
    for i in range(n):
        x, y = 4.0 * (i % 3), 4.0 * (i // 3)
        atoms += [('O', (x, y, 0)), ('H', (x+.757, y, .587)), ('H', (x-.757, y, .587))]
    return atoms

lib.num_threads(4)
mol = gto.M(atom=water_cluster(n), basis=basis, unit='Angstrom', verbose=0, max_memory=8000)
mf = dft.RKS(mol, xc='PBE'); mf.grids.level = level
mf.grids.build(with_non0tab=True)
dm = mf.get_init_guess()
with threadpool_limits(limits=1, user_api='blas'):
    for rep in range(2):
        t = time.perf_counter()
        pool = PartitionedXC(mol, mf.grids, 'PBE', [([0, 1], None), ([2, 3], None)],
                             allowed_cpus=[0, 1, 2, 3], memory_mb=4000,
                             log_prefix=os.path.join(tempfile.mkdtemp(prefix='pyscf-pool-'), f'r{rep}'))
        startup = time.perf_counter() - t
        t = time.perf_counter(); pool.evaluate(dm); first = time.perf_counter() - t
        t = time.perf_counter(); pool.evaluate(dm); second = time.perf_counter() - t
        pool.close()
        t = time.perf_counter(); mf._numint.nr_rks(mol, mf.grids, 'PBE', dm); native = time.perf_counter() - t
        print(f'rep {rep}: pool startup {startup:.2f}s, first eval {first:.2f}s, '
              f'second eval {second:.2f}s, native nr_rks {native:.2f}s  '
              f'(nao={mol.nao}, ngrids={len(mf.grids.weights)})')
