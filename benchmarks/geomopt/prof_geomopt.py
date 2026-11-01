"""Profile two geometry-optimization steps through the same gradient scanner
that pyscf.geomopt uses.  Usage: prof_geomopt.py ROOT N XC BASIS LEVEL THREADS
"""
import sys, time, json
from collections import defaultdict

root, n, xc, basis, level, threads = sys.argv[1], int(sys.argv[2]), sys.argv[3], \
    sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
sys.path.insert(0, root)
import numpy as np
from pyscf import gto, dft, lib
from pyscf.dft import numa, gen_grid, numint
from pyscf.grad import rks as rks_grad, rhf as rhf_grad
from pyscf.scf import hf
from threadpoolctl import threadpool_limits

lib.num_threads(threads)

def water_cluster(n, shift=None):
    atoms = []
    for i in range(n):
        x, y = 4.0 * (i % 3), 4.0 * (i // 3)
        atoms += [('O', (x, y, 0)), ('H', (x+.757, y, .587)), ('H', (x-.757, y, .587))]
    return atoms

timers = defaultdict(float)
counts = defaultdict(int)
numa_decisions = []

def wrap(mod, name, label=None):
    label = label or name
    orig = getattr(mod, name)
    def f(*a, **k):
        t = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            timers[label] += time.perf_counter() - t
            counts[label] += 1
    setattr(mod, name, f)

# NUMA eligibility as seen inside the scanner
orig_supported = numa._supported
def supported(ks, mol, dm, hermi):
    r = orig_supported(ks, mol, dm, hermi)
    numa_decisions.append(dict(result=bool(r), type_is_RKS=type(ks) is dft.rks.RKS,
                               type=type(ks).__name__, converged=bool(ks.converged)))
    return r
numa._supported = supported

import os
FUSED = os.environ.get('FUSED') == '1'
if FUSED:
    sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
    from xc_grad_contracted import get_xc_grad_fused
    _xc_forces = {}
    def fused_get_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
                      max_memory=2000, verbose=None):
        _xc_forces['de'] = get_xc_grad_fused(ni, mol, grids, xc_code,
                                             dms.mo_coeff, dms.mo_occ, max_memory)
        return None, np.zeros((3, mol.nao, mol.nao))
    rks_grad.get_vxc = fused_get_vxc
    _orig_extra = rks_grad.Gradients.extra_force
    def extra_force(self, atom_id, envs):
        return _orig_extra(self, atom_id, envs) + _xc_forces['de'][atom_id]
    rks_grad.Gradients.extra_force = extra_force

if os.environ.get('XC_FORCE') == '0':
    rks_grad.XC_FORCE = False  # matrix path (get_vxc), for A/B in one build
if os.environ.get('JK_FORCE') == '0':
    rks_grad.JK_FORCE = False  # matrix path (get_jk), for A/B in one build
if hasattr(rhf_grad, 'get_jk_force'):
    wrap(rhf_grad, 'get_jk_force', 'grad.jk (get_jk_force)')
if hasattr(rks_grad, 'get_xc_force'):
    wrap(rks_grad, 'get_xc_force', 'grad.xc (get_xc_force)')

wrap(gen_grid.Grids, 'build', 'grids.build')
wrap(numa, 'nr_rks', 'scf.xc (numa.nr_rks)')
wrap(hf.SCF, 'get_jk', 'scf.get_jk')
wrap(rks_grad, 'get_vxc', 'grad.xc (get_vxc)')
wrap(rhf_grad, 'get_jk', 'grad.get_jk')
wrap(rhf_grad, 'get_j', 'grad.get_j')
wrap(rhf_grad, 'get_k', 'grad.get_k')

mol = gto.M(atom=water_cluster(n), basis=basis, unit='Angstrom', verbose=0, max_memory=8000)
mf = dft.RKS(mol, xc=xc)
mf.grids.level = level
mf.max_memory = 8000
cycles = []
mf.callback = lambda env: None
g = mf.nuc_grad_method().as_scanner()

rng = np.random.default_rng(7)
coords0 = mol.atom_coords(unit='Angstrom')
results = []
with threadpool_limits(limits=1, user_api='blas'):
    for step in range(2):
        timers.clear(); counts.clear()
        coords = coords0 if step == 0 else coords0 + rng.normal(scale=0.02, size=coords0.shape)
        m = mol.set_geom_(coords, unit='Angstrom', inplace=False)
        t0 = time.perf_counter()
        e, de = g(m)
        wall = time.perf_counter() - t0
        scf = g.base
        results.append(dict(step=step, wall=wall, e=e, cycles=scf.cycles,
                            de=np.asarray(de).round(10).tolist(),
                            ngrids=len(scf.grids.weights), nao=m.nao,
                            timers=dict(timers), counts=dict(counts)))
print(json.dumps(dict(n=n, xc=xc, basis=basis, level=level, threads=threads,
                      results=results, numa_decisions=numa_decisions[:3] + numa_decisions[-2:],
                      n_numa_calls=len(numa_decisions)), indent=1))
