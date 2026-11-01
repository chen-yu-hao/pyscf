"""SCF cycles per geometry-optimization step: stock scanner vs. projected guess
vs. projected guess + gradient-adaptive SCF tolerance.
Usage: guess_experiment.py ROOT VARIANT XC BASIS LEVEL THREADS
VARIANT in {stock, project, adaptive}
"""
import sys, time, json
root, variant, xc, basis, level, threads = sys.argv[1], sys.argv[2], sys.argv[3], \
    sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
sys.path.insert(0, root)
import numpy as np
import scipy.linalg
from pyscf import gto, dft, lib
from pyscf.scf import hf
from pyscf.grad import rhf as rhf_grad
from pyscf.geomopt import berny_solver
from threadpoolctl import threadpool_limits
lib.num_threads(threads)

# Glycine, deliberately distorted start (bond lengths/angles off by a few %)
GLY = '''
N   -1.9530  0.1700  0.0300
H   -2.0200  1.1500 -0.2500
H   -2.6500 -0.3000 -0.5500
C   -0.6500 -0.3800  0.2800
H   -0.6000 -1.4200  0.0200
H   -0.4300 -0.2500  1.3800
C    0.4600  0.3600 -0.4800
O    0.3400  1.4800 -0.9200
O    1.6600 -0.2500 -0.5100
H    2.3000  0.3000 -0.9700
'''
mol = gto.M(atom=GLY, basis=basis, verbose=0, max_memory=8000)
mf = dft.RKS(mol, xc=xc)
mf.grids.level = level

log = dict(cycles=[], scf_wall=[], grad_wall=[], conv_tol=[], gmax=[])
orig_scf_call = hf.SCF_Scanner.__call__
orig_grad_call = rhf_grad.SCF_GradScanner.__call__
state = dict(prev_gmax=None)


def projected_dm0(scanner, mol_new):
    occ = scanner.mo_occ > 0
    c = scanner.mo_coeff[:, occ]
    s = mol_new.intor_symmetric('int1e_ovlp')
    w, v = scipy.linalg.eigh(c.T @ s @ c)
    c = c @ (v * w**-.5) @ v.T          # Loewdin-orthonormalize in the new metric
    return (c * scanner.mo_occ[occ]) @ c.T


def scf_call(self, mol_or_geom, **kwargs):
    if variant != 'stock' and self.mo_coeff is not None and 'dm0' not in kwargs:
        kwargs['dm0'] = projected_dm0(self, mol_or_geom)
    t = time.perf_counter()
    e = orig_scf_call(self, mol_or_geom, **kwargs)
    log['scf_wall'].append(time.perf_counter() - t)
    log['cycles'].append(self.cycles)
    log['conv_tol'].append(self.conv_tol)
    return e


def grad_call(self, mol_or_geom, **kwargs):
    if variant == 'adaptive' and state['prev_gmax'] is not None:
        tol_grad = float(np.clip(.03 * state['prev_gmax'], 10**-4.5, 1e-3))
        self.base.conv_tol_grad = tol_grad
        self.base.conv_tol = max(tol_grad**2, 1e-9)
    if isinstance(mol_or_geom, gto.MoleBase):
        mol_new = mol_or_geom
    else:
        mol_new = self.mol.set_geom_(mol_or_geom, inplace=False)
    self.reset(mol_new)
    e_tot = self.base(mol_new)
    t = time.perf_counter()
    de = self.kernel(**kwargs)
    log['grad_wall'].append(time.perf_counter() - t)
    state['prev_gmax'] = float(abs(de).max())
    log['gmax'].append(state['prev_gmax'])
    return e_tot, de


hf.SCF_Scanner.__call__ = scf_call
rhf_grad.SCF_GradScanner.__call__ = grad_call

with threadpool_limits(limits=1, user_api='blas'):
    t0 = time.perf_counter()
    mol_eq = berny_solver.optimize(mf, maxsteps=100)
    wall = time.perf_counter() - t0
    # Tight reference single point at the final geometry
    ref = dft.RKS(mol_eq, xc=xc); ref.grids.level = level; ref.conv_tol = 1e-11
    e_final = ref.kernel()
print(json.dumps(dict(variant=variant, xc=xc, basis=basis, nao=mol.nao,
                      steps=len(log['cycles']), total_scf_cycles=int(sum(log['cycles'])),
                      cycles=log['cycles'], wall=wall,
                      scf_wall=sum(log['scf_wall']), grad_wall=sum(log['grad_wall']),
                      e_final_tight=e_final,
                      coords_final=mol_eq.atom_coords().round(6).tolist(),
                      conv_tol=log['conv_tol'], gmax=log['gmax'])))
