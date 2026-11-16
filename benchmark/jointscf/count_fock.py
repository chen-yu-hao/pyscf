#!/usr/bin/env python
'''Fock build counts of JointSCF (no timings).  Usage: count_fock.py case [key=value ...]'''
import sys
from pyscf import dft, lib
from pyscf.scf import jointscf
import ensembles

lib.num_threads(1)
name = sys.argv[1]
mols, xc, level = ensembles.build_case(name)
def make(mol):
    mf = dft.RKS(mol, xc=xc)
    mf.grids.level = level
    return mf
ref = jointscf.JointSCF([make(mol) for mol in mols])
ref.schedule = 'sync'
ref.max_workers = 1
e_ref = ref.kernel()
jmf = jointscf.JointSCF([make(mol) for mol in mols])
jmf.max_workers = 1
for arg in sys.argv[2:]:
    key, val = arg.split('=')
    setattr(jmf, key, float(val) if '.' in val else int(val))
e = jmf.kernel()
lv = [(int((jmf.levels == l).sum()), int(jmf.promoted_at[jmf.levels == l][0]),
       round(float(jmf.nfock[jmf.levels == l].mean()), 2)) for l in range(jmf.levels.max()+1)]
print('%-18s %s sync %.2f cascade %.2f ratio %.2f conv %d/%d max|dE| %.1e levels(size,cycle,nfock) %s' % (
    name, ' '.join(sys.argv[2:]), ref.nfock.mean(), jmf.nfock.mean(), ref.nfock.sum() / jmf.nfock.sum(),
    jmf.converged.sum(), len(mols), abs(e - e_ref).max(), lv))
print('   promotions', [(p['cycle'], p['nsrc'], p['ndst'], '%.1e' % p['model_error'], p['saturated']) for p in jmf.models[0].promotions])
