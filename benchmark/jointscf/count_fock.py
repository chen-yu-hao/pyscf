#!/usr/bin/env python
'''Fock build counts of JointSCF (no timings).  Usage: count_fock.py case [key=value ...]'''
import sys
from pyscf import dft, lib
from pyscf.scf import jointscf
import ensembles

lib.num_threads(1)
name = sys.argv[1]
mols, xc, level = ensembles.build_case(name)

def _parse_val(val):
    if val == 'True':
        return True
    if val == 'False':
        return False
    if val == 'None':
        return None
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val

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
    key, val = arg.split('=', 1)
    setattr(jmf, key, _parse_val(val))
e = jmf.kernel()
de = abs(e - e_ref)
if jmf.schedule == 'cascade' and jmf.models:
    lv = [(int((jmf.levels == l).sum()), int(jmf.promoted_at[jmf.levels == l][0]),
           round(float(jmf.nfock[jmf.levels == l].mean()), 2)) for l in range(jmf.levels.max()+1)]
    print('%-18s %s sync %.2f cascade %.2f ratio %.2f conv %d/%d max|dE| %.1e levels(size,cycle,nfock) %s' % (
        name, ' '.join(sys.argv[2:]), ref.nfock.mean(), jmf.nfock.mean(), ref.nfock.sum() / jmf.nfock.sum(),
        jmf.converged.sum(), len(mols), de.max(), lv))
    print('   promotions', [(p['cycle'], p['nsrc'], p['ndst'], '%.1e' % p['model_error'], p['saturated']) for p in jmf.models[0].promotions])
elif jmf.schedule == 'transferable':
    print('%-18s %s sync %.2f trans %.2f (sh %.2f + po %.2f) ratio %.2f conv %d/%d mean|dE| %.2e max|dE| %.2e est %.2e' % (
        name, ' '.join(sys.argv[2:]), ref.nfock.mean(), jmf.nfock.mean(),
        jmf.nfock_shared.mean(), jmf.nfock_polish.mean(),
        ref.nfock.sum() / jmf.nfock.sum(), jmf.converged.sum(), len(mols),
        de.mean(), de.max(), jmf.e_err_est.mean()))
else:
    print('%-18s %s sync %.2f %s %.2f ratio %.2f conv %d/%d max|dE| %.1e' % (
        name, ' '.join(sys.argv[2:]), ref.nfock.mean(), jmf.schedule, jmf.nfock.mean(),
        ref.nfock.sum() / jmf.nfock.sum(), jmf.converged.sum(), len(mols), de.max()))
