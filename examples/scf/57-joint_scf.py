#!/usr/bin/env python

'''
Joint SCF: converge the wavefunctions of many geometries in a single SCF loop.

JointSCF takes a list of mean-field objects (or one object and a list of
molecules).  All wavefunctions are optimized in the same loop starting from
the initial guess.  Every configuration is converged with the iterations and
the convergence criteria of mf.kernel().  The results are ordinary
single-point SCF solutions.

schedule = 'sync'
    One Fock build per configuration in every iteration.  The same
    arithmetic as independent kernel() calls.

schedule = 'cascade' (default)
    In the first iterations only a subset of the batch builds Fock matrices.
    The other configurations follow through a model which interpolates
    density and Fock matrices across the geometries.  They receive their
    first Fock build when the SCF error of the subset is as small as the
    error of the model, and they inherit the DIIS subspace.

The configurations are distributed over worker processes.  As for any
program that starts processes, the main module needs the __main__ guard
only if it is imported; the workers do not import this script.
'''

import numpy
from pyscf import gto, scf, dft

#
# 1. A 2D scan of the water molecule
#
mols = []
for r in numpy.linspace(0.90, 1.10, 6):
    for theta in numpy.linspace(96, 114, 6):
        t = numpy.deg2rad(theta)
        mols.append(gto.M(
            atom=[['O', (0, 0, 0)], ['H', (r, 0, 0)],
                  ['H', (r*numpy.cos(t), r*numpy.sin(t), 0)]],
            basis='def2-svp', verbose=0))

mf = dft.RKS(mols[0], xc='pbe')
# Optional: keep the AO values on the grids in memory.  The results are
# identical; the XC matrix is computed faster in all iterations but the first.
mf._numint = dft.numint.CachedNumInt()

jmf = scf.JointSCF(mf, mols)
jmf.verbose = 4
e_tot = jmf.kernel()

print('Total energies', e_tot)
print('All converged:', jmf.converged.all())
print('Fock builds per configuration', jmf.nfock)
print('Level of each configuration in the cascade', jmf.levels)

# The mean-field objects hold the usual results
mf0 = jmf.mfs[0]
print('E =', mf0.e_tot, ' HOMO =', mf0.mo_energy[mf0.mo_occ > 0][-1])

# Independent calculations for comparison
nfock = 0
for i in (0, 17, 35):
    ref = dft.RKS(mols[i], xc='pbe').run()
    nfock += ref.cycles + 2  # initial guess + SCF cycles + final check
    print('config %2d  E(joint) - E(kernel) = %.1e' % (i, e_tot[i] - ref.e_tot))
print('Fock builds of mf.kernel(): %.1f per configuration.  JointSCF: %.1f'
      % (nfock / 3, jmf.nfock.mean()))

#
# 2. The synchronous schedule reproduces mf.kernel() for any batch of
# mean-field objects, e.g. different molecules and methods
#
mfs = [scf.RHF(mols[0]), scf.UHF(gto.M(atom='O 0 0 0; H 0 0 0.97', spin=1,
                                       basis='def2-svp', verbose=0)),
       dft.RKS(mols[1], xc='b3lyp')]
jmf = scf.JointSCF(mfs)
jmf.schedule = 'sync'
jmf.kernel()
print(jmf.e_tot, jmf.nfock)

#
# 3. User defined descriptors of the geometries (the default descriptors are
# the inverse interatomic distances), and other settings of the cascade
#
jmf = scf.JointSCF(scf.RHF(mols[0]), mols)
jmf.descriptors = numpy.array(
    [[r, theta] for r in numpy.linspace(0.90, 1.10, 6)
     for theta in numpy.linspace(96, 114, 6)])
jmf.n_lead = 6          # number of configurations treated exactly from the start
jmf.max_workers = 2     # worker processes
jmf.kernel()
print(jmf.nfock.mean(), jmf.converged.all())
