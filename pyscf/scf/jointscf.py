#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Joint (batched) single-loop SCF for many molecular configurations.

All wavefunctions of a batch of mean-field objects are optimized inside one
SCF loop, starting from the initial guess.  Every configuration is converged
to exactly the same criteria as an independent ``mf.kernel()`` call (the
loop body is the body of :func:`pyscf.scf.hf.kernel`), so the results are
ordinary single-point SCF solutions and each ``mf`` can be handed to any
post-SCF module afterwards.

Two schedules are available:

``sync``
    Every configuration receives one exact Fock build in every iteration.
    The numbers produced are identical to independent SCF calculations; the
    benefit is purely technical (configuration-level parallelism, shared
    setup).

``cascade``
    Multi-fidelity synchronous loop.  In the first iterations only a
    space-filling subset of the batch (the leaders) builds exact Fock
    matrices.  The states ``(D, F)`` of all other configurations follow the
    leaders through a shared cross-configuration model: a kernel regression
    over geometry descriptors, formulated for density and Fock matrices in a
    symmetrically orthogonalized, orientation-aligned AO basis in which these
    matrices are gauge-free and smooth functions of the geometry.  Once the
    SCF residual of the exactly treated configurations has dropped to the
    leave-one-out error of the model, the next level of the batch is promoted
    to exact Fock builds.  A promoted configuration inherits a synthetic DIIS
    history from the model, so it keeps the convergence momentum of the
    configurations it was following instead of restarting the extrapolation.

Example::

    >>> from pyscf import gto, dft
    >>> from pyscf.scf import jointscf
    >>> mols = [gto.M(atom=f'O 0 0 0; H 0 0 {r}; H 0 {r} 0', basis='def2-svp')
    ...         for r in (0.90, 0.95, 1.00, 1.05)]
    >>> jmf = jointscf.JointSCF([dft.RKS(mol, xc='pbe') for mol in mols])
    >>> e_tot = jmf.kernel()
'''

import copy
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy
import scipy.linalg

from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf
from pyscf.scf import chkfile
from pyscf.scf import diis as scf_diis

# Hyper-parameter grid of the cross-configuration kernel regression.  The
# optimal point is selected by leave-one-out cross validation.
_LENGTH_SCALES = (0.5, 1., 2., 4., 8., 16., 32.)
_RIDGES = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4)
# Number of matrix elements sampled for the hyper-parameter selection
_NSAMPLE_CV = 1500


def default_descriptor(mol):
    '''Rotation and translation invariant geometry descriptor: the inverse
    interatomic distances (in 1/Bohr) of all atom pairs.'''
    coords = mol.atom_coords()
    natm = len(coords)
    if natm < 2:
        return numpy.zeros(1)
    iu = numpy.triu_indices(natm, 1)
    rr = numpy.linalg.norm(coords[:,None,:] - coords[None,:,:], axis=-1)
    return 1. / rr[iu]


def kabsch_rotation(coords, ref_coords, weights=None):
    '''The proper rotation R which minimizes
    sum_i w_i |R (x_i - x_c) - (y_i - y_c)|^2  for x = coords, y = ref_coords.
    '''
    if weights is None:
        weights = numpy.ones(len(coords))
    w = weights / weights.sum()
    x = coords - w.dot(coords)
    y = ref_coords - w.dot(ref_coords)
    u, _, vt = numpy.linalg.svd((y * w[:,None]).T.dot(x))
    d = numpy.sign(numpy.linalg.det(u.dot(vt)))
    return u.dot(numpy.diag([1., 1., d])).dot(vt)


def farthest_point_order(z):
    '''Greedy max-min ordering.  The first point is the one closest to the
    centroid.  Every prefix of the returned sequence is a space-filling
    subset of the samples.'''
    n = len(z)
    first = int(numpy.argmin(((z - z.mean(axis=0))**2).sum(axis=1)))
    order = [first]
    dist = ((z - z[first])**2).sum(axis=1)
    dist[first] = -1.
    for _ in range(n - 1):
        nxt = int(numpy.argmax(dist))
        order.append(nxt)
        dist = numpy.minimum(dist, ((z - z[nxt])**2).sum(axis=1))
        dist[nxt] = -1.
    return order


def _reduce_descriptors(desc, tol=1e-6):
    '''Standardize the descriptors and remove linear dependencies.  The
    result is scaled to unit mean squared norm.'''
    desc = numpy.asarray(desc, dtype=float)
    desc = desc.reshape(len(desc), -1)
    z = desc - desc.mean(axis=0)
    std = z.std(axis=0)
    keep = std > 1e-10 * max(abs(desc).max(), 1.)
    if not keep.any():
        return numpy.zeros((len(desc), 1))
    z = z[:,keep] / std[keep]
    u, s, _ = numpy.linalg.svd(z, full_matrices=False)
    mask = s > tol * s[0]
    z = u[:,mask] * s[mask]
    return z / numpy.sqrt((z**2).sum(axis=1).mean())


def _kernel_matrix(za, zb, length):
    '''Constant + linear + Gaussian covariance function'''
    d2 = ((za**2).sum(axis=1)[:,None] + (zb**2).sum(axis=1)[None,:]
          - 2 * za.dot(zb.T))
    return 1. + za.dot(zb.T) + numpy.exp(-.5 * numpy.maximum(d2, 0) / length**2)


def _regularized_inverse(kmat, ridge):
    n = len(kmat)
    kmat = kmat + numpy.eye(n) * (ridge * numpy.trace(kmat) / n)
    return scipy.linalg.pinvh(kmat)


class SharedModel(lib.StreamObject):
    '''Cross-configuration model of density and Fock matrices for a group of
    configurations which share the atoms and the basis set.

    Matrices are stored in the representation

        P~ = R^T S^{1/2}  D S^{1/2}  R
        F~ = R^T S^{-1/2} F S^{-1/2} R

    where R is the AO rotation matrix of the rigid rotation that aligns a
    configuration with the reference geometry (Kabsch).  In this
    representation the matrices do not depend on any orbital gauge, they are
    invariant to rigid motions of the molecule and they are smooth functions
    of the internal coordinates.  A matrix of an unseen configuration is
    predicted as the linear combination  sum_j w_j(z) X~_j  with the weights
    of a kernel ridge regression over reduced geometry descriptors z.
    '''
    def __init__(self, mols, descriptors=None, align=True, window=16):
        self.mol = mols[0]
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.nconf = len(mols)
        self.window = window
        if descriptors is None:
            descriptors = [default_descriptor(mol) for mol in mols]
        self.z = _reduce_descriptors(descriptors)
        self.order = farthest_point_order(self.z)
        self.rank = numpy.argsort(self.order)
        self.rot = [None] * self.nconf
        if align:
            self._init_alignment(mols)
        # config id -> (S^{-1/2} R, S^{1/2} R)
        self._metric = {}
        # config id -> {entry id: (P~, F~)}
        self._entries = [{} for _ in range(self.nconf)]
        self._cv_cols = None
        self._hyper = None
        self.levels = [self.order]
        self.next_level = 1
        self.promotions = []

    def _init_alignment(self, mols):
        mol0 = mols[self.order[0]]
        if mol0.cart or mol0.natm < 2:
            # The AO rotation matrix of Cartesian GTOs is not orthogonal.
            return
        ref = mol0.atom_coords()
        mass = mol0.atom_mass_list()
        eye = numpy.eye(3)
        for i, mol in enumerate(mols):
            rmat = kabsch_rotation(mol.atom_coords(), ref, mass)
            if abs(rmat - eye).max() > 1e-9:
                # S(rotated) = u^T S u
                self.rot[i] = mol.ao_rotation_matrix(rmat)

    def set_overlap(self, i, s1e):
        e, v = scipy.linalg.eigh(s1e)
        e = numpy.maximum(e, 1e-14)
        s_isqrt = (v / numpy.sqrt(e)).dot(v.T)
        s_sqrt = (v * numpy.sqrt(e)).dot(v.T)
        if self.rot[i] is not None:
            s_isqrt = s_isqrt.dot(self.rot[i])
            s_sqrt = s_sqrt.dot(self.rot[i])
        self._metric[i] = (s_isqrt, s_sqrt)

    def has_entry(self, i, k):
        return k in self._entries[i]

    def record(self, i, k, dm, fock):
        '''Store the pair (density matrix, Fock matrix) of entry k'''
        s_isqrt, s_sqrt = self._metric[i]
        p = numpy.matmul(s_sqrt.T, numpy.matmul(numpy.asarray(dm), s_sqrt))
        f = numpy.matmul(s_isqrt.T, numpy.matmul(numpy.asarray(fock), s_isqrt))
        self._store(i, k, p, f)

    def _store(self, i, k, p, f):
        entries = self._entries[i]
        entries[k] = (p, f)
        for key in [key for key in entries if key <= k - self.window]:
            del entries[key]

    def _get(self, i, k):
        entries = self._entries[i]
        if k in entries:
            return entries[k]
        # A converged configuration does not produce new entries.
        older = [key for key in entries if key < k]
        if older:
            return entries[max(older)]
        return entries[min(entries)]

    def _targets(self, src, k):
        p = numpy.array([self._get(i, k)[0] for i in src])
        f = numpy.array([self._get(i, k)[1] for i in src])
        return p, f

    def select_hyper(self, src, k):
        '''Choose the length scale and the ridge by leave-one-out cross
        validation over the density matrices of entry k.'''
        z = self.z[src]
        p = self._targets(src, k)[0].reshape(len(src), -1)
        if self._cv_cols is None:
            rng = numpy.random.default_rng(7)
            ncol = p.shape[1]
            self._cv_cols = numpy.sort(rng.choice(
                ncol, min(ncol, _NSAMPLE_CV), replace=False))
        y = p[:,self._cv_cols]
        best = None
        for length in _LENGTH_SCALES:
            kmat = _kernel_matrix(z, z, length)
            for ridge in _RIDGES:
                kinv = _regularized_inverse(kmat, ridge)
                diag = kinv.diagonal()
                if (diag <= 0).any():
                    continue
                err = numpy.linalg.norm(kinv.dot(y) / diag[:,None])
                if best is None or err < best[0]:
                    best = (err, length, ridge)
        if best is None:
            best = (numpy.inf, _LENGTH_SCALES[-1], _RIDGES[-1])
        self._hyper = best[1:]
        logger.debug(self, 'shared model: nsrc=%d length=%g ridge=%g',
                     len(src), *self._hyper)
        return self._hyper

    def loo_error(self, src, k):
        '''Residual of the exactly treated configurations and the
        leave-one-out prediction error of the model, both measured as the
        norm of the commutator [F~, P~].

        Returns:
            (residuals, errors) : two arrays of length len(src)
        '''
        length, ridge = self._hyper
        z = self.z[src]
        kinv = _regularized_inverse(_kernel_matrix(z, z, length), ridge)
        diag = numpy.maximum(kinv.diagonal(), 1e-300)
        p, f = self._targets(src, k)
        shape = p.shape
        nsrc = len(src)
        p_loo = p - (kinv.dot(p.reshape(nsrc, -1)) / diag[:,None]).reshape(shape)
        f_loo = f - (kinv.dot(f.reshape(nsrc, -1)) / diag[:,None]).reshape(shape)
        comm = numpy.matmul(f, p)
        comm = comm - comm.swapaxes(-1, -2)
        comm_loo = numpy.matmul(f_loo, p_loo)
        comm_loo = comm_loo - comm_loo.swapaxes(-1, -2)
        res = numpy.linalg.norm(comm.reshape(nsrc, -1), axis=1)
        err = numpy.linalg.norm((comm_loo - comm).reshape(nsrc, -1), axis=1)
        return res, err

    def residuals(self, src, k):
        '''Norm of the commutator [F~, P~] of entry k'''
        p, f = self._targets(src, k)
        comm = numpy.matmul(f, p)
        comm = comm - comm.swapaxes(-1, -2)
        return numpy.linalg.norm(comm.reshape(len(src), -1), axis=1)

    def weights(self, src, dst):
        '''Regression weights, shape (len(dst), len(src))'''
        length, ridge = self._hyper
        z = self.z[src]
        kinv = _regularized_inverse(_kernel_matrix(z, z, length), ridge)
        return _kernel_matrix(self.z[dst], z, length).dot(kinv)

    def predict(self, src, dst, k, wts=None, store=True):
        '''Predict entry k of the configurations dst from the configurations
        src.  The prediction can be stored as a (synthetic) entry of dst.

        Returns:
            A list of (dm, fock) in the AO basis of each configuration in dst.
        '''
        if wts is None:
            wts = self.weights(src, dst)
        p, f = self._targets(src, k)
        p_pred = numpy.tensordot(wts, p, axes=1)
        f_pred = numpy.tensordot(wts, f, axes=1)
        out = []
        for n, i in enumerate(dst):
            if store:
                self._store(i, k, p_pred[n], f_pred[n])
            out.append(self.to_ao(i, p_pred[n], f_pred[n]))
        return out

    def to_ao(self, i, p, f):
        s_isqrt, s_sqrt = self._metric[i]
        dm = numpy.matmul(s_isqrt, numpy.matmul(p, s_isqrt.T))
        fock = numpy.matmul(s_sqrt, numpy.matmul(f, s_sqrt.T))
        return dm, fock


class _State:
    '''SCF state of one configuration.  The attributes mirror the local
    variables of :func:`pyscf.scf.hf.kernel`.'''
    def __init__(self, idx, mf):
        self.idx = idx
        self.mf = mf
        self.group = None
        self.level = 0
        self.started = False    # exact Fock builds are carried out
        self.done = False
        self.scf_conv = False
        self.cycle = 0
        self.nfock = 0
        self.promoted_at = 0
        self.inherited = None   # synthetic (dm, fock) history from the model
        self.s1e = self.h1e = self.x_orth = self.mf_diis = None
        self.dm = self.vhf = self.fock_last = None
        self.e_tot = numpy.nan
        self.mo_energy = self.mo_coeff = self.mo_occ = None
        self.norm_gorb = numpy.inf
        self.conv_tol = mf.conv_tol
        self.conv_tol_grad = mf.conv_tol_grad
        if self.conv_tol_grad is None:
            self.conv_tol_grad = numpy.sqrt(self.conv_tol)

    def plain_fock(self):
        return self.h1e + numpy.asarray(self.vhf)


def _norm_gorb(mf, mo_coeff, mo_occ, fock):
    gorb = mf.get_grad(mo_coeff, mo_occ, fock)
    norm_gorb = numpy.linalg.norm(gorb)
    if not hf.TIGHT_GRAD_CONV_TOL:
        norm_gorb = norm_gorb / numpy.sqrt(gorb.size)
    return norm_gorb


def _setup(st):
    '''Integrals and DIIS of one configuration.  No Fock build.'''
    mf = st.mf
    mol = mf.mol
    mf.dump_flags()
    mf.build(mol)
    st.s1e = mf.get_ovlp(mol)
    st.h1e = mf.get_hcore(mol)
    st.x_orth = mf.check_linear_dependency(st.s1e)
    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        assert issubclass(mf.DIIS, lib.diis.DIIS)
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback
        mf_diis.damp = mf.diis_damp
        mf_diis.Corth = st.x_orth
    else:
        mf_diis = None
    st.mf_diis = mf_diis
    return st


def _init_cycle(st, dm0=None, dump_chk=True):
    '''Initial guess and the Fock build of the initial guess.  Equivalent to
    the preamble of :func:`pyscf.scf.hf.kernel`.'''
    mf = st.mf
    mol = mf.mol
    if st.s1e is None:
        _setup(st)
    s1e, h1e = st.s1e, st.h1e
    if dm0 is None:
        dm = mf.get_init_guess(mol, mf.init_guess, s1e=s1e)
    else:
        dm = dm0
    vhf = mf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    logger.info(mf, 'init E= %.15g', e_tot)
    if dump_chk and mf.chkfile:
        chkfile.save_mol(mol, mf.chkfile)
    scf_conv = False
    mo_energy = mo_coeff = mo_occ = None
    cycle = 0
    mf.pre_kernel(locals())
    st.dm, st.vhf, st.e_tot = dm, vhf, e_tot
    st.nfock += 1
    st.cycle = 0
    st.started = True
    return st


def _anchor_cycle(st, dump_chk=True):
    '''First exact Fock build of a promoted configuration.

    The Fock matrix is built for the density matrix predicted by the shared
    model.  The difference between the exact and the predicted Fock matrix
    at this point is the error of the model.  It is added to the predicted
    Fock matrices of the earlier iterations, before they are loaded into the
    DIIS subspace.  The differences between the DIIS vectors, which determine
    the extrapolation, are then accurate to the relative error of the model.
    '''
    mf = st.mf
    mol = mf.mol
    h1e, s1e, mf_diis = st.h1e, st.s1e, st.mf_diis
    entries = st.inherited
    cycle = max(entries)
    dm, fock_model = entries[cycle]
    vhf = mf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    logger.info(mf, 'promoted at cycle %d  E= %.15g', cycle, e_tot)
    if mf_diis is not None:
        shift = h1e + numpy.asarray(vhf) - fock_model
        for k in sorted(entries)[:-1]:
            if k >= mf.diis_start_cycle:
                dm_k, fock_k = entries[k]
                mf.get_fock(h1e, s1e, fock_k + shift - h1e, dm_k, k, mf_diis)
    if dump_chk and mf.chkfile:
        chkfile.save_mol(mol, mf.chkfile)
    st.inherited = None
    st.dm, st.vhf, st.e_tot = dm, vhf, e_tot
    st.nfock += 1
    st.cycle = cycle
    return st


def _scf_cycle(st, callback=None, dump_chk=True, conv_check=True):
    '''One SCF iteration.  Equivalent to the loop body of
    :func:`pyscf.scf.hf.kernel`, followed by the extra cycle if the
    configuration converged in this iteration.'''
    mf = st.mf
    mol = mf.mol
    h1e, s1e, x_orth, mf_diis = st.h1e, st.s1e, st.x_orth, st.mf_diis
    dm, vhf, e_tot = st.dm, st.vhf, st.e_tot
    fock_last, cycle = st.fock_last, st.cycle
    conv_tol, conv_tol_grad = st.conv_tol, st.conv_tol_grad
    scf_conv = False

    dm_last = dm
    last_hf_e = e_tot

    fock = mf.get_fock(h1e, s1e, vhf, dm, cycle, mf_diis, fock_last=fock_last)
    mo_energy, mo_coeff = mf.eig(fock, s1e, x=x_orth)
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    vhf = mf.get_veff(mol, dm, dm_last, vhf)
    e_tot = mf.energy_tot(dm, h1e, vhf)

    fock_last = fock
    fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
    norm_gorb = _norm_gorb(mf, mo_coeff, mo_occ, fock)
    norm_ddm = numpy.linalg.norm(dm-dm_last)
    logger.info(mf, 'cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                cycle+1, e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)

    if callable(mf.check_convergence):
        scf_conv = mf.check_convergence(locals())
    elif abs(e_tot-last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
        scf_conv = True

    if dump_chk and mf.chkfile:
        mf.dump_chk(locals())
    if callable(mf.callback):
        mf.callback(locals())
    if callable(callback):
        callback(locals())

    st.nfock += 1
    st.cycle = cycle + 1
    st.dm, st.vhf, st.e_tot, st.fock_last = dm, vhf, e_tot, fock_last
    st.mo_energy, st.mo_coeff, st.mo_occ = mo_energy, mo_coeff, mo_occ
    st.norm_gorb = norm_gorb
    st.scf_conv = scf_conv
    if scf_conv or st.cycle >= mf.max_cycle:
        st.done = True
        if scf_conv and conv_check:
            _extra_cycle(st, fock, dump_chk)
    return st


def _extra_cycle(st, fock, dump_chk=True):
    '''The extra diagonalization and Fock build after convergence (conv_check
    in :func:`pyscf.scf.hf.kernel`)'''
    mf = st.mf
    mol = mf.mol
    h1e, s1e, x_orth = st.h1e, st.s1e, st.x_orth
    dm, vhf, e_tot = st.dm, st.vhf, st.e_tot
    conv_tol, conv_tol_grad = st.conv_tol, st.conv_tol_grad

    mo_energy, mo_coeff = mf.eig(fock, s1e, x=x_orth)
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
    vhf = mf.get_veff(mol, dm, dm_last, vhf)
    e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf), e_tot

    fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
    norm_gorb = _norm_gorb(mf, mo_coeff, mo_occ, fock)
    norm_ddm = numpy.linalg.norm(dm-dm_last)

    conv_tol = conv_tol * 10
    conv_tol_grad = conv_tol_grad * 3
    if callable(mf.check_convergence):
        scf_conv = mf.check_convergence(locals())
    elif abs(e_tot-last_hf_e) < conv_tol or norm_gorb < conv_tol_grad:
        scf_conv = True
    else:
        scf_conv = False
    logger.info(mf, 'Extra cycle  E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)
    if dump_chk and mf.chkfile:
        mf.dump_chk(locals())

    st.nfock += 1
    st.dm, st.vhf, st.e_tot = dm, vhf, e_tot
    st.mo_energy, st.mo_coeff, st.mo_occ = mo_energy, mo_coeff, mo_occ
    st.norm_gorb = norm_gorb
    st.scf_conv = scf_conv
    return st


class _Pool:
    '''Configuration-level thread pool.  The C kernels of PySCF release the
    GIL.  Each worker thread uses a fixed number of OpenMP threads.'''
    def __init__(self, nworkers, omp_threads):
        self.nworkers = nworkers
        self.omp_threads = omp_threads
        self._executor = None
        self._blas_ctl = None

    def __enter__(self):
        if self.nworkers > 1:
            try:
                from threadpoolctl import threadpool_limits
                # BLAS library bundled with numpy/scipy
                self._blas_ctl = threadpool_limits(self.omp_threads, 'blas')
            except Exception:
                self._blas_ctl = None
            self._executor = ThreadPoolExecutor(
                self.nworkers, initializer=lib.num_threads,
                initargs=(self.omp_threads,))
        return self

    def __exit__(self, *args):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self._blas_ctl is not None:
            self._blas_ctl.restore_original_limits()
            self._blas_ctl = None

    def map(self, fn, tasks):
        if self._executor is None or len(tasks) < 2:
            return [fn(*task) for task in tasks]
        futures = [self._executor.submit(fn, *task) for task in tasks]
        return [fut.result() for fut in futures]


def _call(fn, *args):
    return fn(*args)


def _group_key(mf):
    mol = mf.mol
    return (mf.__class__, getattr(mf, 'xc', None), tuple(mol.elements),
            mol.nelectron, mol.spin, mol.cart, mol.nao, mol.nbas,
            mol._bas[:,1:4].tobytes())


def _supports_cascade(mf):
    '''The shared model handles real AO matrices of shape (nao,nao) or
    (2,nao,nao) of molecules and the plain commutator DIIS.'''
    if mf.istype('GHF') or mf.istype('DHF') or hasattr(mf, 'cell'):
        return False
    if isinstance(mf.diis, lib.diis.DIIS):
        return type(mf.diis) is scf_diis.CDIIS
    return bool(mf.diis) and mf.DIIS is scf_diis.CDIIS


def _check_sanity(mf):
    if not isinstance(mf, hf.SCF):
        raise TypeError(f'{mf.__class__} is not an SCF object')
    cls = mf.__class__
    if cls.scf is not hf.SCF.scf or cls.kernel is not hf.SCF.kernel:
        # e.g. second order SCF
        raise NotImplementedError(
            f'JointSCF drives the DIIS iterations of pyscf.scf.hf.kernel. '
            f'{mf.__class__} defines a different SCF driver.')


def kernel(jmf, dm0=None, callback=None):
    '''Single-loop SCF driver for a batch of mean-field objects.

    Args:
        jmf : an instance of :class:`JointSCF`

    Kwargs:
        dm0 : list of ndarrays
            Initial guess density matrices.  If given, all configurations
            are treated exactly from the first iteration (schedule 'sync').

    Returns:
        A list of the SCF states of all configurations
    '''
    log = logger.new_logger(jmf)
    cput0 = log.init_timer()
    t_start = time.perf_counter()
    mfs = jmf.mfs
    states = [_State(i, mf) for i, mf in enumerate(mfs)]
    dump_chk = jmf.dump_chk
    conv_check = jmf.conv_check
    nworkers, omp_threads = jmf.get_num_workers()
    log.info('workers = %d  OpenMP threads per worker = %d', nworkers, omp_threads)

    schedule = jmf.schedule
    if dm0 is not None:
        schedule = 'sync'
    groups = []
    if schedule == 'cascade':
        groups = jmf.build_groups(states)
    elif schedule != 'sync':
        raise ValueError(f'Unknown schedule {schedule}')
    for model, members in groups:
        log.info('cascade group: %d configurations, %d descriptors, levels %s',
                 len(members), model.z.shape[1],
                 [len(level) for level in model.levels])

    history = []
    def record_round(it, tasks, t0):
        active = [st for st in states if st.started and not st.done]
        gmax = max([st.norm_gorb for st in active
                    if numpy.isfinite(st.norm_gorb)], default=0.)
        history.append({
            'round': it, 'nfock': len(tasks),
            'nconverged': sum(st.scf_conv and st.done for st in states),
            'npending': sum(not st.started for st in states),
            'max_norm_gorb': gmax, 'time': time.perf_counter() - t_start})
        log.info('round %3d  fock builds= %4d  converged= %4d  pending= %4d  '
                 'max|g|= %.3g', it, len(tasks), history[-1]['nconverged'],
                 history[-1]['npending'], gmax)
        return log.timer_debug1('round %d' % it, *t0)

    with _Pool(nworkers, omp_threads) as pool:
        in_cascade = set(st.idx for _, members in groups for st in members)
        # One- and two-electron integrals, DIIS objects
        pool.map(_setup, [(st,) for st in states])
        for model, members in groups:
            for n, st in enumerate(members):
                model.set_overlap(n, st.s1e)

        # round 0: Fock matrices of the initial guess
        t0 = (logger.process_clock(), logger.perf_counter())
        tasks = []
        for st in states:
            if st.idx in in_cascade and st.level > 0:
                continue
            tasks.append((st, None if dm0 is None else dm0[st.idx], dump_chk))
        pool.map(_init_cycle, tasks)
        for model, members in groups:
            for n, st in enumerate(members):
                if st.started:
                    model.record(n, 0, st.dm, st.plain_fock())
        t0 = record_round(0, tasks, t0)
        cput1 = log.timer('initialize joint scf', *cput0)

        it = 0
        while True:
            # Promotion of the next levels of the cascade groups
            for model, members in groups:
                _promote(jmf, model, members, log)

            active = [st for st in states if st.started and not st.done]
            if not active:
                break
            it += 1
            anchors = [st for st in active if st.inherited is not None]
            cycles = [st for st in active if st.inherited is None]
            tasks = ([(_anchor_cycle, st, dump_chk) for st in anchors] +
                     [(_scf_cycle, st, callback, dump_chk, conv_check) for st in cycles])
            pool.map(_call, tasks)
            for model, members in groups:
                for n, st in enumerate(members):
                    if st.started and not model.has_entry(n, st.cycle):
                        model.record(n, st.cycle, st.dm, st.plain_fock())
            t0 = record_round(it, tasks, t0)

    for st in states:
        _finalize_state(st)
    jmf.scf_history = history
    jmf.models = [model for model, _ in groups]
    log.timer('joint scf', *cput1)
    return states


def _finalize_state(st):
    '''Store the results in the mean-field object as SCF.scf does'''
    mf = st.mf
    mol = mf.mol
    scf_conv, e_tot, cycle = st.scf_conv, st.e_tot, st.cycle
    mo_energy, mo_coeff, mo_occ = st.mo_energy, st.mo_coeff, st.mo_occ
    dm, vhf, h1e, s1e = st.dm, st.vhf, st.h1e, st.s1e
    mf.converged = scf_conv
    mf.e_tot = e_tot
    mf.mo_energy, mf.mo_coeff, mf.mo_occ = mo_energy, mo_coeff, mo_occ
    mf.cycles = cycle
    mf.post_kernel(locals())
    mf._finalize()
    return mf


def _promote(jmf, model, members, log):
    '''Promote the next level of a cascade group if the SCF residual of the
    exactly treated configurations reached the accuracy of the model.'''
    if model.next_level >= len(model.levels):
        return
    src = [n for n, st in enumerate(members) if st.started]
    if any(members[n].inherited is not None for n in src):
        # The previous level has not been evaluated yet
        return
    running = numpy.array([not members[n].done for n in src])
    if running.any():
        # The latest SCF cycle which all running configurations completed
        it = min(members[n].cycle for n, run in zip(src, running) if run)
    else:
        it = max(members[n].cycle for n in src)
    model.select_hyper(src, it)
    res, err = model.loo_error(src, it)
    res_max = res[running].max() if running.any() else 0.
    eps = numpy.median(err)
    log.debug('cascade: cycle %d  residual= %.3g  model error= %.3g', it, res_max, eps)
    if res_max > jmf.promote_ratio * eps and it < jmf.max_lead_cycle:
        return

    # Learning curve of the model.  If twice the number of sources does not
    # reduce the error substantially, more exact configurations will not
    # improve the predictions and the rest of the group is promoted at once.
    nhalf = len(src) // 2
    saturated = not running.any()
    if not saturated and nhalf >= 2:
        half = sorted(src, key=lambda n: (members[n].level, model.rank[n]))[:nhalf]
        eps_half = numpy.median(model.loo_error(half, it)[1])
        saturated = eps_half < jmf.min_model_gain * eps
        log.debug('cascade: model error with %d sources= %.3g', nhalf, eps_half)

    # The DIIS vectors inherited by the promoted configurations.  Only
    # iterations in which the sources moved by more than the error of
    # the model provide usable differences.
    first = max(it - jmf.diis_inherit, 0)
    inherit = [k for k in range(first, it)
               if numpy.median(model.residuals(src, k)) > jmf.inherit_ratio * eps]
    dst = []
    while model.next_level < len(model.levels):
        dst.extend(model.levels[model.next_level])
        model.next_level += 1
        if not saturated:
            break
    wts = model.weights(src, dst)
    wts /= wts.sum(axis=1, keepdims=True)
    for n in dst:
        members[n].inherited = {}
    for k in range(first, it + 1):
        preds = model.predict(src, dst, k, wts, store=(k < it))
        if k == it or k in inherit:
            for n, pred in zip(dst, preds):
                members[n].inherited[k] = pred
    for n in dst:
        st = members[n]
        st.cycle = it
        st.promoted_at = it
        st.started = True
    model.promotions.append({
        'cycle': it, 'nsrc': len(src), 'ndst': len(dst), 'residual': res_max,
        'model_error': eps, 'ninherit': len(inherit), 'saturated': bool(saturated)})
    log.info('cascade: cycle %d  promoted %d configurations '
             '(residual %.3g, model error %.3g)', it, len(dst), res_max, eps)


class JointSCF(lib.StreamObject):
    '''Single-loop SCF for a batch of mean-field objects.

    Args:
        mfs : list of SCF objects, or one SCF object used as a template
        mols : list of Mole objects
            Required if ``mfs`` is a template.

    Attributes:
        schedule : str
            'sync' or 'cascade'.  See the documentation of the module.
        max_workers : int
            Number of configurations which are processed concurrently.  By
            default it equals the number of OpenMP threads for small
            molecules.  Each worker runs lib.num_threads()//max_workers
            OpenMP threads.
        descriptors : ndarray, shape (nconf, ndesc)
            Geometry descriptors of the cross-configuration model.  Inverse
            interatomic distances by default.
        n_lead : int
            Number of leaders of a cascade group.
        level_growth : float
            Ratio between the number of exactly treated configurations after
            and before a promotion.
        promote_ratio : float
            A level is promoted when the largest SCF residual of the exactly
            treated configurations is smaller than promote_ratio times the
            leave-one-out error of the model.
        diis_inherit : int
            Maximum number of DIIS vectors given to a promoted configuration.
        inherit_ratio : float
            An iteration contributes an inherited DIIS vector only if the
            SCF residual of the sources in that iteration is larger than
            inherit_ratio times the error of the model.
        min_model_gain : float
            If the error of the model with all sources is not at least this
            factor smaller than with half of the sources, all remaining
            configurations are promoted at once.
        max_lead_cycle : int
            The remaining configurations are promoted at this SCF cycle at
            the latest.
        min_group_size : int
            Groups with fewer configurations are treated in 'sync' mode.
        align : bool
            Align the orientations of the configurations (Kabsch) before
            comparing AO matrices.

    Saved results:
        e_tot : ndarray
            Total energies
        converged : ndarray of bool
        nfock : ndarray of int
            Number of exact Fock builds of each configuration
        cycles : ndarray of int
        mfs : list
            The mean-field objects.  They carry the same results (mo_coeff,
            mo_energy, mo_occ, e_tot, converged) as after mf.kernel().
    '''
    schedule = 'cascade'
    max_workers = None
    descriptors = None
    n_lead = None
    level_growth = 2.
    promote_ratio = 1.
    inherit_ratio = 3.
    min_model_gain = 1.5
    max_lead_cycle = 30
    diis_inherit = 8
    min_group_size = 6
    align = True
    dump_chk = True
    conv_check = True

    def __init__(self, mfs, mols=None):
        if mols is not None:
            mfs = [_clone(mfs, mol) for mol in mols]
        self.mfs = list(mfs)
        if len(self.mfs) == 0:
            raise ValueError('No configuration')
        for mf in self.mfs:
            _check_sanity(mf)
        mol = self.mfs[0].mol
        self.mol = mol
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self.e_tot = None
        self.converged = None
        self.nfock = None
        self.cycles = None
        self.levels = None
        self.promoted_at = None
        self.scf_history = None
        self.models = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s ********', self.__class__)
        log.info('number of configurations = %d', len(self.mfs))
        log.info('schedule = %s', self.schedule)
        if self.schedule == 'cascade':
            log.info('n_lead = %s  level_growth = %g  promote_ratio = %g  '
                     'diis_inherit = %d', self.n_lead, self.level_growth,
                     self.promote_ratio, self.diis_inherit)
        return self

    def get_num_workers(self):
        nthreads = lib.num_threads()
        nworkers = self.max_workers
        if nworkers is None:
            nao = max(mf.mol.nao for mf in self.mfs)
            nworkers = nthreads if nao <= 400 else 1
        nworkers = max(1, min(int(nworkers), len(self.mfs)))
        if nworkers == 1:
            return 1, nthreads
        return nworkers, max(1, nthreads // nworkers)

    def build_groups(self, states):
        '''Partition the batch into groups of configurations which share
        atoms, basis and method, and set up their cross-configuration models.
        '''
        buckets = {}
        for st in states:
            if _supports_cascade(st.mf):
                buckets.setdefault(_group_key(st.mf), []).append(st)
        groups = []
        for members in buckets.values():
            nmem = len(members)
            if nmem < self.min_group_size:
                continue
            desc = None
            if self.descriptors is not None:
                desc = numpy.asarray(self.descriptors)[[st.idx for st in members]]
            model = SharedModel([st.mf.mol for st in members], desc, self.align)
            model.verbose, model.stdout = self.verbose, self.stdout
            n_lead = self.n_lead
            if n_lead is None:
                n_lead = max(8, 2 * model.z.shape[1] + 2)
            n_lead = max(2, min(int(n_lead), nmem))
            levels = []
            nexact = 0
            size = n_lead
            while nexact < nmem:
                size = min(size, nmem - nexact)
                levels.append(model.order[nexact:nexact+size])
                nexact += size
                size = max(1, int(round(nexact * (self.level_growth - 1))))
            model.levels = levels
            model.next_level = 1
            # Each promotion delays the promoted configurations by one cycle
            model.window = self.diis_inherit + len(levels) + 2
            for ilevel, level in enumerate(levels):
                for n in level:
                    members[n].level = ilevel
                    members[n].group = model
            groups.append((model, members))
        return groups

    def kernel(self, dm0=None, **kwargs):
        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        states = kernel(self, dm0, **kwargs)
        self.e_tot = numpy.array([st.e_tot for st in states])
        self.converged = numpy.array([st.scf_conv for st in states])
        self.nfock = numpy.array([st.nfock for st in states])
        self.cycles = numpy.array([st.cycle for st in states])
        self.levels = numpy.array([st.level for st in states])
        self.promoted_at = numpy.array([st.promoted_at for st in states])
        logger.timer(self, 'JointSCF', *cput0)
        self._finalize()
        return self.e_tot
    scf = kernel

    def _finalize(self):
        nconf = len(self.mfs)
        nconv = int(self.converged.sum())
        logger.note(self, 'JointSCF: %d/%d configurations converged, '
                    '%d Fock builds (%.2f per configuration)',
                    nconv, nconf, self.nfock.sum(), self.nfock.mean())
        return self

    @property
    def mo_coeff(self):
        return [mf.mo_coeff for mf in self.mfs]

    @property
    def mo_energy(self):
        return [mf.mo_energy for mf in self.mfs]

    @property
    def mo_occ(self):
        return [mf.mo_occ for mf in self.mfs]

    def make_rdm1(self):
        return [mf.make_rdm1() for mf in self.mfs]

BatchSCF = JointSCF


def _clone(mf, mol):
    '''A copy of the mean-field object for a different geometry'''
    new = mf.copy()
    for key in ('grids', 'nlcgrids', '_numint', 'with_df', 'with_x2c', 'with_solvent'):
        obj = getattr(new, key, None)
        if obj is not None:
            setattr(new, key, copy.copy(obj))
    new.reset(mol)
    new._chkfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    new.chkfile = new._chkfile.name
    new.mo_coeff = new.mo_energy = new.mo_occ = None
    new.converged = False
    new.verbose = mol.verbose
    new.stdout = mol.stdout
    return new
