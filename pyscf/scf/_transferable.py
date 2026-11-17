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
Transferable shared-subspace joint SCF optimization internals.

Implements the ``schedule='transferable'`` solver for :class:`JointSCF`:
each configuration's occupied orbitals are parameterized as

    C_R = X_R @ qf(Phi_theta(z_R) @ A_R + Delta_R)

in an orientation-aligned symmetric-orthogonalized AO frame ``X_R``, with
single-Fock exact thin-QR reverse-mode gradients (via ``dE/dD = F``),
mini-batch shared-parameter updates (L-BFGS with Barzilai-Borwein scaling
or Adam), closed-form per-geometry subspace diagonalization, and optional
exact SCF polishing.
'''

import numpy
import scipy.linalg
from pyscf.scf import diis as scf_diis


class IncompatibleGroupError(RuntimeError):
    '''Raised when a group cannot use the transferable parameterization.'''


def canonical_ref_coords(mol):
    '''Rigid-motion-invariant centered coordinates for ``mol``.'''
    coords = numpy.asarray(mol.atom_coords(), dtype=float)
    mass = numpy.asarray(mol.atom_mass_list(), dtype=float)
    w = mass / mass.sum()
    c = coords - w.dot(coords)
    natm = len(coords)
    if natm < 2:
        return c

    e1 = None
    for i in range(1, natm):
        v1 = coords[i] - coords[0]
        n1 = float(numpy.linalg.norm(v1))
        if n1 > 1e-8:
            e1 = v1 / n1
            break
    if e1 is None:
        return c

    e2 = None
    for j in range(1, natm):
        vj = coords[j] - coords[0]
        u2 = vj - float(numpy.dot(vj, e1)) * e1
        n2 = float(numpy.linalg.norm(u2))
        if n2 > 1e-6:
            e2 = u2 / n2
            break
    if e2 is None:
        tmp = numpy.array([1., 0., 0.]) if abs(e1[0]) < 0.9 else numpy.array([0., 1., 0.])
        u2 = tmp - float(numpy.dot(tmp, e1)) * e1
        e2 = u2 / float(numpy.linalg.norm(u2))
    e3 = numpy.cross(e1, e2)
    frame = numpy.column_stack([e1, e2, e3])
    return c.dot(frame)


def aligned_orthogonalizer(mol, s1e, ref_coords=None, mass=None, align=True,
                           lindep=1e-10):
    '''Return ``(x_orth, s_x, rot)`` in the Kabsch-aligned Loewdin frame.

    ``x_orth`` has shape ``(nao, n_orth)`` with ``x_orth.T @ s1e @ x_orth = I``.
    ``s_x = s1e @ x_orth`` maps AO orbitals to aligned orthonormal coordinates
    via ``U = s_x.T @ C``.
    '''
    from pyscf.scf import jointscf

    s1e = numpy.asarray(s1e, dtype=float)
    rot = None
    if align and ref_coords is not None and not mol.cart and mol.natm >= 2:
        rmat = jointscf.kabsch_rotation(mol.atom_coords(), ref_coords, mass)
        if abs(rmat - numpy.eye(3)).max() > 1e-12:
            rot = mol.ao_rotation_matrix(rmat)

    s_al = rot.T.dot(s1e).dot(rot) if rot is not None else s1e
    s_al = 0.5 * (s_al + s_al.T)
    e, v = scipy.linalg.eigh(s_al)
    keep = e > lindep * max(float(e.max()), 1.0)
    if not numpy.any(keep):
        raise IncompatibleGroupError('Overlap matrix is numerically singular')
    if numpy.all(keep):
        x_al = (v / numpy.sqrt(e)).dot(v.T)
    else:
        x_al = v[:, keep] / numpy.sqrt(e[keep])
    x_orth = rot.dot(x_al) if rot is not None else x_al
    s_x = s1e.dot(x_orth)
    return x_orth, s_x, rot


def qf_with_r(y, rank_tol=1e-12):
    '''Thin QR factor ``(Q, R)`` with strictly positive diagonal on ``R``.

    Positive ``diag(R)`` ensures smooth left-orthogonal equivariance:
    ``qf_with_r(O @ y)[0] == O @ qf_with_r(y)[0]``.
    '''
    y = numpy.asarray(y, dtype=float)
    q, r = numpy.linalg.qr(y, mode='reduced')
    d = numpy.diag(r)
    scale = max(float(numpy.abs(r).max()), 1.0)
    if numpy.any(numpy.abs(d) <= rank_tol * scale):
        raise ValueError('Rank-deficient orbital matrix in qf_with_r')
    signs = numpy.where(d >= 0.0, 1.0, -1.0)
    q = q * signs[None, :]
    r = r * signs[:, None]
    return q, r


def qr_vjp(u, r, grad_u):
    '''Exact reverse-mode VJP of ``u = qf_with_r(y)[0]``.'''
    m_mat = u.conj().T.dot(grad_u)
    proj_grad = grad_u - u.dot(m_mat)
    skew_lower = numpy.tril(m_mat - m_mat.conj().T, k=-1)
    rhs = proj_grad + u.dot(skew_lower)
    grad_y_h = scipy.linalg.solve_triangular(
        r, rhs.conj().T, trans='N', lower=False, check_finite=False)
    return grad_y_h.conj().T


def procrustes_align(ref, tgt):
    '''Orthogonally rotate ``tgt`` to maximize ``Tr(ref.T @ tgt @ R)``.'''
    u, _, vt = numpy.linalg.svd(tgt.conj().T.dot(ref), full_matrices=False)
    rot = u.dot(vt)
    return tgt.dot(rot), rot


def _orthonormal_columns(mat, tol=1e-12):
    '''Extract orthonormal column basis via SVD.'''
    u, s, _ = numpy.linalg.svd(mat, full_matrices=False)
    if s.size == 0 or s[0] <= 0:
        return u[:, :0]
    keep = s > tol * max(float(s[0]), 1.0)
    return u[:, keep]


def _orthogonal_complement_vectors(basis, n_need):
    '''Return ``n_need`` deterministic orthonormal vectors orthogonal to ``basis``.'''
    n_rows = basis.shape[0]
    proj = numpy.eye(n_rows) - basis.dot(basis.T)
    e, v = scipy.linalg.eigh(0.5 * (proj + proj.T))
    # Largest eigenvalues (near 1.0) lie in the complement
    idx = numpy.argsort(-e)[:n_need]
    return v[:, idx]


class GeometryConditionedSubspace:
    '''Shared descriptor-conditioned subspace map ``Phi_theta(z_R)``.

        Phi_theta(z_R) = T_0 + sum_{h=1}^H tanh(W_1 @ z_R + b_1)_h * T_h
    '''

    def __init__(self, n_orth, latent_rank, desc_dim, n_hidden=8):
        self.n_orth = int(n_orth)
        self.latent_rank = int(latent_rank)
        self.desc_dim = int(desc_dim)
        self.n_hidden = int(n_hidden)
        self._t0_size = self.n_orth * self.latent_rank
        self._th_size = self.n_hidden * self.n_orth * self.latent_rank
        self._w1_size = self.n_hidden * self.desc_dim
        self._b1_size = self.n_hidden
        self.n_params = self._t0_size + self._th_size + self._w1_size + self._b1_size

    def unpack(self, theta):
        theta = numpy.asarray(theta, dtype=float).ravel()
        if theta.size != self.n_params:
            raise ValueError(f'Expected theta of size {self.n_params}, got {theta.size}')
        idx = 0
        t0 = theta[idx:idx + self._t0_size].reshape(self.n_orth, self.latent_rank)
        idx += self._t0_size
        th = theta[idx:idx + self._th_size].reshape(
            self.n_hidden, self.n_orth, self.latent_rank)
        idx += self._th_size
        w1 = theta[idx:idx + self._w1_size].reshape(self.n_hidden, self.desc_dim)
        idx += self._w1_size
        b1 = theta[idx:idx + self._b1_size]
        return t0, th, w1, b1

    def pack(self, t0, th, w1, b1):
        return numpy.concatenate([
            numpy.asarray(t0, dtype=float).ravel(),
            numpy.asarray(th, dtype=float).ravel(),
            numpy.asarray(w1, dtype=float).ravel(),
            numpy.asarray(b1, dtype=float).ravel(),
        ])

    def eval(self, z_r, theta):
        t0, th, w1, b1 = self.unpack(theta)
        a = numpy.tanh(w1.dot(z_r) + b1)
        return t0 + numpy.einsum('h,hrk->rk', a, th)

    def eval_batch(self, z_mat, theta):
        t0, th, w1, b1 = self.unpack(theta)
        a = numpy.tanh(z_mat.dot(w1.T) + b1[None, :])
        return t0[None, :, :] + numpy.einsum('mh,hrk->mrk', a, th)

    def vjp(self, z_r, theta, grad_phi):
        _, th, w1, b1 = self.unpack(theta)
        a = numpy.tanh(w1.dot(z_r) + b1)
        sech2 = 1.0 - a * a
        g_t0 = grad_phi
        g_th = a[:, None, None] * grad_phi[None, :, :]
        g_a = numpy.einsum('hrk,rk->h', th, grad_phi)
        g_pre = g_a * sech2
        g_w1 = numpy.outer(g_pre, z_r)
        g_b1 = g_pre
        return numpy.concatenate([
            g_t0.ravel(), g_th.ravel(), g_w1.ravel(), g_b1.ravel()
        ])


class _LocalDIIS:
    '''Minimal Pulay DIIS extrapolator in aligned Loewdin space.'''

    def __init__(self, max_space=6, reg=1e-13):
        self.max_space = max(int(max_space), 1)
        self.reg = float(reg)
        self._focks = []
        self._errors = []

    def reset(self):
        self._focks.clear()
        self._errors.clear()

    def update(self, fock, error):
        f_arr = numpy.asarray(fock, dtype=float).copy()
        e_vec = numpy.asarray(error, dtype=float).ravel().copy()
        if self._focks and self._focks[-1].shape != f_arr.shape:
            self._focks.clear()
            self._errors.clear()
        self._focks.append(f_arr)
        self._errors.append(e_vec)
        if len(self._focks) > self.max_space:
            self._focks.pop(0)
            self._errors.pop(0)
        n = len(self._focks)
        if n < 2:
            return f_arr

        b_mat = numpy.empty((n + 1, n + 1), dtype=float)
        b_mat[-1, :] = -1.0
        b_mat[:, -1] = -1.0
        b_mat[-1, -1] = 0.0
        for i in range(n):
            for j in range(i, n):
                val = float(numpy.dot(self._errors[i], self._errors[j]))
                b_mat[i, j] = val
                b_mat[j, i] = val
            b_mat[i, i] += self.reg
        rhs = numpy.zeros(n + 1, dtype=float)
        rhs[-1] = -1.0
        try:
            sol = numpy.linalg.solve(b_mat, rhs)
            weights = sol[:n]
        except numpy.linalg.LinAlgError:
            return f_arr
        if (not numpy.all(numpy.isfinite(weights)) or
                float(numpy.max(numpy.abs(weights))) > 3.0):
            return f_arr
        extrap = numpy.zeros_like(f_arr)
        for w_i, f_i in zip(weights, self._focks):
            extrap += w_i * f_i
        return 0.5 * (extrap + extrap.T)


def _fit_local_a(phi, target, reg=1e-12):
    k = phi.shape[1]
    gram = phi.T.dot(phi) + reg * numpy.eye(k)
    rhs = phi.T.dot(target)
    return numpy.linalg.solve(gram, rhs)


def _assemble_delta(delta_entry, correction_rank):
    if correction_rank == 0 or delta_entry is None:
        return None
    if isinstance(delta_entry, tuple):
        b_mat, g_mat = delta_entry
        return b_mat.dot(g_mat.T)
    return numpy.asarray(delta_entry, dtype=float)


def _is_factored_delta(correction_rank, m):
    return (correction_rank is not None and correction_rank != 0
            and int(correction_rank) < m)


def _resolve_latent_ranks(n_orth, nelec_channels, latent_rank):
    ranks = []
    for m in nelec_channels:
        if m <= 0:
            ranks.append(0)
        elif latent_rank is None:
            ranks.append(min(n_orth, m + max(4, m // 2)))
        else:
            ranks.append(min(n_orth, max(int(latent_rank), m)))
    return ranks


def evaluate_transferable_loss_and_grad(
    mfs, theta, a_list, delta_list=None, *,
    descriptors=None, weights=None, latent_rank=None, correction_rank=0,
    n_hidden=8, reg_delta=1e-6, reg_theta=1e-6, align=True,
    providers=None, ref_coords=None, mass=None, z=None
):
    '''Evaluate the transferable joint objective and exact single-Fock gradients.

    Uses ``dE/dD = F`` from a single ``mf.get_veff`` call per configuration and
    backpropagates through ``qf_with_r`` and ``Phi_theta(z_R)`` to return exact
    analytical gradients for ``theta``, ``a_list``, and ``delta_list``.

    Returns ``(loss, grad_theta, grad_a_list, grad_delta_list, energies)``.
    '''
    from pyscf.scf import jointscf

    mfs = list(mfs)
    nconf = len(mfs)
    if nconf == 0:
        raise ValueError('No configurations')
    mols = [mf.mol for mf in mfs]
    if z is None:
        if descriptors is None:
            descriptors = [jointscf.default_descriptor(mol) for mol in mols]
        z = jointscf._reduce_descriptors(descriptors)
    else:
        z = numpy.asarray(z, dtype=float)

    if weights is None:
        w = numpy.full(nconf, 1.0 / nconf)
    else:
        w = numpy.asarray(weights, dtype=float)
        w = w / w.sum()

    order = jointscf.farthest_point_order(z)
    if ref_coords is None:
        ref_coords = canonical_ref_coords(mols[order[0]])
    if mass is None:
        mass = mols[order[0]].atom_mass_list()

    nspin = 2 if mfs[0].istype('UHF') else 1
    occ_val = 1.0 if nspin == 2 else 2.0
    nelec_channels = (list(mfs[0].nelec) if nspin == 2
                      else [mols[0].nelectron // 2])
    active_spins = [s for s, m in enumerate(nelec_channels) if m > 0]

    metrics = []
    s1e_list = []
    h1e_list = []
    for mf in mfs:
        mol = mf.mol
        if getattr(mf, 'grids', None) is not None and mf.grids.coords is None:
            mf.build(mol)
        s1e = mf.get_ovlp(mol)
        h1e = mf.get_hcore(mol)
        s1e_list.append(s1e)
        h1e_list.append(h1e)
        metrics.append(aligned_orthogonalizer(mol, s1e, ref_coords, mass, align))

    n_orth = metrics[0][0].shape[1]
    ranks = _resolve_latent_ranks(n_orth, nelec_channels, latent_rank)

    if providers is None:
        providers = {}
        for s in active_spins:
            providers[s] = GeometryConditionedSubspace(
                n_orth, ranks[s], z.shape[1], n_hidden)

    theta_is_list = isinstance(theta, (list, tuple))
    if theta_is_list:
        theta_by_spin = {s: numpy.asarray(theta[s], dtype=float).ravel()
                         for s in active_spins}
    elif len(active_spins) == 1:
        theta_by_spin = {active_spins[0]: numpy.asarray(theta, dtype=float).ravel()}
    else:
        theta_flat = numpy.asarray(theta, dtype=float).ravel()
        theta_by_spin = {}
        off = 0
        for s in active_spins:
            sz = providers[s].n_params
            theta_by_spin[s] = theta_flat[off:off + sz]
            off += sz

    grad_theta_by_spin = {
        s: reg_theta * theta_by_spin[s] for s in active_spins
    }
    loss_reg_theta = sum(
        0.5 * reg_theta * float(numpy.dot(theta_by_spin[s], theta_by_spin[s]))
        for s in active_spins
    )

    grad_a_list = []
    grad_delta_list = None if correction_rank == 0 else []
    energies = numpy.empty(nconf, dtype=float)
    loss_e = 0.0
    loss_reg_delta = 0.0

    for i, mf in enumerate(mfs):
        mol = mf.mol
        x_orth = metrics[i][0]
        nao = mol.nao
        a_entry = a_list[i]
        a_by_spin = [a_entry] if nspin == 1 else list(a_entry)
        if correction_rank == 0 or delta_list is None:
            d_by_spin = [None] * nspin
        else:
            d_entry = delta_list[i]
            d_by_spin = [d_entry] if nspin == 1 else list(d_entry)

        phi_s = {}
        u_s = {}
        r_s = {}
        c_s = {}
        delta_mat_s = {}
        dm_spins = []

        for s in range(nspin):
            m_s = nelec_channels[s]
            if m_s == 0:
                dm_spins.append(numpy.zeros((nao, nao), dtype=float))
                continue
            phi = providers[s].eval(z[i], theta_by_spin[s])
            a_mat = numpy.asarray(a_by_spin[s], dtype=float)
            y = phi.dot(a_mat)
            d_mat = _assemble_delta(d_by_spin[s], correction_rank)
            if d_mat is not None:
                y = y + d_mat
                loss_reg_delta += 0.5 * reg_delta * float(numpy.sum(d_mat * d_mat))
            u, r_qr = qf_with_r(y)
            c = x_orth.dot(u)
            phi_s[s] = phi
            u_s[s] = u
            r_s[s] = r_qr
            c_s[s] = c
            delta_mat_s[s] = d_mat
            dm_spins.append(occ_val * c.dot(c.T))

        dm = dm_spins[0] if nspin == 1 else numpy.array(dm_spins)
        vhf = mf.get_veff(mol, dm)
        e_tot = float(numpy.real(mf.energy_tot(dm, h1e_list[i], vhf)))
        fock = numpy.asarray(mf.get_fock(h1e_list[i], s1e_list[i], vhf, dm))
        energies[i] = e_tot
        loss_e += w[i] * e_tot

        ga_spins = []
        gd_spins = []
        for s in range(nspin):
            m_s = nelec_channels[s]
            if m_s == 0:
                ga_spins.append(numpy.zeros((0, 0)))
                if grad_delta_list is not None:
                    gd_spins.append(None)
                continue
            f_ao = fock if nspin == 1 else fock[s]
            f_ao = 0.5 * (f_ao + f_ao.T)
            f_orth = x_orth.T.dot(f_ao).dot(x_orth)
            f_orth = 0.5 * (f_orth + f_orth.T)

            grad_u = 2.0 * occ_val * f_orth.dot(u_s[s])
            g_y = w[i] * qr_vjp(u_s[s], r_s[s], grad_u)

            a_mat = numpy.asarray(a_by_spin[s], dtype=float)
            ga_spins.append(phi_s[s].T.dot(g_y))
            grad_phi = g_y.dot(a_mat.T)
            grad_theta_by_spin[s] += providers[s].vjp(z[i], theta_by_spin[s], grad_phi)

            if grad_delta_list is not None:
                g_delta = g_y + reg_delta * delta_mat_s[s]
                if isinstance(d_by_spin[s], tuple):
                    b_mat, g_mat = d_by_spin[s]
                    gd_spins.append((g_delta.dot(g_mat), g_delta.T.dot(b_mat)))
                else:
                    gd_spins.append(g_delta)

        grad_a_list.append(ga_spins[0] if nspin == 1 else ga_spins)
        if grad_delta_list is not None:
            grad_delta_list.append(gd_spins[0] if nspin == 1 else gd_spins)

    loss = loss_e + loss_reg_theta + loss_reg_delta
    if theta_is_list:
        grad_theta = [grad_theta_by_spin[s] for s in active_spins]
    elif len(active_spins) == 1:
        grad_theta = grad_theta_by_spin[active_spins[0]]
    else:
        grad_theta = numpy.concatenate([grad_theta_by_spin[s] for s in active_spins])
    return loss, grad_theta, grad_a_list, grad_delta_list, energies


class _LBFGSOptimizer:
    '''Two-loop L-BFGS with Barzilai-Borwein scaling on shared parameters.'''

    def __init__(self, max_hist=8):
        self.max_hist = max_hist
        self.s_hist = []
        self.y_hist = []
        self.prev_theta = None

    def direction(self, grad):
        q = grad.copy()
        k = len(self.s_hist)
        if k == 0:
            gn = float(numpy.linalg.norm(grad))
            return -0.15 * grad / max(gn, 1.0)

        alphas = []
        rhos = []
        for s_i, y_i in zip(reversed(self.s_hist), reversed(self.y_hist)):
            rho = 1.0 / float(numpy.dot(y_i, s_i))
            alpha = rho * float(numpy.dot(s_i, q))
            q = q - alpha * y_i
            alphas.append(alpha)
            rhos.append(rho)

        s_last, y_last = self.s_hist[-1], self.y_hist[-1]
        ss = float(numpy.dot(s_last, s_last))
        yy = float(numpy.dot(y_last, y_last))
        gamma = float(numpy.clip(numpy.sqrt(ss / max(yy, 1e-16)), 1e-3, 2.0))
        r = gamma * q

        for (s_i, y_i), alpha, rho in zip(
            zip(self.s_hist, self.y_hist), reversed(alphas), reversed(rhos)
        ):
            beta = rho * float(numpy.dot(y_i, r))
            r = r + s_i * (alpha - beta)

        if float(numpy.dot(r, grad)) >= 0.0:
            gn = float(numpy.linalg.norm(grad))
            return -0.15 * grad / max(gn, 1.0)
        return -r

    def record(self, s_vec, y_vec):
        sn = float(numpy.linalg.norm(s_vec))
        yn = float(numpy.linalg.norm(y_vec))
        sy = float(numpy.dot(s_vec, y_vec))
        if sy > 1e-10 * sn * yn:
            self.s_hist.append(s_vec.copy())
            self.y_hist.append(y_vec.copy())
            if len(self.s_hist) > self.max_hist:
                self.s_hist.pop(0)
                self.y_hist.pop(0)


class _AdamOptimizer:
    '''Bias-corrected Adam optimizer on shared parameters.'''

    def __init__(self, size, lr=0.02, beta1=0.9, beta2=0.99, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = numpy.zeros(size, dtype=float)
        self.v = numpy.zeros(size, dtype=float)

    def step(self, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad * grad)
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        return -self.lr * m_hat / (numpy.sqrt(v_hat) + self.eps)


class TransferableGroupModel:
    '''Shared transferable orbital model for one homogeneous configuration group.'''

    def __init__(self, mfs, descriptors=None, align=True, latent_rank=None,
                 correction_rank=0, n_hidden=8, reg_delta=1e-6, reg_theta=1e-6,
                 optimizer='lbfgs', seed=0):
        from pyscf.scf import jointscf

        self.mfs = list(mfs)
        self.mol = self.mfs[0].mol
        self.nconf = len(self.mfs)
        mols = [mf.mol for mf in self.mfs]
        if descriptors is None:
            descriptors = [jointscf.default_descriptor(mol) for mol in mols]
        self.z = jointscf._reduce_descriptors(descriptors)
        self.order = jointscf.farthest_point_order(self.z)
        self.ref_idx = int(self.order[0])
        self.ref_coords = canonical_ref_coords(mols[self.ref_idx])
        self.mass = mols[self.ref_idx].atom_mass_list()
        self.align = bool(align)
        self.latent_rank_req = latent_rank
        self.correction_rank = correction_rank
        self.n_hidden = int(n_hidden)
        self.reg_delta = float(reg_delta)
        self.reg_theta = float(reg_theta)
        self.optimizer_name = str(optimizer).lower()
        if self.optimizer_name not in ('lbfgs', 'adam'):
            raise ValueError(f'Unknown transferable optimizer {optimizer}')
        self.seed = int(seed)

        self.nspin = 2 if self.mfs[0].istype('UHF') else 1
        self.occ_val = 1.0 if self.nspin == 2 else 2.0
        self.nelec_channels = (list(self.mfs[0].nelec) if self.nspin == 2
                               else [self.mol.nelectron // 2])
        self.active_spins = [s for s, m in enumerate(self.nelec_channels) if m > 0]
        self.weights = numpy.full(self.nconf, 1.0 / self.nconf, dtype=float)

        self.metrics = []
        self.n_orth = 0
        self.latent_ranks = []
        self.providers = {}
        self.theta = {}
        self.a_mats = {}
        self.delta_entries = {}
        self.diis_buffers = {}
        self.cached_f_orth = {}
        self.cached_f_diis = {}
        self.opt_state = {}
        self.prev_theta = {}
        self.last_energies = numpy.full(self.nconf, numpy.nan, dtype=float)
        self.visits = numpy.zeros(self.nconf, dtype=int)
        self.nfock_shared_total = 0
        self.history = []

    def initialize(self, setup_data):
        '''Zero-Fock initialization of ``(theta, A_R, Delta_R)`` from SAD guesses.'''
        mols = [mf.mol for mf in self.mfs]
        metrics = []
        s1e_list = []
        h1e_list = []
        dm_init_list = []
        for i, (s1e, h1e, dm0) in enumerate(setup_data):
            s1e_list.append(s1e)
            h1e_list.append(h1e)
            dm_init_list.append(dm0)
            metrics.append(aligned_orthogonalizer(
                mols[i], s1e, self.ref_coords, self.mass, self.align))

        n_orths = {m[0].shape[1] for m in metrics}
        if len(n_orths) != 1:
            raise IncompatibleGroupError(
                'Inconsistent linearly independent AO counts across geometries')
        self.metrics = metrics
        self.n_orth = next(iter(n_orths))
        self.latent_ranks = _resolve_latent_ranks(
            self.n_orth, self.nelec_channels, self.latent_rank_req)

        d_dim = self.z.shape[1]
        n_hid = self.n_hidden

        for s in self.active_spins:
            m = self.nelec_channels[s]
            k = self.latent_ranks[s]
            provider = GeometryConditionedSubspace(self.n_orth, k, d_dim, n_hid)
            self.providers[s] = provider

            u0_occ = numpy.empty((self.nconf, self.n_orth, m), dtype=float)
            n_vir = k - m
            u0_extra = numpy.empty((self.nconf, self.n_orth, n_vir), dtype=float)
            for i in range(self.nconf):
                x_orth, s_x, _ = self.metrics[i]
                dm_s = dm_init_list[i] if self.nspin == 1 else dm_init_list[i][s]
                p0 = s_x.T.dot(dm_s).dot(s_x)
                h0 = x_orth.T.dot(h1e_list[i]).dot(x_orth)
                h_scale = max(float(numpy.linalg.norm(h0)), 1.0)
                mat = -0.5 * (p0 + p0.T) + 1e-2 * (0.5 * (h0 + h0.T)) / h_scale
                _, vecs = scipy.linalg.eigh(mat)
                u0_occ[i] = vecs[:, :m]
                if n_vir > 0:
                    u0_extra[i] = vecs[:, m:k]

            ref_u = u0_occ[self.ref_idx].copy()
            for i in range(self.nconf):
                u0_occ[i], _ = procrustes_align(ref_u, u0_occ[i])

            if n_vir > 0:
                u_mean = qf_with_r(numpy.mean(u0_occ, axis=0))[0]
                diff_list = [u0_occ[i] - u_mean.dot(u_mean.T.dot(u0_occ[i]))
                             for i in range(self.nconf)]
                blocks = [numpy.hstack(diff_list), 0.5 * numpy.hstack(list(u0_extra))]
                u_svd, _, _ = numpy.linalg.svd(numpy.hstack(blocks), full_matrices=False)
                v_pol_glob = u_svd[:, :n_vir]
                u_targets = numpy.empty((self.nconf, self.n_orth, k), dtype=float)
                for i in range(self.nconf):
                    v_perp = v_pol_glob - u0_occ[i].dot(u0_occ[i].T.dot(v_pol_glob))
                    try:
                        v_orth, _ = qf_with_r(v_perp)
                    except ValueError:
                        v_orth = _orthogonal_complement_vectors(u0_occ[i], n_vir)
                    v_al, _ = procrustes_align(v_pol_glob, v_orth)
                    u_targets[i] = numpy.hstack([u0_occ[i], v_al])
            else:
                u_targets = u0_occ

            rng = numpy.random.default_rng(self.seed + 101 * s)
            w1 = numpy.zeros((n_hid, d_dim), dtype=float)
            b1 = numpy.zeros((n_hid,), dtype=float)
            n_lin = min(d_dim, n_hid)
            w1[:n_lin, :n_lin] = 0.35 * numpy.eye(n_lin)
            if n_hid > n_lin:
                rem = n_hid - n_lin
                w1[n_lin:] = rng.normal(
                    scale=0.4 / numpy.sqrt(max(d_dim, 1)), size=(rem, d_dim))
                b1[n_lin:] = rng.uniform(-0.5, 0.5, size=(rem,))

            z_act = numpy.tanh(self.z.dot(w1.T) + b1[None, :])
            design = numpy.hstack([numpy.ones((self.nconf, 1)), z_act])
            reg_diag = numpy.full(1 + n_hid, 1e-4, dtype=float)
            reg_diag[0] = 1e-10
            reg_diag[1:1 + n_lin] = 1e-5
            gram = design.T.dot(design) + numpy.diag(reg_diag)
            rhs = design.T.dot(u_targets.reshape(self.nconf, -1))
            w_fit = numpy.linalg.solve(gram, rhs).reshape(1 + n_hid, self.n_orth, k)
            theta_s = provider.pack(w_fit[0], w_fit[1:], w1, b1)
            self.theta[s] = theta_s

            a_s = []
            d_s = []
            factored = _is_factored_delta(self.correction_rank, m)
            r_del = int(self.correction_rank) if factored else None
            for i in range(self.nconf):
                phi_i = provider.eval(self.z[i], theta_s)
                a_i = _fit_local_a(phi_i, u0_occ[i])
                a_s.append(a_i)
                if self.correction_rank == 0:
                    d_s.append(None)
                else:
                    res = u0_occ[i] - phi_i.dot(a_i)
                    if not factored:
                        d_s.append(res)
                    else:
                        q_phi = _orthonormal_columns(phi_i)
                        res_perp = res - q_phi.dot(q_phi.T.dot(res))
                        u_d, sv_d, vt_d = numpy.linalg.svd(res_perp, full_matrices=False)
                        if sv_d.size > 0 and sv_d[0] > 1e-8:
                            keep_d = min(r_del, sv_d.size)
                            b_i = u_d[:, :keep_d] * numpy.sqrt(sv_d[:keep_d])[None, :]
                            g_i = vt_d[:keep_d, :].T * numpy.sqrt(sv_d[:keep_d])[None, :]
                            if keep_d < r_del:
                                comp = _orthogonal_complement_vectors(
                                    numpy.hstack([q_phi, b_i]), r_del - keep_d)
                                b_i = numpy.hstack([b_i, comp])
                                g_i = numpy.hstack([g_i, numpy.full((m, r_del - keep_d), 1e-4)])
                        else:
                            b_i = _orthogonal_complement_vectors(q_phi, r_del)
                            g_i = rng.normal(scale=1e-4, size=(m, r_del))
                        d_s.append((b_i, g_i))
            self.a_mats[s] = a_s
            self.delta_entries[s] = d_s
            self.diis_buffers[s] = [_LocalDIIS(max_space=6) for _ in range(self.nconf)]
            self.cached_f_orth[s] = [None] * self.nconf
            self.cached_f_diis[s] = [None] * self.nconf
            if self.optimizer_name == 'lbfgs':
                self.opt_state[s] = _LBFGSOptimizer(max_hist=8)
            else:
                self.opt_state[s] = _AdamOptimizer(provider.n_params)

    def select_batch(self, step, batch_size):
        '''Stratified round-robin batch over ``farthest_point_order``.'''
        b = min(max(1, int(batch_size)), self.nconf)
        if b == self.nconf:
            return list(self.order)
        start = (step * b) % self.nconf
        return [int(self.order[(start + k) % self.nconf]) for k in range(b)]

    def forward_batch(self, indices):
        '''Build AO densities and cache QR intermediates for ``indices``.'''
        unvis = [idx for idx in indices if self.visits[idx] == 0]
        if unvis and numpy.any(self.visits > 0):
            self._interpolate_a_indices(unvis)

        dms = []
        cache = []
        for idx in indices:
            x_orth = self.metrics[idx][0]
            nao = self.mfs[idx].mol.nao
            dm_spins = []
            spin_cache = {}
            for s in range(self.nspin):
                m_s = self.nelec_channels[s]
                if m_s == 0:
                    dm_spins.append(numpy.zeros((nao, nao), dtype=float))
                    continue
                phi = self.providers[s].eval(self.z[idx], self.theta[s])
                a_mat = self.a_mats[s][idx]
                d_mat = _assemble_delta(self.delta_entries[s][idx], self.correction_rank)
                y = phi.dot(a_mat) if d_mat is None else (phi.dot(a_mat) + d_mat)
                try:
                    u, r_qr = qf_with_r(y)
                except ValueError:
                    # Recover smoothly if a local correction became collinear
                    y = phi.dot(a_mat)
                    u, r_qr = qf_with_r(y)
                c = x_orth.dot(u)
                dm_spins.append(self.occ_val * c.dot(c.T))
                spin_cache[s] = (phi, u, r_qr)
            dms.append(dm_spins[0] if self.nspin == 1 else numpy.array(dm_spins))
            cache.append(spin_cache)
        return dms, cache

    def _batch_surrogate_val_and_grad(self, indices, f_orth_batch, theta_s, s, w_batch,
                                      need_grad=True):
        '''Evaluate linearized fixed-Fock KS surrogate on shared subspace for ``theta``.'''
        provider = self.providers[s]
        val = 0.5 * self.reg_theta * float(numpy.dot(theta_s, theta_s))
        g_th = (self.reg_theta * theta_s) if need_grad else None
        for pos, idx in enumerate(indices):
            phi = provider.eval(self.z[idx], theta_s)
            a_mat = self.a_mats[s][idx]
            y = phi.dot(a_mat)
            u, r_qr = qf_with_r(y)
            fu = f_orth_batch[pos].dot(u)
            val += w_batch[pos] * self.occ_val * float(numpy.sum(u * fu))
            if need_grad:
                grad_u = 2.0 * self.occ_val * fu
                g_y = w_batch[pos] * qr_vjp(u, r_qr, grad_u)
                g_th += provider.vjp(self.z[idx], theta_s, g_y.dot(a_mat.T))
        return val, g_th

    def _select_replay_indices(self, indices):
        '''Select up to 32 visited geometries (always including ``indices``) in deterministic order.'''
        idx_set = set(int(i) for i in indices)
        ref_spin = self.active_spins[0]
        visited_ordered = [
            int(j) for j in self.order
            if int(j) in idx_set or self.cached_f_orth[ref_spin][int(j)] is not None
        ]
        if len(visited_ordered) <= 32:
            return visited_ordered
        others = [j for j in visited_ordered if j not in idx_set]
        k_keep = max(0, 32 - len(idx_set))
        if k_keep > 0 and others:
            picked = {others[int(k * len(others) / k_keep)] for k in range(k_keep)}
        else:
            picked = set()
        keep_set = idx_set | picked
        return [j for j in visited_ordered if j in keep_set]

    def _solve_local_subspace(self, s, idx, f_mat, w_idx, phi_eval, u_ref):
        '''Closed-form subspace diagonalization for ``(A_R, Delta_R)`` on ``f_mat``.'''
        m_s = self.nelec_channels[s]
        q_phi = _orthonormal_columns(phi_eval)
        if q_phi.shape[1] < m_s and self.correction_rank == 0:
            q_phi = qf_with_r(phi_eval[:, :m_s])[0]

        if self.correction_rank == 0:
            f_sub = q_phi.T.dot(f_mat).dot(q_phi)
            f_sub = 0.5 * (f_sub + f_sub.T)
            _, v_sub = scipy.linalg.eigh(f_sub)
            u_new = q_phi.dot(v_sub[:, :m_s])
            u_al, _ = procrustes_align(phi_eval[:, :m_s], u_new)
            return _fit_local_a(phi_eval, u_al), None, u_new

        p_phi = q_phi.dot(q_phi.T)
        if self.reg_delta > 0.0:
            penalty = self.reg_delta / (2.0 * max(w_idx, 1e-12) * self.occ_val)
            f_eff = f_mat + min(penalty, 0.1) * (numpy.eye(self.n_orth) - p_phi)
        else:
            f_eff = f_mat
        f_eff = 0.5 * (f_eff + f_eff.T)

        if not _is_factored_delta(self.correction_rank, m_s):
            _, v_full = scipy.linalg.eigh(f_eff)
            u_new = v_full[:, :m_s]
            u_al, _ = procrustes_align(phi_eval[:, :m_s], u_new)
            a_fit = _fit_local_a(phi_eval, u_al)
            d_new = (u_al - phi_eval.dot(a_fit)) / (1.0 + self.reg_delta)
            return a_fit, d_new, u_new

        r_del = int(self.correction_rank)
        b_cur, _ = self.delta_entries[s][idx]
        b_perp = b_cur - p_phi.dot(b_cur)

        f_sub0 = q_phi.T.dot(f_eff).dot(q_phi)
        f_sub0 = 0.5 * (f_sub0 + f_sub0.T)
        e_sub0, v_sub0 = scipy.linalg.eigh(f_sub0)
        u_0 = q_phi.dot(v_sub0[:, :m_s])
        eps_0 = e_sub0[:m_s]

        fu0 = f_eff.dot(u_0)
        r_0 = fu0 - p_phi.dot(fu0)
        fur = f_eff.dot(u_ref)
        r_ref = fur - p_phi.dot(fur)

        q_res = _orthonormal_columns(numpy.hstack([b_perp, r_0, r_ref]))
        if q_res.shape[1] > 0:
            f_res = q_res.T.dot(f_eff).dot(q_res)
            f_res = 0.5 * (f_res + f_res.T)
            eps_vir, v_res = scipy.linalg.eigh(f_res)
            psi_vir = q_res.dot(v_res)
            coup = psi_vir.T.dot(fu0)
            denom = numpy.sqrt(numpy.maximum(eps_vir[:, None] - eps_0[None, :], 0.05))
            m_pt2 = coup / denom
            u_m, _, _ = numpy.linalg.svd(m_pt2, full_matrices=False)
            keep_r = min(r_del, u_m.shape[1])
            w_del = psi_vir.dot(u_m[:, :keep_r])
        else:
            w_del = numpy.zeros((self.n_orth, 0), dtype=float)

        if w_del.shape[1] < r_del:
            basis_used = numpy.hstack([q_phi, w_del]) if w_del.shape[1] > 0 else q_phi
            w_extra = _orthogonal_complement_vectors(basis_used, r_del - w_del.shape[1])
            w_del = numpy.hstack([w_del, w_extra])

        q_aug = numpy.hstack([q_phi, w_del])
        f_aug = q_aug.T.dot(f_eff).dot(q_aug)
        f_aug = 0.5 * (f_aug + f_aug.T)
        _, v_aug = scipy.linalg.eigh(f_aug)
        u_new = q_aug.dot(v_aug[:, :m_s])
        u_al, _ = procrustes_align(phi_eval[:, :m_s], u_new)
        u_in = q_phi.dot(q_phi.T.dot(u_al))
        v_del = w_del.T.dot(u_al)
        return _fit_local_a(phi_eval, u_in), (w_del, v_del.T), u_new

    def _update_local_subspace(self, s, idx, f_diis, w_idx, phi_eval, u_cur=None):
        '''Update local ``(A_R, Delta_R)`` via closed-form subspace diagonalization.'''
        m_s = self.nelec_channels[s]
        if u_cur is None:
            a_old = self.a_mats[s][idx]
            d_old = _assemble_delta(self.delta_entries[s][idx], self.correction_rank)
            y_old = phi_eval.dot(a_old) if d_old is None else (phi_eval.dot(a_old) + d_old)
            try:
                u_ref, _ = qf_with_r(y_old)
            except ValueError:
                u_ref = qf_with_r(phi_eval[:, :m_s])[0]
        else:
            u_ref = u_cur

        a_new, d_new, _ = self._solve_local_subspace(
            s, idx, f_diis, w_idx, phi_eval, u_ref)
        self.a_mats[s][idx] = a_new
        if d_new is not None:
            self.delta_entries[s][idx] = d_new

    def backward_and_step(self, indices, e_vec, fock_list, cache):
        '''Update ``theta`` via VJP + optimizer and closed-form local ``(A_R, Delta_R)``.

        Returns ``(mean_de, grad_rms)`` diagnostics for stopping criteria.
        '''
        nb = len(indices)
        w_sub = self.weights[indices]
        w_batch = w_sub / w_sub.sum()

        f_orth_by_spin = {s: [] for s in self.active_spins}
        for pos, idx in enumerate(indices):
            x_orth = self.metrics[idx][0]
            fock = fock_list[pos]
            for s in self.active_spins:
                f_ao = fock if self.nspin == 1 else fock[s]
                f_ao = 0.5 * (f_ao + f_ao.T)
                f_orth = x_orth.T.dot(f_ao).dot(x_orth)
                f_orth = 0.5 * (f_orth + f_orth.T)
                f_orth_by_spin[s].append(f_orth)

        # Energy change diagnostic: exact on revisited configs, linearized on first visit
        de_list = []
        has_prev = len(self.prev_theta) == len(self.active_spins)
        for pos, idx in enumerate(indices):
            if numpy.isfinite(self.last_energies[idx]):
                de_list.append(abs(float(e_vec[pos]) - float(self.last_energies[idx])))
            elif has_prev:
                de_lin = 0.0
                for s in self.active_spins:
                    u_cur = cache[pos][s][1]
                    phi_p = self.providers[s].eval(self.z[idx], self.prev_theta[s])
                    d_mat = _assemble_delta(self.delta_entries[s][idx], self.correction_rank)
                    y_p = phi_p.dot(self.a_mats[s][idx])
                    if d_mat is not None:
                        y_p = y_p + d_mat
                    try:
                        u_p, _ = qf_with_r(y_p)
                    except ValueError:
                        u_p = u_cur
                    f_o = f_orth_by_spin[s][pos]
                    dp = u_cur.dot(u_cur.T) - u_p.dot(u_p.T)
                    de_lin += self.occ_val * float(numpy.sum(f_o * dp))
                de_list.append(abs(de_lin))
            self.last_energies[idx] = float(e_vec[pos])

        surr_idx = self._select_replay_indices(indices)
        n_surr = len(surr_idx)
        w_surr = numpy.full(n_surr, 1.0 / n_surr, dtype=float)
        z_surr = self.z[surr_idx]

        grad_sq = 0.0
        grad_cnt = 0

        for s in self.active_spins:
            m_s = self.nelec_channels[s]
            provider = self.providers[s]
            theta_cur = self.theta[s]
            f_orth_b = f_orth_by_spin[s]
            _, g_theta = self._batch_surrogate_val_and_grad(
                indices, f_orth_b, theta_cur, s, w_batch, need_grad=True)

            for pos, idx in enumerate(indices):
                phi, u, _ = cache[pos][s]
                f_orth = f_orth_b[pos]
                fu = f_orth.dot(u)
                if self.correction_rank == 0:
                    q_phi = _orthonormal_columns(phi)
                    if q_phi.shape[1] < m_s:
                        q_phi = qf_with_r(phi[:, :m_s])[0]
                    g_sub = q_phi.T.dot(fu) - (q_phi.T.dot(u)).dot(u.T.dot(fu))
                    grad_sq += float(numpy.sum(g_sub * g_sub))
                elif _is_factored_delta(self.correction_rank, m_s):
                    q_phi = _orthonormal_columns(phi)
                    b_cur = self.delta_entries[s][idx][0]
                    b_perp = b_cur - q_phi.dot(q_phi.T.dot(b_cur))
                    q_b = _orthonormal_columns(b_perp)
                    q_aug = numpy.hstack([q_phi, q_b]) if q_b.shape[1] > 0 else q_phi
                    g_sub = q_aug.T.dot(fu) - (q_aug.T.dot(u)).dot(u.T.dot(fu))
                    grad_sq += float(numpy.sum(g_sub * g_sub))
                else:
                    g_proj = fu - u.dot(u.T.dot(fu))
                    grad_sq += float(numpy.sum(g_proj * g_proj))
                grad_cnt += 1

            if self.correction_rank is not None:
                grad_sq += float(numpy.dot(g_theta, g_theta)) / max(provider.n_params, 1) * nb

            # Cache Fock/fixed-frame DIIS and update local (A_R, Delta_R) on current batch
            for pos, idx in enumerate(indices):
                phi_cur, u_cur, _ = cache[pos][s]
                f_orth = f_orth_b[pos]
                p_orth = u_cur.dot(u_cur.T)
                comm_full = self.occ_val * (f_orth.dot(p_orth) - p_orth.dot(f_orth))
                f_diis = self.diis_buffers[s][idx].update(f_orth, comm_full)
                self.cached_f_orth[s][idx] = f_orth
                self.cached_f_diis[s][idx] = f_diis
                self._update_local_subspace(
                    s, idx, f_diis, w_batch[pos], phi_cur, u_cur=u_cur)

            # Cached-Fock replay steps on shared parameters (0 extra Focks)
            opt = self.opt_state[s]
            f_surr = [self.cached_f_orth[s][j] for j in surr_idx]
            a_surr = numpy.array([self.a_mats[s][j] for j in surr_idx])
            theta_k = theta_cur.copy()
            val_k = None
            g_k = None
            n_inner = 5
            for inner_it in range(n_inner):
                if val_k is None:
                    val_k, g_k = self._batch_surrogate_val_and_grad(
                        surr_idx, f_surr, theta_k, s, w_surr, need_grad=True)
                    if self.optimizer_name == 'lbfgs' and opt.prev_theta is not None:
                        try:
                            _, g_prev = self._batch_surrogate_val_and_grad(
                                surr_idx, f_surr, opt.prev_theta, s, w_surr, need_grad=True)
                            opt.record(theta_k - opt.prev_theta, g_k - g_prev)
                        except ValueError:
                            pass

                gn = float(numpy.linalg.norm(g_k))
                if gn <= 1e-14:
                    break

                if self.optimizer_name == 'lbfgs':
                    d_theta = opt.direction(g_k)
                else:
                    d_theta = opt.step(g_k)

                slope = float(numpy.dot(g_k, d_theta))
                if slope >= 0.0:
                    d_theta = -0.04 * g_k / max(gn, 1.0)
                    slope = float(numpy.dot(g_k, d_theta))

                phi_cur_s = provider.eval_batch(z_surr, theta_k)
                phi_cand_s = provider.eval_batch(z_surr, theta_k + d_theta)
                diff_y = numpy.matmul(phi_cand_s - phi_cur_s, a_surr)
                rms_rot = float(numpy.sqrt(numpy.mean(
                    numpy.sum(diff_y * diff_y, axis=(-2, -1)) / m_s)))
                max_rot = 0.05
                if rms_rot > max_rot:
                    scale_rot = max_rot / rms_rot
                    d_theta = d_theta * scale_rot
                    slope *= scale_rot

                alpha = 1.0
                accepted_step = numpy.zeros_like(d_theta)
                for _ls in range(5):
                    try:
                        cand_val, _ = self._batch_surrogate_val_and_grad(
                            surr_idx, f_surr, theta_k + alpha * d_theta, s, w_surr,
                            need_grad=False)
                    except ValueError:
                        alpha *= 0.5
                        continue
                    if cand_val <= val_k + 1e-4 * alpha * slope:
                        accepted_step = alpha * d_theta
                        break
                    alpha *= 0.5
                else:
                    if self.optimizer_name == 'lbfgs':
                        opt.s_hist.clear()
                        opt.y_hist.clear()
                    d_sd = -0.02 * g_k / max(gn, 1.0)
                    try:
                        cand_val, _ = self._batch_surrogate_val_and_grad(
                            surr_idx, f_surr, theta_k + d_sd, s, w_surr, need_grad=False)
                        if cand_val < val_k:
                            accepted_step = d_sd
                    except ValueError:
                        pass

                if not numpy.any(accepted_step):
                    break

                theta_next = theta_k + accepted_step
                if inner_it + 1 < n_inner:
                    try:
                        val_next, g_next = self._batch_surrogate_val_and_grad(
                            surr_idx, f_surr, theta_next, s, w_surr, need_grad=True)
                    except ValueError:
                        if self.optimizer_name == 'lbfgs':
                            opt.prev_theta = theta_k.copy()
                        theta_k = theta_next
                        break
                    if self.optimizer_name == 'lbfgs':
                        opt.record(accepted_step, g_next - g_k)
                        opt.prev_theta = theta_k.copy()
                    theta_k, val_k, g_k = theta_next, val_next, g_next
                else:
                    if self.optimizer_name == 'lbfgs':
                        opt.prev_theta = theta_k.copy()
                    theta_k = theta_next

            self.prev_theta[s] = theta_cur.copy()
            self.theta[s] = theta_k

            # Refresh local (A_R, Delta_R) on all visited geometries at updated theta
            all_vis_idx = [int(j) for j in self.order if self.cached_f_diis[s][int(j)] is not None]
            w_vis = 1.0 / max(len(all_vis_idx), 1)
            phi_new_vis = provider.eval_batch(self.z[all_vis_idx], self.theta[s])
            for pos, idx in enumerate(all_vis_idx):
                self._update_local_subspace(
                    s, idx, self.cached_f_diis[s][idx], w_vis, phi_new_vis[pos])

        for idx in indices:
            self.visits[idx] += 1
        self.nfock_shared_total += nb

        mean_de = float(numpy.mean(de_list)) if de_list else numpy.inf
        grad_rms = float(numpy.sqrt(grad_sq / max(grad_cnt, 1)))
        self.history.append({
            'step': len(self.history),
            'batch': list(indices),
            'mean_energy': float(numpy.mean(e_vec)),
            'mean_de': mean_de,
            'grad_rms': grad_rms,
            'nfock_shared': self.nfock_shared_total,
        })
        return mean_de, grad_rms

    def _interpolate_a_indices(self, target_indices):
        '''Transfer local ``A`` coefficients to unvisited geometries via RBF.'''
        vis = numpy.where(self.visits > 0)[0]
        unvis = numpy.asarray(list(target_indices), dtype=int)
        if vis.size == 0 or unvis.size == 0:
            return
        z_vis = self.z[vis]
        z_unvis = self.z[unvis]
        d2 = ((z_unvis[:, None, :] - z_vis[None, :, :]) ** 2).sum(axis=-1)
        if vis.size >= 2:
            d2_vv = ((z_vis[:, None, :] - z_vis[None, :, :]) ** 2).sum(axis=-1)
            triu = d2_vv[numpy.triu_indices(vis.size, 1)]
            scale = max(float(numpy.median(triu)), 1e-4)
        else:
            scale = 1.0
        wts = numpy.exp(-0.5 * d2 / scale)
        wts = wts / numpy.maximum(wts.sum(axis=1, keepdims=True), 1e-16)
        for s in self.active_spins:
            a_vis = numpy.array([self.a_mats[s][i] for i in vis])
            a_pred = numpy.tensordot(wts, a_vis, axes=1)
            for pos, idx in enumerate(unvis):
                self.a_mats[s][idx] = a_pred[pos]
        if self.correction_rank != 0:
            for s in self.active_spins:
                m_s = self.nelec_channels[s]
                factored = _is_factored_delta(self.correction_rank, m_s)
                for idx in unvis:
                    if self.visits[idx] == 0:
                        if not factored:
                            self.delta_entries[s][idx] = numpy.zeros(
                                (self.n_orth, m_s), dtype=float)
                        else:
                            b_cur, g_cur = self.delta_entries[s][idx]
                            self.delta_entries[s][idx] = (b_cur, numpy.zeros_like(g_cur))

    def build_all_densities(self):
        '''Assemble final AO density matrices and occupied orbitals for all configs.'''
        unvis = numpy.where(self.visits == 0)[0]
        if unvis.size > 0:
            self._interpolate_a_indices(unvis)
        dms = []
        c_occs = []
        for idx in range(self.nconf):
            x_orth = self.metrics[idx][0]
            nao = self.mfs[idx].mol.nao
            dm_spins = []
            c_spins = []
            for s in range(self.nspin):
                m_s = self.nelec_channels[s]
                if m_s == 0:
                    dm_spins.append(numpy.zeros((nao, nao), dtype=float))
                    c_spins.append(numpy.zeros((nao, 0), dtype=float))
                    continue
                phi = self.providers[s].eval(self.z[idx], self.theta[s])
                a_mat = self.a_mats[s][idx]
                d_mat = _assemble_delta(self.delta_entries[s][idx], self.correction_rank)
                y = phi.dot(a_mat) if d_mat is None else (phi.dot(a_mat) + d_mat)
                try:
                    u, _ = qf_with_r(y)
                except ValueError:
                    u, _ = qf_with_r(phi.dot(a_mat))
                c = x_orth.dot(u)
                c_spins.append(c)
                dm_spins.append(self.occ_val * c.dot(c.T))
            dms.append(dm_spins[0] if self.nspin == 1 else numpy.array(dm_spins))
            c_occs.append(c_spins[0] if self.nspin == 1 else c_spins)
        return dms, c_occs


def semicanonical_mo_and_err_est(mf, s1e, fock, c_occ, lindep=1e-10):
    '''Construct full semicanonical ``(mo_coeff, mo_energy, mo_occ)`` and
    second-order perturbation energy error estimate from ``c_occ`` and ``fock``.
    '''
    s1e = numpy.asarray(s1e, dtype=float)
    e_s, v_s = scipy.linalg.eigh(0.5 * (s1e + s1e.T))
    keep = e_s > lindep * max(float(e_s.max()), 1.0)
    x_orth = (v_s / numpy.sqrt(e_s)).dot(v_s.T) if keep.all() else v_s[:, keep] / numpy.sqrt(e_s[keep])
    n_orth = x_orth.shape[1]
    s_x = s1e.dot(x_orth)

    nspin = 2 if mf.istype('UHF') else 1
    occ_val = 1.0 if nspin == 2 else 2.0

    def _one_spin(f_ao, c_occ_s):
        f_ao = 0.5 * (f_ao + f_ao.T)
        f_orth = x_orth.T.dot(f_ao).dot(x_orth)
        f_orth = 0.5 * (f_orth + f_orth.T)
        m_s = c_occ_s.shape[1]
        if m_s == 0:
            e_vir, v_vir = scipy.linalg.eigh(f_orth)
            mo_c = x_orth.dot(v_vir)
            mo_o = numpy.zeros(n_orth, dtype=float)
            return mo_c, e_vir, mo_o, 0.0

        u_occ = s_x.T.dot(c_occ_s)
        u_occ, _ = qf_with_r(u_occ)
        c_occ_clean = x_orth.dot(u_occ)

        f_occ = u_occ.T.dot(f_orth).dot(u_occ)
        f_occ = 0.5 * (f_occ + f_occ.T)
        e_occ, v_occ = scipy.linalg.eigh(f_occ)
        c_occ_semi = c_occ_clean.dot(v_occ)

        n_vir = n_orth - m_s
        if n_vir > 0:
            proj_vir = numpy.eye(n_orth) - u_occ.dot(u_occ.T)
            _, v_p = scipy.linalg.eigh(0.5 * (proj_vir + proj_vir.T))
            u_vir = v_p[:, m_s:]
            f_vir = u_vir.T.dot(f_orth).dot(u_vir)
            f_vir = 0.5 * (f_vir + f_vir.T)
            e_vir, v_vir = scipy.linalg.eigh(f_vir)
            c_vir_semi = x_orth.dot(u_vir.dot(v_vir))
            mo_c = numpy.hstack([c_occ_semi, c_vir_semi])
            mo_e = numpy.concatenate([e_occ, e_vir])
            mo_o = numpy.zeros(n_orth, dtype=float)
            mo_o[:m_s] = occ_val
            f_ia = c_occ_semi.T.dot(f_ao).dot(c_vir_semi)
            denom = numpy.maximum(e_vir[None, :] - e_occ[:, None], 0.1)
            err = float(occ_val * numpy.sum((f_ia * f_ia) / denom))
        else:
            mo_c, mo_e = c_occ_semi, e_occ
            mo_o = numpy.full(m_s, occ_val, dtype=float)
            err = 0.0
        return mo_c, mo_e, mo_o, err

    if nspin == 1:
        return _one_spin(numpy.asarray(fock, dtype=float), c_occ)

    res0 = _one_spin(numpy.asarray(fock[0], dtype=float), c_occ[0])
    res1 = _one_spin(numpy.asarray(fock[1], dtype=float), c_occ[1])
    mo_coeff = numpy.array([res0[0], res1[0]])
    mo_energy = numpy.array([res0[1], res1[1]])
    mo_occ = numpy.array([res0[2], res1[2]])
    return mo_coeff, mo_energy, mo_occ, res0[3] + res1[3]


def estimate_pt2_err(mf, fock, mo_coeff, mo_energy, mo_occ):
    '''Second-order energy error estimate from converged or approximate MOs.'''
    if mo_coeff is None or mo_energy is None or mo_occ is None:
        return 0.0
    nspin = 2 if mf.istype('UHF') else 1

    def _one(f_ao, c, e, o):
        occ_mask = o > 1e-8
        vir_mask = ~occ_mask
        if not numpy.any(occ_mask) or not numpy.any(vir_mask):
            return 0.0
        c_o = c[:, occ_mask]
        c_v = c[:, vir_mask]
        f_ia = c_o.T.dot(0.5 * (f_ao + f_ao.T)).dot(c_v)
        denom = numpy.maximum(e[vir_mask][None, :] - e[occ_mask][:, None], 0.1)
        occ_w = o[occ_mask][:, None]
        return float(numpy.sum(occ_w * (f_ia * f_ia) / denom))

    if nspin == 1:
        return _one(numpy.asarray(fock, dtype=float), mo_coeff, mo_energy, mo_occ)
    return (
        _one(numpy.asarray(fock[0], dtype=float), mo_coeff[0], mo_energy[0], mo_occ[0])
        + _one(numpy.asarray(fock[1], dtype=float), mo_coeff[1], mo_energy[1], mo_occ[1])
    )


def supports_transferable(mf):
    '''Return True if ``mf`` is closed-shell RHF/RKS or open-shell UHF/UKS.'''
    if mf.istype('GHF') or mf.istype('DHF') or mf.istype('ROHF') or 'cell' in mf.__dict__:
        return False
    if mf.istype('UHF'):
        return mf.mol.nelectron >= 1
    return mf.mol.spin == 0 and mf.mol.nelectron >= 2 and mf.mol.nelectron % 2 == 0
