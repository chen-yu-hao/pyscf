"""Prototype: RKS XC nuclear gradient contracted with the density first.

Reference (pyscf.grad.rks.get_vxc) builds vmat[x] (3 x nao x nao) with 6 (GGA)
or 15 (MGGA) ngrid*nao^2 GEMMs per block, on top of the 1 (GGA) or 4 (MGGA)
GEMMs inside eval_rho, and only afterwards contracts with D per atom.

Here we need only the per-AO quantity f[x, mu] = sum_nu vmat[x]_mu,nu D_mu,nu:
    P   = phi   D          (also gives rho, grad rho)
    F   = aow   D          aow = sum_k wv_k phi^(k)
    Q_j = d_j phi D        (MGGA only; also gives tau)
With D = C n C^T every product is two thin GEMMs (ngrid x nao x nocc).
"""
import numpy as np
from pyscf import lib
from pyscf.dft import numint

D2 = ((4, 5, 6), (5, 7, 8), (6, 8, 9))  # index of d_x d_j phi in ao[4:10]


def _dm_apply(a, c, cn):
    # a @ D with D = c @ cn.T, as two thin GEMMs; Fortran order like ao[i]
    return lib.dot(cn, lib.dot(c.T, a.T)).T


_cr = numint._contract_rho  # threaded C kernel: sum_i a[g,i] b[g,i]


def get_xc_grad_contracted(ni, mol, grids, xc_code, mo_coeff, mo_occ, max_memory=2000):
    xctype = ni._xc_type(xc_code)
    occ = mo_occ > 0
    c = np.asarray(mo_coeff[:, occ], order='C')
    cn = c * mo_occ[occ]
    nao = mol.nao
    f = np.zeros((3, nao))
    ao_deriv = 1 if xctype == 'LDA' else 2
    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        phi = ao[0]
        P = _dm_apply(phi, c, cn)
        if xctype == 'LDA':
            rho = _cr(P, phi)
            vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[1]
            G = P * (weight * vxc[0])[:, None]
            for x in range(3):
                f[x] += np.einsum('gi,gi->i', ao[1 + x], G)
            continue

        nvar = 4 if xctype == 'GGA' else 5
        rho = np.empty((nvar, phi.shape[0]))
        rho[0] = _cr(P, phi)
        for j in range(3):
            rho[1 + j] = 2 * _cr(P, ao[1 + j])
        if xctype == 'MGGA':
            Q = [_dm_apply(ao[1 + j], c, cn) for j in range(3)]
            rho[4] = .5 * sum(_cr(Q[j], ao[1 + j]) for j in range(3))
        vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[1]
        wv = weight * vxc
        wv[0] *= .5
        if xctype == 'MGGA':
            wv[4] *= .5
        aow = numint._scale_ao(ao[:4], wv[:4])
        F = _dm_apply(aow, c, cn)
        G = F + P * wv[0][:, None]
        R = [P * wv[1 + j][:, None] for j in range(3)]
        if xctype == 'MGGA':
            for j in range(3):
                R[j] += Q[j] * wv[4][:, None]
        for x in range(3):
            fx = np.einsum('gi,gi->i', ao[1 + x], G)
            for j in range(3):
                fx += np.einsum('gi,gi->i', ao[D2[x][j]], R[j])
            f[x] += fx
    # de[A] += 2 * sum_{mu in A} (-vmat) . D  ==> -2 f
    aoslices = mol.aoslice_by_atom()
    return np.array([-2 * f[:, p0:p1].sum(axis=1) for p0, p1 in aoslices[:, 2:]])


def reference_xc_grad(ni, mol, grids, xc_code, dm, max_memory=2000):
    from pyscf.grad import rks as rks_grad
    vxc = rks_grad.get_vxc(ni, mol, grids, xc_code, dm, max_memory=max_memory)[1]
    aoslices = mol.aoslice_by_atom()
    return np.array([2 * np.einsum('xij,ij->x', vxc[:, p0:p1], dm[p0:p1])
                     for p0, p1 in aoslices[:, 2:]])


# ---- fused C contraction (grad_xc_kernel.c) ---------------------------------
import ctypes, os, subprocess
_here = os.path.dirname(os.path.abspath(__file__))
_so = os.path.join(_here, 'libgradxc.so')
if not os.path.exists(_so):
    subprocess.run(['gcc', '-O3', '-fopenmp', '-fPIC', '-shared', '-o', _so,
                    os.path.join(_here, 'grad_xc_kernel.c')], check=True)
_lib = ctypes.CDLL(_so)


def _thin(a, c, cn, out):
    # out (nao x ngrid, C order) = (a @ D).T  with D = c cn^T
    lib.dot(cn, lib.dot(c.T, a.T), c=out)
    return out


def get_xc_grad_fused(ni, mol, grids, xc_code, mo_coeff, mo_occ, max_memory=2000):
    xctype = ni._xc_type(xc_code)
    assert xctype in ('GGA', 'MGGA')
    mgga = xctype == 'MGGA'
    occ = mo_occ > 0
    c = np.asarray(mo_coeff[:, occ], order='C')
    cn = c * mo_occ[occ]
    nao = mol.nao
    f = np.zeros((3, nao))
    ao_loc = mol.ao_loc_nr()
    ao_shell = np.repeat(np.arange(mol.nbas, dtype=np.int32), np.diff(ao_loc)).astype(np.int32)
    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, 2, max_memory):
        ng = weight.size
        P = _thin(ao[0], c, cn, np.empty((nao, ng)))
        rho = np.empty((5 if mgga else 4, ng))
        rho[0] = _cr(P.T, ao[0])
        for j in range(3):
            rho[1 + j] = 2 * _cr(P.T, ao[1 + j])
        Q = None
        if mgga:
            Q = np.empty((3, nao, ng))
            for j in range(3):
                _thin(ao[1 + j], c, cn, Q[j])
            rho[4] = .5 * sum(_cr(Q[j].T, ao[1 + j]) for j in range(3))
        vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[1]
        wv = weight * vxc
        wv[0] *= .5
        if mgga:
            wv[4] *= .5
        wv = np.ascontiguousarray(wv)
        aow = numint._scale_ao(ao[:4], wv[:4])
        F = _thin(aow, c, cn, np.empty((nao, ng)))
        if mask is not None:
            mask = np.ascontiguousarray(mask, dtype=np.uint8)
        _lib.grad_xc_contract(
            f.ctypes.data_as(ctypes.c_void_p), ao.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_ssize_t(ao.strides[0] // 8),
            P.ctypes.data_as(ctypes.c_void_p), F.ctypes.data_as(ctypes.c_void_p),
            Q.ctypes.data_as(ctypes.c_void_p) if mgga else None,
            wv.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(wv.shape[0]),
            ctypes.c_int(nao), ctypes.c_int(ng),
            mask.ctypes.data_as(ctypes.c_void_p) if mask is not None else None,
            ao_shell.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas))
    aoslices = mol.aoslice_by_atom()
    return np.array([-2 * f[:, p0:p1].sum(axis=1) for p0, p1 in aoslices[:, 2:]])
