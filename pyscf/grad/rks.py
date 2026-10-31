#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''Non-relativistic RKS analytical nuclear gradients'''


import numpy
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad import rhf as rhf_grad
from pyscf.dft import numint, radi, gen_grid, xc_deriv
from pyscf.dft.numint import libdft
from pyscf.dft.gen_grid import BLKSIZE
from pyscf import __config__
import ctypes

# Contract the XC term with the density on the grid inside grad_elec instead
# of building the 3 x nao x nao matrix of get_vxc (see get_xc_force).
XC_FORCE = getattr(__config__, 'grad_rks_xc_force', True)


def get_veff(ks_grad, mol=None, dm=None):
    '''
    First order derivative of DFT effective potential matrix (wrt electron coordinates)

    Args:
        ks_grad : grad.uhf.Gradients or grad.uks.Gradients object
    '''
    if mol is None: mol = ks_grad.mol
    if dm is None: dm = ks_grad.base.make_rdm1()
    t0 = (logger.process_clock(), logger.perf_counter())

    mf = ks_grad.base
    ni = mf._numint
    grids, nlcgrids = _initialize_grids(ks_grad)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory*.9-mem_now)
    if ks_grad.grid_response:
        exc, vxc = get_vxc_full_response(ni, mol, grids, mf.xc, dm,
                                         max_memory=max_memory,
                                         verbose=ks_grad.verbose)
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = get_nlc_vxc_full_response(
                ni, mol, nlcgrids, xc, dm,
                max_memory=max_memory, verbose=ks_grad.verbose)
            exc += enlc
            vxc += vnlc
        logger.debug1(ks_grad, 'sum(grids response) %s', exc.sum(axis=0))
    else:
        xc_force = getattr(ks_grad, '_xc_force', None)
        if xc_force is not None and _xc_force_supported(ni, dm):
            # Called from grad_elec: the XC term is returned to grad_elec
            # through extra_force as per-atom forces.
            exc = None
            xc_force['de'] = get_xc_force(ni, mol, grids, mf.xc, dm,
                                          max_memory=max_memory,
                                          verbose=ks_grad.verbose)
            vxc = numpy.zeros((3,) + dm.shape[-2:])
        else:
            exc, vxc = get_vxc(ni, mol, grids, mf.xc, dm,
                               max_memory=max_memory, verbose=ks_grad.verbose)
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = get_nlc_vxc(
                ni, mol, nlcgrids, xc, dm,
                max_memory=max_memory, verbose=ks_grad.verbose)
            vxc += vnlc
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    if not ni.libxc.is_hybrid_xc(mf.xc):
        vj = ks_grad.get_j(mol, dm)
        vxc += vj
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
        vj, vk = ks_grad.get_jk(mol, dm)
        vk *= hyb
        if omega != 0:
            vk += ks_grad.get_k(mol, dm, omega=omega) * (alpha - hyb)
        vxc += vj - vk * .5
        """
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks_grad.xc, spin=mol.spin)
        if omega == 0:
            vj, vk = ks_grad.get_jk(mol, dm)
            vk *= hyb
        elif alpha == 0: # LR=0, only SR exchange
            vj = ks_grad.get_j(mol, dm)
            vk = ks_grad.get_k(mol, dm, omega=-omega)
            vk *= hyb
        elif hyb == 0: # SR=0, only LR exchange
            vj = ks_grad.get_j(mol, dm)
            vk = ks_grad.get_k(mol, dm, omega=omega)
            vk *= alpha
        else: # SR and LR exchange with different ratios
            vj, vk = ks_grad.get_jk(mol, dm)
            vk *= hyb
            vklr = ks_grad.get_k(mol, dm, omega=omega)
            vklr *= (alpha - hyb)
            vk += vklr
        """
    return lib.tag_array(vxc, exc1_grid=exc)

def _initialize_grids(ks_grad):
    mf = ks_grad.base
    if ks_grad.grids is not None:
        grids = ks_grad.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)

    nlcgrids = None
    if mf.do_nlc():
        if ks_grad.nlcgrids is not None:
            nlcgrids = ks_grad.nlcgrids
        else:
            nlcgrids = mf.nlcgrids
        if nlcgrids.coords is None:
            nlcgrids.build(with_non0tab=True)
    return grids, nlcgrids


def get_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
            max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_loc = mol.ao_loc_nr()

    vmat = numpy.zeros((nset,3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao[0], mask, xctype)
                vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[1]
                wv = weight * vxc[0]
                aow = numint._scale_ao(ao[0], wv)
                _d1_dot_(vmat[idm], mol, ao[1:4], aow, mask, ao_loc, True)

    elif xctype == 'GGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao[:4], mask, xctype)
                vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[1]
                wv = weight * vxc
                wv[0] *= .5
                _gga_grad_sum_(vmat[idm], mol, ao, wv, mask, ao_loc)

    elif xctype == 'MGGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao[:10], mask, xctype)
                vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[1]
                wv = weight * vxc
                wv[0] *= .5
                wv[4] *= .5  # for the factor 1/2 in tau
                _gga_grad_sum_(vmat[idm], mol, ao, wv, mask, ao_loc)
                _tau_grad_dot_(vmat[idm], mol, ao, wv[4], mask, ao_loc, True)

    exc = None
    if nset == 1:
        vmat = vmat[0]
    # - sign because nabla_X = -nabla_x
    return exc, -vmat

def _xc_force_supported(ni, dm, nspin=1):
    '''Whether get_xc_force reproduces get_vxc for this integrator and density.

    The force path evaluates the density itself, so integrators that change
    how densities are evaluated keep the matrix path. Custom AO evaluators,
    block loops and XC functionals are used through ni and remain supported.
    '''
    if not isinstance(ni, numint.NumInt) or numint.MGGA_DENSITY_LAPL:
        return False
    cls = type(ni)
    for name in ('_gen_rho_evaluator', 'eval_rho', 'eval_rho1', 'eval_rho2'):
        if name in ni.__dict__ or getattr(cls, name) is not getattr(numint.NumInt, name):
            return False
    shape = dm.shape if isinstance(dm, numpy.ndarray) else ()
    if nspin == 2:
        ok = len(shape) == 3 and shape[0] == 2
    else:
        ok = len(shape) == 2
    return ok and dm.dtype == numpy.double and shape[-1] == shape[-2]

def _xc_force_density(dm, nao):
    '''Density factors for P = ao.dot(D):  D = c.dot(cn.T) through occupied
    orbitals when they are cheaper than the AO density matrix.'''
    mo_coeff = getattr(dm, 'mo_coeff', None)
    mo_occ = getattr(dm, 'mo_occ', None)
    if mo_coeff is not None and mo_occ is not None:
        mo_occ = numpy.asarray(mo_occ)
        occ = mo_occ != 0
        nocc = numpy.count_nonzero(occ)
        if 0 < nocc and 2 * nocc <= nao and numpy.asarray(mo_coeff).dtype == numpy.double:
            c = numpy.asarray(mo_coeff[:,occ], order='C')
            return c, numpy.asarray(c * mo_occ[occ], order='C')
    return None, numpy.asarray(dm, order='C')

def _xc_force(ni, mol, grids, xc_code, dms, spin, max_memory=2000, verbose=None):
    '''Per-AO XC force f[x,i] = sum_j V^x_ij D_ij summed over spins, where
    V^x is the matrix returned by get_vxc (before the sign flip).'''
    xctype = ni._xc_type(xc_code)
    nao = mol.nao
    f = numpy.zeros((3, nao))
    if xctype not in ('LDA', 'GGA', 'MGGA'):
        return f
    log = logger.new_logger(mol, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    lda = xctype == 'LDA'
    mgga = xctype == 'MGGA'
    nvar = 1 if lda else (5 if mgga else 4)
    ao_deriv = 1 if lda else 2
    comp = 4 if lda else 10
    factors = [_xc_force_density(dm, nao) for dm in dms]
    nspin = len(factors)

    # Working arrays per block: AO (comp), aow, and per spin P, F, Q_xyz.
    nextra = (0 if lda else 1) + nspin * (1 if lda else (5 if mgga else 2))
    ngrids = grids.weights.size
    blksize = int(max_memory*1e6/((comp+1+nextra)*nao*8*BLKSIZE))
    blksize = max(4, min(blksize, ngrids//BLKSIZE+1, 1200)) * BLKSIZE
    nmo = max([nao] + [c.shape[1] for c, _ in factors if c is not None])
    xbuf = numpy.empty(nmo * blksize)
    pbuf = numpy.empty((nspin, nao * blksize))
    fbuf = numpy.empty((nspin, nao * blksize)) if not lda else None
    qbuf = numpy.empty((nspin, 3, nao * blksize)) if mgga else None
    aowbuf = numpy.empty(nao * blksize) if not lda else None

    ao_loc = mol.ao_loc_nr()
    shls_slice = (0, mol.nbas)
    ao_shell = numpy.asarray(numpy.repeat(numpy.arange(mol.nbas), numpy.diff(ao_loc)),
                             dtype=numpy.int32)
    xctype_id = 0 if lda else (2 if mgga else 1)

    def apply_dm(a, mask, c, d, out):
        '''(a.dot(D)).T as a C-ordered (nao, ngrids) array, D = c.dot(d.T)'''
        ng = a.shape[0]
        if c is None:
            return _ao_dot_dm_t(mol, a, d, mask, shls_slice, ao_loc, out)
        x = _ao_dot_dm_t(mol, a, c, mask, shls_slice, ao_loc, xbuf)
        return lib.dot(d, x, c=numpy.ndarray((nao, ng), buffer=out))

    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory, blksize=blksize):
        ng = weight.size
        if not all(ao[i].flags.f_contiguous for i in range(comp)):
            ao = numpy.array(ao.transpose(0,2,1), order='C').transpose(0,2,1)
        if mask is not None:
            mask = numpy.asarray(mask, dtype=numpy.uint8, order='C')
        P = []
        Q = []
        rho = numpy.empty((nspin, nvar, ng))
        for s, (c, d) in enumerate(factors):
            p = apply_dm(ao[0], mask, c, d, pbuf[s])
            rho[s,0] = numint._contract_rho(p.T, ao[0])
            if not lda:
                for j in range(1, 4):
                    rho[s,j] = numint._contract_rho(p.T, ao[j]) * 2
            if mgga:
                q = numpy.ndarray((3, nao, ng), buffer=qbuf[s])
                rho[s,4] = 0
                for j in range(3):
                    apply_dm(ao[j+1], mask, c, d, q[j])
                    rho[s,4] += numint._contract_rho(q[j].T, ao[j+1])
                rho[s,4] *= .5
                Q.append(q)
            P.append(p)
        if nspin == 1:
            vxc = ni.eval_xc_eff(xc_code, rho[0,0] if lda else rho[0], 1,
                                 xctype=xctype)[1]
            wv = (weight * vxc).reshape(1, nvar, ng)
        else:
            vxc = ni.eval_xc_eff(xc_code, rho if not lda else rho[:,0], 1,
                                 xctype=xctype, spin=1)[1]
            wv = (weight * vxc).reshape(2, nvar, ng)
        if not lda:
            wv[:,0] *= .5  # same convention as _gga_grad_sum_
        if mgga:
            wv[:,4] *= .5  # factor 1/2 in tau
        for s, (c, d) in enumerate(factors):
            ws = numpy.ascontiguousarray(wv[s])
            F = None
            if not lda:
                aow = numint._scale_ao(ao[:4], ws[:4], out=aowbuf)
                F = apply_dm(aow, mask, c, d, fbuf[s])
            libdft.VXCxc_grad_contract(
                f.ctypes.data_as(ctypes.c_void_p),
                ao.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_ssize_t(ao.strides[0] // ao.itemsize),
                P[s].ctypes.data_as(ctypes.c_void_p),
                F.ctypes.data_as(ctypes.c_void_p) if F is not None else None,
                Q[s].ctypes.data_as(ctypes.c_void_p) if mgga else None,
                ws.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(xctype_id), ctypes.c_int(nao), ctypes.c_int(ng),
                mask.ctypes.data_as(ctypes.c_void_p) if mask is not None else None,
                ao_shell.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas))
    log.timer_debug1('xc force', *t0)
    return f

def _ao_dot_dm_t(mol, ao, dm, mask, shls_slice, ao_loc, out):
    '''(ao.dot(dm)).T as a C-ordered (dm.shape[1], ngrids) array in out'''
    ngrids, nao = ao.shape
    n = dm.shape[1]
    res = numpy.ndarray((n, ngrids), buffer=out)
    if mask is not None and nao >= numint.SWITCH_SIZE and ao.flags.f_contiguous:
        # Shell-screened product; writes the Fortran-ordered (ngrids, n) array
        numint._dot_ao_dm(mol, ao, dm, mask, shls_slice, ao_loc, out=res)
    else:
        lib.dot(dm.T, ao.T, c=res)
    return res

def get_xc_force(ni, mol, grids, xc_code, dm, max_memory=2000, verbose=None):
    '''XC contribution to restricted Kohn-Sham nuclear gradients.

    Equivalent to contracting get_vxc with the density for each atom,
    ``2 * einsum('xij,ij->x', get_vxc(...)[1][:,p0:p1], dm[p0:p1])``, but the
    density is contracted on the grid first. Only P = ao.dot(dm) and, for
    (meta-)GGA, aow.dot(dm) (and the three ao_xyz.dot(dm) for meta-GGA) are
    formed, as thin products through the occupied orbitals when dm carries
    mo_coeff/mo_occ tags. No 3 x nao x nao matrix is built. Without grid
    response; NLC terms are not included.

    Returns:
        (natm, 3) array.
    '''
    f = _xc_force(ni, mol, grids, xc_code, [dm], 0, max_memory, verbose)
    aoslices = mol.aoslice_by_atom()
    return numpy.array([-2 * f[:,p0:p1].sum(axis=1) for p0, p1 in aoslices[:,2:]])

def get_nlc_vxc(ni, mol, grids, xc_code, dm, relativity=0, hermi=1,
                max_memory=2000, verbose=None):
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, hermi, False, grids)
    assert nset == 1
    ao_loc = mol.ao_loc_nr()

    vmat = numpy.zeros((3,nao,nao))
    nlc_coefs = ni.nlc_coeff(xc_code)
    if len(nlc_coefs) != 1:
        raise NotImplementedError('Additive NLC')
    nlc_pars, fac = nlc_coefs[0]
    ao_deriv = 2
    vvrho = []
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        vvrho.append(make_rho(0, ao[:4], mask, 'GGA'))
    rho = numpy.hstack(vvrho)

    vxc = numint._vv10nlc(rho, grids.coords, rho, grids.weights,
                          grids.coords, nlc_pars)[1]
    vv_vxc = xc_deriv.transform_vxc(rho, vxc, 'GGA', spin=0)

    p1 = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        p0, p1 = p1, p1 + weight.size
        wv = vv_vxc[:,p0:p1] * weight
        wv[0] *= .5  # *.5 because vmat + vmat.T at the end
        _gga_grad_sum_(vmat, mol, ao, wv, mask, ao_loc)

    exc = None
    # - sign because nabla_X = -nabla_x
    return exc, -vmat

def _make_dR_dao_w(ao, wv):
    #:aow = numpy.einsum('npi,p->npi', ao[1:4], wv[0])
    aow = [
        numint._scale_ao(ao[1], wv[0]),  # dX nabla_x
        numint._scale_ao(ao[2], wv[0]),  # dX nabla_y
        numint._scale_ao(ao[3], wv[0]),  # dX nabla_z
    ]
    # XX, XY, XZ = 4, 5, 6
    # YX, YY, YZ = 5, 7, 8
    # ZX, ZY, ZZ = 6, 8, 9
    aow[0] += numint._scale_ao(ao[4], wv[1])  # dX nabla_x
    aow[0] += numint._scale_ao(ao[5], wv[2])  # dX nabla_y
    aow[0] += numint._scale_ao(ao[6], wv[3])  # dX nabla_z
    aow[1] += numint._scale_ao(ao[5], wv[1])  # dY nabla_x
    aow[1] += numint._scale_ao(ao[7], wv[2])  # dY nabla_y
    aow[1] += numint._scale_ao(ao[8], wv[3])  # dY nabla_z
    aow[2] += numint._scale_ao(ao[6], wv[1])  # dZ nabla_x
    aow[2] += numint._scale_ao(ao[8], wv[2])  # dZ nabla_y
    aow[2] += numint._scale_ao(ao[9], wv[3])  # dZ nabla_z
    return aow

def _d1_dot_(vmat, mol, ao1, ao2, mask, ao_loc, dR1_on_bra=True):
    shls_slice = (0, mol.nbas)
    if dR1_on_bra:
        vmat[0] += numint._dot_ao_ao(mol, ao1[0], ao2, mask, shls_slice, ao_loc)
        vmat[1] += numint._dot_ao_ao(mol, ao1[1], ao2, mask, shls_slice, ao_loc)
        vmat[2] += numint._dot_ao_ao(mol, ao1[2], ao2, mask, shls_slice, ao_loc)
    else:
        vmat[0] += numint._dot_ao_ao(mol, ao1, ao2[0], mask, shls_slice, ao_loc)
        vmat[1] += numint._dot_ao_ao(mol, ao1, ao2[1], mask, shls_slice, ao_loc)
        vmat[2] += numint._dot_ao_ao(mol, ao1, ao2[2], mask, shls_slice, ao_loc)

def _gga_grad_sum_(vmat, mol, ao, wv, mask, ao_loc):
    #:aow = numpy.einsum('npi,np->pi', ao[:4], wv[:4])
    aow = numint._scale_ao(ao[:4], wv[:4])
    _d1_dot_(vmat, mol, ao[1:4], aow, mask, ao_loc, True)
    aow = _make_dR_dao_w(ao, wv[:4])
    _d1_dot_(vmat, mol, aow, ao[0], mask, ao_loc, True)
    return vmat

# XX, XY, XZ = 4, 5, 6
# YX, YY, YZ = 5, 7, 8
# ZX, ZY, ZZ = 6, 8, 9
def _tau_grad_dot_(vmat, mol, ao, wv, mask, ao_loc, dR1_on_bra=True):
    '''The tau part of MGGA functional'''
    aow = numint._scale_ao(ao[1], wv)
    _d1_dot_(vmat, mol, [ao[4], ao[5], ao[6]], aow, mask, ao_loc, True)
    aow = numint._scale_ao(ao[2], wv, aow)
    _d1_dot_(vmat, mol, [ao[5], ao[7], ao[8]], aow, mask, ao_loc, True)
    aow = numint._scale_ao(ao[3], wv, aow)
    _d1_dot_(vmat, mol, [ao[6], ao[8], ao[9]], aow, mask, ao_loc, True)

def _vv10nlc_grad(rho, coords, vvrho, vvweight, vvcoords, nlc_pars):
    # VV10 gradient term from Vydrov and Van Voorhis 2010 eq. 25-26
    # https://doi.org/10.1063/1.3521275
    thresh=1e-8

    #constants and parameters
    Pi=numpy.pi
    Pi43=4.*Pi/3.
    Bvv, Cvv = nlc_pars
    Kvv=Bvv*1.5*Pi*((9.*Pi)**(-1./6.))
    Beta=((3./(Bvv*Bvv))**(0.75))/32.

    #output
    exc=numpy.zeros((rho[0,:].size,3))

    if vvcoords.size == 0:
        return exc, Beta

    #outer grid needs threshing
    threshind=rho[0,:]>=thresh
    coords=coords[threshind]
    R=rho[0,:][threshind]
    Gx=rho[1,:][threshind]
    Gy=rho[2,:][threshind]
    Gz=rho[3,:][threshind]
    G=Gx**2.+Gy**2.+Gz**2.

    #inner grid needs threshing
    innerthreshind=vvrho[0,:]>=thresh
    vvcoords=vvcoords[innerthreshind]
    vvweight=vvweight[innerthreshind]
    Rp=vvrho[0,:][innerthreshind]
    RpW=Rp*vvweight
    Gxp=vvrho[1,:][innerthreshind]
    Gyp=vvrho[2,:][innerthreshind]
    Gzp=vvrho[3,:][innerthreshind]
    Gp=Gxp**2.+Gyp**2.+Gzp**2.

    assert vvcoords.size > 0

    #inner grid
    W0p=Gp/(Rp*Rp)
    W0p=Cvv*W0p*W0p
    W0p=(W0p+Pi43*Rp)**0.5
    Kp=Kvv*(Rp**(1./6.))

    #outer grid
    W0tmp=G/(R**2)
    W0tmp=Cvv*W0tmp*W0tmp
    W0=(W0tmp+Pi43*R)**0.5
    K=Kvv*(R**(1./6.))

    vvcoords = numpy.asarray(vvcoords, order='C')
    coords = numpy.asarray(coords, order='C')
    F = numpy.empty(R.shape +(3,), order='C')
    libdft.VXC_vv10nlc_grad(F.ctypes.data_as(ctypes.c_void_p),
                            vvcoords.ctypes.data_as(ctypes.c_void_p),
                            coords.ctypes.data_as(ctypes.c_void_p),
                            W0p.ctypes.data_as(ctypes.c_void_p),
                            W0.ctypes.data_as(ctypes.c_void_p),
                            K.ctypes.data_as(ctypes.c_void_p),
                            Kp.ctypes.data_as(ctypes.c_void_p),
                            RpW.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(vvcoords.shape[0]),
                            ctypes.c_int(coords.shape[0]))
    #exc is multiplied by Rho later
    exc[threshind] = F
    return exc, Beta

def get_vxc_full_response(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
                          max_memory=2000, verbose=None):
    '''Full response including the response of the grids'''
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_loc = mol.ao_loc_nr()

    excsum = numpy.zeros((mol.natm,3))
    vmat = numpy.zeros((3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        vtmp = numpy.empty((3,nao,nao))
        for atm_id, (coords, weight, weight1) in enumerate(grids_response_cc(grids)):
            mask = gen_grid.make_mask(mol, coords)
            ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask,
                            cutoff=grids.cutoff)
            rho = make_rho(0, ao[0], mask, xctype)
            exc, vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[:2]
            wv = weight * vxc[0]

            vtmp = numpy.zeros((3,nao,nao))
            aow = numint._scale_ao(ao[0], wv)
            _d1_dot_(vtmp, mol, ao[1:4], aow, mask, ao_loc, True)
            vmat += vtmp

            # response of weights
            excsum += numpy.einsum('r,r,nxr->nx', exc, rho, weight1)
            # response of grids coordinates
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dms) * 2
            rho = vxc = aow = None

    elif xctype == 'GGA':
        ao_deriv = 2
        for atm_id, (coords, weight, weight1) in enumerate(grids_response_cc(grids)):
            mask = gen_grid.make_mask(mol, coords)
            ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask,
                            cutoff=grids.cutoff)
            rho = make_rho(0, ao[:4], mask, xctype)
            exc, vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[:2]
            wv = weight * vxc
            wv[0] *= .5

            vtmp = numpy.zeros((3,nao,nao))
            _gga_grad_sum_(vtmp, mol, ao, wv, mask, ao_loc)
            vmat += vtmp

            # response of weights
            excsum += numpy.einsum('r,r,nxr->nx', exc, rho[0], weight1)
            # response of grids coordinates
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dms) * 2
            rho = vxc = wv = None

    elif xctype == 'MGGA':
        ao_deriv = 2
        for atm_id, (coords, weight, weight1) in enumerate(grids_response_cc(grids)):
            mask = gen_grid.make_mask(mol, coords)
            ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask,
                            cutoff=grids.cutoff)
            rho = make_rho(0, ao[:10], mask, xctype)
            exc, vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[:2]
            wv = weight * vxc
            wv[0] *= .5
            wv[4] *= .5  # for the factor 1/2 in tau

            vtmp = numpy.zeros((3,nao,nao))
            _gga_grad_sum_(vtmp, mol, ao, wv, mask, ao_loc)
            _tau_grad_dot_(vtmp, mol, ao, wv[4], mask, ao_loc, True)
            vmat += vtmp

            # response of weights
            excsum += numpy.einsum('r,r,nxr->nx', exc, rho[0], weight1)
            # response of grids coordinates
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dms) * 2
            rho = vxc = wv = None

    # - sign because nabla_X = -nabla_x
    return excsum, -vmat

def get_nlc_vxc_full_response(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
                              max_memory=2000, verbose=None):
    '''Full NLC functional response including the response of the grids'''
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_loc = mol.ao_loc_nr()

    excsum = numpy.zeros((mol.natm,3))
    vmat = numpy.zeros((3,nao,nao))
    nlc_coefs = ni.nlc_coeff(xc_code)
    if len(nlc_coefs) != 1:
        raise NotImplementedError('Additive NLC')
    nlc_pars, fac = nlc_coefs[0]
    ao_deriv = 2
    vvrho = []
    vvcoords = []
    vvweights = []
    for atm_id, (coords, weight) in enumerate(grids_noresponse_cc(grids)):
        mask = gen_grid.make_mask(mol, coords)
        ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask,
                        cutoff=grids.cutoff)
        vvrho.append(make_rho(0, ao[:4], mask, 'GGA'))
        vvcoords.append(coords)
        vvweights.append(weight)
    vvcoords_flat = numpy.vstack(vvcoords)
    vvweights_flat = numpy.concatenate(vvweights)
    vvrho_flat = numpy.hstack(vvrho)

    vv_vxc = []
    for atm_id, (coords, weight, weight1) in enumerate(grids_response_cc(grids)):
        rho = vvrho[atm_id]
        mask = gen_grid.make_mask(mol, coords)
        ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask,
                        cutoff=grids.cutoff)

        exc, vxc = numint._vv10nlc(rho, coords, vvrho_flat, vvweights_flat,
                                   vvcoords_flat, nlc_pars)
        vv_vxc = xc_deriv.transform_vxc(rho, vxc, 'GGA', spin=0)
        wv = vv_vxc * weight
        wv[0] *= .5
        vtmp = numpy.zeros((3,nao,nao))
        _gga_grad_sum_(vtmp, mol, ao, wv, mask, ao_loc)
        vmat += vtmp

        if mol.natm <= 1:
            vvrho_sub = numpy.zeros(0)
            vvcoords_sub = numpy.zeros([0,3])
            vvweights_sub = numpy.zeros(0)
        else:
            vvrho_sub = numpy.hstack(
                [r for i, r in enumerate(vvrho) if i != atm_id]
            )
            vvcoords_sub = numpy.vstack(
                [r for i, r in enumerate(vvcoords) if i != atm_id]
            )
            vvweights_sub = numpy.concatenate(
                [r for i, r in enumerate(vvweights) if i != atm_id]
            )
        egrad, Beta = _vv10nlc_grad(rho, coords, vvrho_sub,
                                    vvweights_sub, vvcoords_sub, nlc_pars)
        # account for factor of 2 in double integration
        exc -= 0.5 * Beta
        # response of weights
        excsum += 2 * numpy.einsum('r,r,nxr->nx', exc, rho[0], weight1)
        # response of grids coordinates
        excsum[atm_id] += 2 * numpy.einsum('xij,ji->x', vtmp, dms)
        excsum[atm_id] += numpy.einsum('r,rx->x', rho[0]*weight, egrad)
    # - sign because nabla_X = -nabla_x
    return excsum, -vmat


def _grid_response_reciprocal(x):
    # Match the GPU cutoff; avoid large ratios for saturated partition weights.
    return numpy.divide(1., x, out=numpy.zeros_like(x), where=numpy.abs(x) > 1e-14)


# JCP 98, 5612 (1993); DOI:10.1063/1.464906
def grids_response_becke(grids):
    mol = grids.mol
    atom_grids_tab = grids.gen_atomic_grids(mol, grids.atom_grid,
                                            grids.radi_method,
                                            grids.level, grids.prune)
    atm_coords = numpy.asarray(mol.atom_coords() , order='C')
    atm_dist = gto.inter_distance(mol, atm_coords)

    def _radii_adjust(mol, atomic_radii):
        charges = mol.atom_charges()
        if grids.radii_adjust == radi.treutler_atomic_radii_adjust:
            rad = numpy.sqrt(atomic_radii[charges]) + 1e-200
        elif grids.radii_adjust == radi.becke_atomic_radii_adjust:
            rad = atomic_radii[charges] + 1e-200
        else:
            fadjust = lambda i, j, g: g
            gadjust = lambda *args: 1
            return fadjust, gadjust

        rr = rad.reshape(-1,1) * (1./rad)
        a = .25 * (rr.T - rr)
        a[a<-.5] = -.5
        a[a>0.5] = 0.5

        def fadjust(i, j, g):
            return g + a[i,j]*(1-g**2)

        #: d[g + a[i,j]*(1-g**2)] /dg = 1 - 2*a[i,j]*g
        def gadjust(i, j, g):
            return 1 - 2*a[i,j]*g
        return fadjust, gadjust

    fadjust, gadjust = _radii_adjust(mol, grids.atomic_radii)

    def get_smooth_function_derivative(mu):
        if grids.becke_scheme == gen_grid.original_becke:
            p0 = mu
            p1 = (3 - p0**2) * p0 * .5
            p2 = (3 - p1**2) * p1 * .5
            p3 = (3 - p2**2) * p2 * .5
            t_uab = 27./8. * (1-p2**2) * (1-p1**2) * (1-p0**2)
            return p3, t_uab
        elif grids.becke_scheme == gen_grid.stratmann:
            a = .64
            ma = mu/a
            ma2 = ma * ma
            g1 = (1/16.)*(ma*(35 + ma2*(-35 + ma2*(21 - 5 *ma2))))
            g1[mu<=-a] = -1
            g1[mu>= a] =  1

            t_uab = (35.0/16.0 / a) * (1.0 - ma2 * (3.0 - ma2 * (3.0 - ma2)))
            t_uab[mu<=-a] = 0
            t_uab[mu>= a] = 0
            return g1, t_uab
        else:
            raise NotImplementedError(f"Grid response for becke_scheme = {grids.becke_scheme} is not implemented")

    def gen_grid_partition(coords, atom_id):
        ngrids = coords.shape[0]
        grid_dist = []
        grid_norm_vec = []
        for ia in range(mol.natm):
            v = (atm_coords[ia] - coords).T
            normv = numpy.linalg.norm(v,axis=0)
            v *= _grid_response_reciprocal(normv)
            grid_dist.append(normv)
            grid_norm_vec.append(v)

        def get_du(ia, ib):  # JCP 98, 5612 (1993); (B10)
            uab = atm_coords[ia] - atm_coords[ib]
            duab = 1./atm_dist[ia,ib] * grid_norm_vec[ia]
            duab-= uab[:,None]/atm_dist[ia,ib]**3 * (grid_dist[ia]-grid_dist[ib])
            return duab

        pbecke = numpy.ones((mol.natm,ngrids))
        dpbecke = numpy.zeros((mol.natm,mol.natm,3,ngrids))
        for ia in range(mol.natm):
            for ib in range(ia):
                g = 1/atm_dist[ia,ib] * (grid_dist[ia]-grid_dist[ib])
                g_radii_adjusted = fadjust(ia, ib, g)
                smoothed, t_uab = get_smooth_function_derivative(g_radii_adjusted)
                t_uab *= 0.5 * gadjust(ia, ib, g)

                s_uab = .5 * (1 - smoothed)
                s_uba = .5 * (1 + smoothed)
                pbecke[ia] *= s_uab
                pbecke[ib] *= s_uba
                pt_uab =-t_uab * _grid_response_reciprocal(s_uab)
                pt_uba = t_uab * _grid_response_reciprocal(s_uba)

# * When grid is on atom ia/ib, ua/ub == 0, d_uba/d_uab may have huge error
#   How to remove this error?
                duab = get_du(ia, ib)
                duba = get_du(ib, ia)
                if ia == atom_id:
                    dpbecke[ia,ia] += pt_uab * duba
                    dpbecke[ia,ib] += pt_uba * duba
                else:
                    dpbecke[ia,ia] += pt_uab * duab
                    dpbecke[ia,ib] += pt_uba * duab

                if ib == atom_id:
                    dpbecke[ib,ib] -= pt_uba * duab
                    dpbecke[ib,ia] -= pt_uab * duab
                else:
                    dpbecke[ib,ib] -= pt_uba * duba
                    dpbecke[ib,ia] -= pt_uab * duba

# * JCP 98, 5612 (1993); (B8) (B10) miss many terms
                if ia != atom_id and ib != atom_id:
                    ua_ub = grid_norm_vec[ia] - grid_norm_vec[ib]
                    ua_ub /= atm_dist[ia,ib]
                    dpbecke[atom_id,ia] -= pt_uab * ua_ub
                    dpbecke[atom_id,ib] -= pt_uba * ua_ub

        for ia in range(mol.natm):
            dpbecke[:,ia] *= pbecke[ia]

        return pbecke, dpbecke

    natm = mol.natm
    for ia in range(natm):
        coords, vol = atom_grids_tab[mol.atom_symbol(ia)]
        coords = coords + atm_coords[ia]
        pbecke, dpbecke = gen_grid_partition(coords, ia)
        z = _grid_response_reciprocal(pbecke.sum(axis=0))
        w1 = dpbecke[:,ia] * z
        w1 -= pbecke[ia] * z**2 * dpbecke.sum(axis=1)
        w1 *= vol
        w0 = vol * pbecke[ia] * z
        yield coords, w0, w1


def _get_lko_partition_functions(mol, ia, radii_adjust, atomic_radii):
    if callable(radii_adjust) and atomic_radii is not None:
        f_radii_adjust = radii_adjust(mol, atomic_radii)
    else:
        f_radii_adjust = None
    atm_coords = numpy.asarray(mol.atom_coords() , order='C')
    gen_grid_fn = gen_grid.libdft.VXCgen_grid_lko
    gen_deriv_fn = gen_grid.libdft.VXCgen_grid_lko_deriv

    if f_radii_adjust is None:
        p_radii_table = lib.c_null_ptr()
    else:
        f_radii_table = numpy.asarray([f_radii_adjust(i, j, 0)
                                       for i in range(mol.natm)
                                       for j in range(mol.natm)])
        p_radii_table = f_radii_table.ctypes.data_as(ctypes.c_void_p)

    def gen_grid_partition(coords):
        coords = numpy.asarray(coords, order='F')
        ngrids = coords.shape[0]
        pbecke = numpy.empty((mol.natm,ngrids))
        gen_grid_fn(pbecke.ctypes.data_as(ctypes.c_void_p),
                    coords.ctypes.data_as(ctypes.c_void_p),
                    atm_coords.ctypes.data_as(ctypes.c_void_p),
                    p_radii_table,
                    ctypes.c_int(mol.natm), ctypes.c_int(ngrids))
        return pbecke

    def gen_grid_deriv(coords, dbecke):
        if isinstance(ia, int):
            ialist = numpy.repeat(numpy.int32(ia), coords.shape[0])
        else:
            assert isinstance(ia, numpy.ndarray)
            ialist = ia.astype(numpy.int32, order="C", copy=True)
        coords = numpy.asarray(coords, order='F')
        ngrids = coords.shape[0]
        pbecke = numpy.empty((4,mol.natm,ngrids))
        gen_deriv_fn(pbecke.ctypes.data_as(ctypes.c_void_p),
                     dbecke.ctypes.data_as(ctypes.c_void_p),
                     coords.ctypes.data_as(ctypes.c_void_p),
                     atm_coords.ctypes.data_as(ctypes.c_void_p),
                     p_radii_table,
                     ctypes.c_int(mol.natm), ctypes.c_int(ngrids),
                     ialist.ctypes.data_as(ctypes.c_void_p))
        return pbecke[1:]

    return gen_grid_partition, gen_grid_deriv, atm_coords


def get_dw_partition(mol, ia, atom_grids_tab,
                     radii_adjust=None, atomic_radii=radi.BRAGG_RADII,
                     concat=True, wtonly=False):
    gen_grid_partition, gen_grid_deriv, atm_coords = _get_lko_partition_functions(
        mol, ia, radii_adjust, atomic_radii
    )
    coords, vol = atom_grids_tab[mol.atom_symbol(ia)]
    coords = coords + atm_coords[ia]
    pbecke = gen_grid_partition(coords)
    invsum = _grid_response_reciprocal(pbecke.sum(axis=0))
    weights = vol * pbecke[ia] * invsum
    if wtonly:
        return coords, weights
    else:
        dbecke = -pbecke * weights * invsum
        dbecke[ia, :] += weights
        dbecke = numpy.ascontiguousarray(dbecke)
        weights1 = gen_grid_deriv(coords, dbecke)
        return coords, weights, weights1


def get_dw_partition_sorted(mol, ialist, coords, weights,
                            radii_adjust=None, atomic_radii=radi.BRAGG_RADII):
    gen_grid_partition, gen_grid_deriv = _get_lko_partition_functions(
        mol, ialist, radii_adjust, atomic_radii
    )[:2]
    pbecke = gen_grid_partition(coords)
    invsum = _grid_response_reciprocal(pbecke.sum(axis=0))
    dbecke = -pbecke * weights * invsum
    for g in range(weights.size):
        dbecke[ialist[g], g] += weights[g]
    #for ia in range(mol.natm):
    #    cond = numpy.nonzero(ialist == ia)
    #    dbecke[ia][cond] += weights[cond]
    dbecke = numpy.ascontiguousarray(dbecke)
    weights1 = gen_grid_deriv(coords, dbecke)
    return weights1


def grids_response_lko(grids):
    mol = grids.mol
    atom_grids_tab = grids.gen_atomic_grids(mol, grids.atom_grid,
                                            grids.radi_method,
                                            grids.level, grids.prune)
    for ia in range(mol.natm):
        coord, weight, weight1 = get_dw_partition(mol, ia, atom_grids_tab,
                                                  grids.radii_adjust,
                                                  grids.atomic_radii)
        yield coord, weight, weight1.transpose(1,0,2)


def grids_noresponse_cc(grids):
    # same as above but without the response, for nlc grids response routine
    mol = grids.mol
    atom_grids_tab = grids.gen_atomic_grids(mol, grids.atom_grid,
                                            grids.radi_method,
                                            grids.level, grids.prune)
    coords_all, weights_all = gen_grid.get_partition(mol, atom_grids_tab,
                                                     grids.radii_adjust,
                                                     grids.atomic_radii,
                                                     grids.becke_scheme,
                                                     concat=False)
    natm = mol.natm
    for ia in range(natm):
        yield coords_all[ia], weights_all[ia]


def grids_response_cc(grids):
    if grids.becke_scheme == gen_grid.becke_lko:
        gr_fn = grids_response_lko
    else:
        gr_fn = grids_response_becke
    for coord, weight, weight1 in gr_fn(grids):
        yield coord, weight, weight1


class Gradients(rhf_grad.Gradients):

    # This parameter has no effects for HF gradients. Add this attribute so that
    # the kernel function can be reused in the DFT gradients code.
    grid_response = getattr(__config__, 'grad_rks_Gradients_grid_response', False)

    _keys = {'grid_response', 'grids', 'nlcgrids'}

    def __init__(self, mf):
        rhf_grad.Gradients.__init__(self, mf)
        self.grids = None
        self.nlcgrids = None

    def dump_flags(self, verbose=None):
        rhf_grad.Gradients.dump_flags(self, verbose)
        logger.info(self, 'grid_response = %s', self.grid_response)
        #if callable(self.base.grids.prune):
        #    logger.info(self, 'Grid pruning %s may affect DFT gradients accuracy.'
        #                'Call mf.grids.run(prune=False) to mute grid pruning',
        #                self.base.grids.prune)
        return self

    get_veff = get_veff

    # Set only while grad_elec runs. get_veff then stores the XC forces here
    # (key 'de') instead of adding the XC matrix, and extra_force adds them.
    # Direct calls to get_veff keep returning the complete matrix.
    _xc_force = None

    def grad_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        with lib.temporary_env(self, _xc_force={} if XC_FORCE else None):
            return rhf_grad.grad_elec(self, mo_energy, mo_coeff, mo_occ, atmlst)

    def extra_force(self, atom_id, envs):
        '''Hook for extra contributions in analytical gradients.

        Contributions like the response of auxiliary basis in density fitting
        method, the grid response in DFT numerical integration can be put in
        this function.
        '''
        de = 0
        if self.grid_response:
            vhf = envs['vhf']
            log = envs['log']
            log.debug('grids response for atom %d %s',
                      atom_id, vhf.exc1_grid[atom_id])
            de = vhf.exc1_grid[atom_id]
        if self._xc_force and 'de' in self._xc_force:
            de = de + self._xc_force['de'][atom_id]
        return de

Grad = Gradients

from pyscf import dft
dft.rks.RKS.Gradients = dft.rks_symm.RKS.Gradients = lib.class_as_method(Gradients)
dft.roks.ROKS.Gradients = lib.invalid_method('Gradients')
