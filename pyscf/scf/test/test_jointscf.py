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

import io
import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import dft
from pyscf.scf import jointscf
from pyscf.scf import _transferable


def water(r1, r2, ang, charge=0, spin=0, basis='6-31g'):
    ang = numpy.deg2rad(ang)
    return gto.M(
        atom=[('O', (0, 0, 0)), ('H', (r1, 0, 0)),
              ('H', (r2*numpy.cos(ang), r2*numpy.sin(ang), 0))],
        basis=basis, charge=charge, spin=spin, verbose=0)

def random_rotation(mol, seed):
    rng = numpy.random.default_rng(seed)
    q = numpy.linalg.qr(rng.normal(size=(3, 3)))[0]
    if numpy.linalg.det(q) < 0:
        q[:,0] *= -1
    mol1 = mol.copy()
    mol1.set_geom_(mol.atom_coords().dot(q.T) + rng.normal(size=3), unit='Bohr')
    return mol1, q

def rks(mol, xc='pbe'):
    mf = dft.RKS(mol, xc=xc)
    mf.grids.level = 1
    return mf

def setUpModule():
    global geoms, mols, e_rhf, nfock_rhf
    rng = numpy.random.default_rng(1)
    geoms = [(rng.uniform(0.9, 1.1), rng.uniform(0.9, 1.1), rng.uniform(98, 112))
             for _ in range(20)]
    mols = [water(*g) for g in geoms]
    mfs = [scf.RHF(mol).run() for mol in mols]
    e_rhf = numpy.array([mf.e_tot for mf in mfs])
    # Fock builds of scf.hf.kernel: initial guess + SCF cycles + extra cycle
    nfock_rhf = numpy.array([mf.cycles + 2 for mf in mfs])

def tearDownModule():
    global geoms, mols, e_rhf, nfock_rhf
    del geoms, mols, e_rhf, nfock_rhf


class KnownValues(unittest.TestCase):
    def test_sync_identical_to_kernel(self):
        ref = [rks(mol).run() for mol in mols[:4]]
        jmf = jointscf.JointSCF([rks(mol) for mol in mols[:4]])
        jmf.schedule = 'sync'
        jmf.max_workers = 1
        e = jmf.kernel()
        self.assertTrue(jmf.converged.all())
        # The same arithmetic operations as scf.hf.kernel.  Deviations are
        # only caused by the thread scheduling of OpenMP reductions.
        for i, mf in enumerate(ref):
            self.assertAlmostEqual(e[i], mf.e_tot, 12)
            self.assertEqual(jmf.cycles[i], mf.cycles)
            self.assertEqual(jmf.nfock[i], mf.cycles + 2)
            self.assertAlmostEqual(abs(jmf.mfs[i].mo_coeff - mf.mo_coeff).max(), 0, 10)
            self.assertAlmostEqual(abs(jmf.mfs[i].mo_energy - mf.mo_energy).max(), 0, 10)
            self.assertTrue(jmf.mfs[i].converged)

    def test_sync_mixed_methods(self):
        def build():
            return [scf.RHF(mols[0]), scf.UHF(water(1., 1., 105., 1, 1)),
                    rks(mols[1], 'b3lyp'), scf.RHF(gto.M(atom='N 0 0 0; N 0 0 1.1',
                                                         basis='6-31g', verbose=0))]
        ref = [mf.run() for mf in build()]
        jmf = scf.JointSCF(build())
        jmf.schedule = 'sync'
        e = jmf.kernel()
        self.assertTrue(jmf.converged.all())
        self.assertAlmostEqual(abs(e - [mf.e_tot for mf in ref]).max(), 0, 11)
        self.assertEqual(list(jmf.nfock), [mf.cycles + 2 for mf in ref])
        self.assertEqual(jmf.mo_coeff[1].shape, ref[1].mo_coeff.shape)

    def test_cascade_rhf(self):
        jmf = jointscf.JointSCF([scf.RHF(mol) for mol in mols])
        e = jmf.kernel()
        self.assertEqual(jmf.schedule, 'cascade')
        self.assertTrue(jmf.converged.all())
        self.assertAlmostEqual(abs(e - e_rhf).max(), 0, 8)
        # leaders are identical to independent calculations
        lead = jmf.levels == 0
        self.assertEqual(lead.sum(), 8)
        self.assertAlmostEqual(abs(e - e_rhf)[lead].max(), 0, 11)
        self.assertEqual(list(jmf.nfock[lead]), list(nfock_rhf[lead]))
        # promoted configurations need fewer Fock builds
        self.assertTrue(jmf.nfock[~lead].mean() < nfock_rhf[~lead].mean() - 1.5)
        self.assertTrue(jmf.nfock.sum() < nfock_rhf.sum() * 0.85)
        self.assertTrue((jmf.promoted_at[~lead] > 0).all())
        self.assertEqual(len(jmf.models), 1)
        self.assertEqual(sum(p['ndst'] for p in jmf.models[0].promotions), 12)
        # orbitals of a promoted configuration
        i = numpy.where(~lead)[0][0]
        mf = scf.RHF(mols[i]).run()
        dm = jmf.mfs[i].make_rdm1()
        self.assertAlmostEqual(abs(dm - mf.make_rdm1()).max(), 0, 4)
        self.assertAlmostEqual(abs(jmf.mfs[i].mo_energy - mf.mo_energy).max(), 0, 5)

    def test_cascade_orientation(self):
        # Rigid motions of the configurations must not change the iterations
        jmf = jointscf.JointSCF([scf.RHF(mol) for mol in mols])
        jmf.kernel()
        mols1 = [random_rotation(mol, i)[0] for i, mol in enumerate(mols)]
        jmf1 = jointscf.JointSCF([scf.RHF(mol) for mol in mols1])
        e = jmf1.kernel()
        self.assertTrue(jmf1.converged.all())
        self.assertAlmostEqual(abs(e - e_rhf).max(), 0, 8)
        self.assertEqual(list(jmf1.levels), list(jmf.levels))
        self.assertTrue(abs(jmf1.nfock.sum() - jmf.nfock.sum()) <= 4)

    def test_cascade_uks(self):
        molc = [water(*g, charge=1, spin=1) for g in geoms[:12]]
        def uks(mol):
            mf = dft.UKS(mol, xc='pbe')
            mf.grids.level = 1
            return mf
        ref = [uks(mol).run() for mol in molc]
        jmf = jointscf.JointSCF([uks(mol) for mol in molc])
        e = jmf.kernel()
        self.assertTrue(jmf.converged.all())
        self.assertAlmostEqual(abs(e - [mf.e_tot for mf in ref]).max(), 0, 8)
        self.assertTrue(jmf.nfock.sum() < sum(mf.cycles + 2 for mf in ref))

    def test_cascade_rohf(self):
        molc = [water(*g, charge=1, spin=1) for g in geoms[:12]]
        ref = [scf.ROHF(mol).run() for mol in molc]
        jmf = jointscf.JointSCF([scf.ROHF(mol) for mol in molc])
        e = jmf.kernel()
        self.assertTrue(jmf.converged.all())
        self.assertAlmostEqual(abs(e - [mf.e_tot for mf in ref]).max(), 0, 8)

    def test_template_and_workers(self):
        jmf = jointscf.JointSCF(rks(mols[0], 'b3lyp'), mols[:12])
        jmf.max_workers = 2
        e = jmf.kernel()
        ref = [rks(mol, 'b3lyp').run().e_tot for mol in mols[:12]]
        self.assertTrue(jmf.converged.all())
        self.assertAlmostEqual(abs(e - ref).max(), 0, 8)
        self.assertTrue(all(mf.mol is mol for mf, mol in zip(jmf.mfs, mols)))
        self.assertEqual(len(set(mf.chkfile for mf in jmf.mfs)), 12)
        self.assertEqual(len(set(id(mf.grids) for mf in jmf.mfs)), 12)

    def test_parallel_backends(self):
        ref = None
        for parallel, nworkers in (('thread', 1), ('thread', 2), ('process', 2)):
            jmf = jointscf.JointSCF([scf.RHF(mol) for mol in mols])
            jmf.parallel = parallel
            jmf.max_workers = nworkers
            e = jmf.kernel()
            self.assertTrue(jmf.converged.all())
            self.assertAlmostEqual(abs(e - e_rhf).max(), 0, 8)
            if ref is None:
                ref = jmf
            # the backend does not change the iterations
            self.assertEqual(list(jmf.nfock), list(ref.nfock))
            self.assertEqual(list(jmf.promoted_at), list(ref.promoted_at))
            self.assertAlmostEqual(abs(e - ref.e_tot).max(), 0, 10)
            for mf, mf0 in zip(jmf.mfs, ref.mfs):
                self.assertTrue(mf.converged)
                self.assertAlmostEqual(
                    abs(mf.make_rdm1() - mf0.make_rdm1()).max(), 0, 7)
            self.assertTrue(jmf.timings['total'] > jmf.timings['startup'])

    def test_backend_fallback(self):
        # Density-fitting objects are instances of a dynamically created
        # class. They are handled by worker threads.
        mfs = [scf.RHF(mol).density_fit() for mol in mols[:3]]
        ref = [scf.RHF(mol).density_fit().run().e_tot for mol in mols[:3]]
        jmf = jointscf.JointSCF(mfs)
        jmf.max_workers = 2
        e = jmf.kernel()
        self.assertAlmostEqual(abs(e - ref).max(), 0, 10)
        # callbacks run in the calling process
        seen = []
        jmf = jointscf.JointSCF([scf.RHF(mol) for mol in mols[:3]])
        jmf.max_workers = 2
        jmf.kernel(callback=lambda envs: seen.append(envs['cycle']))
        self.assertEqual(len(seen), jmf.cycles.sum())

    def test_user_descriptors_and_dm0(self):
        jmf = jointscf.JointSCF([scf.RHF(mol) for mol in mols])
        jmf.descriptors = numpy.array(geoms)
        jmf.n_lead = 6
        e = jmf.kernel()
        self.assertTrue(jmf.converged.all())
        self.assertAlmostEqual(abs(e - e_rhf).max(), 0, 8)
        self.assertEqual((jmf.levels == 0).sum(), 6)
        # restart from the converged density matrices
        dm0 = jmf.make_rdm1()
        jmf1 = jointscf.JointSCF([scf.RHF(mol) for mol in mols])
        e = jmf1.kernel(dm0=dm0)
        self.assertAlmostEqual(abs(e - e_rhf).max(), 0, 8)
        self.assertTrue(jmf1.nfock.max() <= 4)

    def test_small_batch_and_unsupported(self):
        # too few configurations for the shared model
        jmf = jointscf.JointSCF([scf.RHF(mol) for mol in mols[:3]])
        e = jmf.kernel()
        self.assertEqual(list(jmf.nfock), list(nfock_rhf[:3]))
        self.assertAlmostEqual(abs(e - e_rhf[:3]).max(), 0, 11)
        self.assertEqual(jmf.models, [])
        self.assertRaises(NotImplementedError, jointscf.JointSCF,
                          [scf.RHF(mols[0]).newton()])
        self.assertRaises(ValueError, jointscf.JointSCF, [])

    def test_max_cycle(self):
        mfs = [scf.RHF(mol) for mol in mols[:3]]
        mfs[1].max_cycle = 2
        jmf = jointscf.JointSCF(mfs)
        jmf.kernel()
        self.assertEqual(list(jmf.converged), [True, False, True])
        self.assertEqual(jmf.cycles[1], 2)
        self.assertFalse(mfs[1].converged)

    def test_kabsch_rotation(self):
        mol1, q = random_rotation(mols[0], 3)
        r = jointscf.kabsch_rotation(mol1.atom_coords(), mols[0].atom_coords())
        self.assertAlmostEqual(abs(r - q.T).max(), 0, 12)
        # AO matrices in the aligned frame
        u = mol1.ao_rotation_matrix(r)
        s0 = mols[0].intor('int1e_ovlp')
        s1 = mol1.intor('int1e_ovlp')
        self.assertAlmostEqual(abs(u.T.dot(s1).dot(u) - s0).max(), 0, 12)

    def test_shared_model(self):
        z = numpy.random.default_rng(2).normal(size=(30, 2))
        order = jointscf.farthest_point_order(z)
        self.assertEqual(sorted(order), list(range(30)))
        self.assertEqual(order[0], numpy.argmin(((z - z.mean(axis=0))**2).sum(axis=1)))
        # The model reproduces matrices which are linear in the descriptors
        model = jointscf.SharedModel(mols, descriptors=numpy.array(geoms), align=False)
        nao = mols[0].nao
        a, b = numpy.random.default_rng(3).normal(size=(2, nao, nao))
        a, b = a + a.T, b + b.T
        for i, mol in enumerate(mols):
            model.set_overlap(i, mol.intor('int1e_ovlp'))
            model._store(i, 0, a + model.z[i,0] * b, b - model.z[i,1] * a)
        src = model.order[:12]
        dst = model.order[12:]
        hyper = model.select_hyper(src, 0)
        wts = model.weights(src, dst, hyper)
        self.assertAlmostEqual(abs(wts.sum(axis=1) - 1).max(), 0, 12)
        p, f = model.predict(src, 0, wts)
        ref = numpy.array([a + model.z[i,0] * b for i in dst])
        self.assertAlmostEqual(abs(p - ref).max(), 0, 3)
        ref = numpy.array([b - model.z[i,1] * a for i in dst])
        self.assertAlmostEqual(abs(f - ref).max(), 0, 3)
        res, err = model.loo_error(src, 0, hyper)
        self.assertTrue(err.max() < 1e-3 * res.max())
        self.assertAlmostEqual(abs(res - model.residuals(src, 0)).max(), 0, 12)
        # increments between two entries
        for i in src:
            model._store(i, 1, 2 * a + model.z[i,0] * b, b)
        dp, df = model.predict_increment(src, 1, 0, wts)
        self.assertAlmostEqual(abs(dp - a).max(), 0, 3)
        # AO representation
        dm, fock = model.to_ao(dst[0], p[0], f[0])
        s1e = mols[dst[0]].intor('int1e_ovlp')
        self.assertAlmostEqual(numpy.einsum('ij,ji', dm, s1e), numpy.trace(p[0]), 9)
        model.record(dst[0], 2, dm, fock)
        self.assertAlmostEqual(abs(model._get(dst[0], 2)[0] - p[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(model._get(dst[0], 2)[1] - f[0]).max(), 0, 9)

    def test_transferable_polish(self):
        jmf = jointscf.JointSCF([scf.RHF(mol) for mol in mols])
        jmf.schedule = 'transferable'
        jmf.polish = True
        jmf.max_workers = 2
        e = jmf.kernel()
        self.assertTrue(jmf.converged.all())
        self.assertTrue(all(mf.converged for mf in jmf.mfs))
        self.assertAlmostEqual(abs(e - e_rhf).max(), 0, 8)
        self.assertTrue((jmf.nfock == jmf.nfock_shared + jmf.nfock_polish).all())
        self.assertTrue((jmf.nfock_shared >= 0).all())
        self.assertTrue((jmf.nfock_polish > 0).all())

        # DFT (RKS/PBE)
        ref_ks = [rks(mol, 'pbe').run().e_tot for mol in mols[:10]]
        jmf_ks = jointscf.JointSCF([rks(mol, 'pbe') for mol in mols[:10]])
        jmf_ks.schedule = 'transferable'
        jmf_ks.polish = True
        jmf_ks.max_workers = 1
        e_ks = jmf_ks.kernel()
        self.assertTrue(jmf_ks.converged.all())
        self.assertAlmostEqual(abs(e_ks - ref_ks).max(), 0, 8)

    def test_transferable_nopolish_variational_and_ranks(self):
        errs = []
        for r_delta in (0, 1, 2, None):
            jmf = jointscf.JointSCF([scf.RHF(mol) for mol in mols[:8]])
            jmf.schedule = 'transferable'
            jmf.polish = False
            jmf.correction_rank = r_delta
            jmf.molecule_batch = 4
            jmf.max_shared_fock = 80
            jmf.max_workers = 1
            e = jmf.kernel()
            self.assertFalse(jmf.converged.any())
            self.assertTrue((jmf.nfock_polish == 0).all())
            self.assertTrue((jmf.nfock == jmf.nfock_shared).all())
            self.assertTrue((jmf.e_err_est >= 0).all())
            self.assertTrue(numpy.isfinite(jmf.e_err_est).all())
            # Exact variational upper bound
            self.assertTrue((e >= e_rhf[:8] - 1e-10).all())
            for mf in jmf.mfs:
                occ = mf.mo_coeff[:, mf.mo_occ > 0]
                s1e = mf.get_ovlp()
                numpy.testing.assert_allclose(
                    occ.T.dot(s1e).dot(occ), numpy.eye(occ.shape[1]), atol=1e-10)
            errs.append(float((e - e_rhf[:8]).mean()))
        self.assertGreater(errs[0], errs[1] - 1e-9)
        self.assertGreater(errs[1], errs[2] - 1e-9)
        self.assertGreater(errs[2], errs[3] - 1e-9)
        self.assertGreater(errs[0], errs[-1] + 1e-5)

    def test_transferable_gradients(self):
        mols3 = mols[:3]
        weights = numpy.array([0.2, 0.5, 0.3])
        rng = numpy.random.default_rng(19)
        factories = (
            lambda m: scf.RHF(m),
            lambda m: rks(m, 'svwn'),
            lambda m: rks(m, 'pbe'),
            lambda m: rks(m, 'b3lyp'),
        )
        for factory in factories:
            for r_delta in (None, 1):
                mfs = [factory(m) for m in mols3]
                for mf in mfs:
                    if hasattr(mf, 'grids'):
                        mf.grids.prune = None
                    mf.build(mf.mol)
                setup_data = [(mf.get_ovlp(), mf.get_hcore(), mf.get_init_guess())
                              for mf in mfs]
                model = _transferable.TransferableGroupModel(
                    mfs, correction_rank=r_delta, latent_rank=7, n_hidden=3,
                    reg_delta=1e-3, reg_theta=1e-3, seed=7)
                model.initialize(setup_data)
                theta0 = model.theta[0] + rng.normal(scale=0.02, size=model.theta[0].shape)
                a_list = [model.a_mats[0][i] + rng.normal(scale=0.02, size=model.a_mats[0][i].shape)
                          for i in range(3)]
                if r_delta is None:
                    d_list = [model.delta_entries[0][i] + rng.normal(
                        scale=0.02, size=model.delta_entries[0][i].shape)
                        for i in range(3)]
                else:
                    d_list = [(b + rng.normal(scale=0.02, size=b.shape),
                               g + rng.normal(scale=0.02, size=g.shape))
                              for b, g in model.delta_entries[0]]

                calls = [0]
                for mf in mfs:
                    orig = mf.get_veff
                    def wrapped(mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1, _orig=orig):
                        calls[0] += 1
                        return _orig(mol, dm, dm_last, vhf_last, hermi)
                    mf.get_veff = wrapped

                _, gt, ga, gd, _ = _transferable.evaluate_transferable_loss_and_grad(
                    mfs, theta0, a_list, d_list,
                    weights=weights, latent_rank=7, correction_rank=r_delta, n_hidden=3,
                    reg_delta=1e-3, reg_theta=1e-3,
                    providers=model.providers, ref_coords=model.ref_coords,
                    mass=model.mass, z=model.z)
                self.assertEqual(calls[0], 3)

                eps = 1e-6
                def check_rel(grad_vec, eval_shifted):
                    idx_s = rng.choice(grad_vec.size, size=min(8, grad_vec.size), replace=False)
                    ana = numpy.array([grad_vec.ravel()[k] for k in idx_s])
                    fd = numpy.array([(eval_shifted(k, eps) - eval_shifted(k, -eps)) / (2 * eps)
                                      for k in idx_s])
                    rel = numpy.linalg.norm(ana - fd) / max(numpy.linalg.norm(fd), 1e-12)
                    self.assertLess(rel, 1e-5)

                check_rel(gt, lambda k, d: _transferable.evaluate_transferable_loss_and_grad(
                    mfs, theta0 + numpy.eye(theta0.size)[k] * d, a_list, d_list,
                    weights=weights, latent_rank=7, correction_rank=r_delta, n_hidden=3,
                    reg_delta=1e-3, reg_theta=1e-3, providers=model.providers,
                    ref_coords=model.ref_coords, mass=model.mass, z=model.z)[0])

                def shift_a(k, d):
                    al = [x.copy() for x in a_list]
                    al[1].ravel()[k] += d
                    return _transferable.evaluate_transferable_loss_and_grad(
                        mfs, theta0, al, d_list, weights=weights, latent_rank=7,
                        correction_rank=r_delta, n_hidden=3, reg_delta=1e-3, reg_theta=1e-3,
                        providers=model.providers, ref_coords=model.ref_coords,
                        mass=model.mass, z=model.z)[0]
                check_rel(ga[1], shift_a)

                if r_delta is None:
                    def shift_d(k, d):
                        dl = [x.copy() for x in d_list]
                        dl[1].ravel()[k] += d
                        return _transferable.evaluate_transferable_loss_and_grad(
                            mfs, theta0, a_list, dl, weights=weights, latent_rank=7,
                            correction_rank=r_delta, n_hidden=3, reg_delta=1e-3, reg_theta=1e-3,
                            providers=model.providers, ref_coords=model.ref_coords,
                            mass=model.mass, z=model.z)[0]
                    check_rel(gd[1], shift_d)
                else:
                    def shift_b(k, d):
                        dl = [(b.copy(), g.copy()) for b, g in d_list]
                        dl[1][0].ravel()[k] += d
                        return _transferable.evaluate_transferable_loss_and_grad(
                            mfs, theta0, a_list, dl, weights=weights, latent_rank=7,
                            correction_rank=r_delta, n_hidden=3, reg_delta=1e-3, reg_theta=1e-3,
                            providers=model.providers, ref_coords=model.ref_coords,
                            mass=model.mass, z=model.z)[0]
                    def shift_g(k, d):
                        dl = [(b.copy(), g.copy()) for b, g in d_list]
                        dl[1][1].ravel()[k] += d
                        return _transferable.evaluate_transferable_loss_and_grad(
                            mfs, theta0, a_list, dl, weights=weights, latent_rank=7,
                            correction_rank=r_delta, n_hidden=3, reg_delta=1e-3, reg_theta=1e-3,
                            providers=model.providers, ref_coords=model.ref_coords,
                            mass=model.mass, z=model.z)[0]
                    check_rel(gd[1][0], shift_b)
                    check_rel(gd[1][1], shift_g)

    def test_transferable_orientation_invariance(self):
        mols1 = [random_rotation(mol, i + 10)[0] for i, mol in enumerate(mols[:10])]
        for polish, tol in ((False, 10), (True, 8)):
            jmf0 = jointscf.JointSCF([scf.RHF(mol) for mol in mols[:10]])
            jmf0.schedule = 'transferable'
            jmf0.polish = polish
            jmf0.max_workers = 1
            e0 = jmf0.kernel()

            jmf1 = jointscf.JointSCF([scf.RHF(mol) for mol in mols1])
            jmf1.schedule = 'transferable'
            jmf1.polish = polish
            jmf1.max_workers = 1
            e1 = jmf1.kernel()

            self.assertAlmostEqual(abs(e1 - e0).max(), 0, tol)
            self.assertEqual(list(jmf1.nfock_shared), list(jmf0.nfock_shared))
            self.assertEqual(list(jmf1.nfock_polish), list(jmf0.nfock_polish))
            self.assertEqual(list(jmf1.nfock), list(jmf0.nfock))

    def test_transferable_backends_and_batch_size(self):
        for polish in (False, True):
            ref = None
            for parallel, nworkers in (('thread', 1), ('thread', 2), ('process', 2)):
                jmf = jointscf.JointSCF([scf.RHF(mol) for mol in mols[:12]])
                jmf.schedule = 'transferable'
                jmf.polish = polish
                jmf.parallel = parallel
                jmf.max_workers = nworkers
                jmf.seed = 42
                e = jmf.kernel()
                if polish:
                    self.assertTrue(jmf.converged.all())
                if ref is None:
                    ref = jmf
                self.assertEqual(list(jmf.nfock_shared), list(ref.nfock_shared))
                self.assertEqual(list(jmf.nfock_polish), list(ref.nfock_polish))
                self.assertEqual(list(jmf.nfock), list(ref.nfock))
                self.assertAlmostEqual(abs(e - ref.e_tot).max(), 0, 10)

        for mb in (1, 4, 12):
            jmf = jointscf.JointSCF([scf.RHF(mol) for mol in mols[:12]])
            jmf.schedule = 'transferable'
            jmf.polish = True
            jmf.molecule_batch = mb
            jmf.max_workers = 2
            e = jmf.kernel()
            self.assertTrue(jmf.converged.all())
            self.assertAlmostEqual(abs(e - e_rhf[:12]).max(), 0, 8)

    def test_transferable_uks_and_fallback(self):
        molc = [water(*g, charge=1, spin=1) for g in geoms[:8]]
        def uks(mol):
            mf = dft.UKS(mol, xc='pbe')
            mf.grids.level = 1
            return mf
        ref_uks = [uks(mol).run().e_tot for mol in molc]
        jmf_p = jointscf.JointSCF([uks(mol) for mol in molc])
        jmf_p.schedule = 'transferable'
        jmf_p.polish = True
        jmf_p.max_workers = 2
        e_p = jmf_p.kernel()
        self.assertTrue(jmf_p.converged.all())
        self.assertAlmostEqual(abs(e_p - ref_uks).max(), 0, 8)

        jmf_np = jointscf.JointSCF([uks(mol) for mol in molc])
        jmf_np.schedule = 'transferable'
        jmf_np.polish = False
        jmf_np.max_workers = 1
        e_np = jmf_np.kernel()
        self.assertFalse(jmf_np.converged.any())
        self.assertTrue((e_np >= numpy.asarray(ref_uks) - 1e-10).all())

        # ROHF falls back to sync with a log message
        buf = io.StringIO()
        mfs_rohf = [scf.ROHF(mol) for mol in molc[:4]]
        for mf in mfs_rohf:
            mf.stdout = buf
        ref_rohf = [scf.ROHF(mol).run().e_tot for mol in molc[:4]]
        jmf_ro = jointscf.JointSCF(mfs_rohf)
        jmf_ro.stdout = buf
        jmf_ro.verbose = 4
        jmf_ro.schedule = 'transferable'
        jmf_ro.max_workers = 1
        e_ro = jmf_ro.kernel()
        self.assertTrue(jmf_ro.converged.all())
        self.assertAlmostEqual(abs(e_ro - ref_rohf).max(), 0, 10)
        self.assertIn('falling back to sync', buf.getvalue())

if __name__ == "__main__":
    print("Full Tests for joint SCF")
    unittest.main()
