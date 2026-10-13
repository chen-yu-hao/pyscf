# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.

"""Memory and execution paths against independent, nonzero RCCSD equations."""
import unittest
from unittest import mock

import numpy as np
from pyscf import gto, lib, scf
from pyscf.cc import ccsd, rccsd


def setUpModule():
    global mf, old_threads
    old_threads = lib.num_threads()
    lib.num_threads(1)
    mol = gto.M(atom='N 0 0 0; H 0 0 .99; H .94 0 -.33; H -.47 .81 -.33',
                basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)


def tearDownModule():
    global mf
    del mf
    lib.num_threads(old_threads)


class KnownValues(unittest.TestCase):
    def check_update(self, budget, outcore=False, direct=False, incore=False,
                     async_io=True):
        solver = ccsd.CCSD(mf)
        solver.direct = direct
        solver.incore_complete = incore
        solver.async_io = async_io
        eris = ccsd._make_eris_outcore(solver) if outcore else solver.ao2mo()
        nocc, nvir = solver.nocc, solver.nmo - solver.nocc
        rng = np.random.default_rng(7201)
        t1 = rng.normal(scale=.02, size=(nocc, nvir))
        t2 = rng.normal(scale=.02, size=(nocc, nocc, nvir, nvir))
        t2 += t2.transpose(1, 0, 3, 2)
        reference = rccsd.RCCSD(mf)
        expected = reference.update_amps(t1, t2, reference.ao2mo())
        t1_saved, t2_saved = t1.copy(), t2.copy()
        solver.max_memory = budget
        disk_intermediates = []
        add_ovvv = ccsd._add_ovvv_

        def observe(*args):
            disk_intermediates.append(args[-1] is not None)
            return add_ovvv(*args)

        with mock.patch.object(lib, 'current_memory', return_value=(0., 0.)), \
                mock.patch.object(ccsd, '_add_ovvv_', side_effect=observe), \
                mock.patch.object(lib, 'einsum', wraps=lib.einsum) as contractions, \
                mock.patch.object(lib, 'call_in_background',
                                  wraps=lib.call_in_background) as background:
            actual = solver.update_amps(t1, t2, eris)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, atol=2e-13, rtol=2e-13)
        np.testing.assert_array_equal(t1, t1_saved)
        np.testing.assert_array_equal(t2, t2_saved)
        self.assertEqual(solver.async_io, async_io)
        self.assertEqual(solver.incore_complete, incore)
        batches = [call.args[1].shape[0] for call in contractions.call_args_list
                   if call.args[0] == 'ijcd,cdbk->ijbk']
        slices = sum(call.args[0] == 'jcd,cdbk->jbk'
                     for call in contractions.call_args_list)
        execution = [(call.args[0].__name__, call.kwargs['sync'])
                     for call in background.call_args_list
                     if call.args[0].__name__ in
                     ('load_ovvv', 'load_oovv', 'load_ovvo',
                      'update_wVooV', 'update_wVOov', 'update_t2')]
        return disk_intermediates, batches, slices, execution

    def test_resident_full_batch(self):
        disk, batches, slices, execution = self.check_update(4000)
        self.assertEqual(disk, [False])
        self.assertEqual(batches, [5])
        self.assertEqual(slices, 0)
        self.assertEqual(len(execution), 7)
        self.assertTrue(all(sync for _, sync in execution))

    def test_occupied_and_virtual_tail_blocks(self):
        for async_io in (False, True):
            with self.subTest(async_io=async_io):
                disk, batches, slices, execution = self.check_update(.037, async_io=async_io)
                self.assertEqual(disk, [True])
                self.assertEqual(batches, [2, 2, 1] * 3)
                self.assertEqual(slices, 0)
                self.assertTrue(all(sync == (not async_io) for _, sync in execution))

    def test_low_memory_scalar_fallback_and_explicit_incore(self):
        for incore in (False, True):
            with self.subTest(incore=incore):
                disk, batches, slices, _ = self.check_update(0, incore=incore)
                self.assertEqual(disk, [not incore])
                self.assertFalse(batches)
                self.assertEqual(slices, 15)

    def test_hdf5_integrals_keep_background_execution(self):
        disk, batches, _, execution = self.check_update(4000, outcore=True)
        self.assertEqual(disk, [False])
        self.assertEqual(batches, [5])
        self.assertEqual(execution[0], ('load_ovvv', True))
        self.assertTrue(all(not sync for _, sync in execution[1:]))

    def test_ao_direct(self):
        disk, batches, slices, _ = self.check_update(4000, direct=True)
        self.assertEqual(disk, [False])
        self.assertFalse(batches)
        self.assertEqual(slices, 0)


if __name__ == '__main__':
    unittest.main()
