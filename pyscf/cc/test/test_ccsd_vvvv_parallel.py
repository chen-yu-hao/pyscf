# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.

"""Exact packed contractions and lifetime of the private parallel scratch."""
import copy
import ctypes
import gc
import pickle
import tempfile
import threading
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import h5py
import numpy as np
from pyscf import cc, gto, lib, scf
from pyscf.cc import ccsd


class KnownValues(unittest.TestCase):
    def setUp(self):
        self.old_threads = lib.num_threads()
        lib.num_threads(1)
        self.mol = gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g', verbose=0)
        self.cc = cc.CCSD(scf.RHF(self.mol))
        self.cc.max_memory = 4000
        self.rng = np.random.default_rng(11029)

    def tearDown(self):
        ccsd._clear_vvvv_workspace(self.cc)
        lib.num_threads(self.old_threads)

    def arrays(self, na=33, nb=27, nx=3):
        packed = self.rng.normal(size=(na*(na+1)//2, nb*(nb+1)//2))
        t2 = self.rng.normal(size=(nx, na, nb))
        a = lib.square_mat_in_trilu_indices(na)
        b = lib.square_mat_in_trilu_indices(nb)
        full = packed[a[:, :, None, None], b[None, None, :, :]]
        expected = np.einsum('xcd,acbd->xab', t2, full)
        return packed, t2, expected

    def contract(self, packed, t2, solver=None, out=None):
        return ccsd._contract_s4vvvv_t2(
            self.cc if solver is None else solver, self.mol, packed, t2, out)

    def test_rectangular_threaded_contractions(self):
        for threads in (1, 8):
            lib.num_threads(threads)
            for na, nb, nx in ((1, 1, 1), (9, 13, 2), (33, 27, 3), (25, 33, 2)):
                with self.subTest(threads=threads, shape=(nx, na, nb)):
                    packed, t2, expected = self.arrays(na, nb, nx)
                    packed.flags.writeable = False
                    backing = np.full(t2.size + 2, 123456.)
                    out = backing[1:-1].reshape(t2.shape)
                    actual = self.contract(packed, t2, out=out)
                    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-13)
                    self.assertTrue(np.shares_memory(actual, out))
                    np.testing.assert_array_equal(backing[[0, -1]], 123456.)

    def test_fused_outcore_and_strided_input(self):
        packed, t2, expected = self.arrays(17, 13, 2)
        # Both strides are non-contiguous, without changing logical elements.
        storage = np.empty((packed.shape[0]*2, packed.shape[1]*2))
        storage[::2, ::2] = packed
        with tempfile.TemporaryFile() as stream, h5py.File(stream, 'w') as handle:
            dataset = handle.create_dataset('vvvv', data=packed)
            for async_io in (False, True):
                self.cc.async_io = async_io
                for source in (dataset, storage[::2, ::2]):
                    with self.subTest(async_io=async_io, source=type(source)):
                        np.testing.assert_allclose(self.contract(source, t2), expected,
                                                   atol=3e-12, rtol=3e-13)

    def test_resident_path_is_used(self):
        packed, t2, expected = self.arrays()
        for threads in (1, 8):
            lib.num_threads(threads)
            result = ccsd._contract_s4vvvv_incore(self.cc, packed, t2, None)
            self.assertIsNotNone(result)
            np.testing.assert_allclose(result, expected, atol=3e-12, rtol=3e-13)

    def test_t2_layouts_and_empty_sectors(self):
        packed, t2, expected = self.arrays()
        for x in (np.asfortranarray(t2), t2.copy()):
            x.flags.writeable = False
            np.testing.assert_allclose(self.contract(packed, x), expected,
                                       atol=3e-12, rtol=3e-13)
        for shape in ((0, 33, 27), (2, 0, 27), (2, 33, 0)):
            np.testing.assert_array_equal(self.contract(packed, np.empty(shape)),
                                          np.zeros(shape))

    def test_copy_scanner_pickle_and_reset(self):
        packed, t2, expected = self.arrays()
        self.contract(packed, t2)
        original = ccsd._vvvv_workspaces[self.cc]
        for clone in (self.cc.copy(), copy.deepcopy(self.cc), self.cc.as_scanner(),
                      pickle.loads(pickle.dumps(self.cc))):
            self.assertNotIn(clone, ccsd._vvvv_workspaces)
            np.testing.assert_allclose(self.contract(packed, t2, clone), expected,
                                       atol=3e-12, rtol=3e-13)
            self.assertIsNot(ccsd._vvvv_workspaces[clone], original)
            clone.reset()
            self.assertNotIn(clone, ccsd._vvvv_workspaces)
            self.assertIs(ccsd._vvvv_workspaces[self.cc], original)
        self.cc.reset()
        self.assertNotIn(self.cc, ccsd._vvvv_workspaces)

    def test_owner_collection_releases_scratch(self):
        packed, t2, _ = self.arrays()
        other = self.cc.copy()
        self.contract(packed, t2, other)
        owner = weakref.ref(other)
        buffer = weakref.ref(ccsd._vvvv_workspaces[other].buffer)
        del other
        gc.collect()
        self.assertIsNone(owner())
        self.assertIsNone(buffer())

    def check_concurrent_call(self, clear):
        packed, t2, expected = self.arrays()
        entered, resume = threading.Event(), threading.Event()
        native = ccsd._ccsd.libcc.CCcontract_s4vvvv

        def pause_once(*args):
            if not entered.is_set():
                entered.set()
                if not resume.wait(20):
                    raise RuntimeError('Timed out waiting for concurrent caller')
            return native(*args)

        def first_call():
            lib.num_threads(1)
            return self.contract(packed, t2)

        with ThreadPoolExecutor(max_workers=1) as executor, \
                mock.patch.object(ccsd._ccsd.libcc, 'CCcontract_s4vvvv',
                                  side_effect=pause_once) as calls:
            future = executor.submit(first_call)
            try:
                self.assertTrue(entered.wait(20))
                buffer = weakref.ref(ccsd._vvvv_workspaces[self.cc].buffer)
                if clear:
                    ccsd._clear_vvvv_workspace(self.cc)
                    gc.collect()
                    self.assertIsNotNone(buffer())
                actual = self.contract(packed, t2)
                np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-13)
                self.assertEqual(calls.call_count, 2 if clear else 1)
                if clear:
                    self.assertIsNot(ccsd._vvvv_workspaces[self.cc].buffer, buffer())
            finally:
                resume.set()
            np.testing.assert_allclose(future.result(timeout=20), expected,
                                       atol=3e-12, rtol=3e-13)

    def test_busy_workspace_uses_independent_fallback(self):
        self.check_concurrent_call(clear=False)

    def test_clear_during_active_call_keeps_buffer_alive(self):
        self.check_concurrent_call(clear=True)

    def test_memory_budget_and_allocation_failure(self):
        packed, t2, expected = self.arrays()
        required = ((8 * t2.shape[-1])**2 + t2.size) * 8
        with mock.patch.object(lib, 'current_memory', return_value=(50., 50.)), \
                mock.patch.object(ccsd._ccsd.libcc, 'CCcontract_s4vvvv',
                                  wraps=ccsd._ccsd.libcc.CCcontract_s4vvvv) as native:
            self.cc.max_memory = 50 + (required + t2.nbytes - 64) / 1e6
            # Supplied output pages also need an explicit reservation.
            actual = self.contract(packed, t2, out=np.empty_like(t2))
            self.assertEqual(native.call_count, 0)
            np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-13)
            self.cc.max_memory = 50 + (required + t2.nbytes + 64) / 1e6
            actual = self.contract(packed, t2)
            self.assertEqual(native.call_count, 1)
            np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-13)
        self.cc.max_memory = 4000
        ccsd._clear_vvvv_workspace(self.cc)
        empty = np.empty

        def fail_workspace(shape, *args, **kwargs):
            if shape == required // 8:
                raise MemoryError('injected workspace failure')
            return empty(shape, *args, **kwargs)

        with mock.patch.object(ccsd.numpy, 'empty', side_effect=fail_workspace):
            np.testing.assert_allclose(self.contract(packed, t2), expected,
                                       atol=3e-12, rtol=3e-13)
        self.assertIsNone(ccsd._vvvv_workspaces[self.cc].buffer)

    def test_native_insufficient_scratch_does_not_write(self):
        packed, t2, _ = self.arrays()
        out = np.full_like(t2, 7.)
        status = ccsd._ccsd.libcc.CCcontract_s4vvvv(
            out.ctypes.data_as(ctypes.c_void_p),
            packed.ctypes.data_as(ctypes.c_void_p),
            t2.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(3),
            ctypes.c_int(33), ctypes.c_int(27), ctypes.c_int(8),
            ctypes.c_void_p(), ctypes.c_size_t(0))
        self.assertEqual(status, 1)
        np.testing.assert_array_equal(out, 7.)

    def test_ao_direct(self):
        t2 = self.rng.normal(size=(3, self.mol.nao, self.mol.nao))
        expected = np.einsum('xcd,acbd->xab', t2, self.mol.intor('int2e'))
        np.testing.assert_allclose(self.contract(None, t2), expected,
                                   atol=3e-12, rtol=3e-13)


if __name__ == '__main__':
    unittest.main()
