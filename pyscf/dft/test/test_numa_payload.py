"""Native-library-independent tests of the exact NUMA payload protocol."""
import importlib.util
from multiprocessing.connection import Connection
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
import socket
import unittest
from unittest import mock
import numpy as np

# Load only this pure utility: the tests can also run before native libraries
# have been built, without importing pyscf.__init__.
_spec = importlib.util.spec_from_file_location(
    '_tested_numa_payload', Path(__file__).parents[1] / '_numa_payload.py')
payload = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(payload)


class Tagged(np.ndarray):
    pass


class PayloadTests(unittest.TestCase):
    def density(self, occ):
        rng = np.random.default_rng(13)
        dm = np.eye(12).view(Tagged)
        dm.mo_occ = np.asarray(occ)
        dm.mo_coeff = rng.normal(size=(12, len(occ)))
        return dm

    def test_drop_only_exact_zeros(self):
        dm = self.density([2., 0., 1e-20, -.5, 0., .25])
        got = payload._density_payload(dm)
        np.testing.assert_array_equal(got['mo_coeff'], dm.mo_coeff[:, [0, 2, 3, 5]])
        np.testing.assert_array_equal(got['mo_occ'], [2., 1e-20, -.5, .25])
        self.assertTrue(got['mo_coeff'].flags.c_contiguous)
        self.assertTrue(np.shares_memory(got['density'], dm))

    def test_no_input_mutation(self):
        dm = self.density([2., 0., 0., 2.])
        coeff, occ, density = dm.mo_coeff.copy(), dm.mo_occ.copy(), dm.copy()
        payload._density_payload(dm)
        np.testing.assert_array_equal(dm.mo_coeff, coeff)
        np.testing.assert_array_equal(dm.mo_occ, occ)
        np.testing.assert_array_equal(dm, density)

    def test_all_occupied_no_extra_copy(self):
        dm = self.density([2., -.1, .25])
        got = payload._density_payload(dm)
        self.assertIs(got['mo_coeff'], dm.mo_coeff)
        self.assertIs(got['mo_occ'], dm.mo_occ)

    def test_all_zero_preserves_shape(self):
        dm = self.density([0., 0., 0.])
        self.assertIs(payload._density_payload(dm)['mo_coeff'], dm.mo_coeff)

    def test_plain_density(self):
        dm = np.eye(3)
        got = payload._density_payload(dm)
        self.assertIsNone(got['mo_coeff'])
        self.assertIsNone(got['mo_occ'])
        np.testing.assert_array_equal(got['density'], dm)

    def test_nonstandard_tags_untouched(self):
        dm = self.density([2., 0.])
        dm.mo_occ = [2., 0.]
        got = payload._density_payload(dm)
        self.assertIs(got['mo_occ'], dm.mo_occ)
        self.assertIs(got['mo_coeff'], dm.mo_coeff)
        dm.mo_occ = np.array([2., 0., 0.])
        self.assertIs(payload._density_payload(dm)['mo_coeff'], dm.mo_coeff)

    def test_noncontiguous_coefficients(self):
        dm = self.density([2., 0., .5, 0.])
        dm.mo_coeff = dm.mo_coeff[::-1]
        got = payload._density_payload(dm)
        np.testing.assert_array_equal(got['mo_coeff'], dm.mo_coeff[:, [0, 2]])

    def test_orbital_density_and_gradient(self):
        dm = self.density([2., 0., .25, -.5, 0., 1e-20])
        got = payload._density_payload(dm)
        rng = np.random.default_rng(12)
        ao = rng.normal(size=(4, 31, 12))
        full = ao @ dm.mo_coeff
        small = ao @ got['mo_coeff']
        np.testing.assert_allclose((full[0]**2*dm.mo_occ).sum(axis=-1),
                                   (small[0]**2*got['mo_occ']).sum(axis=-1),
                                   rtol=1e-13, atol=1e-13)
        for c in range(1, 4):
            np.testing.assert_allclose((full[0]*full[c]*dm.mo_occ).sum(axis=-1),
                                       (small[0]*small[c]*got['mo_occ']).sum(axis=-1),
                                       rtol=1e-13, atol=1e-13)

    def test_serialize_once_for_many_workers(self):
        dm = self.density([2., 0., 0.])
        conns = [mock.Mock() for _ in range(3)]
        with mock.patch.object(ForkingPickler, 'dumps', wraps=ForkingPickler.dumps) as dumps:
            payload.send_payload(conns, dm)
            self.assertEqual(dumps.call_count, 1)
        messages = [c.send_bytes.call_args.args[0] for c in conns]
        self.assertTrue(all(x is messages[0] for x in messages))
        for message in messages:
            got = ForkingPickler.loads(message)
            np.testing.assert_array_equal(got['density'], dm)
            np.testing.assert_array_equal(got['mo_occ'], [2.])
        # Release exported BytesIO buffers before mock teardown on Python 3.13.
        for c in conns:
            c.reset_mock()
        del got, message, messages

    def test_connection_recv_wire_compatibility(self):
        a, b = socket.socketpair()
        left, right = Connection(a.detach()), Connection(b.detach())
        try:
            dm = self.density([0., 2., 0., .5])
            payload.send_payload([left], dm)
            got = right.recv()
            np.testing.assert_array_equal(got['density'], dm)
            np.testing.assert_array_equal(got['mo_coeff'], dm.mo_coeff[:, [1, 3]])
        finally:
            left.close()
            right.close()


if __name__ == '__main__':
    unittest.main()
