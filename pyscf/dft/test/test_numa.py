"""Selection, invalidation and lifecycle checks for default NUMA RKS."""
import os
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
from pyscf import dft, gto, lib, scf
from pyscf.dft import numa


class AutoNumaTests(unittest.TestCase):
    def setUp(self):
        self.mol = gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g', verbose=0)
        self.mf = dft.RKS(self.mol)
        self.mf.grids.level = 0
        self.mf.grids.build(with_non0tab=True)
        self.dm = np.eye(self.mol.nao)
        self.value = (2., -1., np.eye(self.mol.nao))
        self.native = mock.Mock(return_value=self.value)
        self.mf._numint.nr_rks = self.native
        self.pool = mock.Mock()
        self.pool.evaluate.return_value = self.value
        self.addCleanup(numa.close, self.mf)

    def invoke(self):
        return numa.nr_rks(self.mf, self.mol, self.dm, 1, 3000)

    def calibration(self, candidate_seconds=2., setup=1.):
        stack = __import__('contextlib').ExitStack()
        stack.enter_context(mock.patch.dict(os.environ, {'PYSCF_NUMA': 'auto'}))
        stack.enter_context(mock.patch.object(numa, '_supported', return_value=True))
        stack.enter_context(mock.patch.object(numa, '_partitions', return_value=([([0, 1], 0), ([2, 3], 1)], [0, 1, 2, 3])))
        stack.enter_context(mock.patch.object(numa.shutil, 'which', return_value='/numactl'))
        stack.enter_context(mock.patch.object(numa.subprocess, 'run'))
        stack.enter_context(mock.patch.object(numa, 'PartitionedXC', return_value=self.pool))
        stack.enter_context(mock.patch.object(numa, '_memory_headroom', return_value=10000))
        stack.enter_context(mock.patch.object(numa.time, 'perf_counter', side_effect=[0, 10, 10, 10+setup, 10+setup, 10+setup+candidate_seconds]))
        return stack

    def test_success_reuses_workers(self):
        with self.calibration():
            self.invoke()
        with mock.patch.object(numa, '_supported', return_value=True):
            self.invoke()
        self.assertEqual(self.native.call_count, 1)
        self.assertEqual(self.pool.evaluate.call_count, 2)

    def test_slow_or_unamortized_keeps_native(self):
        for elapsed, setup in ((9, 1), (2, 100)):
            with self.subTest(elapsed=elapsed, setup=setup):
                numa.close(self.mf)
                with self.calibration(elapsed, setup):
                    self.invoke()
                self.assertTrue(numa._states[self.mf].disabled)
                self.assertIsNone(numa._states[self.mf].pool)

    def test_wrong_matrix_and_nan_rejected(self):
        for value in (np.eye(2) * 2, np.full((2, 2), np.nan)):
            numa.close(self.mf)
            self.pool.evaluate.return_value = (2., -1., value)
            with self.calibration():
                result = self.invoke()
            np.testing.assert_array_equal(result[2], self.value[2])
            self.assertTrue(numa._states[self.mf].disabled)

    def test_dead_worker_uses_native(self):
        with self.calibration():
            self.invoke()
        self.pool.evaluate.side_effect = EOFError('worker exited')
        with mock.patch.object(numa, '_supported', return_value=True):
            result = self.invoke()
        np.testing.assert_array_equal(result[2], self.value[2])
        self.assertTrue(numa._states[self.mf].disabled)
        self.pool.close.assert_called()

    def test_failed_binding_uses_native(self):
        with self.calibration(), mock.patch.object(numa.subprocess, 'run', side_effect=OSError('binding denied')):
            result = self.invoke()
        np.testing.assert_array_equal(result[2], self.value[2])
        self.pool.evaluate.assert_not_called()

    def test_low_memory_does_not_start_workers(self):
        with self.calibration(), mock.patch.object(numa, '_memory_headroom', return_value=100):
            self.invoke()
        self.pool.evaluate.assert_not_called()

    def test_switch_off_closes_active_pool(self):
        with self.calibration():
            self.invoke()
        with mock.patch.dict(os.environ, {'PYSCF_NUMA': 'off'}):
            self.invoke()
        self.assertNotIn(self.mf, numa._states)
        self.pool.close.assert_called()

    def test_reset_and_exception_cleanup(self):
        with self.calibration():
            self.invoke()
        self.mf.reset()
        self.assertNotIn(self.mf, numa._states)
        self.mf.grids.build(with_non0tab=True)
        with self.calibration():
            self.invoke()
        with mock.patch.object(scf.hf.SCF, 'scf', side_effect=RuntimeError('interrupted')):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                self.mf.kernel()
        self.assertNotIn(self.mf, numa._states)

    def test_grid_and_environment_mutation_invalidate(self):
        before = numa._signature(self.mf, self.mol)
        self.mf.grids.weights[0] += .1
        self.assertNotEqual(before, numa._signature(self.mf, self.mol))
        before = numa._signature(self.mf, self.mol)
        self.mol._env[-1] += .01
        self.assertNotEqual(before, numa._signature(self.mf, self.mol))

    def test_custom_small_and_uks_declined(self):
        self.assertFalse(numa._supported(self.mf, self.mol, self.dm, 1))
        self.assertFalse(numa._supported(dft.UKS(self.mol), self.mol, self.dm, 1))
        self.assertFalse(numa._supported(self.mf, self.mol, self.dm.astype(complex), 1))

    def test_builtin_large_rks_eligibility(self):
        molecule = gto.M(atom=[('H', (i * 1.4, 0, 0)) for i in range(20)], basis='cc-pvtz', verbose=0)
        mf = dft.RKS(molecule)
        mf.xc = 'MN15'
        mf.grids.weights = np.ones(100000)
        dm = lib.tag_array(np.eye(molecule.nao), mo_coeff=np.eye(molecule.nao), mo_occ=np.ones(molecule.nao))
        self.assertTrue(numa._supported(mf, molecule, dm, 1))
        mf.converged = True
        self.assertFalse(numa._supported(mf, molecule, dm, 1))
        mf.converged = False
        with mock.patch.dict(numa.libxc._CUSTOM_FUNC_R, {'changed': object()}):
            self.assertFalse(numa._supported(mf, molecule, dm, 1))
        with mock.patch.object(numa.libxc, '__name__', 'pyscf.dft.xcfun'):
            self.assertFalse(numa._supported(mf, molecule, dm, 1))
        mf._numint.eval_xc = lambda *args: None
        self.assertFalse(numa._supported(mf, molecule, dm, 1))


class ResourceTests(unittest.TestCase):
    def test_ancestor_cgroup_limits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'proc/self').mkdir(parents=True)
            (root / 'proc/self/cgroup').write_text('0::/parent/job\n')
            (root / 'proc/meminfo').write_text('MemAvailable: 8000000 kB\n')
            cg = root / 'sys/fs/cgroup'
            (cg / 'parent/job').mkdir(parents=True)
            (cg / 'cgroup.controllers').write_text('cpu memory')
            (cg / 'cpu.max').write_text('max 100000')
            (cg / 'parent/cpu.max').write_text('250000 100000')
            (cg / 'parent/job/cpu.max').write_text('800000 100000')
            (cg / 'parent/memory.max').write_text('2000000000')
            (cg / 'parent/memory.current').write_text('1000000000')
            with mock.patch.object(numa, 'Path', side_effect=lambda p: root / str(p).lstrip('/')):
                self.assertEqual(numa._quota(), 2.5)
                self.assertEqual(numa._memory_headroom(), 750.)

    def test_v1_cpu_quota_declines(self):
        with mock.patch.object(numa.Path, 'read_text', return_value='2:cpu,cpuacct:/job\n'):
            self.assertEqual(numa._quota(), 0)

    def test_mpi_rank_declines(self):
        with mock.patch.dict(os.environ, {'OMPI_COMM_WORLD_SIZE': '2'}):
            self.assertEqual(numa._partitions(), ([], []))


if __name__ == '__main__':
    unittest.main(verbosity=2)
