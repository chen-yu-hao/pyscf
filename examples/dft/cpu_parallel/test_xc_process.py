"""Numerical checks for the optional CPU worker example."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Capture the allocation before NumPy/OpenMP can narrow the main thread.
ALLOCATION = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else []

import numpy as np
from pyscf import dft, gto, lib
from xc_process import PartitionedXC
from phase_threads import limit_xc_threads


@unittest.skipUnless(len(ALLOCATION) >= 3, 'Requires three allocated CPUs on Linux')
class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.mol = gto.M(atom='O 0 0 0; H 0 0 .96; H 0 .93 -.23',
                         basis='cc-pvdz', verbose=0)
        self.grids = dft.gen_grid.Grids(self.mol)
        self.grids.level = 1
        self.grids.build(with_non0tab=True)
        mf = dft.RKS(self.mol)
        _, coeff = mf.eig(mf.get_hcore(), mf.get_ovlp())
        occ = np.zeros(coeff.shape[1])
        occ[:self.mol.nelectron // 2] = 2
        dm = (coeff * occ) @ coeff.T
        self.dm = lib.tag_array(dm, mo_coeff=coeff, mo_occ=occ)
        self.ni = dft.numint.NumInt()

    def test_updated_densities_and_functionals(self):
        for xc in ('LDA,VWN', 'PBE', 'MN15'):
            with self.subTest(xc=xc), PartitionedXC(
                    self.mol, self.grids, xc,
                    [(ALLOCATION[:1], None), (ALLOCATION[1:3], None)],
                    allowed_cpus=ALLOCATION, ni_cutoff=self.ni.cutoff,
                    log_prefix=str(Path(self.directory.name) / xc.replace(',', '-'))) as pool:
                scaled = lib.tag_array(np.asarray(self.dm) * .9,
                                       mo_coeff=self.dm.mo_coeff,
                                       mo_occ=self.dm.mo_occ * .9)
                for dm in (np.asarray(self.dm), self.dm, scaled):
                    expected = self.ni.nr_rks(self.mol, self.grids, xc, dm, hermi=1)
                    actual = pool.evaluate(dm)
                    for x, y in zip(actual, expected):
                        np.testing.assert_allclose(x, y, atol=1e-10, rtol=1e-11)
                self.assertEqual(pool.cuts[1], (len(self.grids.weights) // 3) // 56 * 56)

    def test_no_numactl_and_allocation_guard(self):
        with mock.patch('xc_process.shutil.which', return_value=None), PartitionedXC(
                self.mol, self.grids, 'MN15', [(ALLOCATION[:2], None)],
                allowed_cpus=ALLOCATION, ni_cutoff=self.ni.cutoff,
                log_prefix=str(Path(self.directory.name) / 'fallback')) as pool:
            expected = self.ni.nr_rks(self.mol, self.grids, 'MN15', self.dm, hermi=1)
            actual = pool.evaluate(self.dm)
            for x, y in zip(actual, expected):
                np.testing.assert_allclose(x, y, atol=1e-10, rtol=1e-11)
        with self.assertRaisesRegex(ValueError, 'allocation'):
            PartitionedXC(self.mol, self.grids, 'MN15', [(ALLOCATION[:2], None)],
                          allowed_cpus=ALLOCATION[:1])

    def test_phase_thread_limit_restores_on_error(self):
        mf = dft.RKS(self.mol)
        seen = []
        def failed(*args, **kwargs):
            seen.append(lib.num_threads())
            raise RuntimeError('test failure')
        mf._numint.nr_rks = failed
        with lib.with_omp_threads(3):
            limit_xc_threads(mf, 1)
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                mf._numint.nr_rks()
            self.assertEqual(lib.num_threads(), 3)
        self.assertEqual(seen, [1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
