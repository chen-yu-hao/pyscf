# Copyright 2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Optional TBLIS compact triples contractions and workspace accounting."""

import sys
import unittest
from unittest.mock import patch

import numpy as np

from pyscf import lib
from pyscf.cc import _uccsdt_t3 as fused
from pyscf.cc.test import test_uccsdt_t3_fused as reference

try:
    import pytblis
except ImportError:
    pytblis = None


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reference.KnownValues.setUpClass()
        cls.fixture = reference.KnownValues()

    def setUp(self):
        self.threads = lib.num_threads()
        lib.num_threads(1)
        if pytblis is not None:
            self.tblis_threads = pytblis.get_num_threads()
            pytblis.set_num_threads(1)

    def tearDown(self):
        lib.num_threads(self.threads)
        if pytblis is not None:
            pytblis.set_num_threads(self.tblis_threads)

    def assert_blocks_close(self, actual, expected):
        for block, (a, b) in enumerate(zip(actual, expected)):
            with self.subTest(block=block):
                np.testing.assert_allclose(a, b, atol=2e-12, rtol=2e-12)

    @unittest.skipIf(pytblis is None, 'pytblis is optional')
    def test_all_spin_blocks_and_small_tiles(self):
        mycc, _, tamps, imds = self.fixture._case(True)
        mycc.einsum_backend = 'numpy'
        expected = reference._full_reference(mycc, imds, tamps)
        self.assert_blocks_close(reference._legacy_reference(mycc, imds, tamps), expected)
        full_shapes = [a.shape for a in mycc.tamps_tri2full(tamps[2])]
        mycc.einsum_backend = 'pytblis'
        for threads in (1, 4):
            lib.num_threads(threads)
            pytblis.set_num_threads(threads)
            for budget in (None, 16000):
                with self.subTest(threads=threads, budget=budget):
                    plan = fused.workspace_plan(mycc, imds, *tamps[1:], workspace_bytes=budget)
                    partial = False
                    for block, info in plan['blocks'].items():
                        self.assertLessEqual(info['work_bytes'] + info['cache_bytes'],
                                             plan['budget_bytes'])
                        if budget is not None:
                            partial |= any(n % step for n, step in
                                           zip(full_shapes[block], info['shape']))
                    actual = fused.compute_r3(mycc, imds, *tamps[1:], workspace_bytes=budget)
                    self.assert_blocks_close(actual, expected)
                    if budget is not None:
                        self.assertTrue(partial)

    @unittest.skipIf(pytblis is None, 'pytblis is optional')
    def test_readonly_strided_operands(self):
        mycc, _, tamps, imds = self.fixture._case(True)
        mycc.einsum_backend = 'numpy'
        expected = reference._full_reference(mycc, imds, tamps)
        snapshots = {}
        for index, (name, a) in enumerate(vars(imds).items()):
            if isinstance(a, np.ndarray):
                a = np.asfortranarray(a) if index % 2 else a[::-1].copy()[::-1]
                a.flags.writeable = False
                setattr(imds, name, a)
                snapshots[name] = a.copy()
        inputs = [[a[::-1].copy()[::-1] for a in rank] for rank in tamps[1:]]
        for rank in inputs:
            for a in rank:
                a.flags.writeable = False
        mycc.einsum_backend = 'pytblis'
        actual = fused.compute_r3(mycc, imds, *inputs, workspace_bytes=16000)
        self.assert_blocks_close(actual, expected)
        for name, saved in snapshots.items():
            np.testing.assert_array_equal(getattr(imds, name), saved)
        for actual_rank, saved_rank in zip(inputs, tamps[1:]):
            for actual, saved in zip(actual_rank, saved_rank):
                np.testing.assert_array_equal(actual, saved)

    @unittest.skipIf(pytblis is None, 'pytblis is optional')
    def test_planner_reserves_outputs_for_spin_reuse(self):
        mycc, _, tamps, imds = self.fixture._case(False)
        output_bytes = sum(a.nbytes for a in tamps[2])
        mycc.max_memory = 100.1
        with patch.object(fused.lib, 'current_memory', return_value=(100., 0.)):
            mycc.einsum_backend = 'numpy'
            numpy_plan = fused.workspace_plan(mycc, imds, *tamps[1:])
            mycc.einsum_backend = 'pytblis'
            plan = fused.workspace_plan(mycc, imds, *tamps[1:])
            subset = fused.workspace_plan(mycc, imds, *tamps[1:], spin_blocks=(0, 1))
            self.assertEqual(subset['budget_bytes'], plan['budget_bytes'])
            self.assertEqual(numpy_plan['budget_bytes'], plan['budget_bytes'])
            self.assertLessEqual(plan['budget_bytes'] + output_bytes, 100000)
            for info in plan['blocks'].values():
                self.assertLessEqual(info['work_bytes'] + info['cache_bytes'], plan['budget_bytes'])
            # A direct TBLIS contraction should fit larger tiles within the
            # same cap than a backend that needs result/transpose copies.
            self.assertGreater(sum(np.prod(info['shape']) for info in plan['blocks'].values()),
                               sum(np.prod(info['shape']) for info in numpy_plan['blocks'].values()))
            mycc.max_memory = 100. + output_bytes / 2e6
            with self.assertRaises(MemoryError):
                fused.workspace_plan(mycc, imds, *tamps[1:], spin_blocks=(0, 1))

    def test_missing_tblis_uses_numpy_workspace_and_residuals(self):
        mycc, _, tamps, imds = self.fixture._case(False)
        mycc.einsum_backend = 'numpy'
        expected = reference._full_reference(mycc, imds, tamps)
        numpy_plan = fused.workspace_plan(mycc, imds, *tamps[1:], workspace_bytes=64000)
        mycc.einsum_backend = 'pytblis'
        with patch.dict(sys.modules, {'pytblis': None}):
            plan = fused.workspace_plan(mycc, imds, *tamps[1:], workspace_bytes=64000)
            self.assertEqual(plan, numpy_plan)
            actual = fused.compute_r3(mycc, imds, *tamps[1:], workspace_bytes=64000)
        self.assert_blocks_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
