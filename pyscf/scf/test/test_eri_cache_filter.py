# Copyright 2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Opt-in membership filter regression; never relax ERI or matrix thresholds."""
import unittest
from unittest import mock
import numpy as np
from pyscf import lib
from pyscf.scf import eri_cache
from pyscf.scf.test import test_eri_cache as original


class FilterInherited(original.KnownValues):
    def setUp(self):
        original_context = eri_cache.cached_eri
        def filtered(*args, **kwargs):
            kwargs.setdefault('fast_lookup', True)
            kwargs.setdefault('saturation_filter', True)
            return original_context(*args, **kwargs)
        patcher = mock.patch.object(eri_cache, 'cached_eri', filtered)
        patcher.start()
        self.addCleanup(patcher.stop)


class FilterTests(unittest.TestCase):
    def test_signed_batches_budgets_and_reset(self):
        rng = np.random.default_rng(109)
        for threads in (1, 2):
            lib.num_threads(threads)
            mf = original.make_mf()
            raw = rng.normal(size=(2, mf.mol.nao, mf.mol.nao))
            dm = raw + raw.transpose(0, 2, 1)
            reference = mf.get_jk(dm=dm)
            for budget in (.125, 1, 8):
                with self.subTest(threads=threads, budget=budget):
                    with eri_cache.cached_eri(mf, max_mb=budget, min_primitive_work=0,
                                              fast_lookup=True, saturation_filter=True) as cache:
                        for scale in (1., -.1, .01):
                            for actual, ref in zip(mf.get_jk(dm=dm*scale), reference):
                                np.testing.assert_allclose(actual, ref*scale, atol=1e-9, rtol=1e-10)
                        before = cache.stats()['caches'][0]
                        self.assertGreater(before['saturation_filter']['bytes'], 0)
                        self.assertLessEqual(before['data_bytes'], before['allocated_bytes'])
                        self.assertLessEqual(before['allocated_bytes'], before['max_bytes'])
                        self.assertTrue(all(before['saturation_filter'][k] == 0 for k in
                                            ('queries', 'negatives', 'recompute_bypasses')))
                        cache.stats(diagnostics=True, reset=True)
                        mf.get_jk(dm=dm)
                        measured = cache.stats(diagnostics=False, reset=True)['caches'][0]
                        self.assertGreater(measured['hits'], 0)
                        self.assertGreater(measured['saturation_filter']['bytes'], 0)
                        mf.get_jk(dm=dm)
                        after = cache.stats()['caches'][0]
                        self.assertEqual(after['calls'], 0)
                        self.assertTrue(all(after['saturation_filter'][k] == 0 for k in
                                            ('queries', 'negatives', 'recompute_bypasses')))

    def test_default_does_not_allocate_filter(self):
        mf = original.make_mf()
        dm = mf.get_init_guess()
        with eri_cache.cached_eri(mf, max_mb=1, fast_lookup=True) as cache:
            mf.get_jk(dm=dm)
            self.assertEqual(cache.stats()['caches'][0]['saturation_filter']['bytes'], 0)

    def test_invalid_flags_and_old_native_library(self):
        mf = original.make_mf()
        for value in (1, None, 'yes'):
            with self.assertRaises(TypeError):
                with eri_cache.cached_eri(mf, fast_lookup=True, saturation_filter=value):
                    pass
        with self.assertRaises(ValueError):
            with eri_cache.cached_eri(mf, saturation_filter=True):
                pass
        with mock.patch.object(eri_cache, '_filter_create', None):
            with self.assertRaisesRegex(RuntimeError, 'rebuilding'):
                with eri_cache.cached_eri(mf, fast_lookup=True, saturation_filter=True):
                    pass
        self.assertNotIn('_eri_cache_scope', mf.__dict__)


if __name__ == '__main__':
    unittest.main()
