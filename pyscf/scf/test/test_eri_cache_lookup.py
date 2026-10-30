# Copyright 2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Molecular/native acceptance for opt-in cache lookup, not a speed test."""
import unittest
from unittest import mock
import numpy as np
from pyscf import lib, scf
from pyscf.scf import eri_cache
from pyscf.scf.test import test_eri_cache as existing
make_mf = existing.make_mf


class FastLookupInherited(existing.KnownValues):
    """Re-run every existing assertion with the new policy explicitly enabled."""
    def setUp(self):
        original = eri_cache.cached_eri
        def fast(*args, **kwargs):
            kwargs.setdefault('fast_lookup', True)
            return original(*args, **kwargs)
        self.patcher = mock.patch.object(eri_cache, 'cached_eri', fast)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)


class LookupTests(unittest.TestCase):
    def test_probe_reduction_and_unchanged_matrices(self):
        lib.num_threads(1)
        mf = make_mf()
        dm = mf.get_init_guess()
        reference = mf.get_jk(dm=dm)
        measurements = []
        for fast in (False, True):
            with eri_cache.cached_eri(mf, max_mb=16, min_primitive_work=0,
                                      fast_lookup=fast) as cache:
                # Existing optimizer is attached, then start counters for cold fill.
                cache.stats(diagnostics=True, reset=True)
                cold = mf.get_jk(dm=dm)
                for a,b in zip(cold, reference):
                    np.testing.assert_allclose(a,b,atol=1e-9,rtol=1e-10)
                record = cache.stats(diagnostics=False, reset=True)['caches'][0]
                measurements.append(record['lookup']['probes'])
                self.assertEqual(record['lookup']['fast_lookup'], int(fast))
                warm = mf.get_jk(dm=.25*dm)
                for a,b in zip(warm, reference):
                    np.testing.assert_allclose(a,.25*b,atol=1e-9,rtol=1e-10)
                st = cache.stats()['caches'][0]
                self.assertEqual(st['calls'],0)
                self.assertEqual(st['lookup']['probes'],0)
                self.assertLessEqual(st['data_bytes'],st['allocated_bytes'])
                self.assertLessEqual(st['allocated_bytes'],st['max_bytes'])
        self.assertLess(measurements[1], measurements[0])

    def test_tiny_budget_and_admission_extremes(self):
        mf = make_mf(); dm=mf.get_init_guess(); reference=mf.get_jk(dm=dm)
        for work in (0,1,64,2**32-1):
            for budget in (.125,2):
                with self.subTest(work=work,budget=budget):
                    with eri_cache.cached_eri(mf, max_mb=budget,
                                              min_primitive_work=work, fast_lookup=True):
                        for factor in (1.,.2,-.1):
                            for a,b in zip(mf.get_jk(dm=factor*dm),reference):
                                np.testing.assert_allclose(a,factor*b,atol=1e-9,rtol=1e-10)

    def test_invalid_policy_and_old_library(self):
        mf=make_mf()
        for value in (1,'true',None):
            with self.assertRaises(TypeError):
                with eri_cache.cached_eri(mf,fast_lookup=value):pass
        with mock.patch.object(eri_cache,'_fast_create',None):
            with self.assertRaisesRegex(RuntimeError,'rebuilding'):
                with eri_cache.cached_eri(mf,fast_lookup=True):pass
        self.assertNotIn('_eri_cache_scope',mf.__dict__)


if __name__ == '__main__':
    unittest.main()
