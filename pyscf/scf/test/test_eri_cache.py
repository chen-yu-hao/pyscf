# Copyright 2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
import concurrent.futures
import unittest
import numpy as np
from pyscf import gto, scf, lib
from pyscf.scf import eri_cache


def make_mf():
    mol = gto.M(atom=[('H', (i*1.3, .4*(i%2), 0)) for i in range(6)],
                basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol)
    mf._is_mem_enough = lambda: False  # test direct route, not an accuracy option
    return mf


class KnownValues(unittest.TestCase):
    def test_direct_same_and_changed_density(self):
        rng = np.random.default_rng(71)
        for threads in (1, 2, 4):
            lib.num_threads(threads)
            mf = make_mf()
            raw = rng.normal(size=(2, mf.mol.nao, mf.mol.nao))
            for hermi in (0, 1):
                dm = raw if hermi == 0 else raw+raw.transpose(0, 2, 1)
                refs = [mf.get_jk(dm=x, hermi=hermi) for x in (dm, dm*.25)]
                with eri_cache.cached_eri(mf, max_mb=8) as cache:
                    actual = mf.get_jk(dm=dm, hermi=hermi)
                    cold = cache.stats()
                    self.assertTrue(cold['caches'][0]['enabled'])
                    self.assertEqual(cold['caches'][0]['calls'], 0)
                    cache.stats(diagnostics=True, reset=True)
                    for x, ref in zip((dm, dm*.25), refs):
                        actual = mf.get_jk(dm=x, hermi=hermi)
                        for a, b in zip(actual, ref):
                            np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-10)
                    st = cache.stats()['caches'][0]
                    self.assertGreater(st['hits'], 0)
                    self.assertLessEqual(st['allocated_bytes'], st['max_bytes'])
                    self.assertLessEqual(st['data_bytes'], st['allocated_bytes'])
                self.assertNotIn('init_direct_scf', mf.__dict__)

    def test_range_and_separate_j_k(self):
        mf = make_mf()
        dm = mf.get_init_guess()
        refs = [mf.get_jk(dm=dm, omega=w) for w in (None, .2, -.2)]
        with eri_cache.cached_eri(mf, max_mb=4):
            for w, ref in zip((None, .2, -.2), refs):
                for a, b in zip(mf.get_jk(dm=dm, omega=w), ref):
                    np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-10)
            np.testing.assert_allclose(mf.get_jk(dm=dm, with_k=False)[0], refs[0][0], atol=1e-9, rtol=1e-10)
            np.testing.assert_allclose(mf.get_jk(dm=dm, with_j=False)[1], refs[0][1], atol=1e-9, rtol=1e-10)

    def test_exception_zero_budget_nested(self):
        mf = make_mf()
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with eri_cache.cached_eri(mf, max_mb=0) as c:
                mf.get_jk(dm=mf.get_init_guess())
                self.assertEqual(c.stats()['caches'], [])
                with self.assertRaises(RuntimeError):
                    with eri_cache.cached_eri(mf):
                        pass
                raise RuntimeError('intentional')
        self.assertNotIn('init_direct_scf', mf.__dict__)
        self.assertNotIn('_eri_cache_scope', mf.__dict__)
        for budget in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                with eri_cache.cached_eri(mf, max_mb=budget):
                    pass

    def test_two_independent_calculations(self):
        def one(shift):
            lib.num_threads(1)
            mf = make_mf()
            dm = mf.get_init_guess()*shift
            ref = mf.get_jk(dm=dm)
            with eri_cache.cached_eri(mf, max_mb=4) as c:
                mf.get_jk(dm=dm)
                c.stats(diagnostics=True)
                for a, b in zip(mf.get_jk(dm=dm), ref):
                    np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-10)
                self.assertGreater(c.stats()['caches'][0]['hits'], 0)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            list(pool.map(one, (1., .37)))


if __name__ == '__main__':
    unittest.main()
