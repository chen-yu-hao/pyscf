"""Run without native PySCF: python test_ao_cache_unit.py path/to/ao_cache.py."""
import importlib.util
import sys
import unittest
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from unittest import mock
import numpy as np

PATH = sys.argv.pop(1) if len(sys.argv) > 1 and sys.argv[1].endswith('.py') else 'pyscf/dft/ao_cache.py'
spec = importlib.util.spec_from_file_location('ao_cache_under_test', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.mol = SimpleNamespace(_atm=np.array([[1, 0]], dtype='int32'),
                                  _bas=np.array([[1, 1]], dtype='int32'),
                                  _env=np.array([.1, .2]), cart=False)
        self.coords = np.arange(45, dtype=float).reshape(-1, 3) / 20
        self.calls = 0
        def evaluate(mol, coords, deriv=0, shls_slice=None, non0tab=None,
                     cutoff=None, out=None, verbose=None):
            self.calls += 1
            comp = (deriv+1)*(deriv+2)*(deriv+3)//6
            nao = 5 if shls_slice is None else 3
            x = np.arange(comp*nao*len(coords), dtype=float).reshape(comp, nao, len(coords))
            x += sum(float(a.sum()) for a in (mol._atm, mol._bas, mol._env))
            x += np.asarray(coords).sum() + float(mol.cart)
            x += 0 if cutoff is None else cutoff
            x += 0 if non0tab is None else non0tab.sum()
            value = x.transpose(0, 2, 1)
            if comp == 1:
                value = value[0]
            return module.AOBlockCache._copy_out(value, out)
        self.evaluate = evaluate
        self.cache = module.AOBlockCache(evaluate, 100000)

    def test_layout_alias_and_copy_out(self):
        for deriv in (0, 1, 2):
            with self.subTest(deriv=deriv):
                self.cache.clear()
                comp = (deriv+1)*(deriv+2)*(deriv+3)//6
                buf = np.zeros(comp*5*15 + 31)
                first = self.cache(self.mol, self.coords, deriv=deriv, out=buf)
                expected = first.copy()
                first[:] = -999
                hit = self.cache(self.mol, self.coords.copy(), deriv=deriv, out=buf)
                np.testing.assert_array_equal(hit, expected)
                self.assertTrue(np.shares_memory(hit, buf))
                self.assertTrue(hit.flags.f_contiguous if deriv == 0 else all(a.flags.f_contiguous for a in hit))
                hit[:] = -123
                independent = self.cache(self.mol, self.coords, deriv=deriv)
                np.testing.assert_array_equal(independent, expected)
                self.assertFalse(np.shares_memory(independent, buf))
                self.assertTrue(independent.flags.writeable)

    def test_input_keys(self):
        self.cache(self.mol, self.coords)
        for coords, kw in [(self.coords+.1, {}), (self.coords, {'deriv':1}),
                           (self.coords, {'cutoff':.2}), (self.coords, {'shls_slice':(0,1)}),
                           (self.coords, {'non0tab':np.ones((1,2), dtype='uint8')})]:
            before = self.calls
            value = self.cache(self.mol, coords, **kw)
            self.assertEqual(self.calls, before+1)
            np.testing.assert_array_equal(value, self.evaluate(self.mol, coords, **kw))
        before = self.calls
        self.cache(self.mol, self.coords)
        self.assertEqual(before, self.calls)

    def test_inplace_molecule_mutations(self):
        self.cache(self.mol, self.coords)
        for name in ('_atm', '_bas', '_env'):
            getattr(self.mol, name).flat[0] += 1
            value = self.cache(self.mol, self.coords)
            np.testing.assert_array_equal(value, self.evaluate(self.mol, self.coords))
        self.mol.cart = True
        self.cache(self.mol, self.coords)
        self.assertEqual(self.cache.snapshot()['invalidations'], 4)

    def test_scan_resistance_and_bound(self):
        cache = module.AOBlockCache(self.evaluate, 15*5*8)
        original = cache(self.mol, self.coords)
        for i in range(1, 8):
            cache(self.mol, self.coords+i)
        before = self.calls
        np.testing.assert_array_equal(cache(self.mol, self.coords), original)
        self.assertEqual(self.calls, before)
        self.assertEqual(cache.snapshot()['entries'], 1)
        self.assertLessEqual(cache.snapshot()['peak_bytes'], cache.max_bytes)
        cache.clear()
        self.assertEqual(cache.snapshot()['retained_bytes'], 0)
        self.assertEqual(cache.snapshot()['entries'], 0)

    def test_zero_budget_and_entry_bound(self):
        for cap in (0, 2):
            cache = module.AOBlockCache(self.evaluate, cap)
            cache(self.mol, self.coords)
            cache(self.mol, self.coords)
            self.assertEqual(cache.snapshot()['hits'], 0)
            self.assertEqual(cache.snapshot()['entries'], 0)
        cache = module.AOBlockCache(self.evaluate, 100000, max_entries=2)
        for i in range(9):
            cache(self.mol, self.coords+i)
        self.assertEqual(cache.snapshot()['entries'], 2)

    def test_bad_budget(self):
        for budget in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                module.AOBlockCache(self.evaluate, budget)

    def test_allocation_failure_preserves_result(self):
        expected = self.evaluate(self.mol, self.coords)
        # Supply out so the evaluator does not call np.array itself.
        buf = np.empty(75)
        with mock.patch.object(module.np, 'array', side_effect=MemoryError):
            value = self.cache(self.mol, self.coords, out=buf)
        np.testing.assert_array_equal(value, expected)
        self.assertEqual(self.cache.snapshot()['entries'], 0)

    def test_parallel_reads_do_not_duplicate_admission(self):
        with ThreadPoolExecutor(4) as pool:
            values = list(pool.map(lambda _: self.cache(self.mol, self.coords), range(16)))
        self.assertEqual(self.calls, 1)
        self.assertEqual(self.cache.snapshot()['hits'], 15)
        for value in values:
            np.testing.assert_array_equal(value, values[0])

    def test_mask_mutation_and_noncontiguous_coords(self):
        mask = np.ones((1, 2), dtype=np.uint8)
        self.cache(self.mol, self.coords, non0tab=mask)
        mask[0,0] = 0
        a = self.cache(self.mol, self.coords, non0tab=mask)
        b = self.evaluate(self.mol, self.coords, non0tab=mask)
        np.testing.assert_array_equal(a, b)
        view = np.empty((15,6))[:, ::2]
        view[:] = self.coords
        before = self.calls
        self.cache(self.mol, view, non0tab=mask)
        self.assertEqual(self.calls, before)

if __name__ == '__main__':
    unittest.main(verbosity=2)
