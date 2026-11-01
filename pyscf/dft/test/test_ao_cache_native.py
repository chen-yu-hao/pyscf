"""Native cache integration tests; no numerical tolerance is relaxed."""
import os
import unittest
from unittest import mock
import numpy as np
from pyscf import dft, gto
from pyscf.dft import numint
from pyscf.dft.ao_cache import AOBlockCache, cached_ao


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.mol = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587',
                         basis='def2-svp', verbose=0)

    def test_native_derivatives_masks_slices_buffers(self):
        rng = np.random.default_rng(88)
        coords = rng.normal(size=(120,3))
        mask = numint.make_mask(self.mol, coords)
        cache = AOBlockCache(numint.eval_ao, 8*1024**2)
        for deriv in (0,1,2):
            for shells in (None, (0,3)):
                expected = numint.eval_ao(self.mol, coords, deriv, shells, mask, cutoff=1e-14)
                out = np.empty(expected.size + 8)
                first = cache(self.mol, coords, deriv, shells, mask, cutoff=1e-14, out=out)
                np.testing.assert_array_equal(first, expected)
                first[:] = 7
                again = cache(self.mol, coords, deriv, shells, mask, cutoff=1e-14, out=out)
                np.testing.assert_array_equal(again, expected)
        self.assertEqual(cache.snapshot()['hits'], 6)

    def test_xc_same_density_and_context_cleanup(self):
        for xc in ('LDA,VWN','PBE','MN15'):
            mf = dft.RKS(self.mol)
            mf.xc, mf.grids.level = xc, 1
            mf.grids.build(with_non0tab=True)
            dm = mf.get_init_guess()
            expected = mf._numint.nr_rks(self.mol, mf.grids, xc, dm)
            with mock.patch.dict(os.environ, PYSCF_NUMA='off'):
                with cached_ao(mf, 64) as cache:
                    for _ in range(2):
                        result = mf._numint.nr_rks(self.mol, mf.grids, xc, dm)
                        for a,b in zip(expected, result):
                            np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12)
                    self.assertGreater(cache.snapshot()['hits'],0)
            self.assertIs(mf._numint.eval_ao,numint.eval_ao)
            self.assertEqual(cache.snapshot()['retained_bytes'],0)

    def test_scf_and_exception_restoration(self):
        mf = dft.RKS(self.mol)
        mf.xc, mf.grids.level = 'MN15', 1
        with mock.patch.dict(os.environ, PYSCF_NUMA='off'):
            with self.assertRaisesRegex(RuntimeError,'test cleanup'):
                with cached_ao(mf, 32):
                    raise RuntimeError('test cleanup')
            self.assertIs(mf._numint.eval_ao,numint.eval_ao)
            energy = mf.kernel()
            other = dft.RKS(self.mol)
            other.xc, other.grids.level = 'MN15', 1
            with cached_ao(other, 64) as cache:
                actual = other.kernel()
                self.assertGreater(cache.snapshot()['hits'],0)
            self.assertTrue(mf.converged and other.converged)
            self.assertLess(abs(energy-actual),1e-10)

    def test_guards(self):
        mf = dft.RKS(self.mol)
        with mock.patch.dict(os.environ, PYSCF_NUMA='on'):
            with self.assertRaises(ValueError):
                with cached_ao(mf):
                    pass
        with mock.patch.dict(os.environ, PYSCF_NUMA='off'):
            with cached_ao(mf):
                with self.assertRaises(TypeError):
                    with cached_ao(mf):
                        pass

if __name__ == '__main__':
    unittest.main(verbosity=2)
