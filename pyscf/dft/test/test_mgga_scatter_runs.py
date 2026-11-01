"""Default-off sparse MGGA scatter must preserve old per-element additions."""
import os
from unittest import mock
import unittest
import numpy as np
from pyscf.dft.test import test_mgga_coalesce as old


class KnownValues(old.KnownValues):
    def setUp(self):
        super().setUp()
        env = mock.patch.dict(os.environ, {'PYSCF_DFT_MGGA_SCATTER_RUNS':'0'})
        env.start()
        self.addCleanup(env.stop)

    def evaluate(self, flag, *args, **kwargs):
        with mock.patch.dict(os.environ, {'PYSCF_DFT_MGGA_SCATTER_RUNS':flag}):
            return self.call(*args, **kwargs)

    def test_scatter_bitwise_and_input_safety(self):
        rng = np.random.default_rng(601)
        for sizes in ([1,3,5,7,11], [70,3,70,3], [1,3,5,7]*15):
            loc = np.asarray([0]+list(np.cumsum(sizes)),dtype=np.int32)
            n,ns = int(loc[-1]),len(sizes)
            for g in (55,56,57,224,449):
                ao = rng.normal(size=(8,n,g))[::2]
                if g==449:ao=ao[::-1]
                w = rng.normal(size=(5,g)); w[:,::9]=0
                s = rng.choice([0,1,9,10,11,19,20,255],((g+55)//56,ns)).astype(np.uint8)
                pairs = rng.integers(0,2,(ns,ns),dtype=np.uint8);pairs |= pairs.T
                initial=rng.normal(size=(n,n))
                saved=[x.copy() for x in (ao,w,s,pairs,loc)]
                for t in (1,2,4):
                    for merge in ('0','1'):
                        with self.subTest(nao=n,g=g,threads=t,merge=merge):
                            args=(ao,w,s,pairs,loc,initial,merge,t)
                            a=self.evaluate('0',*args);b=self.evaluate('1',*args)
                            self.assertEqual((a[0],b[0]),(1,1))
                            np.testing.assert_array_equal(a[1],b[1])
                            for x,y in zip((ao,w,s,pairs,loc),saved):np.testing.assert_array_equal(x,y)

    def test_scatter_unaffected_and_malformed_flags(self):
        rng=np.random.default_rng(602);ao=rng.normal(size=(4,67,449));w=rng.normal(size=(5,449))
        loc=np.asarray([0,3,10,67],dtype=np.int32);pair=np.ones((3,3),dtype=np.uint8)
        mask=np.full((9,3),25,dtype=np.uint8);out=np.zeros((67,67))
        for nc,s in ((1,mask),(2,mask),(4,None),(4,mask)):
            a=self.evaluate('0',ao,w,s,pair,loc,out,'0',ncomp=nc)
            for flag in ('','true','01','2') + (('1',) if nc!=4 or s is None else ()):
                b=self.evaluate(flag,ao,w,s,pair,loc,out,'0',ncomp=nc)
                self.assertEqual(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])

    def test_scatter_low_workspace(self):
        rng=np.random.default_rng(603);ao=rng.normal(size=(4,160,448));w=rng.normal(size=(5,448))
        loc=np.asarray([0,80,160],dtype=np.int32);s=np.full((8,2),25,dtype=np.uint8)
        p=np.ones((2,2),dtype=np.uint8);out=np.zeros((160,160))
        for budget in ('1','2','3'):
            for t in (1,2,4):
                with mock.patch.dict(os.environ,{'PYSCF_DFT_VMAT_MAX_MEMORY_MB':budget}):
                    a=self.evaluate('0',ao,w,s,p,loc,out,'0',t)
                    b=self.evaluate('1',ao,w,s,p,loc,out,'0',t)
                self.assertEqual((a[0],b[0]),(1,1))
                np.testing.assert_allclose(a[1],b[1],atol=1e-9,rtol=1e-10)

    def test_scatter_asymmetric_fallback(self):
        with mock.patch.dict(os.environ,{'PYSCF_DFT_MGGA_SCATTER_RUNS':'1'}):
            self.test_asymmetric_fallback_does_not_write()


if __name__=='__main__':
    unittest.main()
