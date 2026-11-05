import unittest
import numpy as np
from walks import _weights

class WeightsTests(unittest.TestCase):
    def test_vectorization_matches_rowwise(self):
        rng=np.random.default_rng(6)
        ref=rng.normal(size=(18,4)); q=rng.normal(size=(10,4))
        ref[rng.random(ref.shape)<.2]=np.nan
        q[rng.random(q.shape)<.2]=np.nan
        gamma=np.array([.5,1.,2.,1.7])
        ids=np.array([-1,0,1,2,3,4,5,6,7,8])
        observed=_weights(q,ref,gamma,ids)
        expected=np.zeros_like(observed)
        for i in range(len(q)):
            for a in range(ref.shape[1]):
                if not np.isfinite(q[i,a]): continue
                choices=np.flatnonzero(np.isfinite(ref[:,a]) & (np.arange(len(ref))!=ids[i]))
                if not len(choices): continue
                lw=-gamma[a]*(q[i,a]-ref[choices,a])**2
                expected[i,a,choices]=np.exp(lw-lw.max())
        np.testing.assert_allclose(observed,expected,atol=1e-14)

if __name__=='__main__': unittest.main()
