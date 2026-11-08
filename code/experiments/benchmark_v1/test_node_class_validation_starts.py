"""Pure unit checks of the new start estimator: no generator and no training.

python -m unittest -v code/experiments/benchmark_v1/test_node_class_validation_starts.py
"""
import pathlib, sys, unittest
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from node_class_validation_starts import start_probabilities,METHODS

class StartEstimationChecks(unittest.TestCase):
    def setUp(self):
        # K=2, N=4, D=3. Feature 0 always observed.
        self.x=np.array([[0.,1.,np.nan],[1.,np.nan,2.],
                         [4.,2.,3.],[5.,2.2,np.nan]])
        self.y=np.array([0,0,1,1])
        self.q=np.array([[.2,1.1,np.nan],[4.2,np.nan,3.1]])
        self.z=np.arange(2*4*4*3,dtype=float).reshape(2,4,4,3)*.01
    def test_normalization_query_mask_and_hypotheses(self):
        for method in METHODS:
            with self.subTest(method=method):
                p=start_probabilities(self.z,self.x,self.y,self.q,method)
                self.assertEqual(p.shape,(2,2,3))
                np.testing.assert_allclose(p.sum(2),1,atol=1e-12,rtol=0)
                self.assertTrue(np.isfinite(p).all())
                self.assertTrue((p[:,:,2][:,0]==0).all())
                self.assertTrue((p[:,:,1][:,1]==0).all())
    def test_classwise_donors_do_not_use_validation_labels(self):
        # Editing logits of class-1 TRAIN sources has no impact on
        # the class-0 candidate estimate for any query sample.
        changed=self.z.copy()
        changed[0,2:,:]=99.
        # 0-class donor ids are 0,1; hypothesis c=0 uses only them.
        for method in METHODS:
            p0=start_probabilities(self.z,self.x,self.y,self.q,method)
            p1=start_probabilities(changed,self.x,self.y,self.q,method)
            np.testing.assert_allclose(p0[0],p1[0],atol=1e-12)
    def test_uniform_policy(self):
        p=start_probabilities(self.z,self.x,self.y,self.q,"uniform")
        np.testing.assert_allclose(p[:,:,0],.5,atol=1e-12)
        np.testing.assert_allclose(p[0,0,1],.5,atol=1e-12)
        np.testing.assert_allclose(p[1,1,2],.5,atol=1e-12)

if __name__=="__main__":
    unittest.main()
