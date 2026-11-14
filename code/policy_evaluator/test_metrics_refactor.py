import unittest
import numpy as np
from metrics import (class_centers, center_metrics, classify, jsd, sqrt_jsd, js_similarity,
                     pairwise_metric, kl_divergence, report, sample_weights_vector,
                     normalized_nll, margin_criterion)

class MetricTests(unittest.TestCase):
    def setUp(self):
        self.p = np.array([[.8,.2],[.7,.3],[.3,.7],[.2,.8]])
        self.y = np.array([0,0,1,1])
        self.centers = class_centers(self.p,self.y)

    def test_three_decisions_agree_for_symmetric_example(self):
        for method in ('kl','sqrt_jsd','js_similarity'):
            pred=classify(self.p,self.centers,method)
            np.testing.assert_array_equal(pred['prediction'],self.y)
            self.assertEqual(pred['decision_values'].shape,(4,2))

    def test_js_identity_and_support(self):
        a=np.array([1.,0.]); b=np.array([0.,1.])
        self.assertAlmostEqual(float(js_similarity(a,b)),0)
        self.assertAlmostEqual(float(sqrt_jsd(a,b)),np.sqrt(np.log(2)))
        self.assertTrue(np.isinf(kl_divergence(a,b)))

    def test_center_margins_and_weights(self):
        c = np.eye(3)
        geom = center_metrics(c,np.array([1.,1.,1.]))
        self.assertAlmostEqual(geom['T'],1)
        self.assertAlmostEqual(geom['BT'],1)
        self.assertAlmostEqual(geom['I'],np.log(3))
        self.assertEqual(geom['TNL'],0.)
        self.assertEqual(geom['BTNL'],0.)
        self.assertAlmostEqual(center_metrics(self.centers)['BT'],.75)

    def test_balanced_metrics_and_weighted_centers(self):
        sw=np.array([1.,3.,1.,1.])
        c=class_centers(self.p,self.y,sample_weights=sw)
        np.testing.assert_allclose(c[0],[.725,.275])
        w=sample_weights_vector(self.y,2,[3,1])
        np.testing.assert_allclose(w,[.375,.375,.125,.125])
        result,pred=report(self.p,self.centers,self.y,metric='kl',class_weights=[3,1])
        self.assertAlmostEqual(result['balanced_accuracy'],1.)
        self.assertAlmostEqual(result['balanced_error'],0.)
        self.assertEqual(result['decision_metric'],'kl')
        self.assertTrue(np.isfinite(result['NLL_norm']))
        self.assertTrue(np.isfinite(result['J_margin']))

    def test_uniform_distances(self):
        d=np.ones((3,2)); y=np.array([0,1,0])
        self.assertAlmostEqual(normalized_nll(d,y),np.log(2))

if __name__=='__main__':
    unittest.main(verbosity=2)
