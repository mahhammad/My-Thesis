"""Read-only mathematical tests of the §4.14 postfit Softmax buffer metrics."""
from __future__ import annotations
import unittest
import numpy as np
from softmax_buffer_analysis import softmax,row_jacobian,row_stats,metrics

class BufferMathTests(unittest.TestCase):
    def test_finite_difference_informative_jacobian(self):
        z=np.array([1.2,-1.3,.4,.9,-.5])
        ids=[0,2,4]
        p=softmax(z)
        jac=row_jacobian(p,ids)
        eps=1e-6
        for j in range(len(z)):
            up=z.copy();up[j]+=eps
            down=z.copy();down[j]-=eps
            numeric=(softmax(up)[ids]-softmax(down)[ids])/(2*eps)
            self.assertTrue(np.allclose(jac[:,j],numeric,atol=1e-9))

    def test_conditional_odds_not_changed_by_noise_logits(self):
        z=np.array([1.4,-.4,2.0])
        extended=np.array([1.4,-.4,2.0,3.0,-2.0])
        a=softmax(z);b=softmax(extended)
        self.assertTrue(np.allclose(a,b[:3]/b[:3].sum()))
        self.assertLess(b[:3].sum(),1.0)
        self.assertAlmostEqual(row_stats(z,[0,1,2])["jac_main_to_noise_frob"],0)

    def test_exact_unmasked_selection_mass(self):
        z=np.zeros((6,5),dtype=float)
        out=metrics(z,[0,1,2,3,4])
        for h in (1,2,3):
            self.assertAlmostEqual(out[f"main_share_h{h}"],3/5)
            self.assertAlmostEqual(out[f"noise_share_h{h}"],2/5)
            self.assertAlmostEqual(sum(out[f"feature_{x}_conditional_h{h}"]
                                       for x in (0,1,3)),1)
        self.assertGreater(out["jac_main_to_noise_frob"],0)
        # Noise is NOT a universally beneficial stabilizer; this scalar
        # does not assess Adam's nonlinear optimizer trajectory.

if __name__=="__main__":
    unittest.main()
