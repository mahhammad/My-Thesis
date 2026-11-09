"""Regression check: differentiating centers through all training origins."""
import unittest
from train import check_gradient

class TestSigmoidPolicyGradient(unittest.TestCase):
    def test_exact_twelve_hop_autograd(self):
        result=check_gradient(seed=20261107,nref=72,tau=.005)
        self.assertLess(result['max_absolute_error'],1e-6)

if __name__=='__main__':
    unittest.main()
