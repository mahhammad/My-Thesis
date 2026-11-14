"""Unit checks: class-tied start training architecture, no data generation.

python -m unittest -v code/experiments/benchmark_v1/test_class_tied_start.py
"""
import pathlib,sys,unittest
import torch
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from engine import DT
from train_class_tied_node_intermediate import make_model_logits

class TiedStartTests(unittest.TestCase):
    def setUp(self):
        class Shape: k=3;n=4;d=5
        self.shape=Shape()

    def test_one_start_row_per_class_in_all_origins(self):
        start=torch.arange(15,dtype=DT).reshape(3,5)
        mid=torch.zeros((3,4,5,5),dtype=DT)
        z=make_model_logits(start,mid,self.shape)
        self.assertEqual(tuple(z.shape),(3,4,6,5))
        for c in range(3):
            for i in range(4):
                torch.testing.assert_close(z[c,i,5,:],start[c])
                torch.testing.assert_close(z[c,i,:5,:],mid[c,i])

    def test_gradients_of_all_origins_accumulate_on_same_start(self):
        start=torch.nn.Parameter(torch.zeros((3,5),dtype=DT))
        mid=torch.nn.Parameter(torch.zeros((3,4,5,5),dtype=DT))
        z=make_model_logits(start,mid,self.shape)
        loss=z[:,:,5,:].sum()+z[:,:,:5,:].sum()
        loss.backward()
        torch.testing.assert_close(start.grad,torch.full_like(start,4.))
        torch.testing.assert_close(mid.grad,torch.ones_like(mid))

    def test_mask_changes_distribution_not_learned_logits(self):
        start=torch.tensor([[1.,2.,-1.,0.,3.]],dtype=DT)
        logits=start[0]
        mask_a=torch.tensor([1,1,0,0,1],dtype=torch.bool)
        mask_b=torch.tensor([1,0,1,1,1],dtype=torch.bool)
        a=torch.softmax(logits.masked_fill(~mask_a,-1e25),0)
        b=torch.softmax(logits.masked_fill(~mask_b,-1e25),0)
        self.assertAlmostEqual(float(a.sum()),1)
        self.assertAlmostEqual(float(b.sum()),1)
        self.assertAlmostEqual(float(a[2]+a[3]),0)
        self.assertAlmostEqual(float(b[1]),0)
        self.assertFalse(bool(torch.allclose(a,b)))

if __name__=="__main__":unittest.main()
