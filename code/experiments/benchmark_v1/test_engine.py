"""Golden checks for exact-walk policy families. Run: python -m unittest -v code/experiments/benchmark_v1/test_engine.py"""
import pathlib,sys,unittest,tempfile
import numpy as np
import torch

ROOT=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"code/experiments/benchmark_v1"))
sys.path.insert(0,str(ROOT/"code/data_generator"))
sys.path.insert(0,str(ROOT/"code/experiments/diag_max_full300"))
sys.path.insert(0,str(ROOT/"code/experiments/multistep_policy_target_match"))

from engine import ExactMaskedWalk,DT
from generate import load_config,generate
from stream_graph import StreamGraph

class WalkBenchmarkChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        s=load_config(ROOT/"code/data_generator/config.json")
        with tempfile.TemporaryDirectory() as p:data=generate(s,p,finite_q=False)
        x,_,y=data["train"];val,_,_=data["validation"]
        cls.x=x[:45];cls.y=y[:45];cls.val=val[:12]
        cls.g=ExactMaskedWalk(cls.x,cls.y,s["gamma"],query_x=cls.val)
        cls.ref=StreamGraph(cls.x,cls.y,s["gamma"],cls.val,horizon=12)
        cls.z=torch.tensor(np.random.default_rng(17).normal(0,.5,size=(cls.g.d+1,cls.g.d)),dtype=DT,requires_grad=True)

    def test_shared_train_and_heldout_match_reference(self):
        r=self.g.represent_train(self.z,"shared")
        old=self.ref.center_train(self.z,np.arange(len(self.y)),return_all=True)
        self.assertTrue(torch.allclose(r,old,atol=1e-10,rtol=0))
        v=self.g.represent_query(self.z,"shared")
        self.assertTrue(torch.allclose(v,self.ref.evaluate_val(self.z),atol=1e-10,rtol=0))

    def test_all_families_collapse_to_identical_shared_policy(self):
        ref=self.g.train_outputs(self.z,"shared")
        choices=[("class",self.z.detach().repeat(self.g.k,1,1)),
                 ("node",self.z.detach().repeat(self.g.n,1,1)),
                 ("node_class",self.z.detach().repeat(self.g.k,self.g.n,1,1))]
        for fam,z in choices:
            actual=self.g.train_outputs(z,fam)
            self.assertTrue(torch.allclose(actual.scores,ref.scores,atol=1e-11,rtol=0),fam)

    def test_autograd_gradient_is_finite(self):
        margin=self.g.train_outputs(self.z,"shared").margin
        grad,=torch.autograd.grad(torch.sigmoid(-margin/.005).mean(),self.z)
        self.assertTrue(bool(torch.isfinite(grad).all()))

    def test_current_node_mask_does_not_inherit_origin_mask(self):
        # Feature 1 is absent in origin 0, but present and eligible in intermediate node 1.
        x=np.array([[0,np.nan],[.2,1],[.3,1.1],[.4,1.2]],float)
        g=ExactMaskedWalk(x,[0,0,1,1],[1,1],horizon=3)
        self.assertFalse(bool(g.available[0,1]))
        self.assertTrue(bool(g.available[1,1]))
        valid=(g.mass-g.base[:,:,0].T)>1e-12
        self.assertTrue(bool(valid[1,1]))
        p=g.represent_train(torch.zeros((3,2),dtype=DT),"shared",origin_ids=[0])
        self.assertAlmostEqual(float(p.sum()),1.0,places=10)

    def test_free_node_policy_has_no_unseen_start_without_generator(self):
        with self.assertRaises(ValueError):
            self.g.represent_query(self.z.detach().repeat(self.g.n,1,1),"node")

if __name__=="__main__":unittest.main()
