"""No-training numeric NFI3/stability tests, including safe checkpoint loading."""
import hashlib, io, pathlib, tempfile, unittest
import numpy as np
import torch
from advanced_analysis import policy3, enumerate3, softmax, stability
from checkpoints import load_checkpoint

class NFI3Tests(unittest.TestCase):
 def test_uniform_all_dimensions_exact(self):
    for d in range(3,9):
      z=np.zeros((d+1,d),dtype=np.float64)
      mapping=([0,2,1,3,4,5,6,7][:d] if d>3 else [0,1,3])
      # At d=4, informative indices 0,1,3 must all be present.
      if d>3:mapping=[0,2,1,3,4,5,6,7][:d]
      p=policy3(z,mapping)
      k=d-3
      self.assertAlmostEqual(p["F3"],1-(3/d)**3,places=12)
      self.assertAlmostEqual(p["E3"],3*k/d,places=12)
      for t in ("q1","q2","q3"):self.assertAlmostEqual(p[t],k/d,places=12)
      if k:self.assertAlmostEqual(p["NFI3"],1.,places=12)
      else:
        self.assertIsNone(p["NFI3"]);self.assertEqual(p["F3"],0);self.assertEqual(p["E3"],0)
 def test_independent_enumeration_all_k_with_nonuniform_logits(self):
    original=[0,2,1,3,4,5,6,7]
    for d in range(3,9):
      mapping=([0,1,3] if d==3 else original[:d])
      z=np.sin(np.arange((d+1)*d).reshape(d+1,d)*.71)+np.arange(d)/d
      val=policy3(z,mapping)
      F,E=enumerate3(z,mapping)
      self.assertAlmostEqual(val["F3"],F,places=11)
      self.assertAlmostEqual(val["E3"],E,places=11)
      self.assertGreaterEqual(val["F3"],0);self.assertLessEqual(val["F3"],1)
      self.assertGreaterEqual(val["E3"],0);self.assertLessEqual(val["E3"],3)
      for q in ("q1","q2","q3"):self.assertTrue(0<=val[q]<=1)
 def test_local_mapping_not_trailing_noise(self):
    z=np.zeros((6,5))
    z[-1,1]=5.0  # local index 1 maps to original noise 2
    mixed=policy3(z,[0,2,1,3,5])
    other=policy3(z,[0,1,2,3,5])
    self.assertGreater(mixed["q1"],other["q1"])
 def test_class_three_identical_logits_equals_shared(self):
    z=np.arange(30,dtype=float).reshape(6,5)/7
    mapping=[0,2,1,3,5]
    a=policy3(z,mapping)
    for copy in np.repeat(z[None,:,:],3,axis=0):
      b=policy3(copy,mapping)
      for k in ("F3","E3","q1","q2","q3","NFI3","delta_F3"):
        self.assertAlmostEqual(a[k],b[k],places=13)
 def test_reject_malformed_mapping_or_logits(self):
    z=np.zeros((5,4))
    for mapping in ([0,1,2,3],[0,1,3,3],[0,1,3,5,6]):
      with self.assertRaises(ValueError):policy3(z,mapping)
    z[0,0]=float("nan")
    with self.assertRaises(ValueError):policy3(z,[0,1,3,4])
 def test_sha_pinned_weights_only_tamper(self):
    with tempfile.TemporaryDirectory() as folder:
      path=pathlib.Path(folder)/"step_30.pt"
      payload={"schema":"e-ch4-14-resumable-v1","step":30,
               "logits":torch.zeros((4,3),dtype=torch.float64),
               "optimizer":{"param_groups":[{"params":[0]}],"state":{0:{"step":torch.tensor(30)}}},
               "metadata":{"test_accessed":False}}
      torch.save(payload,path)
      digest=hashlib.sha256(path.read_bytes()).hexdigest()
      path.with_suffix(".pt.sha256").write_text(digest+"\n")
      self.assertEqual(load_checkpoint(path,expected_sha256=digest)["step"],30)
      with self.assertRaises(ValueError):load_checkpoint(path,expected_sha256="f"*64)
      path.write_bytes(path.read_bytes()+b"corrupted")
      with self.assertRaises(ValueError):load_checkpoint(path,expected_sha256=digest)

class StabilityTests(unittest.TestCase):
 def trace(self,ba,loss,grad):
   return [dict(step=i,validation_BA=ba[i],train_softplus_loss=loss[i],gradient_l2=grad[i]) for i in range(31)]
 def test_drawdown_totalvariation_and_joint_loss_ba_fall(self):
   ba=[.95,.96,.85,.92]+[.92]*27
   loss=[3-.01*i for i in range(31)]
   grad=[.04+(.001 if i%2 else 0) for i in range(31)]
   m=stability(self.trace(ba,loss,grad),"N0-01__shared",0,"shared")
   self.assertAlmostEqual(m["max_drawdown_pp"],11.)
   self.assertAlmostEqual(m["total_variation_pp"],19.)
   self.assertEqual(m["ba_down_steps"],1)
   self.assertEqual(m["loss_down_ba_down_steps"],1)
   self.assertEqual(m["loss_up_steps"],0)
   self.assertEqual(m["gradient_peak_count"],15)
   self.assertEqual(m["gradient_trough_count"],14)
   self.assertAlmostEqual(m["gradient_late_early_tv_ratio"],1.)
 def test_reject_step_gap(self):
   t=self.trace([.9]*31,[.5]*31,[.1]*31);t[5]["step"]=9
   with self.assertRaises(ValueError):stability(t,"N0-01__class",0,"class")
 def test_gradient_damping_is_descriptive_not_convergence(self):
   grad=[.15 + (.02/(i+1))*(1 if i%2 else -1) for i in range(31)]
   t=self.trace([.9]*31,[2-i*.001 for i in range(31)],grad)
   m=stability(t,"N0-01__shared",0,"shared")
   self.assertLess(m["gradient_late_early_range_ratio"],1)
   self.assertLess(m["gradient_late_early_tv_ratio"],1)
   self.assertGreater(m["final_gradient"],0)

if __name__=="__main__":unittest.main()
