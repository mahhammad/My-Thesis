from hashlib import sha256
from pathlib import Path
import sys
import tempfile
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from checkpoints import save_checkpoint,load_checkpoint


def optimize(z,opt,steps):
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        (z.square().sum()+.13*z.sum()).backward()
        opt.step()


class CheckpointTests(unittest.TestCase):
    def test_step30_adam_resume_same_as_uninterrupted(self):
        target=torch.nn.Parameter(torch.tensor([[1.,-2.],[.3,.7]],dtype=torch.float64))
        target_opt=torch.optim.Adam([target],lr=.12)
        optimize(target,target_opt,31)
        baseline=target.detach().clone()
        partial=torch.nn.Parameter(torch.tensor([[1.,-2.],[.3,.7]],dtype=torch.float64))
        opt=torch.optim.Adam([partial],lr=.12)
        optimize(partial,opt,30)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'step_30.pt'
            meta={'fixture':True,'family':'shared','test_accessed':False}
            digest=save_checkpoint(p,partial,opt,30,meta,fixture=True)
            self.assertTrue(p.with_suffix('.pt.sha256').is_file())
            with self.assertRaises(FileExistsError):
                save_checkpoint(p,partial,opt,30,meta,fixture=True)
            loaded=load_checkpoint(p,expected_sha256=digest,expected_metadata=meta)
            z=torch.nn.Parameter(loaded['logits'].clone())
            resume=torch.optim.Adam([z],lr=.12)
            resume.load_state_dict(loaded['optimizer'])
            optimize(z,resume,1)
            self.assertTrue(torch.equal(z,baseline))
            self.assertEqual(int(resume.state[z]['step']),31)
            with self.assertRaises(ValueError):
                load_checkpoint(p,expected_sha256='0'*64)
            with self.assertRaises(ValueError):
                load_checkpoint(p,expected_sha256=digest,expected_metadata={'bad':True})

    def test_no_intermediate_checkpoint(self):
        p=torch.nn.Parameter(torch.tensor([2.],dtype=torch.float64))
        optim=torch.optim.Adam([p],lr=.12)
        optimize(p,optim,29)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                save_checkpoint(Path(tmp)/'step_29.pt',p,optim,29,{},fixture=True)
            with self.assertRaises(ValueError):
                save_checkpoint(Path(tmp)/'step_30.pt',p,optim,30,{},fixture=True)

if __name__=='__main__':unittest.main()
