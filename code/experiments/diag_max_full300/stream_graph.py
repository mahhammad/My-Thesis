"""Memory-bounded exact multi-step expectations for all 300 sample-feature origins.

Reproduces ExactGraph semantics but computes forbidden-origin corrections on the
fly from shared normalized feature-wise base transitions.
"""
import numpy as np
import torch
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/experiments/multistep_policy_target_match'))
from run import coefficients, jsd, DT, wgraph, heldout_weights

class StreamGraph:
    def __init__(self,x,y,gamma,validation_x=None,beta=.4,lam=.8,horizon=12):
        self.x=np.asarray(x,float);self.labels=np.asarray(y,int)
        self.n,self.d=self.x.shape;self.k=int(self.labels.max())+1
        self.H=horizon
        self.coeff=torch.as_tensor(coefficients(beta,lam,horizon),dtype=DT)
        self.yhot=torch.as_tensor(np.eye(self.k)[self.labels],dtype=DT)
        self.available=torch.as_tensor(np.isfinite(self.x),dtype=torch.bool)
        self.base=torch.as_tensor(wgraph(self.x,gamma),dtype=DT)
        self.base_mass=self.base.sum(dim=2).T.contiguous()
        self.qbase=torch.einsum('bij,jk->ibk',self.base,self.yhot)
        self.valid_start=self.available
        self.validation_x=np.asarray(validation_x,float) if validation_x is not None else None
        if validation_x is not None:
            self.val_start=torch.as_tensor(heldout_weights(self.validation_x,self.x,gamma),dtype=DT)
            self.val_available=torch.as_tensor(np.isfinite(self.validation_x),dtype=torch.bool)
        self._all_idx=torch.arange(self.n)

    @staticmethod
    def masked_softmax(logits,mask):
        return torch.softmax(logits.masked_fill(~mask,-1e20),dim=-1)

    def _valid_den(self,idx):
        idx=torch.as_tensor(idx,dtype=torch.long)
        src_to_origin=self.base[:,:,idx].permute(2,1,0)
        den=self.base_mass.unsqueeze(0)-src_to_origin
        valid=den>1e-12
        return den,valid

    def _step(self,state,policy,den,idx):
        choose=torch.einsum('mja,mjab->mjb',state,policy)
        result=[]
        for b in range(self.d):
            w=torch.where(den[:,:,b]>1e-12, choose[:,:,b]/den[:,:,b].clamp_min(1e-12),torch.zeros_like(choose[:,:,b]))
            dest=w @ self.base[b]
            dest=dest.scatter(1,idx[:,None],torch.zeros((len(idx),1),dtype=dest.dtype))
            result.append(dest)
        return torch.stack(result,dim=-1)

    def center_train(self,z,indices,return_all=False,max_hops=None):
        idx=torch.as_tensor(indices,dtype=torch.long)
        H=self.H if max_hops is None else max_hops
        den,valid=self._valid_den(idx)
        policy=self.masked_softmax(z[:self.d][None,None,:,:].expand(len(idx),self.n,-1,-1),valid[:,:,None,:].expand(-1,-1,self.d,-1))
        p0=self.masked_softmax(z[self.d][None,:].expand(len(idx),-1),self.valid_start[idx])
        start=self.base[:,idx,:].permute(1,2,0)
        state=start * p0[:,None,:]
        result=torch.zeros((len(idx),self.k),dtype=state.dtype)
        if H==self.H:alpha=self.coeff
        else:alpha=self.coeff[:H]/self.coeff[:H].sum()
        for t in range(H):
            result=result+alpha[t]*torch.einsum('mjb,jk->mk',state,self.yhot)
            if t!=H-1:
                state=self._step(state,policy,den,idx)
        if torch.any(torch.abs(result.sum(-1)-1)>1e-7):
            raise AssertionError('walk mass was lost')
        return result if return_all else result.mean(dim=0)

    def local_q_loss(self,z,indices,target):
        idx=torch.as_tensor(indices,dtype=torch.long)
        den,valid=self._valid_den(idx)
        policy=self.masked_softmax(z[:self.d][None,None,:,:].expand(len(idx),self.n,-1,-1),valid[:,:,None,:].expand(-1,-1,self.d,-1))
        sobs=self.base[:,:,idx].permute(2,1,0)
        q=(self.qbase[None,:,:,:]-sobs[:,:,:,None]*self.yhot[idx,None,None,:])/den[:,:,:,None].clamp_min(1e-12)
        q=torch.where(valid[:,:,:,None],q,torch.zeros_like(q))
        dist=torch.einsum('miab,mibk->miak',policy,q)
        valid_state=self.available[None,:,:]&valid.any(-1)[:,:,None]
        interior=(jsd(dist,target.reshape(1,1,1,-1))*valid_state).sum()/valid_state.sum()
        first=self.masked_softmax(z[self.d][None,:].expand(len(idx),-1),self.valid_start[idx])
        start_q=self.qbase[idx]
        start_dist=torch.einsum('mb,mbk->mk',first,start_q)
        return .5*(interior+jsd(start_dist,target[None,:]).mean())

    def evaluate_val(self,z):
        if self.validation_x is None:raise ValueError('no validation data')
        B=len(self.validation_x)
        pi0=self.masked_softmax(z[self.d][None,:].expand(B,-1),self.val_available)
        state=self.val_start.permute(0,2,1)*pi0[:,None,:]
        available=self.base_mass>1e-12
        policy=self.masked_softmax(z[:self.d][None,None,:,:].expand(B,self.n,-1,-1),available[None,:,None,:].expand(B,-1,self.d,-1))
        out=torch.zeros((B,self.k),dtype=state.dtype)
        for t in range(self.H):
            out=out+self.coeff[t]*torch.einsum('mjb,jk->mk',state,self.yhot)
            if t+1!=self.H:
                choose=torch.einsum('mja,mjab->mjb',state,policy)
                state=torch.stack([choose[:,:,b] @ self.base[b] for b in range(self.d)],dim=-1)
        return out
