"""Exact differentiable sample-feature walks for shared/class/node/node_class logits.
Intermediate logits are evaluated at the current graph node, never frozen at origin.
Free node tables are train-only diagnostics unless query logits are supplied.
"""
from dataclasses import dataclass
import numpy as np
import torch

DT=torch.float64
LN2=np.log(2.0)
FAMILIES=("shared","class","node","node_class")

def coefficients(beta=.4,lam=.8,horizon=12):
    out=np.zeros(horizon,float)
    for length in range(1,horizon+1):
        p=beta*(1-beta)**(length-1) if length<horizon else (1-beta)**(horizon-1)
        norm=sum(lam**j for j in range(length))
        for t in range(length):out[t]+=p*lam**t/norm
    if not np.isclose(out.sum(),1,atol=1e-12):raise AssertionError("walk coefficient mass")
    return torch.as_tensor(out,dtype=DT)

def jsd(p,q):
    m=(p+q)/2
    def ent(x):
        return -torch.where(x>0,x*torch.log(x.clamp_min(1e-300)),torch.zeros_like(x)).sum(-1)
    return (ent(m)-(ent(p)+ent(q))/2).clamp_min(0)

def balanced(v,labels,k):
    return torch.stack([v[labels==c].mean() for c in range(k)]).mean()

def masked_softmax(logits,mask):
    # Unreachable all-false rows have zero state probability.
    return torch.softmax(logits.masked_fill(~mask,-1e25),dim=-1)

def feature_kernel(x,gamma):
    n,d=x.shape;out=np.zeros((d,n,n),float)
    for b in range(d):
        ok=np.isfinite(x[:,b]);v=np.where(ok,x[:,b],0)
        w=np.exp(-gamma[b]*(v[:,None]-v[None,:])**2)
        w*=ok[:,None]&ok[None,:]
        np.fill_diagonal(w,0)
        den=w.sum(1,keepdims=True)
        np.divide(w,den,out=w,where=den>0);out[b]=w
    return out

def query_kernel(q,ref,gamma):
    m,d=q.shape;n=len(ref);out=np.zeros((m,d,n),float)
    for b in range(d):
        qo=np.isfinite(q[:,b]);ro=np.isfinite(ref[:,b])
        qv=np.where(qo,q[:,b],0);rv=np.where(ro,ref[:,b],0)
        w=np.exp(-gamma[b]*(qv[:,None]-rv[None,:])**2)*(qo[:,None]&ro[None,:])
        den=w.sum(1,keepdims=True)
        np.divide(w,den,out=w,where=den>0);out[:,b,:]=w
    return out

@dataclass
class EvalOutputs:
    centers:torch.Tensor
    scores:torch.Tensor
    margin:torch.Tensor
    correct_representations:torch.Tensor

class ExactMaskedWalk:
    def __init__(self,ref_x,ref_y,gamma,beta=.4,lam=.8,horizon=12,query_x=None):
        self.x=np.asarray(ref_x,float)
        self.labels=torch.as_tensor(ref_y,dtype=torch.long)
        self.n,self.d=self.x.shape;self.k=int(self.labels.max().item())+1
        self.gamma=np.asarray(gamma,dtype=float)
        self.alpha=coefficients(beta,lam,horizon);self.horizon=horizon
        self.base=torch.as_tensor(feature_kernel(self.x,self.gamma),dtype=DT)
        self.mass=self.base.sum(2).T.contiguous()
        self.available=torch.as_tensor(np.isfinite(self.x),dtype=torch.bool)
        self.yhot=torch.nn.functional.one_hot(self.labels,self.k).to(DT)
        self.qx=None if query_x is None else np.asarray(query_x,float)
        if query_x is not None:
            self.qweights=torch.as_tensor(query_kernel(self.qx,self.x,self.gamma),dtype=DT)
            self.qavailable=torch.as_tensor(np.isfinite(self.qx),dtype=torch.bool)

    def check_logits(self,z,family):
        expected={"shared":(self.d+1,self.d),
                  "class":(self.k,self.d+1,self.d),
                  "node":(self.n,self.d+1,self.d),
                  "node_class":(self.k,self.n,self.d+1,self.d)}
        if family not in expected or tuple(z.shape)!=expected[family]:
            raise ValueError(f"family {family}: expected {expected.get(family)}, got {tuple(z.shape)}")
        if z.dtype!=DT:raise TypeError("float64 logits required")

    def _select_node_scores(self,z,family,c,idx):
        m=len(idx)
        if family=="shared":
            return z[self.d].expand(m,self.d),z[:self.d].expand(self.n,self.d,self.d)
        if family=="class":
            return z[c,self.d].expand(m,self.d),z[c,:self.d].expand(self.n,self.d,self.d)
        if family=="node":
            return z[idx,self.d],z[:,:self.d]
        return z[c,idx,self.d],z[c,:,:self.d]

    def represent_train(self,z,family,c=0,origin_ids=None):
        self.check_logits(z,family)
        idx=torch.arange(self.n,dtype=torch.long) if origin_ids is None else torch.as_tensor(origin_ids,dtype=torch.long)
        m=len(idx);start_z,mid_z=self._select_node_scores(z,family,c,idx)
        # Reject returning to origin *after* every hop and renormalize each vertical feature.
        den=self.mass.unsqueeze(0)-self.base[:,:,idx].permute(2,1,0)
        valid=(den>1e-12)&self.available.unsqueeze(0)
        pol=masked_softmax(mid_z.unsqueeze(0).expand(m,-1,-1,-1),
                           valid[:,:,None,:].expand(-1,-1,self.d,-1))
        p0=masked_softmax(start_z,self.available[idx])
        state=self.base[:,idx,:].permute(1,2,0)*p0[:,None,:]
        out=torch.zeros((m,self.k),dtype=DT)
        for t in range(self.horizon):
            out=out+self.alpha[t]*(state.sum(2)@self.yhot)
            if t==self.horizon-1:break
            choose=torch.einsum("mja,mjab->mjb",state,pol)
            moved=[]
            for b in range(self.d):
                mass=torch.where(den[:,:,b]>1e-12,
                                 choose[:,:,b]/den[:,:,b].clamp_min(1e-12),
                                 torch.zeros_like(choose[:,:,b]))
                dest=mass@self.base[b]
                dest=dest.scatter(1,idx[:,None],torch.zeros((m,1),dtype=DT))
                moved.append(dest)
            state=torch.stack(moved,dim=2)
        if not torch.allclose(out.sum(1),torch.ones(m,dtype=DT),rtol=0,atol=1e-8):
            raise AssertionError("train walk mass lost")
        return out

    def represent_query(self,z,family,c=0,query_start_logits=None):
        if self.qx is None:raise ValueError("query samples missing")
        self.check_logits(z,family)
        B=len(self.qx)
        if family=="shared":
            mid=z[:self.d].expand(self.n,-1,-1);first=z[self.d].expand(B,-1)
        elif family=="class":
            mid=z[c,:self.d].expand(self.n,-1,-1);first=z[c,self.d].expand(B,-1)
        elif family=="node":
            mid=z[:,:self.d]
            if query_start_logits is None:raise ValueError("free node policies lack unseen start policy")
            first=query_start_logits
        else:
            mid=z[c,:,:self.d]
            if query_start_logits is None:raise ValueError("free node-class policies lack unseen start policy")
            first=query_start_logits[c]
        if first.shape!=(B,self.d):raise ValueError("query start logits shape")
        p0=masked_softmax(first,self.qavailable)
        state=self.qweights.permute(0,2,1)*p0[:,None,:]
        valid=(self.mass>1e-12)&self.available
        pol=masked_softmax(mid,valid[:,None,:].expand(-1,self.d,-1))
        out=torch.zeros((B,self.k),dtype=DT)
        for t in range(self.horizon):
            out=out+self.alpha[t]*(state.sum(2)@self.yhot)
            if t==self.horizon-1:break
            choose=torch.einsum("mja,jab->mjb",state,pol)
            state=torch.stack([choose[:,:,b]@self.base[b] for b in range(self.d)],dim=2)
        if not torch.allclose(out.sum(1),torch.ones(B,dtype=DT),rtol=0,atol=1e-8):
            raise AssertionError("query walk mass lost")
        return out

    def train_outputs(self,z,family):
        centers=[]
        if family in ("shared","node"):
            reps=self.represent_train(z,family)
            for c in range(self.k):centers.append(reps[self.labels==c].mean(0))
            C=torch.stack(centers)
            S=1-jsd(reps[:,None,:],C[None,:,:])/LN2
            own=reps
        else:
            by_c=[]
            for c in range(self.k):
                reps=self.represent_train(z,family,c)
                by_c.append(reps);centers.append(reps[self.labels==c].mean(0))
            C=torch.stack(centers)
            S=torch.stack([1-jsd(by_c[c],C[c])/LN2 for c in range(self.k)],dim=1)
            own=torch.stack([by_c[int(c)][i] for i,c in enumerate(self.labels)])
        truth=S.gather(1,self.labels[:,None]).squeeze(1)
        rival=S.masked_fill(torch.nn.functional.one_hot(self.labels,self.k).bool(),-1e20).max(1).values
        return EvalOutputs(C,S,truth-rival,own)

    def metrics(self,out,tau=.005):
        """Always emit the canonical full diagnostic set, including three R variants."""
        from metrics import summarize
        return summarize(out,self.labels,tau,center_is_class_mean=True)
