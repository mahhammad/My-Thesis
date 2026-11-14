#!/usr/bin/env python3
"""Exact multi-step target matching versus local one-step Q matching.

Experimental tool (not model training): fixed target centers, a shared masked
class-conditional feature-policy matrix (d+1,d), and deterministic expectations.
Run from the repository root after installing numpy and torch:
  python code/experiments/multistep_policy_target_match/run.py
Data are re-generated from code/data_generator/config.json. No test labels are
used in fitting. All seed / hyper-parameters are listed below.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/data_generator'))
from generate import generate, load_config  # noqa: E402

DT = torch.float64
TORCH_DEVICE = 'cpu'


def coefficients(beta=0.4, lam=0.8, horizon=12):
    """E[weight of step t / within-walk weight sum] for capped geometric L."""
    a = np.zeros(horizon, dtype=float)
    for ell in range(1, horizon + 1):
        prob = beta * (1-beta) ** (ell-1) if ell < horizon else (1-beta) ** (horizon-1)
        denom = sum(lam ** (t-1) for t in range(1, ell+1))
        for t in range(1, ell+1):
            a[t-1] += prob * (lam ** (t-1)) / denom
    assert np.isclose(a.sum(),1)
    return a


def jsd(p, q):
    """JSD with analytic gradients for positive probabilities and exact zeros."""
    mid = 0.5 * (p+q)
    def ent(x):
        return -(torch.where(x > 0, x * torch.log(x.clamp_min(1e-300)), torch.zeros_like(x))).sum(-1)
    return (ent(mid) - (ent(p) + ent(q))*0.5).clamp_min(0)


def wgraph(x, gamma):
    """Kernels normalized over eligible destinations separately for each feature."""
    n, d = x.shape
    w = np.zeros((d,n,n),dtype=float)
    for b in range(d):
        good = np.isfinite(x[:,b]); value = np.where(good,x[:,b],0)
        diff = value[:,None]-value[None,:]
        wb = np.exp(-gamma[b]*diff**2)
        wb *= good[:,None] & good[None,:]
        np.fill_diagonal(wb,0)
        den = wb.sum(axis=1,keepdims=True)
        np.divide(wb,den,out=wb,where=den>0)
        w[b]=wb
    return w


def heldout_weights(queries, ref, gamma):
    m,d = queries.shape
    n = len(ref)
    w = np.zeros((m,d,n),float)
    for b in range(d):
        qok = np.isfinite(queries[:,b]); rok = np.isfinite(ref[:,b])
        diff = np.where(qok,queries[:,b],0)[:,None]-np.where(rok,ref[:,b],0)[None,:]
        wb = np.exp(-gamma[b]*diff**2)*(qok[:,None]&rok[None,:])
        den=wb.sum(axis=-1,keepdims=True)
        np.divide(wb,den,out=wb,where=den>0)
        w[:,b,:]=wb
    return w


class ExactGraph:
    def __init__(self, ref, labels, gamma, selected_origins, val_queries, beta=.4, lam=.8, horizon=12):
        self.x=np.asarray(ref,float);self.labels=np.asarray(labels,int)
        self.n,self.d=self.x.shape;self.k=int(self.labels.max())+1
        self.origins=np.asarray(selected_origins,int)
        self.coeff=torch.tensor(coefficients(beta,lam,horizon),dtype=DT)
        self.H=horizon
        self.yhot=torch.tensor(np.eye(self.k)[self.labels],dtype=DT)
        self.available=torch.tensor(np.isfinite(self.x),dtype=torch.bool)
        base=wgraph(self.x,gamma)
        self.base=torch.tensor(base,dtype=DT) # [b,i,j] with diagonal excluded
        self.start=torch.tensor(base[:,self.origins,:].transpose(1,2,0),dtype=DT) # [origin,j,b]
        # For all later steps exclude the starting sample and normalize each feature's destination distribution
        banned=np.broadcast_to(base[None,...],(len(self.origins),self.d,self.n,self.n)).copy()
        for oi, origin in enumerate(self.origins):
            banned[oi,:,:,origin]=0.0
        den=banned.sum(axis=-1,keepdims=True)
        np.divide(banned,den,out=banned,where=den>0)
        self.transitions=torch.tensor(banned,dtype=DT) # [origin,b,i,j]
        self.valid=torch.tensor((den[...,0]>1e-12).transpose(0,2,1),dtype=torch.bool) # [origin,i,b]
        self.valid_start=self.available[self.origins] # [origin,b]
        self.val_start=torch.tensor(heldout_weights(val_queries,self.x,gamma),dtype=DT) # [query,b,j]
        self.val_available=torch.tensor(np.isfinite(val_queries),dtype=torch.bool)
        self.Q=torch.einsum('bij,jk->ibk',self.base,self.yhot)
        self.Qstart=self.Q[self.origins]
        self.Qinter=torch.einsum('sbij,jk->sibk',self.transitions,self.yhot)
        if not self.valid.all(dim=-1).any(dim=-1).all():
            raise ValueError('Some reference states have no legal feature')
        if not self.valid_start.any(-1).all():
            raise ValueError('Some starts have no observed feature')

    @staticmethod
    def masked_softmax(logits,mask):
        return torch.softmax(logits.masked_fill(~mask,-1e20),dim=-1)

    def local_policy(self,z):
        # Policy of (current sample i, previous feature a) for each origin s.
        # Invalid destination features may differ by origin; mask before softmax.
        return self.masked_softmax(z[:self.d][None,None,:,:].expand(len(self.origins),self.n,-1,-1),self.valid[:,:,:,None].transpose(-1,-2).expand(-1,-1,self.d,-1))

    def center_train(self,z,origins_mask,return_all=False):
        """Exact E[normalized walk label histogram], conditional on each train origin.

        Original sampler: choose b with masked softmax Z[a,b], choose destination
        with feature-specific kernel; exclude current node and initial origin.
        """
        idx=torch.as_tensor(origins_mask,dtype=torch.long)
        pos=self.local_policy(z)[idx]  # [m,n,prev_feature,next_feature]
        first_policy=self.masked_softmax(z[self.d][None,:].expand(len(idx),-1),self.valid_start[idx])
        state=self.start[idx]*first_policy[:,None,:] # [m,j,b], state *after* 1st hop
        rec=[]
        for t in range(self.H):
            rec.append(torch.einsum('mjb,jk->mk',state,self.yhot))
            if t+1==self.H:break
            choose=torch.einsum('mja,mjab->mjb',state,pos)
            # [m,b,i] times [m,b,i,j] -> [m,j,b]
            batch=choose.permute(0,2,1).reshape(-1,1,self.n)
            transitions=self.transitions[idx].reshape(-1,self.n,self.n)
            state=torch.bmm(batch,transitions).reshape(len(idx),self.d,self.n).transpose(1,2)
        out=(torch.stack(rec,dim=1)*self.coeff[None,:,None]).sum(dim=1)
        if not torch.allclose(out.sum(-1),torch.ones(len(idx),dtype=DT),atol=1e-8):
            raise AssertionError('Unexpected lost walk probability')
        return out if return_all else out.mean(dim=0)

    def center_one_step(self,z,origins_mask):
        idx=torch.as_tensor(origins_mask,dtype=torch.long)
        p=self.masked_softmax(z[self.d][None,:].expand(len(idx),-1),self.valid_start[idx])
        return torch.einsum('mb,mbk->k',p,self.Qstart[idx])/len(idx)

    def local_q_loss(self,z,origins_mask,target):
        """Closest analog to prior local-Q: each (origin,current,entry-feature) aims at target.

        Both methods optimize the same constrained class-specific (d+1)*d Z.
        This objective matches each *one-step* label distribution against target,
        not the realized multiple-step class center. All states are weighted
        equally (not by their visitation frequency).
        """
        idx=torch.as_tensor(origins_mask,dtype=torch.long)
        reg=self.local_policy(z)[idx] # [m,n,a,b]
        q=self.Qinter[idx] # [m,n,b,k]
        dist=torch.einsum('miab,mibk->miak',reg,q)
        target_b=target.reshape(1,1,1,-1)
        # Previous feature must exist at the reached sample: (i,a) is otherwise
        # not a graph node and must not enter the local-Q loss.
        valid_state=(self.available[None,:,:] & self.valid[idx].any(-1)[:,:,None]) # [m,n,a]
        loss_inter=(jsd(dist,target_b)*valid_state).sum()/valid_state.sum()
        first=self.masked_softmax(z[self.d][None,:].expand(len(idx),-1),self.valid_start[idx])
        dist_start=torch.einsum('mb,mbk->mk',first,self.Qstart[idx])
        loss_start=jsd(dist_start,target[None,:]).mean()
        return 0.5*(loss_inter+loss_start)

    def evaluate_val(self,z):
        # For heldout query, its origin isn't among reference nodes, so no origin banning.
        first_pi=self.masked_softmax(z[self.d][None,:].expand(len(self.val_start),-1),self.val_available)
        state=self.val_start.permute(0,2,1)*first_pi[:,None,:]
        regular=self.masked_softmax(z[:self.d][None,:,:].expand(self.n,-1,-1),self.available[:,None,:].expand(-1,self.d,-1))
        rec=[]
        for t in range(self.H):
            rec.append(torch.einsum('qib,ik->qk',state,self.yhot))
            if t+1==self.H:break
            choose=torch.einsum('qia,iab->qib',state,regular)
            state=torch.stack([choose[:,:,b]@self.base[b] for b in range(self.d)],dim=-1)
        return (torch.stack(rec,dim=1)*self.coeff[None,:,None]).sum(dim=1)


def config_run(nref=72,steps=70,seed=20261107,beta=.4,lam=.8,horizon=12,lr=.12, verbose=False):
    torch.set_num_threads(1)
    spec=load_config(ROOT/'code/data_generator/config.json')
    with tempfile.TemporaryDirectory() as tmp:
        data=generate(spec,tmp,finite_q=False)
    x,y=data['train'][0],data['train'][2]; vx,vy=data['validation'][0],data['validation'][2]
    rng=np.random.default_rng(seed)
    # balanced stratified fixed reference subset, original 300-train split only
    chosen=np.concatenate([rng.choice(np.flatnonzero(y==c),size=nref//len(spec['pi']),replace=False)
                           for c in range(len(spec['pi']))])
    rng.shuffle(chosen)
    x,y=x[chosen],y[chosen]
    origins=np.arange(len(x))
    g=ExactGraph(x,y,spec['gamma'],origins,vx,beta=beta,lam=lam,horizon=horizon)
    k=g.k;d=g.d
    target=(1-.08)*np.eye(k)+.08*np.ones((k,k))/k
    results={'meta':{'description':'Controlled one-step local-Q vs exact-multistep optimization on identical shared masked class-conditional policy matrices',
                     'date':'2026-11-07','seed':seed,'data_seed':spec['seed'],'n_train_original':len(data['train'][2]),
                     'n_ref':len(x),'n_val':len(vx),'n_features':d,'n_classes':k,
                     'beta':beta,'lambda':lam,'horizon':horizon,'steps':steps,'learning_rate':lr,
                     'reference_subset_indices':chosen.tolist(),'targets':target.tolist(),
                     'policy_shape':[d+1,d],'regime':'class-conditional oracle training; candidate-specific validation with no label used in walks',
                     'length_step_coefficients':g.coeff.numpy().tolist()},'results':[]}
    learned={}
    for objective in ['uniform','local_q','exact_multistep']:
        zs=[];centers=[];record=[]
        for c in range(k):
            o=np.flatnonzero(y==c)
            u=torch.tensor(target[c],dtype=DT)
            z=torch.nn.Parameter(torch.zeros((d+1,d),dtype=DT))
            trace=[]
            if objective!='uniform':
                opt=torch.optim.Adam([z],lr=lr)
                for step in range(steps):
                    opt.zero_grad()
                    loss=g.local_q_loss(z,o,u) if objective=='local_q' else jsd(g.center_train(z,o),u)
                    loss.backward()
                    opt.step()
                    if (step+1)%10==0 or step==0 or step==steps-1:
                        with torch.no_grad():
                            actual=jsd(g.center_train(z,o),u).item()
                            trace.append({'step':step+1,'surrogate':float(loss.item()),'actual_center_JSD':actual})
                if verbose: print(objective,c,trace[-1],flush=True)
            with torch.no_grad():
                v=g.center_train(z,o).numpy();err=jsd(torch.tensor(v),u).item()
                zs.append(z.detach().clone());centers.append(v)
                record.append({'class':c,'target_center':target[c].tolist(),'realized_center':v.tolist(),
                               'JSD_to_target':err,'Z':z.detach().numpy().tolist(),'optimization_trace':trace})
        centers=np.array(centers)
        # Validation: evaluate each class hypothesis separately and compare to that class' frozen center.
        with torch.no_grad():
            distances=np.stack([jsd(g.evaluate_val(zs[c]),torch.tensor(centers[c])[None,:]).numpy()
                                for c in range(k)],axis=1)
            predictions=distances.argmin(axis=1)
            BA=float(np.mean([(predictions[vy==c]==c).mean() for c in range(k)]))
            accuracy=float((predictions==vy).mean())
        per_class=np.array([e['JSD_to_target'] for e in record]);mean=per_class.mean()
        item={'objective':objective,'mean_center_target_JSD':float(mean),'max_center_target_JSD':float(per_class.max()),
              'class_target_JSD':per_class.tolist(),'validation_accuracy':accuracy,'validation_balanced_accuracy':BA,
              'centers':centers.tolist(),'per_class':record}
        results['results'].append(item)
        learned[objective]=zs
    return results,g,learned


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--nref',type=int,default=72)
    parser.add_argument('--steps',type=int,default=70)
    parser.add_argument('--seed',type=int,default=20261107)
    parser.add_argument('--lr',type=float,default=.12)
    parser.add_argument('--out',type=Path,default=Path(__file__).parent/'results/comparison.json')
    args=parser.parse_args()
    results,_,_=config_run(nref=args.nref,steps=args.steps,seed=args.seed,lr=args.lr,verbose=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(results,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print('RESULTS',args.out)
    for r in results['results']:
        print(f"{r['objective']:17s} JSD={r['mean_center_target_JSD']:.8f} "
              f"val_BA={r['validation_balanced_accuracy']:.4f}")

if __name__=='__main__':main()
