"""E08 independent forward recursion and independent random-walk Monte Carlo.

All graph kernels are created by the canonical Benchmark engine; the
propagation and path sampler below do not call its representation methods.
Only shared fixed logits; never uses Validation labels for inference.
"""
import numpy as np
import torch

BETA=.4
LAM=.8
H=12
DEPTHS=(2,3,4,6,8,12)
MC_BUDGETS=(128,512,2048)
MC_SEEDS=(20261121,20261122)
MC_ORIGINS={"train":(0,149,299),"validation":(0,44,89)}


def forward_steps(g,z,split,max_depth=H,origin_positions=None):
    from engine import masked_softmax, DT
    if split not in ("train","validation"):
        raise ValueError("Only Train and Validation supported")
    if not 1 <= max_depth <= H:
        raise ValueError("Invalid depth")
    g.check_logits(z,"shared")
    with torch.no_grad():
        if split == "train":
            origins = (torch.arange(g.n,dtype=torch.long) if origin_positions is None
                       else torch.as_tensor(origin_positions,dtype=torch.long))
            if not bool(g.available[origins].any(dim=1).all()):
                raise ValueError("Train origin has no available start feature")
        elif not bool(g.qavailable[(torch.arange(len(g.qx)) if origin_positions is None
                                  else torch.as_tensor(origin_positions,dtype=torch.long))].any(dim=1).all()):
            raise ValueError("Validation origin has no available start feature")
        if split=="train":
            ids=(torch.arange(g.n,dtype=torch.long) if origin_positions is None
                 else torch.as_tensor(origin_positions,dtype=torch.long))
            m=len(ids)
            den=g.mass.unsqueeze(0)-g.base[:,:,ids].permute(2,1,0)
            valid=(den>1e-12)&g.available.unsqueeze(0)
            mid=z[:g.d].expand(g.n,g.d,g.d)
            pol=masked_softmax(mid.unsqueeze(0).expand(m,-1,-1,-1),
                               valid[:,:,None,:].expand(-1,-1,g.d,-1))
            p0=masked_softmax(z[g.d].expand(m,g.d),g.available[ids])
            state=g.base[:,ids,:].permute(1,2,0)*p0[:,None,:]
        else:
            if g.qx is None:
                raise AssertionError("Validation graph unavailable")
            ids=(torch.arange(len(g.qx),dtype=torch.long)
                 if origin_positions is None
                 else torch.as_tensor(origin_positions,dtype=torch.long))
            m=len(ids)
            p0=masked_softmax(z[g.d].expand(m,g.d),g.qavailable[ids])
            state=g.qweights[ids].permute(0,2,1)*p0[:,None,:]
            valid=(g.mass>1e-12)&g.available
            mid=z[:g.d].expand(g.n,g.d,g.d)
            pol=masked_softmax(mid,valid[:,None,:].expand(-1,g.d,-1))
        steps=[]
        for depth in range(max_depth):
            dist=(state.sum(2)@g.yhot).cpu().numpy()
            if not np.isfinite(dist).all() or not np.allclose(
                    dist.sum(1),1,atol=1e-9,rtol=0):
                raise AssertionError("Path probability mass lost")
            steps.append(dist)
            if depth==max_depth-1:
                break
            if split=="train":
                feature=torch.einsum("mja,mjab->mjb",state,pol)
                nxt=[]
                for b in range(g.d):
                    mval=torch.where(den[:,:,b]>1e-12,
                                     feature[:,:,b]/den[:,:,b].clamp_min(1e-12),
                                     torch.zeros_like(feature[:,:,b]))
                    dest=mval@g.base[b]
                    dest=dest.scatter(1,ids[:,None],torch.zeros((m,1),dtype=DT))
                    nxt.append(dest)
                state=torch.stack(nxt,dim=2)
            else:
                feature=torch.einsum("mja,jab->mjb",state,pol)
                state=torch.stack([feature[:,:,b]@g.base[b] for b in range(g.d)],dim=2)
        return np.stack(steps,axis=0)


def expectation(steps,alpha,depth):
    a=np.asarray(alpha[:depth],dtype=float)
    return np.tensordot(a,steps[:depth],axes=(0,0))/float(a.sum())


def tv(a,b):
    return .5*np.abs(np.asarray(a)-np.asarray(b)).sum(-1)


def centers(p,labels,k):
    return np.stack([p[labels==c].mean(0) for c in range(k)])


def balanced_mean(values,labels,k):
    return float(np.mean([np.mean(np.asarray(values)[labels==c]) for c in range(k)]))


def analyze_depths(tr_steps,va_steps,alpha,train_y,val_y):
    k=len(np.unique(train_y))
    base_tr=expectation(tr_steps,alpha,H)
    base_va=expectation(va_steps,alpha,H)
    base_c=centers(base_tr,train_y,k)
    rows=[]
    for depth in DEPTHS:
        tr=expectation(tr_steps,alpha,depth)
        va=expectation(va_steps,alpha,depth)
        c=centers(tr,train_y,k)
        ttr=tv(tr,base_tr);tva=tv(va,base_va);tc=tv(c,base_c)
        eps=float(np.sum(alpha[depth:]))
        if max(ttr.max(),tva.max(),tc.max())>eps+1e-9:
            raise AssertionError("Theory bound violated")
        rows.append({
            "r":depth,"epsilon_tail":eps,
            "train_mean_balanced_tv":balanced_mean(ttr,train_y,k),
            "validation_mean_balanced_tv":balanced_mean(tva,val_y,k),
            "train_max_tv":float(ttr.max()),
            "validation_max_tv":float(tva.max()),
            "class_centers_max_tv":float(tc.max()),
            "train_tv_each":ttr.tolist(),
            "validation_tv_each":tva.tolist(),
            "center_tv_each":tc.tolist(),
            "bound_verified":True
        })
    return rows


def sample_paths(g,z,split,origin,seed,nwalks):
    """Independent path simulator: sample length and feature/destination each hop."""
    rng=np.random.default_rng(np.random.SeedSequence([seed,
        0 if split=="train" else 1,int(origin)]))
    base=g.base.detach().cpu().numpy()
    masses=g.mass.detach().cpu().numpy()
    avail=g.available.detach().cpu().numpy()
    q=g.qweights.detach().cpu().numpy() if split=="validation" else None
    qav=g.qavailable.detach().cpu().numpy() if split=="validation" else None
    labels=g.labels.detach().cpu().numpy()
    logits=z.detach().cpu().numpy()
    weights=LAM**np.arange(H,dtype=float)
    sums=np.cumsum(weights)
    out=np.zeros((nwalks,g.k),dtype=float)
    lengths=np.empty(nwalks,dtype=int)
    for i in range(nwalks):
        length=min(int(rng.geometric(BETA)),H)
        lengths[i]=length
        node=int(origin) if split=="train" else -1
        prev=g.d
        for t in range(length):
            if t==0:
                allowed=avail[node] if split=="train" else qav[origin]
            else:
                allowed=avail[node]&(masses[node]>1e-12)
                if split=="train":
                    allowed=allowed&(masses[node]-base[:,node,origin]>1e-12)
            good=np.flatnonzero(allowed)
            if len(good)==0:
                raise AssertionError("Dead-end in Monte Carlo path")
            p=np.exp(logits[prev,good]-np.max(logits[prev,good]))
            p=p/p.sum()
            feature=int(rng.choice(good,p=p))
            if t==0 and split=="validation":
                dest=q[origin,feature].copy()
            else:
                dest=base[feature,node].copy()
                if split=="train":
                    dest[origin]=0
            if dest.sum()<=1e-12:
                raise AssertionError("No legal destination")
            dest/=dest.sum()
            next_node=int(rng.choice(g.n,p=dest))
            if split=="train" and next_node==origin:
                raise AssertionError("Return to origin")
            if t>0 and next_node==node:
                raise AssertionError("Vertical self-move")
            out[i,labels[next_node]]+=weights[t]/sums[length-1]
            prev=feature
            node=next_node
    return out,lengths


def monte_carlo(g,policies,exact_train,exact_validation):
    items=[]
    for pindex,(name,z) in enumerate(policies.items()):
        for split,origins,truth in (
            ("train",MC_ORIGINS["train"],exact_train[name]),
            ("validation",MC_ORIGINS["validation"],exact_validation[name])
        ):
            for origin in origins:
                for seed in MC_SEEDS:
                    paths,lengths=sample_paths(
                        g,z,split,origin,seed,max(MC_BUDGETS))
                    partial=paths.cumsum(axis=0)
                    for n in MC_BUDGETS:
                        value=partial[n-1]/n
                        items.append({
                            "policy":name,"split":split,"origin_position":int(origin),
                            "mc_seed":seed,"nwalks":n,"estimated":value.tolist(),
                            "exact":truth[origin].tolist(),
                            "tv_to_exact":float(tv(value,truth[origin])),
                            "mean_sampled_path_length":float(lengths[:n].mean())
                        })
    summary=[]
    for name in policies:
        for n in MC_BUDGETS:
            errors=[p["tv_to_exact"] for p in items
                    if p["policy"]==name and p["nwalks"]==n]
            summary.append({"policy":name,"nwalks":n,
                            "mean_tv":float(np.mean(errors)),
                            "max_tv":float(np.max(errors)),
                            "observations":len(errors)})
    return {"raw":items,"summary":summary,
            "nested_budgets":True,
            "mc_seeds_independent_of_data_generator":True}
