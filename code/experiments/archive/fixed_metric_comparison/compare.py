"""Reproduce fixed-policy walk evaluation with extra training metrics; no training.
Base semantics mirror code/policy_evaluator/walks.py fetched from GitHub chapter4.
"""
import json, time
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent
REPO=BASE.parents[2]
D=REPO/'generated_data'
conf=json.loads((REPO/'code/data_generator/config.json').read_text())
gamma=np.asarray(conf['kernel_gamma'],float)
beta=.4
lam=.8
horizon=12
nwalks=300
seed=20261103

def load(name):
    with np.load(D/f'{name}.npz') as f: return f['x'].astype(float), f['y'].astype(int)

tr_x,tr_y=load('train')
val_x,val_y=load('validation')
test_x,test_y=load('test')
K=int(tr_y.max())+1
n,d=tr_x.shape
uniform=np.zeros((d+1,d))
guided=np.array([[1.65,.2,-1.2,.9,-1.2],[1.4,.45,-1.2,.9,-1.2],
 [1.4,.2,-.95,.9,-1.2],[1.4,.2,-1.2,1.15,-1.2],
 [1.4,.2,-1.2,.9,-.95],[1.4,.2,-1.2,.9,-1.2]],float)

def weights(x,refs,selfids=None):
    nq,d=x.shape; nr=len(refs); out=np.zeros((nq,d,nr))
    for a in range(d):
        for i in np.flatnonzero(np.isfinite(x[:,a])):
            valid=np.isfinite(refs[:,a]);
            if selfids is not None and selfids[i]>=0:
                valid=valid.copy();valid[selfids[i]]=False
            js=np.flatnonzero(valid)
            if not len(js):continue
            lw=-gamma[a]*(x[i,a]-refs[js,a])**2
            lw-=lw.max()
            out[i,a,js]=np.exp(lw)
    return out

wref=weights(tr_x,tr_x,np.arange(n))
totals=wref.sum(axis=2)
cum=np.cumsum(wref,axis=2)


def representation(queries,logits,train_ids=None):
    nq=len(queries)
    ids=np.full(nq,-1,int) if train_ids is None else np.asarray(train_ids,int)
    starts=weights(queries,tr_x,ids)
    seeds=np.random.SeedSequence(seed).spawn(nq)
    neighborhoods=np.zeros((nq,K))
    for i in range(nq):
        rng=np.random.default_rng(seeds[i]);origin=int(ids[i]);start_weights=starts[i]
        lengths=np.minimum(rng.geometric(beta,size=nwalks),horizon)
        z=(1-lam**lengths)/(1-lam)
        result=np.zeros((nwalks,K),float)
        current=np.full(nwalks,-1,int); prev=np.full(nwalks,d,int)
        for t in range(1,int(lengths.max())+1):
            sel=np.flatnonzero(lengths>=t)
            curr=current[sel]; first=curr<0
            allowed=np.empty((len(sel),d),float)
            if first.any():allowed[first]=start_weights.sum(axis=1)
            if (~first).any():
                allowed[~first]=totals[curr[~first]]
                if origin>=0:allowed[~first]-=wref[curr[~first],:,origin]
            good=allowed>1e-13
            if not np.all(good.any(axis=1)):raise ValueError('dead end')
            zlog=np.where(good,logits[prev[sel]],-np.inf)
            p=np.exp(zlog-np.max(zlog,axis=1,keepdims=True));p/=p.sum(axis=1,keepdims=True)
            v=rng.random(len(sel)); feat=(np.cumsum(p,axis=1)<v[:,None]).sum(axis=1)
            cc=np.empty((len(sel),n),float); ft=np.empty(len(sel),float); ban=np.zeros(len(sel))
            if first.any():
                cc[first]=np.cumsum(start_weights[feat[first]],axis=1);ft[first]=cc[first,-1]
            if (~first).any():
                cc[~first]=cum[curr[~first],feat[~first],:]
                ft[~first]=totals[curr[~first],feat[~first]]
                if origin>=0:ban[~first]=wref[curr[~first],feat[~first],origin]
            v2=rng.random(len(sel)); thresh=np.maximum(np.finfo(float).tiny,v2*(ft-ban))
            if origin>=0:
                prefix=cc[:,origin]-ban;thresh+=(thresh>=prefix)*ban
            dest=(cc<thresh[:,None]).sum(axis=1)
            if (dest>=n).any() or (dest==curr).any() or (origin>=0 and (dest==origin).any()):raise ValueError('destination')
            wt=lam**(t-1)/z[sel]
            np.add.at(result,(sel,tr_y[dest]),wt)
            current[sel]=dest;prev[sel]=feat
        neighborhoods[i]=result.mean(axis=0)
    return neighborhoods

def entropy(p):return -np.sum(np.where(p>0,p*np.log(np.maximum(p,1e-300)),0),axis=-1)
def jsd(p,q):
    m=.5*(p+q);return np.clip(entropy(m)-.5*(entropy(p)+entropy(q)),0,np.log(2))
def kl(p,q):
    return np.sum(np.where(p>0,p*(np.log(np.maximum(p,1e-300))-np.log(np.maximum(q,1e-300))),0),axis=-1)
def metrics(train_p, eval_p, y, center=None):
    center=np.stack([train_p[tr_y==c].mean(axis=0) for c in range(K)]) if center is None else center
    j=jsd(eval_p[:,None,:],center[None,:,:]);score=1-j/np.log(2)
    preds=np.argmax(score,axis=1)
    ba=np.mean([(preds[y==c]==c).mean() for c in range(K)])
    accuracy=np.mean(preds==y)
    weight=np.array([1.0/(K*(y==c).sum()) for c in y]);
    mine=score[np.arange(len(y)),y]
    maxr=np.max(np.where(np.arange(K)[None,:]==y[:,None],-np.inf,score),axis=1)
    margin=mine-maxr
    losses={}
    for tau in [.03,.08,.15]:
        # stable positive-softplus over all class rivals, surrogate 0-1 margin
        dif=(score-mine[:,None])/tau
        dif[np.arange(len(y)),y]=-np.inf
        mx=np.maximum(0,dif.max(axis=1));ell=tau*(mx+np.log(np.exp(-mx)+np.exp(dif-mx[:,None]).sum(axis=1)))
        losses[f'margin_softplus_{tau}']=float(weight@ell)
    dist=kl(eval_p[:,None,:],center[None,:,:])
    nll=np.logaddexp.reduce(-dist,axis=1)+dist[np.arange(len(y)),y]
    sigma=dist.std(axis=1)
    norm_scores=np.divide(dist.mean(axis=1,keepdims=True)-dist,sigma[:,None],where=(sigma[:,None]>1e-14),out=np.zeros_like(dist))
    nllnorm=np.logaddexp.reduce(norm_scores,axis=1)-norm_scores[np.arange(len(y)),y]
    o={
    'accuracy':float(accuracy),'balanced_acc':float(ba),'balanced_error':float(1-ba),
    'mean_true_margin':float(weight@margin),
    'error_margin_nonpositive':float(np.sum(weight*(margin<=0))),
    'mean_win_margin':float(np.mean(np.sort(score,axis=1)[:,-1]-np.sort(score,axis=1)[:,-2])),
    'nll_bal':float(weight@nll),'nllnorm_bal':float(weight@nllnorm),
    'within_jsd':float(weight@j[np.arange(len(y)),y]),
    'mean_correct_jsd':float(j[np.arange(len(y)),y].mean()),
    **losses
    }
    return o,center

def global_geometry(P,C):
    avg=C.mean(axis=0)
    mi=entropy(avg)-entropy(C).mean()
    dij=jsd(C[:,None,:],C[None,:,:])
    minsep=np.sqrt(dij[~np.eye(K,dtype=bool)].min())
    # original theoretical KL margin G_{c,r}=var/(var+mean^2)
    vals=[]
    for c in range(K):
        for r in range(K):
            if c==r:continue
            coeff=np.log(np.maximum(C[c],1e-300)/np.maximum(C[r],1e-300))
            arr=P[tr_y==c]@coeff;mean=arr.mean();var=arr.var()
            vals.append(var/(var+mean**2) if var+mean**2>1e-15 else 1.0)
    return dict(center_MI=float(mi),min_center_sqrt_JSD=float(minsep),J_margin=float(np.mean(vals)*K*(K-1)/K))

if __name__=='__main__':
    t0=time.time();out=[]
    for alpha in [0,.25,.5,.75,1]:
        z=alpha*guided
        Pt=representation(tr_x,z,np.arange(n))
        Pv=representation(val_x,z)
        Ptest=representation(test_x,z)
        Rtr,C=metrics(Pt,Pt,tr_y)
        Rval,_=metrics(Pt,Pv,val_y,C)
        Rtest,_=metrics(Pt,Ptest,test_y,C)
        G=global_geometry(Pt,C)
        o=dict(alpha=alpha,train=Rtr,validation=Rval,test=Rtest,geometry=G)
        out.append(o)
        print(f'alpha={alpha:.2f} train={Rtr["accuracy"]:.4f} val={Rval["accuracy"]:.4f} test={Rtest["accuracy"]:.4f} valNLLnorm={Rval["nllnorm_bal"]:.4f} I={G["center_MI"]:.4f} J={G["J_margin"]:.4f} sec={time.time()-t0:.2f}',flush=True)
    (BASE/'results').mkdir(parents=True,exist_ok=True)
    (BASE/'results/fixed_metric_comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))