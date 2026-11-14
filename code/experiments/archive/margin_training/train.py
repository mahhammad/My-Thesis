"""Exact class-independent sigmoid JS-margin policy training.
Standalone experiment; does not optimize I, R or a target center.
Run from repository root: python code/experiments/margin_training/train.py ...
"""
from pathlib import Path
import argparse, json, sys, tempfile, time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'code/data_generator'))
sys.path.insert(0,str(ROOT/'code/experiments/multistep_policy_target_match'))
sys.path.insert(0,str(ROOT/'code/experiments/diag_max_full300'))
from generate import load_config, generate
from run import DT, jsd
from stream_graph import StreamGraph

LN2=np.log(2.0)

def build_data(seed=20261107,nref=72):
    spec=load_config(ROOT/'code/data_generator/config.json')
    with tempfile.TemporaryDirectory() as outdir:
        splits=generate(spec,outdir,finite_q=False)
    xx,_,yy=splits['train']
    val,_,vy=splits['validation']
    rng=np.random.default_rng(seed)
    if nref==len(yy):
        ids=np.arange(nref)
    else:
        assert nref%len(spec['pi'])==0
        ids=np.concatenate([
            rng.choice(np.flatnonzero(yy==c),nref//len(spec['pi']),replace=False)
            for c in range(len(spec['pi']))])
    return spec,xx[ids],yy[ids],val,vy,ids

def outputs(graph,z,y):
    idx=np.arange(len(y))
    reps=graph.center_train(z,idx,return_all=True)
    centers=torch.stack([
        reps[torch.as_tensor(y==c)].mean(dim=0)
        for c in range(graph.k)])
    scores=1-jsd(reps[:,None,:],centers[None,:,:])/LN2
    target=torch.as_tensor(y,dtype=torch.long)
    true=scores.gather(1,target[:,None]).squeeze(-1)
    nearest_rival=scores.masked_fill(
        torch.nn.functional.one_hot(target,graph.k).bool(),-1e9).max(1).values
    return reps,centers,true-nearest_rival

def balanced(x,y,k):
    return torch.stack([
        x[torch.as_tensor(y==c)].mean() for c in range(k)]).mean()

def training_metrics(reps,centers,margin,y,tau):
    cent=centers.detach().numpy()
    entropy=lambda x:-(x*np.log(np.maximum(x,1e-300))).sum(-1)
    j=jsd(reps,centers[torch.as_tensor(y,dtype=torch.long)])
    pair=[jsd(centers[c],centers[r]) for c in range(len(cent))
          for r in range(c+1,len(cent))]
    return {'loss':float(balanced(torch.sigmoid(-margin/tau),y,len(cent)).detach()),
            'balanced_accuracy':float(1-balanced((margin<=0).double(),y,len(cent)).detach()),
            'I':float(entropy(cent.mean(0))-entropy(cent).mean()),
            'R_JS':float(balanced(j,y,len(cent)).detach()),
            'minimum_center_sqrt_JSD':float(torch.sqrt(torch.stack(pair).min()).detach()),
            'margin_q10':float(np.quantile(margin.detach().numpy(),.10)),
            'centers':cent.tolist()}

def validation_metrics(graph,z,centers,vy,tau):
    with torch.no_grad():
        reps=graph.evaluate_val(z)
        score=1-jsd(reps[:,None,:],centers[None,:,:])/LN2
        y=torch.as_tensor(vy,dtype=torch.long)
        pred=score.argmax(1)
        true=score.gather(1,y[:,None]).squeeze(1)
        rival=score.masked_fill(
            torch.nn.functional.one_hot(y,graph.k).bool(),-1e9).max(1).values
        m=true-rival
        return {'balanced_accuracy':float(balanced((pred==y).double(),vy,graph.k)),
                'ordinary_accuracy':float((pred==y).double().mean()),
                'sigmoid_loss':float(balanced(torch.sigmoid(-m/tau),vy,graph.k)),
                'margin_q10':float(np.quantile(m.numpy(),.10)),
                'predicted':pred.tolist()}

def fit(seed=20261107,nref=72,tau=.005,steps=50,lr=.12):
    torch.set_num_threads(1)
    spec,x,y,val,vy,ids=build_data(seed,nref)
    graph=StreamGraph(x,y,spec['gamma'],val,beta=.4,lam=.8,horizon=12)
    z=torch.nn.Parameter(torch.zeros((graph.d+1,graph.d),dtype=DT))
    optim=torch.optim.Adam([z],lr=lr)
    trace=[]
    t0=time.perf_counter()
    for step in range(steps+1):
        reps,centers,margin=outputs(graph,z,y)
        loss=balanced(torch.sigmoid(-margin/tau),y,graph.k)
        if step==0 or step==steps or step%10==0:
            trace.append({'step':step,'train':training_metrics(reps,centers,margin,y,tau),
                          'validation':validation_metrics(graph,z,centers.detach(),vy,tau),
                          'elapsed_s':time.perf_counter()-t0})
            row=trace[-1]
            print('STEP',step,'LOSS',round(row['train']['loss'],6),
                  'TRAIN_BA',round(row['train']['balanced_accuracy'],6),
                  'VAL_BA',round(row['validation']['balanced_accuracy'],6),flush=True)
        if step==steps:break
        optim.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(z.grad).all():raise ValueError('Nonfinite gradient')
        optim.step()
    return {'meta':{'data_seed':spec['seed'],'reference_seed':seed,'nref':nref,
                    'tau':tau,'steps':steps,'lr':lr,'beta':.4,'lambda':.8,'H':12,
                    'policy':'single shared Z; zero start',
                    'objective':'balanced sigmoid JS margin only',
                    'validation_used_for_training':False,
                    'test_used':False},
            'history':trace,'Z':z.detach().tolist(),
            'reference_ids':ids.tolist()}

def check_gradient(seed=20261107,nref=72,tau=.005):
    torch.set_num_threads(1)
    spec,x,y,val,vy,ids=build_data(seed,nref)
    graph=StreamGraph(x,y,spec['gamma'],val,horizon=12)
    z=torch.tensor(np.random.default_rng(11).normal(0,.3,size=(graph.d+1,graph.d)),
                   dtype=DT,requires_grad=True)
    def loss(v):
        _,_,margin=outputs(graph,v,y)
        return balanced(torch.sigmoid(-margin/tau),y,graph.k)
    auto=torch.autograd.grad(loss(z),z)[0].detach().numpy()
    eps=1e-4
    checks=[]
    for idx in [(0,0),(1,2),(4,4),(5,0),(5,3)]:
        plus=z.detach().clone();minus=plus.clone()
        plus[idx]+=eps;minus[idx]-=eps
        fd=float((loss(plus)-loss(minus))/(2*eps))
        checks.append({'index':idx,'autograd':float(auto[idx]),
                       'finite_difference':fd,
                       'absolute_error':abs(fd-float(auto[idx]))})
    assert max(v['absolute_error'] for v in checks)<2e-5
    return {'checks':checks,'max_absolute_error':max(v['absolute_error'] for v in checks)}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=20261107)
    p.add_argument('--nref',type=int,default=72)
    p.add_argument('--tau',type=float,default=.005)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--lr',type=float,default=.12)
    p.add_argument('--check-gradient',action='store_true')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    obj=check_gradient(a.seed,a.nref,a.tau) if a.check_gradient else fit(
        seed=a.seed,nref=a.nref,tau=a.tau,steps=a.steps,lr=a.lr)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if a.check_gradient:print(json.dumps(obj,indent=2))
