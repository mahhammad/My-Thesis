"""Observe iterative center-target matching under full multi-step random walks.

This is a *diagnostic experiment*, not a trained deployable classifier.
Each training-origin trajectory knows its origin class, and each held-out
candidate is simulated under that hypothesized class without using test labels.
All feature decisions, including intermediate nodes, optimize one-step Q toward
current target class center, recomputing Q with forbidden original origin.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'code' / 'policy_evaluator'))
from walks import ReferenceGraph, load_data, load_policy, _weights  # noqa: E402
from metrics import class_centers, center_metrics, jsd, report  # noqa: E402
from projection import optimize_mixtures  # noqa: E402


def normalized_q(numerators, totals, *, zero_tol=1e-12):
    """Normalized one-step label distributions, NaN for forbidden features."""
    q = np.full(numerators.shape,np.nan,dtype=float)
    np.divide(numerators,totals[...,None],out=q,where=totals[...,None]>zero_tol)
    return q


class CenterPolicyWalker:
    """Exact origin-aware Q + batched local policy optimization + MC walks."""
    def __init__(self, graph, beta=0.4, lam=0.8, horizon=12, nwalks=300,
                 optimizer_steps=65, optimizer_tol=1e-7):
        self.graph=graph
        self.beta,self.lam,self.horizon,self.nwalks=beta,lam,horizon,nwalks
        self.optimizer_steps=optimizer_steps
        self.optimizer_tol=optimizer_tol
        self.n,self.d=graph.x.shape;self.k=graph.nclasses
        self.oh=np.eye(self.k)[graph.y]
        self.num=graph.weights @ self.oh
        self.den=graph.totals
        self.q_ref=normalized_q(self.num,self.den)
        if not np.isfinite(self.q_ref).any(axis=2).any(axis=1).all():
            raise ValueError('reference contains a node without permissible transition')
        self._base_policy_cache={}

    def _opt(self, q, target):
        flat=q.reshape(-1,self.d,self.k)
        policies, residuals, gaps=optimize_mixtures(flat,target,steps=self.optimizer_steps,tol=self.optimizer_tol)
        return policies.reshape(q.shape[:-1]), residuals.reshape(q.shape[:-2]), gaps.reshape(q.shape[:-2])

    def origin_q(self, origin):
        """Q for all reference states, excluding current and original origin."""
        banned=self.graph.weights[:,:,origin]
        numer=self.num-banned[...,None]*self.oh[origin]
        denom=self.den-banned
        return normalized_q(np.maximum(numer,0),np.maximum(denom,0))

    def origin_policies(self, origin, center):
        return self._opt(self.origin_q(origin),center)

    def reference_policies(self, center):
        return self._opt(self.q_ref,center)

    def start_q(self, start_weights):
        return normalized_q(start_weights@self.oh,start_weights.sum(axis=2))

    def _sample(self, start_weights, start_policy, node_policy, origin, seed):
        """Sample vectorized walks using different feature policies at every state."""
        rng=np.random.default_rng(seed)
        length=rng.geometric(self.beta,size=self.nwalks)
        if self.horizon is not None:length=np.minimum(length,self.horizon)
        z=(-np.expm1(length*np.log(self.lam))/(1-self.lam)
           if self.lam!=1 else length.astype(float))
        res=np.zeros((self.nwalks,self.k))
        current=np.full(self.nwalks,-1,dtype=int)
        for t in range(1,int(length.max())+1):
            sel=np.flatnonzero(length>=t)
            now=current[sel]
            first=now<0
            probabilities=np.empty((len(sel),self.d))
            if first.any():probabilities[first]=start_policy
            if (~first).any():probabilities[~first]=node_policy[now[~first]]
            if not np.allclose(probabilities.sum(axis=1),1,atol=1e-7):
                raise RuntimeError('invalid feature policy')
            feat=(np.cumsum(probabilities,axis=1)<rng.random(len(sel))[:,None]).sum(axis=1)
            cc=np.empty((len(sel),self.n))
            full=np.empty(len(sel));banned=np.zeros(len(sel))
            if first.any():
                cc[first]=np.cumsum(start_weights[feat[first]],axis=1)
                full[first]=cc[first,-1]
            if (~first).any():
                cc[~first]=self.graph.cumulatives[now[~first],feat[~first]]
                full[~first]=self.graph.totals[now[~first],feat[~first]]
                if origin>=0:
                    banned[~first]=self.graph.weights[now[~first],feat[~first],origin]
            if np.any(full-banned<=1e-12):
                raise RuntimeError('selected feature has no allowed destination')
            threshold=np.maximum(np.finfo(float).tiny,rng.random(len(sel))*(full-banned))
            if origin>=0:
                prefix=cc[:,origin]-banned
                threshold+=(threshold>=prefix)*banned
            dest=(cc<threshold[:,None]).sum(axis=1)
            if ((dest>=self.n).any() or (dest==now).any() or
                    (origin>=0 and (dest==origin).any())):
                raise RuntimeError('forbidden destination reached')
            np.add.at(res,(sel,self.graph.y[dest]),self.lam**(t-1)/z[sel])
            current[sel]=dest
        return res.mean(axis=0)

    def train_representations(self, centers, *, seed=20261104):
        """Optimize all origin-current Q states in one vectorized convex batch."""
        # Weights/current exclusion are precomputed in the ReferenceGraph.
        # Additional exclusion of origin is required for every intermediate step.
        banned = self.graph.weights.transpose(2, 0, 1)  # (origin,current,feature)
        numerator = self.num[None] - banned[..., None]*self.oh[:,None,None,:]
        denominator = self.den[None] - banned
        q = normalized_q(np.maximum(numerator, 0.), np.maximum(denominator, 0.))
        # The origin itself is technically unreachable after leaving it, but
        # its Q is also the correct first-step distribution from that origin.
        target = np.broadcast_to(centers[self.graph.y,None,:],
                                 (self.n,self.n,self.k)).reshape(-1,self.k)
        policies, residuals, gaps = optimize_mixtures(
            q.reshape(-1,self.d,self.k), target,
            steps=self.optimizer_steps, tol=self.optimizer_tol)
        policies = policies.reshape(self.n,self.n,self.d)
        residuals=residuals.reshape(self.n,self.n)
        gaps=gaps.reshape(self.n,self.n)
        out=np.empty((self.n,self.k))
        seeds=np.random.SeedSequence(seed).spawn(self.n)
        for origin in range(self.n):
            pi=policies[origin]
            out[origin]=self._sample(self.graph.weights[origin],pi[origin],pi,
                                     origin,np.random.default_rng(seeds[origin]).integers(0,2**63))
        return out,dict(local_Q_projection_jsd=float(np.mean(residuals)),
                        max_FW_gap=float(np.max(gaps)),
                        p95_FW_gap=float(np.quantile(gaps,.95)))

    def heldout_hypotheses(self,queries,centers,*,seed=20261105):
        """Simulate every class assumption; query labels are NEVER consulted."""
        nq=len(queries)
        start_weights=_weights(queries,self.graph.x,self.graph.gamma)
        start_q=self.start_q(start_weights)
        all_ps=np.empty((nq,self.k,self.k))
        diagnostic={'first_Q_jsd':[],'max_FW_gap':[]}
        seeds=np.random.SeedSequence(seed).spawn(nq)
        child_seeds=[int(np.random.default_rng(ss).integers(0,2**63)) for ss in seeds]
        for c in range(self.k):
            base_p,base_r,base_g=self.reference_policies(centers[c])
            first_p,first_r,first_g=self._opt(start_q,centers[c])
            for i in range(nq):
                all_ps[i,c]=self._sample(start_weights[i],first_p[i],base_p,-1,child_seeds[i])
            diagnostic['first_Q_jsd'].append(float(np.mean(first_r)))
            diagnostic['max_FW_gap'].append(float(max(np.max(base_g),np.max(first_g))))
        return all_ps, diagnostic


def hypothesis_metrics(representations, centers, labels):
    """Candidate-conditional classification: no label enters the feature policy."""
    y=np.asarray(labels,int)
    n,k,_=representations.shape
    distances=np.stack([jsd(representations[:,c,:],centers[c]) for c in range(k)],axis=1)
    pred=distances.argmin(axis=1)
    balanced=float(np.mean([np.mean(pred[y==c]==c) for c in range(k)]))
    own=distances[np.arange(n),y]
    rival=np.min(np.where(np.eye(k,dtype=bool)[y],np.inf,distances),axis=1)
    return {'accuracy':float(np.mean(pred==y)), 'balanced_accuracy':balanced,
            'within_correct_hypothesis_jsd':float(np.mean([own[y==c].mean() for c in range(k)])),
            'mean_true_margin_jsd':float(np.mean([np.mean((rival-own)[y==c]) for c in range(k)])),
            'fraction_nonpositive_true_margin':float(np.mean(rival<=own)),
            'mean_best_candidate_jsd':float(np.mean(np.min(distances,axis=1))),
            'mean_candidate_gap':float(np.mean(np.sort(distances,axis=1)[:,1]-np.sort(distances,axis=1)[:,0]))},pred


def run_experiment(train_path, val_path, policy_path, *, rounds=6, eta=.5,
                   epsilon=.08, output=None, optimizer_steps=65, walkers_per_sample=300,
                   seed=20261104, resume=False):
    train_x,train_y=load_data(train_path)
    val_x,val_y=load_data(val_path)
    baseline=load_policy(policy_path,train_x.shape[1])
    g=ReferenceGraph(train_x,train_y,baseline['kernel_gamma'])
    runner=CenterPolicyWalker(g,beta=baseline['beta'],lam=baseline['lambda'],
            horizon=baseline['max_steps'],nwalks=walkers_per_sample,
            optimizer_steps=optimizer_steps)
    k=g.nclasses
    # Baseline uses the SAME walk parameters and samples per query.
    uniform_policy={**baseline,'logits':np.zeros_like(baseline['logits']),
                    'walks_per_sample':walkers_per_sample,'seed':seed}
    p_uni,_=g.representation(train_x,uniform_policy,np.arange(len(train_x)))
    v_uni=class_centers(p_uni,train_y)
    p_val_uni,_=g.representation(val_x,uniform_policy)
    base_train,_=report(p_uni,v_uni,train_y)
    base_val,_=report(p_val_uni,v_uni,val_y)
    v=(1-epsilon)*np.eye(k)+epsilon*np.ones((k,k))/k
    history=[]
    config={'rounds':rounds,'eta':eta,'epsilon':epsilon,
            'walks_per_sample':walkers_per_sample,'seed':seed,
            'beta':baseline['beta'],'lambda':baseline['lambda'],
            'max_steps':baseline['max_steps'],'optimizer_steps':optimizer_steps}
    output_data={'experiment':'iterative multi-step class-target center matching (observational, not learning)',
            'config':config,
            'baseline':{'train':base_train,'validation':base_val,'centers':v_uni.tolist()},
            'history':history,
            'warnings':['Training policies are conditioned on known origin label; train/oracleBA is a diagnostic, not deployable accuracy.',
              'Validation uses separate class-hypothesis trajectories and never uses validation labels to generate them.',
              'The matching policy minimizes immediate Q-JSD, not the global multi-step objective.',
              'All reported centers are constructed from training origin walks; targets are separate moving quantities.',
              'MC randomness and approximate FW optimization preclude theoretical convergence claims.']}
    out=Path(output) if output is not None else None
    if resume and out is None:
        raise ValueError('resuming requires an output path')
    if resume and out is not None and out.exists():
        old=json.loads(out.read_text(encoding='utf-8'))
        for key in ('eta','epsilon','walks_per_sample','seed','beta','lambda','max_steps','optimizer_steps'):
            if old['config'][key] != config[key]:
                raise ValueError(f'resume configuration differs: {key}')
        history.extend(old['history'])
        if history:
            v=np.asarray(history[-1]['next_targets'],float)
    def persist():
        if out is None:
            return
        out.parent.mkdir(parents=True,exist_ok=True)
        output_data['completed_rounds']=len(history)
        out.write_text(json.dumps(output_data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
        fields=['round','mean_target_realized_jsd','target_update_l1','center_information',
                'min_center_js_distance','T','train_oracle_balanced_accuracy',
                'validation_hypothesis_balanced_accuracy','projection_max_FW_gap']
        with out.with_suffix('.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields)
            writer.writeheader()
            for r in history:
                item={key:r.get(key) for key in fields}
                item['train_oracle_balanced_accuracy']=r['train_oracle_target']['balanced_accuracy']
                item['validation_hypothesis_balanced_accuracy']=r['validation_hypothesis']['balanced_accuracy']
                item['projection_max_FW_gap']=r['projection']['max_FW_gap']
                writer.writerow(item)
    persist()
    for step in range(len(history),rounds):
        p,diag=runner.train_representations(v,seed=seed)
        realized=class_centers(p,train_y)
        heldout,hold_diag=runner.heldout_hypotheses(val_x,v,seed=seed+1)
        # This is a hypothesis-conditional score, NOT comparable to treating
        # a known validation label as a target. Center is fitted on train only.
        val_res,_=hypothesis_metrics(heldout,realized,val_y)
        tr_res,_=report(p,realized,train_y)
        geometry=center_metrics(realized)
        target_geometry=center_metrics(v)
        target_gap=np.mean([jsd(realized[c],v[c]) for c in range(k)])
        new_target=(1-eta)*v+eta*realized
        record={'round':step,'targets':v.tolist(),'realized_centers':realized.tolist(),
                'next_targets':new_target.tolist(),
                'mean_target_realized_jsd':float(target_gap),
                'target_update_l1':float(np.max(np.abs(new_target-v))),
                'measured_center_update_l1':float(np.max(np.abs(realized-v))),
                'center_information':geometry['I'],
                'min_center_js_distance':geometry['min_center_js_distance'],
                'T':geometry['T'],
                'target_center_information':target_geometry['I'],
                'train_oracle_target':{n:tr_res.get(n) for n in ('accuracy','balanced_accuracy','within_class_jsd','NLL_norm','J_margin')},
                'validation_hypothesis':val_res,
                'projection':diag,'validation_projection':hold_diag}
        history.append(record)
        persist()
        print(f'iter={step} gap={target_gap:.5f} move={record["target_update_l1"]:.4f} I={geometry["I"]:.5f} '
              f'sep={geometry["min_center_js_distance"]:.4f} '
              f'train/oracleBA={tr_res["balanced_accuracy"]:.3f} val/hypBA={val_res["balanced_accuracy"]:.3f}',flush=True)
        v=new_target
    persist()
    return output_data


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--train',type=Path,required=True)
    cli.add_argument('--validation',type=Path,required=True)
    cli.add_argument('--policy',type=Path,required=True)
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--rounds',type=int,default=6)
    cli.add_argument('--eta',type=float,default=.5)
    cli.add_argument('--epsilon',type=float,default=.08)
    cli.add_argument('--walks',type=int,default=300)
    cli.add_argument('--optimizer-steps',type=int,default=65)
    cli.add_argument('--seed',type=int,default=20261104)
    cli.add_argument('--resume',action='store_true',help='Continue from a matching JSON checkpoint')
    args=cli.parse_args()
    if args.rounds<1 or args.walks<1 or args.optimizer_steps<1 or not (0<args.eta<=1) or not(0<=args.epsilon<1):
        cli.error('rounds, walks and optimizer-steps must be positive; eta in (0,1], epsilon in [0,1)')
    run_experiment(args.train,args.validation,args.policy,rounds=args.rounds,eta=args.eta,
                   epsilon=args.epsilon,output=args.out,walkers_per_sample=args.walks,
                   optimizer_steps=args.optimizer_steps,seed=args.seed,resume=args.resume)

if __name__=='__main__':main()
