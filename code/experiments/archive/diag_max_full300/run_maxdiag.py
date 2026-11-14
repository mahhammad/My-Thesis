"""Exact full-300 target fit with Q warm-start and damped GN; origin-safe streaming."""
from pathlib import Path
import json,argparse,time,tempfile,sys
import numpy as np,torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'code/data_generator'))
sys.path.insert(0,str(ROOT/'code/experiments/multistep_policy_target_match'))
sys.path.insert(0,str(ROOT/'code/experiments/bayes_center_inverse'))
from generate import load_config,generate
from stream_graph import StreamGraph
from run import DT,jsd
from run_bayes import jsdiv,kl,center_geometry,bayes_class_centers,bayes_posterior,heldout_candidate_metrics
from bootstrap_inverse import warm_q,damped_gauss_newton,add_counts
RESULT=HERE/'results'

def inputs(target_kind='equal'):
 spec=load_config(ROOT/'code/data_generator/config.json')
 with tempfile.TemporaryDirectory() as outdir:
  dat=generate(spec,outdir,finite_q=False)
 x,_,y=dat['train'];vx,_,vy=dat['validation'];k=len(spec['pi'])
 bayes,_=bayes_class_centers(spec,x,y)
 bounds=json.loads((ROOT/'code/experiments/bayes_center_inverse/results/full_reference_bounds.json').read_text())['by_reference_size']['300']['bounds']
 lower=np.array(bounds['lower']); upper=np.array(bounds['upper']); diag=np.diag(upper)
 target=np.zeros((k,k));np.fill_diagonal(target,diag)
 for c in range(k):
  j=np.arange(k)!=c
  if target_kind in ('equal','bounded_equal'):target[c,j]=(1-diag[c])/(k-1)
  else:target[c,j]=(1-diag[c])*bayes[c,j]/bayes[c,j].sum()
  if target_kind=='bounded_equal':
   a,b=np.flatnonzero(j);r=1-diag[c]
   lo=max(lower[c,a],r-upper[c,b]); hi=min(upper[c,a],r-lower[c,b])
   if lo>hi+1e-12:raise ValueError('Impossible component bounds at fixed diagonal')
   target[c,a]=np.clip(r/2,lo,hi);target[c,b]=r-target[c,a]
 per_lower=[]
 for c in range(k):
  bounds_js=[float(jsdiv(np.array([np.clip(target[c,j],lower[c,j],upper[c,j]),1-np.clip(target[c,j],lower[c,j],upper[c,j])]),np.array([target[c,j],1-target[c,j]]))) for j in range(k)]
  per_lower.append(max(bounds_js))
 return spec,x,y,vx,vy,bayes,target,lower,upper,per_lower

def one_class(args):
 torch.set_num_threads(args.threads)
 spec,x,y,vx,vy,bayes,target,lower,upper,min_js=inputs(args.target_kind)
 g=StreamGraph(x,y,spec['gamma'],vx,beta=.4,lam=.8,horizon=12)
 c=args.c;idx=np.flatnonzero(y==c);u=torch.tensor(target[c],dtype=DT)
 neutral=torch.zeros((g.d+1,g.d),dtype=DT)
 with torch.no_grad():unif=g.center_train(neutral,idx).numpy()
 print('CLASS',c,'n',len(idx),'GOAL',np.round(target[c],7),'INFEASIBLE_COMPONENTS',(target[c]<lower[c]).tolist(),'UNIFORM',np.round(unif,6),'JSD',float(jsdiv(unif,target[c])),flush=True)
 zq,tq,cq=warm_q(g,idx,u,args.q_steps,args.lr)
 with torch.no_grad():qcenter=g.center_train(zq,idx).numpy()
 print('Q WARM',c,'time',round(tq,3),'center',np.round(qcenter,6),'JSD',float(jsdiv(qcenter,target[c])),flush=True)
 zb,tb,cb,trace=damped_gauss_newton(g,idx,u,zq,args.gn_steps,args.tol)
 with torch.no_grad():real=g.center_train(zb,idx).numpy()
 print('GN',c,'time',round(tb,3),'center',np.round(real,6),'JSD',float(jsdiv(real,target[c])),'steps',len(trace)-1,flush=True)
 alt=None
 if args.restart_steps>0:
  za,ta,ca,at=damped_gauss_newton(g,idx,u,neutral,args.restart_steps,args.tol)
  with torch.no_grad():altreal=g.center_train(za,idx).numpy()
  alt={'time':ta,'center':altreal.tolist(),'JSD':float(jsdiv(altreal,target[c])),'trace':at,'counts':ca}
  if alt['JSD']<float(jsdiv(real,target[c])):zb,real=za,altreal
  tb+=ta;cb=add_counts(cb,ca)
 with torch.no_grad():rep=g.center_train(zb,idx,return_all=True).numpy()
 record={'class':c,'n_origins':len(idx),'target':target[c].tolist(),'uniform_center':unif.tolist(),'q_center':qcenter.tolist(),'center':real.tolist(),
  'uniform_JSD':float(jsdiv(unif,target[c])),'q_JSD':float(jsdiv(qcenter,target[c])),'final_JSD':float(jsdiv(real,target[c])),
  'target_diagonal':float(target[c,c]),'realized_diagonal':float(real[c]),'lower_bound_for_JSD_from_component_bounds':min_js[c],
  'infeasible_target_components':(target[c]<lower[c]-1e-12).tolist(),
  'training_R_JS':float(jsdiv(rep,real).mean()),'training_R_KL':float(kl(rep,real).mean()),
  'elapsed_Q_s':tq,'elapsed_GN_s':tb,'total_s':tq+tb,'q_cost':cq,'gn_cost':cb,'gn_trace':trace,'restart':alt,'Z':zb.tolist(),
  'meta':{'target_kind':args.target_kind,'q_steps':args.q_steps,'gn_steps':args.gn_steps,'restart_steps':args.restart_steps,'H':12,'beta':.4,'lambda':.8,'tol':args.tol,'threads':args.threads}}
 RESULT.mkdir(parents=True,exist_ok=True)
 out=RESULT/f'{args.target_kind}_class{c}.json';out.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 print('SAVED',out,flush=True)

def combine(kind):
 spec,x,y,vx,vy,bayes,target,lower,upper,min_js=inputs(kind)
 parts=[json.loads((RESULT/f'{kind}_class{c}.json').read_text()) for c in range(3)]
 g=StreamGraph(x,y,spec['gamma'],vx)
 policies=[torch.tensor(r['Z'],dtype=DT) for r in parts]
 centers=np.asarray([r['center'] for r in parts])
 val=heldout_candidate_metrics(g,policies,centers,vy,bayes_posterior(spec,vx))
 output={'status':'complete','meta':{'n_train':len(y),'class_counts':np.bincount(y).tolist(),'target_kind':kind,'max_upper_diagonal_is_relaxed_not_sharp':True,'origin_exclusion_is_exact':True,'single_base_matrix_without_all_origin_transitions':True,'H':12,'beta':.4,'lambda':.8},
  'target':target.tolist(),'bayes_centers':bayes.tolist(),'upper':upper.tolist(),'lower':lower.tolist(),
  'parts':parts,'summary':{'realized_centers':centers.tolist(),'mean_target_JSD':float(jsdiv(centers,target).mean()),
   'max_target_JSD':float(jsdiv(centers,target).max()),'center_geometry':center_geometry(centers),
   'mean_R_JS':float(np.mean([r['training_R_JS'] for r in parts])),
   'mean_R_KL':float(np.mean([r['training_R_KL'] for r in parts])),
   'relaxed_lower_bound_JSD_per_class':min_js,'validation':val,'time_s':sum(r['total_s'] for r in parts)}}
 out=RESULT/f'full300_{kind}_summary.json';out.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 print('SUMMARY',kind,'GOAL',np.round(target,6),'REALIZED',np.round(centers,6),'JSD',output['summary']['mean_target_JSD'],'VALID',val,'FILE',out,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--c',type=int,choices=[0,1,2]);p.add_argument('--target-kind',choices=['equal','bounded_equal','posterior_ratio'],default='equal');p.add_argument('--q-steps',type=int,default=45);p.add_argument('--gn-steps',type=int,default=30);p.add_argument('--restart-steps',type=int,default=0);p.add_argument('--lr',type=float,default=.12);p.add_argument('--tol',type=float,default=1e-8);p.add_argument('--threads',type=int,default=1);p.add_argument('--combine',action='store_true'); args=p.parse_args()
 combine(args.target_kind) if args.combine else one_class(args)
