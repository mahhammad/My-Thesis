"""Group gradient contributions for trained sigmoid-margin feature policy."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from train import build_data,outputs,DT,StreamGraph,balanced

def inspect(seed=20261107,nref=72,tau=.005,z_values=None):
    torch.set_num_threads(1)
    spec,x,y,vx,vy,ids=build_data(seed,nref)
    graph=StreamGraph(x,y,spec['gamma'],vx,horizon=12)
    d=graph.d
    start=np.zeros((d+1,d)) if z_values is None else np.asarray(z_values)
    z=torch.tensor(start,dtype=DT,requires_grad=True)
    rep,centers,margin=outputs(graph,z,y)
    loss=balanced(torch.sigmoid(-margin/tau),y,graph.k)
    whole=torch.autograd.grad(loss,z,retain_graph=True)[0].flatten().detach().numpy()
    m=margin.detach().numpy()
    weights=np.asarray([1/(graph.k*sum(y==k)) for k in y])
    samples=[]
    for i in range(len(y)):
        gm=torch.autograd.grad(margin[i],z,retain_graph=True)[0].flatten().detach().numpy()
        sigmoid=float(torch.sigmoid(-margin[i]/tau).detach())
        gradient=weights[i]*(-sigmoid*(1-sigmoid)/tau)*gm
        grp=('near' if abs(m[i])<=tau else
             'far_wrong' if m[i]<-2*tau else
             'far_correct' if m[i]>2*tau else
             'mid_wrong' if m[i]<0 else 'mid_correct')
        samples.append({'i':i,'class':int(y[i]),'margin':float(m[i]),
                        'group':grp,'balanced_weight':float(weights[i]),
                        'grad_margin_norm':float(np.linalg.norm(gm)),
                        'grad_loss_norm':float(np.linalg.norm(gradient)),
                        'grad_loss_dot_total':float(gradient @ whole)})
    grouped={}
    for k in ('near','mid_wrong','far_wrong','mid_correct','far_correct'):
        group=[r for r in samples if r['group']==k]
        grouped[k]={'count':len(group),
                    'balanced_mass':sum(v['balanced_weight'] for v in group),
                    'sum_of_individual_gradient_norms':sum(v['grad_loss_norm'] for v in group),
                    'mean_margin_gradient_norm':float(np.mean([v['grad_margin_norm'] for v in group])) if group else None}
    return {'seed':seed,'nref':nref,'tau':tau,'overall_gradient_norm':float(np.linalg.norm(whole)),
            'groups':grouped,'samples':samples}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=20261107)
    p.add_argument('--tau',type=float,default=.005)
    p.add_argument('--nref',type=int,default=72)
    p.add_argument('--trained-result',type=Path)
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    z=None if args.trained_result is None else json.loads(args.trained_result.read_text())['Z']
    record=inspect(args.seed,args.nref,args.tau,z)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'overall_gradient_norm':record['overall_gradient_norm'],'groups':record['groups']},indent=2))
