"""Re-evaluate saved full-300 policies on train, validation and held-out test.

This script does NOT train or select policies. The fixed test split was opened
for retrospective reporting, never used for hyperparameter tuning.
"""
from pathlib import Path
import json, tempfile, sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'code/data_generator'))
sys.path.insert(0,str(ROOT/'code/experiments/diag_max_full300'))
sys.path.insert(0,str(ROOT/'code/experiments/multistep_policy_target_match'))
from generate import generate,load_config
from stream_graph import StreamGraph
from run import DT,jsd

def balanced(pred,y,k):
    return float(np.mean([(pred[y==c]==c).mean() for c in range(k)]))

def evaluate():
    torch.set_num_threads(1)
    spec=load_config(ROOT/'code/data_generator/config.json')
    with tempfile.TemporaryDirectory() as tmp:
        splits=generate(spec,tmp,finite_q=False)
    train_x,_,train_y=splits['train']
    val_x,_,val_y=splits['validation']
    test_x,_,test_y=splits['test']
    saved=json.loads((HERE/'results/full300_policy_snapshots.json').read_text())
    graph=StreamGraph(train_x,train_y,spec['gamma'],val_x,horizon=12)
    test_graph=StreamGraph(train_x,train_y,spec['gamma'],test_x,horizon=12)
    k=len(spec['pi'])
    results={}
    for key,z_values in [('uniform',np.zeros((6,5)))]+list(saved['trained_policies'].items()):
        z=torch.tensor(z_values,dtype=DT)
        with torch.no_grad():
            reps=graph.center_train(z,np.arange(len(train_y)),return_all=True)
            centers=torch.stack([reps[torch.as_tensor(train_y==c)].mean(0) for c in range(k)])
            train_pred=jsd(reps[:,None,:],centers[None,:,:]).argmin(1).numpy()
            val_pred=jsd(graph.evaluate_val(z)[:,None,:],centers[None,:,:]).argmin(1).numpy()
            test_pred=jsd(test_graph.evaluate_val(z)[:,None,:],centers[None,:,:]).argmin(1).numpy()
        results[key]={
            'train_balanced':balanced(train_pred,train_y,k),
            'validation_balanced':balanced(val_pred,val_y,k),
            'test_balanced':balanced(test_pred,test_y,k),
            'test_ordinary':float((test_pred==test_y).mean())}
    return results

if __name__=='__main__':
    print(json.dumps(evaluate(),indent=2,ensure_ascii=False))
