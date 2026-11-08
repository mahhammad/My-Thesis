"""Run the immutable uniform reference check. No parameter fitting or model selection.
Usage: python code/experiments/benchmark_v1/smoke_uniform.py --out /tmp/uniform_v1.json
"""
import argparse,json,pathlib,sys,tempfile,time
import numpy as np,torch
ROOT=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"code/data_generator"))
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from generate import load_config,generate
from engine import ExactMaskedWalk,DT,jsd,LN2
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",default=None);args=parser.parse_args()
    torch.set_num_threads(1);start=time.perf_counter()
    spec=load_config(ROOT/"code/data_generator/config.json")
    with tempfile.TemporaryDirectory() as d:splits=generate(spec,d,finite_q=False)
    x,_,y=splits["train"];vx,_,vy=splits["validation"]
    g=ExactMaskedWalk(x,y,spec["gamma"],query_x=vx)
    z=torch.zeros((g.d+1,g.d),dtype=DT)
    train=g.train_outputs(z,"shared");train_metrics=g.metrics(train)
    # Use the same Train-fitted centers and all three radius measures on Validation.
    from train_margin import validate
    val_metrics=validate(g,z,"shared",train.centers.detach(),vy,tau=.005)
    result={"schema":"benchmark-v1","status":"smoke-baseline-not-model-comparison",
            "family":"shared","objective":"uniform/no-training","split":"old 20261102",
            "n_train":len(y),"class_counts":np.bincount(y).tolist(),"n_validation":len(vy),
            "test_used":False,"train":train_metrics,
            "validation":val_metrics,"elapsed_s":time.perf_counter()-start}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if args.out:pathlib.Path(args.out).write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
if __name__=="__main__":main()
