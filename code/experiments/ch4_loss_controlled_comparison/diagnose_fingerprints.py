"""Read-only E-CH4-13 cross-run fingerprint diagnostic; NEVER run any optimizer."""
import json
from pathlib import Path
import platform
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"code/experiments/benchmark_v1"))
sys.path.insert(0,str(ROOT/"code/data_generator"))
from dataset_adapter import dataset_train_val
from generate import load_config
from snapshot_pipeline import array_fingerprint,config_fingerprint

def main():
    cfg=ROOT/"code/data_generator/config.json"
    spec=load_config(cfg);spec["seed"]=20261102
    data=dataset_train_val(spec)
    x,y=data["train"];vx,vy=data["validation"]
    base=HERE/"results/pilot_seed20261102_shared_two_fits"
    m=json.loads((base/"manifest.json").read_text("utf8"))
    ids=np.asarray(m["reference_ids"],dtype=np.int64)
    fields={
        "train_x":(array_fingerprint(x[ids]),m["reference_x_sha256"]),
        "validation_x":(array_fingerprint(vx),m["validation_x_sha256"]),
        "config":(config_fingerprint(cfg),m["config_sha256"])
    }
    result={
      "python":platform.python_version(),
      "numpy":np.__version__,
      "machine":platform.machine(),
      "platform":platform.platform(),
      "fingerprints":{k:{"current":a,"stored":b,"match":a==b} for k,(a,b) in fields.items()},
      "train_label_counts":np.bincount(y[ids],minlength=3).tolist(),
      "validation_label_counts":np.bincount(vy,minlength=3).tolist(),
      "test_accessed":False
    }
    print(json.dumps(result,indent=2))
    return result
if __name__=="__main__":main()
