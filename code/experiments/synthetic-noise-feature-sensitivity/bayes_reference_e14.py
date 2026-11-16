"""Read-only Bayes MAP reference from original frozen seed-20261102 Train/Validation.

This reuses the established E-CH4-13 Bayes_MAP implementation and same frozen
generative inputs, without generating data or touching Test.
"""
from __future__ import annotations
import hashlib, importlib.util, json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
E13=HERE.parent/"objective-start-diagnostics"
CONFIG=HERE.parents[1]/"data_generator"/"config.json"
FROZEN=E13/"inputs"/"phase2_train_validation"
DATA=FROZEN/"ds20261102_train_val.npz"
ORIGINAL_SHA="e701cf9ef99b709fd5d6faee2d3ac631737d6b0195cf76eefc23d063bccb1b4a"

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def balanced(y,p):
    return float(np.mean([np.mean(p[y==c]==c) for c in range(3)]))

def compute_bayes_reference():
    fm=json.loads((FROZEN/"manifest.json").read_text(encoding="utf8"))
    rec=fm["records"]["20261102"]
    assert fm["test_accessed"] is False
    assert sha256(DATA)==rec["file_sha256"]==ORIGINAL_SHA
    assert sha256(CONFIG)==fm["config_sha256"]
    assert DATA.name==rec["file"]
    spec=importlib.util.spec_from_file_location("e14_original_e13_bayes",
                 E13/"validation_bayes_summary.py")
    assert spec and spec.loader
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config=json.loads(CONFIG.read_text(encoding="utf8"))
    output={"source":"E-CH4-13 frozen seed20261102 original 5-feature Bayes MAP",
            "data_seed":20261102,"frozen_train_validation_sha256":ORIGINAL_SHA,
            "config_sha256":fm["config_sha256"],"test_accessed":False}
    with np.load(DATA,allow_pickle=False) as data:
        for split in ("train","validation"):
            x=np.asarray(data[split+"_x"],dtype=np.float64)
            y=np.asarray(data[split+"_y"],dtype=np.int64)
            assert x.shape==((300 if split=="train" else 90),5)
            assert list(np.bincount(y,minlength=3))==rec[split+"_counts"]
            prediction=module.bayes_map(x,config)
            output[split]={"BA":balanced(y,prediction),
                           "Accuracy":float(np.mean(y==prediction)),
                           "n":int(len(y))}
    assert np.isclose(output["validation"]["BA"],0.9744897959183674,atol=2e-13)
    return output

if __name__=="__main__":
    data=compute_bayes_reference()
    print("E14_MATCHED_BAYES_MAP_REFERENCE_PASS",
          json.dumps({"train":data["train"],"validation":data["validation"],
                      "test_accessed":False},sort_keys=True))
