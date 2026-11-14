"""E-CH4-13 phase2 fixed-grid contract. Does NOT import a trainer or generate Test.
No command in this module can initiate fitting.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

SEEDS=(20261102,20261103,20261104)
FAMILIES=("shared","class")
LOSSES=("sigmoid_JS_margin_tau005","softplus_JS_tau005")
N_TRAIN=300
N_VALIDATION=90
N_UPDATES=25
K=3
D=5
BETA=.4
LAMBDA=.8
H=12
TAU=.005
LR=.12
MAX_FITS=12
SCHEMA="e-ch4-13-phase2-locked-grid-v1"
AUTH_STATUS="PENDING_PRELIMINARY_TECHNICAL_QA_AND_MANAGER_EXECUTION_AUTHORIZATION"
AUTH_FILE=Path(__file__).resolve().parent/"phase2_execution_gate.json"

@dataclass(frozen=True)
class Run:
    seed:int
    family:str
    loss:str
    def key(self):
        return f"ds{self.seed}_{self.family}_{self.loss}_n300_s25"
    def to_dict(self):
        return {"data_seed":self.seed,"family":self.family,
                "objective":self.loss,"n_train":N_TRAIN,"steps":N_UPDATES,
                "tau_width":TAU}

def grid():
    return tuple(Run(s,f,o) for s in SEEDS for f in FAMILIES for o in LOSSES)

def validate_grid(rows):
    rows=tuple(rows)
    if len(rows)!=MAX_FITS or len(set(rows))!=MAX_FITS or rows!=grid():
        raise ValueError("FULL_PAIR_GRID_REQUIRED: exactly 3x2x2 ordered independent fits")
    if any(x.key().find("test")>=0 for x in rows):
        raise ValueError("Test naming prohibited")
    return rows

def reference_ids(y):
    """All 300 Train IDs; there is no sampling for full300."""
    import numpy as np
    y=np.asarray(y)
    if y.shape!=(N_TRAIN,) or y.dtype.kind not in "iu":
        raise ValueError("Expected exactly 300 integer Train labels")
    if set(y.tolist())!={0,1,2}:
        raise ValueError("All 3 Train classes required")
    return np.arange(N_TRAIN,dtype=np.int64)

def expected_data_gate(records):
    """Reject missing/inconsistent data contracts before optimizer invocation."""
    if set(map(int,records))!=set(SEEDS):
        raise ValueError("Missing generator seed fingerprint")
    for seed in SEEDS:
        r=records[str(seed)] if str(seed) in records else records[seed]
        if r["n_train"]!=N_TRAIN or r["n_validation"]!=N_VALIDATION:
            raise ValueError("Split size changed")
        if r["n_classes"]!=K or r["n_features"]!=D:
            raise ValueError("Unexpected K/d")
        if r["reference_ids"]!=list(range(N_TRAIN)):
            raise ValueError("Reference ID order mismatch")
        for name in ("config_sha256","train_x_sha256","train_y_sha256","validation_x_sha256"):
            digest=r.get(name)
            if not isinstance(digest,str) or len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Missing/invalid SHA256: "+name)
        if r.get("test_accessed") is not False:
            raise ValueError("Test must be explicitly untouched")
    return True

def evidence_gate(*,execute,approval_file=AUTH_FILE,cli_token=None):
    """Require PRELIMINARY technical review AND explicit manager execution gate.

    Final scientific QA is postponed until after phase2 results. A local JSON
    file cannot independently prove official GitHub authority: the protected
    workflow MUST fetch and validate Issue #81 official approval and same-SHA
    preflight logs using read-only API before setting external gate verified.
    """
    if not execute:
        return "DRY_RUN_NO_TRAINING"
    if not approval_file.is_file():
        raise PermissionError("PRELIM_TECH_QA_AND_MANAGER_AUTHORIZATION_MISSING")
    approval=json.loads(approval_file.read_text("utf8"))
    technical=approval.get("technical_audit_url")
    manager=approval.get("manager_authorization_url")
    ci=approval.get("ci_url")
    if (approval.get("status")!="PHASE2_EXECUTION_AUTHORIZED"
        or approval.get("technical_audit_status")!="PASS"
        or approval.get("phase2_tests")!="PASS"
        or not isinstance(technical,str)
        or not technical.startswith("https://github.com/mahhammad/My-Thesis/issues/")
        or "#issuecomment-" not in technical
        or not isinstance(manager,str)
        or not manager.startswith("https://github.com/mahhammad/My-Thesis/issues/81#issuecomment-")
        or not isinstance(ci,str)
        or not ci.startswith("https://github.com/mahhammad/My-Thesis/actions/runs/")
        or not isinstance(approval.get("source_sha"),str)
        or len(approval["source_sha"])!=40
        or cli_token is None
        or cli_token!=approval.get("one_time_gate_token")):
        raise PermissionError("FORMAL_PRELIMINARY_TECHNICAL_AND_MANAGER_GATE_NOT_VERIFIED")
    if approval.get("used") is not False or approval.get("training_authorized") is not True:
        raise PermissionError("Phase2 training not authorized or budget already consumed")
    return approval

def budget_ledger_path(root):
    return Path(root)/"fit_budget_ledger.json"

def new_ledger():
    return {"schema":SCHEMA,"cap":MAX_FITS,
            "attempts":[],"test_accessed":False}

def charge_attempt(ledger,run):
    validate_grid(grid())
    if len(ledger.get("attempts",[]))>=MAX_FITS:
        raise RuntimeError("MAX_FITS exceeded")
    key=run.key()
    if key not in {r.key() for r in grid()} or key in [a["run_key"] for a in ledger["attempts"]]:
        raise RuntimeError("Duplicate or unapproved job")
    ledger["attempts"].append({"run_key":key,"status":"ATTEMPT_CHARGED_BEFORE_FIT"})
    return ledger

def atomic_new_json(path,data):
    """Never overwrite evidence or silently resume a charged run."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    encoded=(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode("utf8")
    with path.open("xb") as f:
        f.write(encoded)
    return sha256(encoded).hexdigest()

def plan_manifest():
    validate_grid(grid())
    return {"schema":SCHEMA,"status":AUTH_STATUS,"technical_audit_gate":"preliminary_separate_from_final_scientific_QA",
        "authorization_issue":81,
        "manager_issue":81,"objective":"paired_loss_only",
        "n_train":N_TRAIN,"n_validation":N_VALIDATION,
        "data_seeds":list(SEEDS),"family_set":list(FAMILIES),"objective_set":list(LOSSES),
        "fit_cap":MAX_FITS,"updates_per_fit":N_UPDATES,
        "beta":BETA,"lambda":LAMBDA,"H":H,"tau_width":TAU,
        "learning_rate":LR,"optimizer":"Adam","init":"zeros",
        "precision":"float64","reference_ids":"ordered_all_train_0_through_299",
        "checkpoint_selection":"fixed_step_25","test_accessed":False,
        "fit_jobs":[x.to_dict() for x in grid()]}

if __name__=="__main__":
    print(json.dumps(plan_manifest(),indent=2,ensure_ascii=False))
