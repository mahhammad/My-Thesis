"""Archive already-completed E-CH4-13 phase2 artifact; ZERO TRAINING.

This tool only validates and creates a *new* immutable evidence folder from
the existing GitHub Actions artifact. It never imports torch, optimizer,
generator, or Test.
"""
from __future__ import annotations
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DEST=HERE/"results/phase2_300_25"
RUN_ID=43670524935
ARTIFACT_ID=14144445918
TRAINING_SHA="cdb45644c245e04a276eb2a419e4d64f71c6c18a"
ARTIFACT_ZIP_SHA="26e7a94e36edeae2a3153b76b40cf282c30b550a3fac359558a20d11e72c3e3e"
SEEDS=(20261102,20261103,20261104)
FAMILIES=("shared","class")
LOSSES=("sigmoid_JS_margin_tau005","softplus_JS_tau005")
EXPECTED={f"ds{s}_{f}_{o}_n300_s25" for s in SEEDS for f in FAMILIES for o in LOSSES}
FILE_KEYS=(("trace.json","trace_sha256"),("model.json","model_sha256"),
           ("distributions.npz","distributions_sha256"),
           ("metrics_full.json","metrics_full_sha256"),("coverage.json","coverage_sha256"))

def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()

def archive(zippath):
    zippath=Path(zippath)
    if sha(zippath)!=ARTIFACT_ZIP_SHA:
        raise ValueError("Original artifact ZIP byte digest differs; STOP")
    if DEST.exists():
        raise FileExistsError("Refuse archive overwrite or rerun")
    with zipfile.ZipFile(zippath) as z:
        names=z.namelist()
        if len(names)!=428 or len(set(names))!=428:
            raise ValueError("Unexpected artifact member count")
        for n in names:
            pp=Path(n)
            if pp.is_absolute() or ".." in pp.parts or n.endswith("/"):
                raise ValueError("Unsafe archive member "+n)
            if any(s.lower()=="test" or s.lower().startswith("test_") for s in pp.parts):
                raise ValueError("Unexpected Test data member "+n)
        m=json.loads(z.read("manifest.json"))
        if (m["fit_count"]!=12 or m["updates_per_fit"]!=25 or
            m["source_commit"]!=TRAINING_SHA or m["test_accessed"] is not False):
            raise ValueError("Unexpected training protocol")
        if {a["run_key"] for a in m["runs"]}!=EXPECTED:
            raise ValueError("Missing or extra fit cell")
        for a in m["runs"]:
            for f,k in FILE_KEYS:
                member=a["run_key"]+"/"+f
                if sha256(z.read(member)).hexdigest()!=a[k]:
                    raise ValueError("Run SHA mismatch "+member)
            trace=json.loads(z.read(a["run_key"]+"/trace.json"))
            if len(trace)!=26 or [x["update_completed"] for x in trace]!=list(range(26)):
                raise ValueError("Incorrect training trace length")
            model=json.loads(z.read(a["run_key"]+"/model.json"))
            if model["data"]["test_accessed"] is not False:
                raise ValueError("Test accessed")
        for fn,field in (("data_fingerprints.json","data_fingerprints_sha256"),
                         ("provenance.json","provenance_sha256")):
            if sha256(z.read(fn)).hexdigest()!=m[field]:
                raise ValueError("Central SHA mismatch "+fn)
        ledger=json.loads(z.read("fit_budget_ledger.json"))
        if ledger["attempt_count"]!=12 or ledger["status"]!="COMPLETE_UNAUDITED":
            raise ValueError("Budget contract mismatch")
        if (sum(x.startswith("attempt_charges/") for x in names)!=12 or
            sum(x.startswith("budget_checkpoints/") for x in names)!=12 or
            sum(x.startswith("success_checkpoints/") for x in names)!=12 or
            sum(x.startswith("optimizer_checkpoints/") for x in names)!=312 or
            any(x.startswith("failures/") for x in names)):
            raise ValueError("Incomplete forensic charge/snapshot record")
        pair=json.loads(z.read("paired_summary.json"))
        if pair["n_fits"]!=12 or len(pair["pairs"])!=6 or pair["test_accessed"] is not False:
            raise ValueError("Invalid reducer")
        # All checks performed BEFORE extraction. Preserve raw artifact bytes.
        DEST.mkdir(parents=True,exist_ok=False)
        for name in names:
            fp=DEST/name
            fp.parent.mkdir(parents=True,exist_ok=True)
            fp.write_bytes(z.read(name))
    if any(sha(DEST/a["run_key"]/fn)!=a[key] for a in m["runs"] for fn,key in FILE_KEYS):
        raise ValueError("Post-extraction corruption")
    return m,pair

def make_tables(m,pair):
    tables=HERE/"tables"; figures=HERE/"figures"
    tables.mkdir(exist_ok=True);figures.mkdir(exist_ok=True)
    a=tables/"phase2_runs_12.csv"
    if a.exists():raise FileExistsError(a)
    keys=("run_key","data_seed","family","objective","train_ba","validation_ba",
          "I","R_JS","R_KL","R_sqrt_JS","margin_validation_median",
          "gradient_l2_init","gradient_l2_final","elapsed_s")
    with a.open("x",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader()
        for item in m["runs"]:w.writerow({k:item[k] for k in keys})
    b=tables/"phase2_pairs_6.csv"
    if b.exists():raise FileExistsError(b)
    pk=("data_seed","family","BA_sigmoid","BA_softplus",
        "paired_delta_validation_BA","validation_flip_count","I_sigmoid",
        "I_softplus","R_JS_sigmoid","R_JS_softplus",
        "validation_margin_median_sigmoid","validation_margin_median_softplus")
    with b.open("x",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=pk);w.writeheader()
        for item in pair["pairs"]:w.writerow({k:item[k] for k in pk})
    # SVG is generated only from locked paired_summary.json, no plotting dependency.
    png=figures/"phase2_paired_delta_validation_ba.svg"
    if png.exists():raise FileExistsError(png)
    vals=pair["pairs"]
    def rect(x,y,h,color):
        if h>=0:return f'<rect x="{x}" y="{y-h:.2f}" width="42" height="{h:.2f}" fill="{color}"/>'
        return f'<rect x="{x}" y="{y}" width="42" height="{-h:.2f}" fill="{color}"/>'
    elements=['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="450" viewBox="0 0 900 450">',
       '<rect width="900" height="450" fill="white"/>',
       '<text x="60" y="38" font-size="20" font-family="sans-serif">E-CH4-13 phase2: paired Validation BA delta (Sigmoid minus Softplus)</text>',
       '<line x1="65" y1="245" x2="835" y2="245" stroke="black"/>']
    for i,item in enumerate(vals):
        value=100*item["paired_delta_validation_BA"];h=value*28
        x=100+i*120
        elements.append(rect(x,245,h,"#2b6c9b" if item["family"]=="shared" else "#d08745"))
        elements.append(f'<text x="{x-20}" y="405" font-size="11" font-family="sans-serif">{item["data_seed"]}</text>')
        elements.append(f'<text x="{x}" y="423" font-size="11" font-family="sans-serif">{item["family"]}</text>')
        elements.append(f'<text x="{x-8}" y="{235 if value>=0 else 260-h:.2f}" font-size="12" font-family="sans-serif">{value:+.2f} pp</text>')
    elements.append('</svg>')
    png.write_text("\n".join(elements)+"\n",encoding="utf8")

def main():
    if len(sys.argv)!=2:
        raise SystemExit("Usage: archive_phase2_existing.py /tmp/original_artifact.zip")
    m,pair=archive(sys.argv[1])
    make_tables(m,pair)
    meta={"status":"PHASE2_RAW_ARCHIVED_UNAUDITED","run_id":RUN_ID,
          "artifact_id":ARTIFACT_ID,"artifact_zip_sha256":ARTIFACT_ZIP_SHA,
          "source_commit":TRAINING_SHA,"raw_fits":12,
          "no_new_optimizer_steps":True,"test_accessed":False,
          "unmodified_raw_member_count":428}
    (HERE/"PHASE2_RAW_ARCHIVE_CHECKPOINT.json").write_text(
        json.dumps(meta,indent=2)+"\n",encoding="utf8")
    print(json.dumps(meta))

if __name__=="__main__":main()
