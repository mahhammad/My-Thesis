"""Deterministic post-hoc E-CH4-13 QA fixes v2: no optimizer, Test, imports of ML libs, or raw rewriting.
Usage: python phase2_derived_reproduce_v2.py [--check]
Generates 12 *new* coverage documents, 11 *new* SVGs and numerical plotting tables.
Original 12 coverage.json and original user-reviewable figures stay untouched.
"""
from pathlib import Path
import argparse,hashlib,json,csv,io,math

HERE=Path(__file__).resolve().parent
RAW=HERE/"results/phase2_300_25"
OUT=HERE/"derived/phase2_qa_fixes_v2"
FIG=HERE/"figures/phase2_repro_v2"
SOURCE="cdb45644c245e04a276eb2a419e4d64f71c6c18a"
SEEDS=(20261102,20261103,20261104)
FAMILIES=("shared","class")
LOSSES=("sigmoid_JS_margin_tau005","softplus_JS_tau005")
RS=("R_KL","R_JS","R_sqrt_JS")
FILES={"trace.json":"trace_sha256","model.json":"model_sha256",
       "distributions.npz":"distributions_sha256","metrics_full.json":"metrics_full_sha256",
       "coverage.json":"coverage_sha256"}
C=("#2778a5","#d37b25","#57934e","#9367a4")

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jb(x):return (json.dumps(x,sort_keys=True,ensure_ascii=False,indent=2)+"\n").encode()
def cb(rows):
    f=io.StringIO(newline="");w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
    w.writeheader();w.writerows(rows);return f.getvalue().encode()
def same(a,b):return math.isclose(float(a),float(b),rel_tol=0,abs_tol=3e-10)

def inputs():
    m=json.loads((RAW/"manifest.json").read_text())
    if (m["source_commit"]!=SOURCE or m["fit_count"]!=12 or m["updates_per_fit"]!=25
        or m["test_accessed"] is not False):raise ValueError("Wrong original training protocol")
    expected={(s,f,l) for s in SEEDS for f in FAMILIES for l in LOSSES}
    observed={(x["data_seed"],x["family"],x["objective"]) for x in m["runs"]}
    if observed!=expected or len(m["runs"])!=12:raise ValueError("Wrong 12-cell grid")
    D={}
    for a in m["runs"]:
        p=RAW/a["run_key"]
        for f,k in FILES.items():
            if sha(p/f)!=a[k]:raise ValueError("Original byte SHA mismatch: "+str(p/f))
        metrics=json.loads((p/"metrics_full.json").read_text())
        coverage=json.loads((p/"coverage.json").read_text())
        trace=json.loads((p/"trace.json").read_text())
        if len(trace)!=26 or [t["update_completed"] for t in trace]!=list(range(26)):
            raise ValueError("25-step optimizer trace invalid")
        for split in ("train","validation"):
            s=metrics[split];q=s["confusion_true_rows_pred_cols"]
            recall=[q[c][c]/sum(q[c]) for c in range(3)]
            if not same(s["balanced_accuracy"],sum(recall)/3):raise ValueError("BA mismatch")
            for c in range(3):
                if not same(s["per_class"][c]["accuracy"],recall[c]):
                    raise ValueError("Original recall mismatch")
            if coverage[split+".per_class_recall"]["status"]!="not_measured":
                raise ValueError("Unexpected raw coverage: refuse rewrite")
            if any(not math.isfinite(s[k]) for k in RS):raise ValueError("Nonfinite R metric")
        D[(a["data_seed"],a["family"],a["objective"])]=(a,metrics,coverage,trace)
    return m,D

def corrected(D):
    outputs={};index=[]
    for (_,_,_), (a,m,cov,t) in sorted(D.items()):
        copy=json.loads(json.dumps(cov))
        for split in ("train","validation"):
            copy[split+".per_class_recall"]={"status":"measured",
                "reason":"Derived alias of metrics_full.json "+split+".per_class[].accuracy; independently verified from confusion diagonal / class counts"}
        rel="derived/phase2_qa_fixes_v2/coverage/"+a["run_key"]+".json"
        b=jb(copy);outputs[rel]=b
        index.append({"run_key":a["run_key"],"file":rel,
            "original_coverage_sha256":a["coverage_sha256"],"original_metrics_sha256":a["metrics_full_sha256"],
            "derived_coverage_sha256":hashlib.sha256(b).hexdigest(),
            "train_recall":[v["accuracy"] for v in m["train"]["per_class"]],
            "validation_recall":[v["accuracy"] for v in m["validation"]["per_class"]]})
    outputs["derived/phase2_qa_fixes_v2/coverage_index.json"]=jb({
        "schema":"E-CH4-13-coverage-v2","source_training_SHA":SOURCE,
        "no_original_raw_modification":True,"test_accessed":False,"runs":index})
    return outputs

def pair_data(D):
    rows=[]
    for f in FAMILIES:
      for s in SEEDS:
        a=D[s,f,LOSSES[0]][1];b=D[s,f,LOSSES[1]][1]
        row={"seed":s,"family":f,"validation_BA_sigmoid":a["validation"]["balanced_accuracy"],
          "validation_BA_softplus":b["validation"]["balanced_accuracy"],
          "delta_BA_pp":100*(a["validation"]["balanced_accuracy"]-b["validation"]["balanced_accuracy"]),
          "delta_I":a["train"]["I"]-b["train"]["I"]}
        for split in ("train","validation"):
          for k in RS:
            row[split+"_"+k+"_sigmoid"]=a[split][k]
            row[split+"_"+k+"_softplus"]=b[split][k]
        rows.append(row)
    return rows

def trajectories(D):
    rows=[]
    for f in FAMILIES:
      for loss in LOSSES:
        for i in range(26):
          three=[D[s,f,loss][3] for s in SEEDS]
          if any(t[0]["objective_loss"]<=1e-12 or t[0]["gradient_l2"]<=1e-12 for t in three):
            raise ValueError("Near-zero relative normalization denominator")
          rows.append({"family":f,"loss":loss,"update_completed":i,
            "mean_loss_divided_by_initial":sum(t[i]["objective_loss"]/t[0]["objective_loss"] for t in three)/3,
            "mean_gradient_divided_by_initial":sum(t[i]["gradient_l2"]/t[0]["gradient_l2"] for t in three)/3,
            "mean_train_BA_percent":sum(100*t[i]["train_ba"] for t in three)/3})
    return rows

def begin(title,subtitle):
    return ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="640" viewBox="0 0 1100 640">',
       '<rect width="1100" height="640" fill="white"/>',
       '<text x="75" y="42" font-size="22" font-family="sans-serif" font-weight="bold">'+title+'</text>',
       '<text x="75" y="67" font-size="13" font-family="sans-serif">'+subtitle+'</text>']
def axes(p,lo,hi,ylabel,xlabels,mode="center"):
    p.extend(['<line x1="105" x2="1035" y1="532" y2="532" stroke="#444"/>',
       '<line x1="105" x2="105" y1="107" y2="532" stroke="#444"/>',
       '<text x="105" y="94" font-size="12" font-family="sans-serif">'+ylabel+'</text>'])
    for i in range(6):
      y=532-425*i/5;v=lo+(hi-lo)*i/5
      p.append(f'<line x1="105" x2="1035" y1="{y:.3f}" y2="{y:.3f}" stroke="#eee"/>')
      p.append(f'<text x="95" y="{y+4:.3f}" text-anchor="end" font-size="11" font-family="monospace">{v:.4g}</text>')
    for i,label in enumerate(xlabels):
      x=105+930*(i/(len(xlabels)-1) if mode=="end" else (i+.5)/len(xlabels))
      p.append(f'<text x="{x:.3f}" y="554" text-anchor="middle" font-size="11" font-family="sans-serif">{label}</text>')
def yc(v,lo,hi):return 532-425*(v-lo)/(hi-lo)
def done(p):
    p.append('<text x="75" y="617" font-size="11" font-family="sans-serif">Raw run 43670524935 | exact H=12 | Train 300 | Validation 90 | 25 Adam updates | No Test</text>')
    p.append('</svg>')
    return ("\n".join(p)+"\n").encode()

def plot_R(rows,split,metric):
    valuekey=split+"_"+metric
    hi=max(r[valuekey+"_"+loss] for r in rows for loss in ("sigmoid","softplus"))*1.2
    p=begin(split.title()+" paired "+metric,"Six original seed x policy pairs. Class-balanced density; lower = closer to class center.")
    axes(p,0,hi,metric,[str(r["seed"])[-4:]+" "+r["family"] for r in rows])
    for i,r in enumerate(rows):
      x=105+930*(i+.5)/6
      a=yc(r[valuekey+"_sigmoid"],0,hi);b=yc(r[valuekey+"_softplus"],0,hi)
      p.append(f'<line x1="{x-22:.3f}" y1="{a:.3f}" x2="{x+22:.3f}" y2="{b:.3f}" stroke="#777"/>')
      p.append(f'<circle cx="{x-22:.3f}" cy="{a:.3f}" r="6" fill="{C[0]}"/>')
      p.append(f'<circle cx="{x+22:.3f}" cy="{b:.3f}" r="6" fill="{C[1]}"/>')
    p.extend([f'<circle cx="778" cy="83" r="5" fill="{C[0]}"/><text x="790" y="88" font-size="12" font-family="sans-serif">Sigmoid</text>',
              f'<circle cx="889" cy="83" r="5" fill="{C[1]}"/><text x="900" y="88" font-size="12" font-family="sans-serif">Softplus</text>'])
    return done(p)

def plot_BA(rows):
    y=[r["delta_BA_pp"] for r in rows];lo=min(y)*1.2;hi=max(y)*1.3
    if lo>=0:lo=-1
    if hi<=0:hi=1
    p=begin("Validation BA: Sigmoid minus Softplus","Percentage points; positive values mean Sigmoid better; no cross-seed significance claim")
    axes(p,lo,hi,"Delta Validation BA (pp)",[str(r["seed"])[-4:]+" "+r["family"] for r in rows])
    z=yc(0,lo,hi)
    p.append(f'<line x1="105" x2="1035" y1="{z:.3f}" y2="{z:.3f}" stroke="#222" stroke-width="2"/>')
    for i,r in enumerate(rows):
      x=105+930*(i+.5)/6;v=r["delta_BA_pp"];y=yc(v,lo,hi)
      p.append(f'<rect x="{x-27:.3f}" y="{min(y,z):.3f}" width="54" height="{max(abs(y-z),.25):.3f}" fill="{C[0] if r["family"]=="shared" else C[1]}"/>')
      p.append(f'<text x="{x:.3f}" y="{y-9 if v>=0 else y+18:.3f}" text-anchor="middle" font-size="12" font-family="monospace">{v:+.3f}</text>')
    return done(p)

def plot_I(rows):
    xhi=max(r["delta_I"] for r in rows)*1.18
    ymin=min(r["delta_BA_pp"] for r in rows)*1.2
    ymax=max(r["delta_BA_pp"] for r in rows)*1.3
    p=begin("Center separation does not guarantee Validation BA","Six observed paired deltas; no causal interpretation, only 3 independent seeds")
    axes(p,ymin,ymax,"Delta Validation BA (pp)",[f'{xhi*i/5:.3f}' for i in range(6)],"end")
    for r in rows:
      x=105+930*r["delta_I"]/xhi;y=yc(r["delta_BA_pp"],ymin,ymax)
      p.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="7" fill="{C[0] if r["family"]=="shared" else C[1]}"/>')
      p.append(f'<text x="{x+8:.3f}" y="{y-8:.3f}" font-family="sans-serif" font-size="11">{str(r["seed"])[-4:]} {r["family"]}</text>')
    p.append('<text x="510" y="584" font-size="12" font-family="sans-serif">Delta center separation I (Sigmoid minus Softplus)</text>')
    return done(p)

def plot_trace(rows,key,title):
    lo=0;hi=max(row[key] for row in rows)*1.18
    p=begin(title,"Arithmetic mean over three seeds; normalize per seed before taking mean (BA remains %)")
    axes(p,lo,hi,key,['0','5','10','15','20','25'],"end")
    for j,(f,loss) in enumerate((f,l) for f in FAMILIES for l in LOSSES):
      seq=[v for v in rows if v["family"]==f and v["loss"]==loss]
      pts=" ".join(f'{105+930*v["update_completed"]/25:.3f},{yc(v[key],lo,hi):.3f}' for v in seq)
      p.append(f'<polyline points="{pts}" fill="none" stroke="{C[j]}" stroke-width="3"/>')
      x=620+(j%2)*215;y=84+(j//2)*22
      p.append(f'<line x1="{x}" x2="{x+22}" y1="{y}" y2="{y}" stroke="{C[j]}" stroke-width="3"/>')
      p.append(f'<text x="{x+27}" y="{y+4}" font-size="11" font-family="sans-serif">{f} {"Sigmoid" if loss==LOSSES[0] else "Softplus"}</text>')
    p.append('<text x="510" y="584" font-size="12" font-family="sans-serif">Completed optimizer update (0 .. 25)</text>')
    return done(p)

def produce():
    m,D=inputs()
    outputs=corrected(D);pairs=pair_data(D);tr=trajectories(D)
    outputs["derived/phase2_qa_fixes_v2/plot_pairs.csv"]=cb(pairs)
    outputs["derived/phase2_qa_fixes_v2/plot_trajectories.csv"]=cb(tr)
    figs={}
    for split in ("train","validation"):
      for metric in RS:figs[f"phase2_{split}_{metric}_six_pairs.svg"]=plot_R(pairs,split,metric)
    figs["phase2_validation_BA_paired_delta_6.svg"]=plot_BA(pairs)
    figs["phase2_delta_I_vs_delta_validation_BA.svg"]=plot_I(pairs)
    figs["phase2_train_loss_relative_dynamics.svg"]=plot_trace(tr,"mean_loss_divided_by_initial","Within-fit normalized objective")
    figs["phase2_gradient_norm_relative_dynamics.svg"]=plot_trace(tr,"mean_gradient_divided_by_initial","Within-fit normalized gradient L2")
    figs["phase2_train_BA_optimizer_dynamics.svg"]=plot_trace(tr,"mean_train_BA_percent","Train BA over Adam steps")
    if len(figs)!=11:raise AssertionError("11 primary plots required")
    for name,blob in figs.items():outputs["figures/phase2_repro_v2/"+name]=blob
    outputs["derived/phase2_qa_fixes_v2/plot_figure_manifest.json"]=jb({
       "schema":"E-CH4-13-reproducible-plot-v2","training_sha":SOURCE,
       "original_manifest_sha256":sha(RAW/"manifest.json"),
       "no_optimizer":True,"test_accessed":False,
       "method":"3 seed-wise ratios averaged at each update; all x/y coordinate source values archived as CSV",
       "figures":{name:hashlib.sha256(blob).hexdigest() for name,blob in sorted(figs.items())}})
    return outputs

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--check",action="store_true");a=ap.parse_args()
    outputs=produce()
    for rel,blob in outputs.items():
      path=HERE/rel
      if a.check:
        if not path.is_file() or path.read_bytes()!=blob:raise AssertionError("Output differs on byte-for-byte rebuild "+rel)
      else:
        if path.exists():raise FileExistsError("Refuse to overwrite existing output "+rel)
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob)
    print(json.dumps({"status":"CHECK_PASS" if a.check else "GENERATED_DERIVED_ONLY",
      "outputs":len(outputs),"new_coverage":12,"plots":11,"test_accessed":False,"optimizer_steps":0}))
if __name__=="__main__":main()
