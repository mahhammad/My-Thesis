"""E-CH4-13 figures 1-8: 2-panel seed-matched overlaid dots + per-method dashed min/mean/max.
A presentation-only, reproducible v3. Reads existing SHA-verified 12-fit frozen raw via
phase2_derived_reproduce_v2.inputs(); NEVER trains, imports torch, or accesses Test.
Previous v2 SVGs are retained as immutable presentation evidence.
Usage: python phase2_figures_1_8_v3.py [--check]
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import phase2_derived_reproduce_v2 as source

ROOT=Path(__file__).resolve().parent
TARGET=ROOT/"figures/phase2_repro_v3"
MANIFEST=ROOT/"derived/phase2_figures_1_8_v3_manifest.json"
SEEDS=source.SEEDS
FAMILIES=source.FAMILIES
COLORS={"sigmoid":"#1675AC","softplus":"#CF6B1C"}
NAMES={
  ("validation","BA"):"figure01_validation_BA_methods_by_seed.svg",
  ("train","I"):"figure02_train_I_methods_by_seed.svg",
}
for _split in ("train","validation"):
    for _metric in source.RS:
        NAMES[_split,_metric]=f"phase2_{_split}_{_metric}_six_pairs.svg"

def ycoord(value,hi):
    return 490-315*value/hi

def text(x,y,s,size=12,anchor="start",color="#252525",weight=None):
    q=f' font-weight="{weight}"' if weight else ""
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" font-family="DejaVu Sans, sans-serif" text-anchor="{anchor}" fill="{color}"{q}>{s}</text>'

def render(rows,split,metric):
    if split=="train" and metric=="I":key="train_I"
    elif split=="validation" and metric=="BA":key="validation_BA"
    else:key=f"{split}_{metric}"
    def grab(f,seed,method):
        a=next(row for row in rows if row["family"]==f and row["seed"]==seed)
        if metric=="BA":return 100*a["validation_BA_"+method]
        if metric=="I":
            # plots consume actual I values directly, not reconstruct from Delta I alone
            return a["train_I_"+method]
        return a[key+"_"+method]
    allvalues=[grab(f,s,m) for f in FAMILIES for s in SEEDS for m in COLORS]
    if not all(math.isfinite(v) and v>=0 for v in allvalues):raise ValueError("invalid plot value")
    hi=max(allvalues)*1.16
    if not hi>0:raise ValueError("zero metric")
    unit="%" if metric=="BA" else "class-balanced value"
    title=(("Validation balanced accuracy" if metric=="BA" else
            "Train center separation I" if metric=="I" else split.title()+" "+metric)
           +" — paired overlaid points")
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1210" height="635" viewBox="0 0 1210 635">',
        '<rect x="0" y="0" width="1210" height="635" fill="white"/>',
        text(60,38,title,21,weight="bold"),
        text(60,61,"Three independent DATA SEEDS per policy family; both losses share EXACTLY the same x",12),
        '<circle cx="776" cy="92" r="7" fill="#1675AC" stroke="white" stroke-width="1"/>',
        text(792,97,"Sigmoid",13),
        '<path d="M 914 84 L 923 93 L 914 102 L 905 93 Z" fill="#CF6B1C" stroke="white" stroke-width="1"/>',
        text(934,97,"Softplus",13),
        text(60,98,"Dashed summaries per method/panel: thicker = mean; thin = min and max",11),
        text(60,120,f"Y axis: {metric} ({unit}); same numerical scale in both panels",11)]
    # Same axes for both panels; horizontal summary dashed lines are not time trends.
    lefts={"shared":75,"class":645}
    for f in FAMILIES:
        left=lefts[f];right=left+485
        svg.append(f'<rect x="{left}" y="140" width="485" height="350" fill="#fbfbfc" stroke="#dedede"/>')
        svg.append(text((left+right)/2,158,f"Policy family: {f}",15,"middle",weight="bold"))
        for i in range(6):
            val=hi*i/5;y=ycoord(val,hi)
            svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#ebebeb" stroke-width="1"/>')
            if f=="shared":svg.append(text(left-9,y+4,f"{val:.4g}",11,"end","#505050"))
        # min/mean/max summaries within EACH FAMILY and METHOD, not pooling families
        for method in ("sigmoid","softplus"):
            vals=[grab(f,s,method) for s in SEEDS]
            summary=[("min",min(vals)),("mean",sum(vals)/3),("max",max(vals))]
            for name,val in summary:
                yp=ycoord(val,hi)
                strong=(name=="mean")
                svg.append(f'<line x1="{left+5}" y1="{yp:.3f}" x2="{right-5}" y2="{yp:.3f}" stroke="{COLORS[method]}" stroke-width="{2.2 if strong else 1.1}" stroke-dasharray="{"8 5" if strong else "3 6"}" opacity="{0.66 if strong else 0.38}"/>')
        for i,seed in enumerate(SEEDS):
            x=left+92+i*151
            svg.append(f'<line x1="{x:.2f}" y1="175" x2="{x:.2f}" y2="490" stroke="#dedede" stroke-width=".6"/>')
            for method in ("sigmoid","softplus"):
                v=grab(f,seed,method);yp=ycoord(v,hi)
                if method=="sigmoid":
                    # hollow edge makes coincident values visible; no horizontal dodge
                    svg.append(f'<circle cx="{x:.3f}" cy="{yp:.3f}" r="8" fill="white" stroke="{COLORS[method]}" stroke-width="3.2"/>')
                else:
                    svg.append(f'<path d="M {x:.3f} {yp-7:.3f} L {x+7:.3f} {yp:.3f} L {x:.3f} {yp+7:.3f} L {x-7:.3f} {yp:.3f} Z" fill="{COLORS[method]}" stroke="white" stroke-width=".9"/>')
            svg.append(text(x,516,str(seed),12,"middle"))
        svg.append(text((left+right)/2,545,"Data seed (not sample count or optimizer step)",11,"middle"))
    svg.append(text(60,586,"Each dashed level summarizes the 3 seeds within the SAME family; mean/min/max are descriptive, not confidence bounds.",11))
    svg.append(text(60,608,"Original authorized 12 fits | Train n=300 | Validation n=90 | exact H=12 | Test NOT RUN",11))
    svg.append("</svg>")
    return ("\n".join(svg)+"\n").encode()

def pairs(D):
    base=source.pair_data(D)
    for row in base:
        seed=row["seed"];family=row["family"]
        for label,loss in (("sigmoid",source.LOSSES[0]),("softplus",source.LOSSES[1])):
            row["train_I_"+label]=D[seed,family,loss][1]["train"]["I"]
    return base

def produce():
    manifest,D=source.inputs()  # validates all 60 original file digests
    rows=pairs(D)
    figs={name:render(rows,split,metric) for (split,metric),name in NAMES.items()}
    assert len(figs)==8 and len(set(figs))==8
    digest={name:hashlib.sha256(b).hexdigest() for name,b in sorted(figs.items())}
    meta={
        "schema":"E-CH4-13-figures-1-to-8-overlaid-v3",
        "source_training_sha":source.SOURCE,
        "source_manifest_sha256":source.sha(source.RAW/"manifest.json"),
        "figure_count":8,
        "summary_scope":"per policy family and method, three independent dataset seeds",
        "panels":["shared","class"],
        "raw_untouched":True,"no_optimizer":True,"test_accessed":False,
        "same_x_for_both_losses":True,
        "figures":digest,
    }
    outputs={f"figures/phase2_repro_v3/{name}":blob for name,blob in figs.items()}
    outputs["derived/phase2_figures_1_8_v3_manifest.json"]=source.jb(meta)
    return outputs

def main():
    p=argparse.ArgumentParser();p.add_argument("--check",action="store_true");args=p.parse_args()
    outputs=produce()
    for name,blob in outputs.items():
        path=ROOT/name
        if args.check:
            if not path.exists() or path.read_bytes()!=blob:
                raise AssertionError("Nonreproducible v3 output: "+name)
        else:
            if path.exists():raise FileExistsError("Refusing overwrite of v3 figure "+name)
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(blob)
    print(json.dumps({"status":"CHECK_PASS" if args.check else "GENERATED",
                      "files":len(outputs),"figures":8,"fits":0,"test_accessed":False}))
if __name__=="__main__":main()
