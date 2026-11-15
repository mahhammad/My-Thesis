"""Deterministic presentation-only Figure 11: Train BA in [70,100].

Reads SHA-verified original raw checkpoint traces and never runs fitting or Test.
The v2 SVG remains immutable. This produces only a NEW SVG for publication.
With --check, checks byte-by-byte against the committed output.
"""
from pathlib import Path
import argparse
import html
import phase2_derived_reproduce_v2 as v2

HERE=Path(__file__).resolve().parent
OUT=HERE/"figures/phase2_tex/figure11_train_BA_y70_100.svg"
KEY="mean_train_BA_percent"
COLORS=["#2778a5","#d37b25","#57934e","#9367a4"]

def make():
    _,D=v2.inputs()
    t=v2.trajectories(D)
    if min(x[KEY] for x in t)<70 or max(x[KEY] for x in t)>100:
        raise AssertionError("Observed Train BA lies outside requested [70,100] axis")
    w,h=1100,650
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
         '<rect width="1100" height="650" fill="#ffffff"/>',
         '<text x="80" y="41" font-size="23" font-weight="bold" font-family="DejaVu Sans, sans-serif">Train balanced accuracy during Adam optimization</text>',
         '<text x="80" y="67" font-size="13" font-family="DejaVu Sans, sans-serif">Mean BA across 3 independent dataset seeds; percent; fixed y axis 70–100</text>']
    left,right,top,bottom=105,1035,140,532
    y=lambda z:bottom-(z-70)*(bottom-top)/30
    for val in (70,75,80,85,90,95,100):
        yp=y(val)
        out.append(f'<line x1="{left}" x2="{right}" y1="{yp:.3f}" y2="{yp:.3f}" stroke="#e8e8e8"/>')
        out.append(f'<text x="{left-12}" y="{yp+4:.3f}" text-anchor="end" font-family="DejaVu Sans, sans-serif" font-size="12">{val}</text>')
    out.extend([f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#444"/>',
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#444"/>'])
    for x in (0,5,10,15,20,25):
        px=left+(right-left)*x/25
        out.append(f'<text x="{px:.3f}" y="554" text-anchor="middle" font-family="DejaVu Sans, sans-serif" font-size="12">{x}</text>')
    for j,(f,l) in enumerate((f,l) for f in v2.FAMILIES for l in v2.LOSSES):
        seq=[x for x in t if x["family"]==f and x["loss"]==l]
        points=" ".join(f'{left+(right-left)*x["update_completed"]/25:.3f},{y(x[KEY]):.3f}' for x in seq)
        out.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[j]}" stroke-width="3.2"/>')
        # visible markers at every 5 updates make crossings easier to track
        for x in seq:
            if x["update_completed"]%5==0:
                cx=left+(right-left)*x["update_completed"]/25;cy=y(x[KEY])
                out.append(f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="3.1" fill="white" stroke="{COLORS[j]}" stroke-width="1.5"/>')
        label=f'{f} {"Sigmoid" if l==v2.LOSSES[0] else "Softplus"}'
        lx=610+(j%2)*205;ly=91+(j//2)*25
        out.append(f'<line x1="{lx}" x2="{lx+22}" y1="{ly}" y2="{ly}" stroke="{COLORS[j]}" stroke-width="3"/>')
        out.append(f'<text x="{lx+27}" y="{ly+4}" font-family="DejaVu Sans, sans-serif" font-size="12">{html.escape(label)}</text>')
    out.append('<text x="525" y="585" text-anchor="middle" font-family="DejaVu Sans, sans-serif" font-size="13">Completed Adam updates (0 to 25)</text>')
    out.append('<text x="80" y="625" font-family="DejaVu Sans, sans-serif" font-size="11">Original 12 exact-walk fits | fixed final checkpoint 25 | no validation trajectory | no Test</text>')
    out.append('</svg>')
    return ("\n".join(out)+"\n").encode()

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--check",action="store_true");a=parser.parse_args()
    expected=make()
    if a.check:
        if not OUT.is_file() or OUT.read_bytes()!=expected:raise AssertionError("Figure 11 mismatch")
    else:
        if OUT.exists():raise FileExistsError("Refuse to overwrite existing figure 11")
        OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_bytes(expected)
    print("FIGURE11_CHECK_PASS" if a.check else "FIGURE11_GENERATED",len(expected))
if __name__=="__main__":main()
