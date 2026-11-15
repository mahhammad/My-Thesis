"""Persian publication figures for E-CH4-13. Presentation-only, NO FIT and NO Test.

Source graphics remain byte-for-byte immutable. All original non-text SVG vector
elements are copied without geometry mutation; English text nodes are replaced by
shaped Persian glyph OUTLINES. Persian Unicode originals are stored in data-fa-text
attributes and in the SHA manifest. PDFs are vector exports of these SVGs.
Run: python phase2_persian_figures.py [--check]
"""
from __future__ import annotations
import argparse,hashlib,json,re,xml.etree.ElementTree as ET
from pathlib import Path
import arabic_reshaper
from bidi.algorithm import get_display
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.path import Path as MplPath
import cairosvg
import fitz

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"figures/phase2_tex_fa"
MANIFEST=ROOT/"derived/phase2_persian_figures_manifest.json"
FONT=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
SVG="http://www.w3.org/2000/svg"
ET.register_namespace("",SVG)
SOURCES=[
 ("figures/phase2_repro_v3/figure01_validation_BA_methods_by_seed.svg","e13_01_fa"),
 ("figures/phase2_repro_v3/figure02_train_I_methods_by_seed.svg","e13_02_fa"),
 ("figures/phase2_repro_v3/phase2_train_R_KL_six_pairs.svg","e13_03_fa"),
 ("figures/phase2_repro_v3/phase2_validation_R_KL_six_pairs.svg","e13_04_fa"),
 ("figures/phase2_repro_v3/phase2_train_R_JS_six_pairs.svg","e13_05_fa"),
 ("figures/phase2_repro_v3/phase2_validation_R_JS_six_pairs.svg","e13_06_fa"),
 ("figures/phase2_repro_v3/phase2_train_R_sqrt_JS_six_pairs.svg","e13_07_fa"),
 ("figures/phase2_repro_v3/phase2_validation_R_sqrt_JS_six_pairs.svg","e13_08_fa"),
 ("figures/phase2_repro_v2/phase2_train_loss_relative_dynamics.svg","e13_09_fa"),
 ("figures/phase2_repro_v2/phase2_gradient_norm_relative_dynamics.svg","e13_10_fa"),
 ("figures/phase2_tex/figure11_train_BA_y70_100.svg","e13_11_fa"),
]
PHRASES={
 "Sigmoid":"سیگموئید",
 "Softplus":"سافت‌پلاس",
 "shared Sigmoid":"مشترک، سیگموئید",
 "shared Softplus":"مشترک، سافت‌پلاس",
 "class Sigmoid":"شرطی کلاس، سیگموئید",
 "class Softplus":"شرطی کلاس، سافت‌پلاس",
 "Policy family: shared":"خانواده سیاست: مشترک",
 "Policy family: class":"خانواده سیاست: شرطی کلاس",
 "Three independent DATA SEEDS per policy family; both losses share EXACTLY the same x":"سه بذر مستقل برای هر خانواده؛ دو روش در هر بذر در یک مختصات افقی",
 "Dashed summaries per method/panel: thicker = mean; thin = min and max":"خط‌چین‌های هر روش: ضخیم‌تر میانگین؛ نازک‌تر کمینه و بیشینه",
 "Data seed (not sample count or optimizer step)":"بذر تولید داده (نه تعداد نمونه یا گام آموزش)",
 "Each dashed level summarizes the 3 seeds within the SAME family; mean/min/max are descriptive, not confidence bounds.":"خط‌چین‌ها خلاصه سه بذر همان خانواده‌اند؛ نه بازه اطمینان",
 "Original authorized 12 fits | Train n=300 | Validation n=90 | exact H=12 | Test NOT RUN":"۱۲ آموزش ثبت‌شده؛ آموزش ۳۰۰، اعتبارسنجی ۹۰؛ گشت دقیق، بدون آزمون",
 "Arithmetic mean over three seeds; normalize per seed before taking mean (BA remains %)":"میانگین سه بذر؛ ابتدا نرمال‌سازی جداگانه برای هر بذر",
 "Within-fit normalized objective":"تغییر نسبی تابع زیان در طول آموزش",
 "Within-fit normalized gradient L2":"تغییر نسبی هنجار گرادیان در طول آموزش",
 "mean_loss_divided_by_initial":"زیان نسبت به مقدار آغازین",
 "mean_gradient_divided_by_initial":"هنجار گرادیان نسبت به مقدار آغازین",
 "Completed optimizer update (0 .. 25)":"گام تکمیل‌شده بهینه‌سازی (۰ تا ۲۵)",
 "Raw run 43670524935 | exact H=12 | Train 300 | Validation 90 | 25 Adam updates | No Test":"گشت دقیق با افق ۱۲؛ آموزش ۳۰۰؛ اعتبارسنجی ۹۰؛ ۲۵ گام آدام؛ بدون آزمون",
 "Train balanced accuracy during Adam optimization":"سیر دقت متوازن آموزش در بهینه‌سازی آدام",
 "Mean BA across 3 independent dataset seeds; percent; fixed y axis 70–100":"میانگین سه بذر مستقل؛ درصد؛ محور عمودی ثابت از ۷۰ تا ۱۰۰",
 "Completed Adam updates (0 to 25)":"گام‌های تکمیل‌شده آدام (۰ تا ۲۵)",
 "Original 12 exact-walk fits | fixed final checkpoint 25 | no validation trajectory | no Test":"۱۲ آموزش با گشت دقیق؛ گام نهایی ۲۵؛ بدون روند اعتبارسنجی یا آزمون",
}
TITLE_RE=re.compile(r"^(Train|Validation) (balanced accuracy|center separation I|R_KL|R_JS|R_sqrt_JS) — paired overlaid points$")
Y_RE=re.compile(r"^Y axis: (BA|I|R_KL|R_JS|R_sqrt_JS) \((?:%|class-balanced value)\); same numerical scale in both panels$")
def translate(s):
    if s in PHRASES:return PHRASES[s]
    m=TITLE_RE.fullmatch(s)
    if m:
        split="آموزش" if m.group(1)=="Train" else "اعتبارسنجی"
        metric={"balanced accuracy":"دقت متوازن BA","center separation I":"جدایی مراکز I",
                "R_KL":"تراکم R_KL","R_JS":"تراکم R_JS","R_sqrt_JS":"تراکم R_sqrt_JS"}[m.group(2)]
        return metric+" در مجموعه "+split+"؛ مقایسه جفتی"
    m=Y_RE.fullmatch(s)
    if m:
        return "محور عمودی: "+m.group(1)+"؛ مقیاس عددی یکسان در هر دو پنل"
    # Numeric ticks and seed identifiers retain their scientific format.
    if re.fullmatch(r"[0-9.\-]+",s):return s
    raise ValueError("Untranslated English text: "+repr(s))

def svg_path(tp):
    commands=[]
    for seg,code in tp.iter_segments(curves=True,simplify=False):
        if code==MplPath.MOVETO:commands.append("M %.4f %.4f"%tuple(seg))
        elif code==MplPath.LINETO:commands.append("L %.4f %.4f"%tuple(seg))
        elif code==MplPath.CURVE3:commands.append("Q "+" ".join("%.4f"%v for v in seg))
        elif code==MplPath.CURVE4:commands.append("C "+" ".join("%.4f"%v for v in seg))
        elif code==MplPath.CLOSEPOLY:commands.append("Z")
        else:raise ValueError("Unexpected font path code "+str(code))
    return " ".join(commands)

def translate_one(src,source_name):
    root=ET.fromstring(src)
    texts=[x for x in root.iter() if x.tag==f"{{{SVG}}}text"]
    ntext=len(texts)
    translations={}
    for parent in list(root.iter()):
        for index,element in enumerate(list(parent)):
            if element.tag!=f"{{{SVG}}}text":continue
            raw="".join(element.itertext())
            fa=translate(raw);translations[raw]=fa
            # Preserve numeric labels unchanged, including all original positions.
            if fa==raw:continue
            x=float(element.attrib["x"]);y=float(element.attrib["y"])
            size=float(element.attrib.get("font-size",12))
            anchor=element.attrib.get("text-anchor","start")
            # Keep long right-to-left publication headings/footers right aligned;
            # other labels (legends, panels, axes) retain their original loci.
            if ("— paired overlaid points" in raw or raw in (
                "Within-fit normalized objective","Within-fit normalized gradient L2",
                "Train balanced accuracy during Adam optimization")):
                x=float(root.get("width"))-80;anchor="end"
            elif raw.startswith(("Three independent DATA SEEDS","Arithmetic mean over three seeds",
                                "Mean BA across","Dashed summaries","Each dashed level",
                                "Original authorized 12 fits","Original 12 exact-walk fits",
                                "Raw run 4367","Y axis:")):
                x=float(root.get("width"))-70;anchor="end"
            # Original footer for v3 and v2 must remain one unclipped visual line.
            rtl=get_display(arabic_reshaper.reshape(fa))
            max_width=min(float(root.get("width"))-130,1030)
            prop=FontProperties(fname=str(FONT),weight="bold" if element.attrib.get("font-weight")=="bold" else "normal")
            path=TextPath((0,0),rtl,size=size,prop=prop)
            bounds=path.get_extents()
            if bounds.width>max_width:
                size*=max_width/bounds.width
                path=TextPath((0,0),rtl,size=size,prop=prop)
                bounds=path.get_extents()
            # Align using ink bounds in visual reading order. SVG y-down.
            origin_x=x-(bounds.x1 if anchor=="end" else (bounds.x0+bounds.x1)/2 if anchor=="middle" else bounds.x0)
            glyph=ET.Element(f"{{{SVG}}}path",{
                "d":svg_path(path),
                "transform":f"translate({origin_x:.4f} {y:.4f}) scale(1 -1)",
                "fill":element.attrib.get("fill","#252525"),
                "class":"persian-figure-text",
                "data-fa-text":fa,
                "data-original-text":raw,
                "data-font-size":f"{size:.4f}",
            })
            parent.remove(element);parent.insert(index,glyph)
    if ntext<10:raise AssertionError("Unexpected text node count "+source_name)
    out=ET.tostring(root,encoding="utf-8",xml_declaration=True)
    # Structural proof of unchanged scientific graph geometry:
    # compare nontext old children and non-persian new children as canonical XML.
    oldroot=ET.fromstring(src);newroot=ET.fromstring(out)
    oldgeo=[(el.tag,tuple(sorted(el.attrib.items()))) for el in oldroot.iter() if el.tag!=f"{{{SVG}}}text"]
    newgeo=[(el.tag,tuple(sorted(el.attrib.items()))) for el in newroot.iter() if el.tag!=f"{{{SVG}}}text" and el.attrib.get("class")!="persian-figure-text"]
    if oldgeo!=newgeo:raise AssertionError("Non-text scientific geometry changed "+source_name)
    if not any("سیگموئید" in v for v in translations.values()):raise AssertionError("Method legend not translated")
    return out,translations

def sha(data):return hashlib.sha256(data).hexdigest()
def produce():
    data={};manifest={"schema":"e-ch4-13-fa-figures-v1","source_originals_immutable":True,
                      "data_unchanged":True,"optimizer_executed":False,"test_accessed":False,
                      "font_family":"DejaVu Sans","font_rendering":"shaped Unicode Persian mapped to vector outlines",
                      "records":[]}
    manifest_v3=json.loads((ROOT/"derived/phase2_figures_1_8_v3_manifest.json").read_text())["figures"]
    manifest_v2=json.loads((ROOT/"derived/phase2_qa_fixes_v2/plot_figure_manifest.json").read_text())["figures"]
    for i,(rel,name) in enumerate(SOURCES,1):
        original=(ROOT/rel).read_bytes()
        if i<=8 and sha(original)!=manifest_v3[Path(rel).name]:raise AssertionError("V3 English SVG hash mismatch")
        if i in (9,10) and sha(original)!=manifest_v2[Path(rel).name]:raise AssertionError("V2 English SVG hash mismatch")
        svg,labels=translate_one(original,rel)
        pdf=cairosvg.svg2pdf(bytestring=svg)
        # Source geometry remains vector in PDF (no full-page raster image).
        doc=fitz.open(stream=pdf,filetype="pdf")
        if len(doc)!=1 or len(doc[0].get_drawings())<15:raise AssertionError("PDF geometry too sparse")
        if doc[0].get_images(full=True):raise AssertionError("Publication PDF unexpectedly rasterized")
        record={"figure":i,"source":rel,"source_sha256":sha(original),
                "persian_svg":f"figures/phase2_tex_fa/{name}.svg",
                "persian_svg_sha256":sha(svg),
                "persian_pdf":f"figures/phase2_tex_fa/{name}.pdf",
                "persian_pdf_sha256":sha(pdf),
                "source_text_to_persian":labels,
                "pdf_vector_drawings":len(doc[0].get_drawings())}
        manifest["records"].append(record)
        data[f"{name}.svg"]=svg;data[f"{name}.pdf"]=pdf
    if len(data)!=22:raise AssertionError("Expected exactly 11 Persian SVG/PDF pairs")
    manifest_bytes=(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode()
    return data,manifest_bytes

def main():
    p=argparse.ArgumentParser();p.add_argument("--check",action="store_true");args=p.parse_args()
    files,mb=produce()
    if args.check:
        expected=json.loads(mb)
        actual=json.loads(MANIFEST.read_bytes())
        # Cairo PDF metadata need not be a byte-identical second rendering.
        # Canonical output checks use stored SHA256 for both file types.
        for key,blob in files.items():
            f=OUT/key
            record=next(v for v in actual["records"] if v["persian_svg"].endswith(key) or v["persian_pdf"].endswith(key))
            saved=record["persian_svg_sha256" if key.endswith(".svg") else "persian_pdf_sha256"]
            if not f.is_file() or sha(f.read_bytes())!=saved:raise AssertionError("Corrupted Persian output "+key)
            if key.endswith(".svg") and f.read_bytes()!=blob:raise AssertionError("Nondeterministic SVG "+key)
        for left,right in zip(actual["records"],expected["records"]):
            for k in ("source","source_sha256","persian_svg","persian_svg_sha256","persian_pdf","source_text_to_persian"):
                if left[k]!=right[k]:raise AssertionError("Manifest source mismatch: "+k)
    else:
        if OUT.exists() or MANIFEST.exists():raise FileExistsError("Refusing overwrite of existing Persian outputs")
        OUT.mkdir(parents=True)
        MANIFEST.parent.mkdir(parents=True,exist_ok=True)
        for key,blob in files.items():(OUT/key).write_bytes(blob)
        MANIFEST.write_bytes(mb)
    print(json.dumps({"status":"CHECK_PASS" if args.check else "GENERATED","svg_count":11,
                      "pdf_count":11,"trained":False,"test_used":False},ensure_ascii=False))
if __name__=="__main__":main()
