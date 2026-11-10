"""Static and source-data checks for presentation of Ch.4 §§4.3–4.7.
No walk execution, Torch, generator, network, or Test-set access.
"""
from __future__ import annotations
import csv
import json
import re
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TEX = REPO / "tex/chapter4.tex"
SPECS = (
    ("sec:ch4-q-validation", "one_step_transition_population_convergence"),
    ("sec:ch4-uniform-hyperparameter-sensitivity", "uniform_policy_hyperparameter_sensitivity"),
    ("sec:ch4-fixed-policy-evaluation", "guided_informative_uniform_mixture"),
    ("sec:ch4-center-target-observation", "target_center_guidance_trajectory"),
    ("sec:ch4-sigmoid-margin-evaluation", "sigmoid_margin_observational_evaluation"),
)

def sections():
    manuscript=TEX.read_text(encoding="utf-8")
    first=manuscript.index(r"\section{اعتبارسنجی همگرایی")
    last=manuscript.index(r"\section{اعتبارسنجی محاسبه ماتریسی")
    text=manuscript[first:last]
    return re.findall(r"\\section\{([^}]+)\}(.*?)(?=\\section\{|\Z)", text,re.S)

def active_inputs():
    parts=sections()
    text="".join(x[1] for x in parts)
    return [(path, (REPO / path.lstrip("./"))) for path in
            re.findall(r"\\input\{([^}]+)\}", text)]

class Chapter43to47PresentationTests(unittest.TestCase):
    def test_each_main_section_has_one_appendix_note_at_its_end(self):
        parts=sections()
        self.assertEqual(len(parts),5)
        for (title,section),(label,slug) in zip(parts,SPECS):
            self.assertIn(r"\label{"+label+"}",section,title)
            note=r"\paragraph{پیوست الکترونیکی و بازتولید.}"
            self.assertEqual(section.count(note),1,title)
            before,after=section.split(note)
            self.assertIn(r"\texttt{"+slug.replace("_",r"\_")+"}",after,title)
            self.assertNotIn(r"\subsection{",after,title)
            self.assertLess(len(after),1700,title)
            # Do not write full repository paths in thesis prose.
            for ln in section.splitlines():
                if "code/experiments/" in ln:
                    self.assertTrue(ln.lstrip().startswith(r"\input{"),(title,ln))

    def test_active_tables_and_figures_exist_with_unique_labels(self):
        pairs=active_inputs()
        self.assertGreaterEqual(len(pairs),20)
        labels=[]
        for name,path in pairs:
            self.assertTrue(path.is_file(),name)
            src=path.read_text(encoding="utf-8")
            ls=re.findall(r"\\label\{([^}]+)\}",src)
            self.assertEqual(len(ls),1,name)
            self.assertNotRegex(src,r"\\lr\{(?:BA|Acc|Accuracy)\}")
            labels.extend(ls)
        self.assertEqual(len(labels),len(set(labels)))
        self.assertNotIn("fig:ch4-uniform-balanced-accuracy",labels)
        self.assertNotIn("fig:ch4-guided-mixture-ba",labels)
        # A source file for a redundant chart may remain archived;
        # it must not be imported to the printed chapter.
        appendix="".join(x[1] for x in sections())
        self.assertNotIn("figures/balanced_accuracy.tex}",appendix)

    def test_manuscript_refs_resolve_to_active_assets_or_inline(self):
        text="".join(x[1] for x in sections())
        active=set(re.findall(r"\\label\{([^}]+)\}",text))
        for _,path in active_inputs():
            active.update(re.findall(r"\\label\{([^}]+)\}",path.read_text(encoding="utf-8")))
        refs=set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}",text))
        for ref in refs:
            if ref.startswith(("tab:ch4-","fig:ch4-")):
                self.assertIn(ref,active)

    def test_size_study_is_preserved_not_recomputed(self):
        p=HERE/"sigmoid_margin_observational_evaluation/results/size_study"
        orig=HERE/"archive/margin_training/results/size_study"
        for name in ("runs.csv","aggregate.csv"):
            self.assertEqual((p/name).read_bytes(),(orig/name).read_bytes())
        with (p/"runs.csv").open(encoding="utf-8",newline="") as f:
            runs=list(csv.DictReader(f))
        with (p/"aggregate.csv").open(encoding="utf-8",newline="") as f:
            aggregate=list(csv.DictReader(f))
        self.assertEqual(len(runs),26)
        self.assertEqual(len(aggregate),6)
        self.assertEqual([int(x["n_train"]) for x in aggregate],[50,100,150,200,250,300])
        self.assertEqual([int(x["repeats"]) for x in aggregate],[5,5,5,5,5,1])
        cov=json.loads((HERE/"sigmoid_margin_observational_evaluation/results/coverage.json").read_text("utf-8"))
        self.assertEqual(cov["measurement_counts"]["extra_size_study_historical_runs"],26)
        self.assertTrue(any(x.get("scope")=="size study, 26 archived runs" for x in cov["not_measured"]))

    def test_chapter_47_distinguishes_two_study_phases(self):
        x=sections()[-1][1]
        self.assertIn(r"\subsection{آموزش مقدماتی سیاست",x)
        self.assertIn(r"\subsection{حساسیت دقت آموزش",x)
        for f in ("direct_training_300.tex","direct_training_72.tex"):
            source=(HERE/"sigmoid_margin_observational_evaluation/tables"/f).read_text("utf-8")
            self.assertIn("دقت متوازن آموزش",source)
            self.assertIn("دقت متوازن اعتبارسنجی",source)

if __name__=="__main__":
    unittest.main()
