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

    def test_size_study_uses_independent_generator_seeds(self):
        root=HERE/"sigmoid_margin_observational_evaluation"
        with (HERE/"benchmark_v1/results/master_runs_76.csv").open(encoding="utf8",newline="") as f:
            master=list(csv.DictReader(f))
        expected=[r for r in master if r["study"]=="size_25steps"
                  and r["objective"]=="margin_JSD" and r["tau"]=="0.005"
                  and r["family"] in ("shared","class") and r["steps"]=="25"]
        with (root/"results/benchmark_size_3seeds.csv").open(encoding="utf8",newline="") as f:
            runs=list(csv.DictReader(f))
        with (root/"results/benchmark_size_aggregate.csv").open(encoding="utf8",newline="") as f:
            groups=list(csv.DictReader(f))
        self.assertEqual(len(expected),36)
        self.assertEqual(len(runs),36)
        self.assertEqual(len(groups),12)
        self.assertEqual({int(r["n"]) for r in runs},{50,100,150,200,250,300})
        self.assertEqual({r["family"] for r in runs},{"shared","class"})
        self.assertEqual({r["data_seed"] for r in runs},{"20261102","20261103","20261104"})
        originals={(r["n"],r["family"],r["data_seed"]):r for r in expected}
        for r in runs:
            old=originals[(r["n"],r["family"],r["data_seed"])]
            self.assertEqual(r["source_sha256"],old["sha256"])
            for k in ("train_BA","val_BA","I","R_JS","R_KL","R_sqrt_JS","objective_loss"):
                self.assertAlmostEqual(float(r[k]),float(old[k]),places=13)
        for group in groups:
            subset=[r for r in runs if r["n"]==group["n"] and r["family"]==group["family"]]
            self.assertEqual(len(subset),3)
            for key in ("train_BA","val_BA","I","R_JS","R_KL","R_sqrt_JS"):
                mean=sum(float(r[key]) for r in subset)/3
                sd=(sum((float(r[key])-mean)**2 for r in subset)/2)**.5
                self.assertAlmostEqual(float(group[key+"_mean"]),mean,places=12)
                self.assertAlmostEqual(float(group[key+"_std_sample"]),sd,places=12)
        cov=json.loads((root/"results/coverage.json").read_text("utf8"))
        self.assertEqual(cov["measurement_counts"]["benchmark_size_study_standard_runs"],36)
        self.assertEqual(cov["primary_training_reference_n"],300)

    def test_chapter_47_distinguishes_two_study_phases(self):
        x=sections()[-1][1]
        self.assertIn(r"\subsection{آموزش مقدماتی سیاست",x)
        self.assertIn(r"\subsection{حساسیت عملکرد به اندازه مرجع",x)
        self.assertNotIn("direct_training_72.tex}",x)
        for f in ("direct_training_300.tex",):
            source=(HERE/"sigmoid_margin_observational_evaluation/tables"/f).read_text("utf-8")
            self.assertIn("دقت متوازن آموزش",source)
            self.assertIn("دقت متوازن اعتبارسنجی",source)

if __name__=="__main__":
    unittest.main()
