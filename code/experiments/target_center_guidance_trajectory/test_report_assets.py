"""Evidence-only checks: no Monte Carlo, no fitting, no Test split."""
from __future__ import annotations
import csv
from hashlib import sha1
import json
from pathlib import Path
import unittest
from xml.etree import ElementTree as ET

HERE = Path(__file__).resolve().parent
RAW = HERE / "results"
ARCHIVE = HERE.parent / "archive/center_matching_comparison/results"


def read_csv(name):
    # The byte-preserved legacy CSV embeds two audit metadata lines before
    # the real index,... header. Preserve source bytes; skip only at read time.
    with (RAW / name).open(encoding="utf8", newline="") as f:
        lines = f.readlines()
    offset = next(i for i, line in enumerate(lines)
                  if line.startswith("index,"))
    return list(csv.DictReader(lines[offset:]))


class TrajectoryReportEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = json.loads((RAW/"history_00_48.json").read_text("utf8"))
        cls.rows = read_csv("trajectory_metrics.csv")
        cls.detail = json.loads((RAW/"detailed_three_cases.json").read_text("utf8"))
        cls.manifest = json.loads((RAW/"provenance.json").read_text("utf8"))

    def test_all_archival_files_exact_bytes(self):
        self.assertEqual(len(self.manifest["raw_files"]),11)
        for name,item in self.manifest["raw_files"].items():
            data=(RAW/name).read_bytes()
            self.assertEqual(data,(ARCHIVE/name).read_bytes())
            self.assertEqual(len(data),item["bytes"])
            sha=sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
            self.assertEqual(sha,item["git_blob_sha"])

    def test_49_rounds_25_measured_and_best_round4(self):
        h=self.history["history"]
        self.assertEqual(len(h),49)
        self.assertEqual(len(self.rows),49)
        self.assertEqual([r["round"] for r in h],list(range(49)))
        observed=[]
        for i,row in enumerate(self.rows):
            self.assertEqual(int(row["round"]),i)
            self.assertAlmostEqual(float(row["I"]),h[i]["center_information"],delta=1e-12)
            self.assertAlmostEqual(float(row["mean_target_realized_jsd"]),h[i]["mean_target_realized_jsd"],delta=1e-12)
            val=h[i].get("validation_hypothesis")
            if val is None:
                self.assertEqual(row["validation_balanced_accuracy"],"")
            else:
                self.assertAlmostEqual(float(row["validation_balanced_accuracy"]),val["balanced_accuracy"],delta=1e-12)
                observed.append(row)
        self.assertEqual(len(observed),25)
        self.assertEqual(max(observed,key=lambda x:float(x["validation_balanced_accuracy"]))["round"],"4")

    def test_three_state_table_is_grounded(self):
        src=(HERE/"tables/three_cases.tex").read_text("utf8")
        self.assertEqual(len(self.detail["cases"]),3)
        for case in self.detail["cases"]:
            self.assertIn(f"{case['geometry']['I']:.5f}",src)
            self.assertIn(f"{case['training_within']['JSD_within']:.5f}",src)
            self.assertIn(f"{100*case['validation_prediction']['balanced_accuracy']:.2f}",src)
        self.assertEqual(self.detail["cases"][0]["inference_protocol"],"class_independent")
        for case in self.detail["cases"][1:]:
            self.assertEqual(case["inference_protocol"],"candidate_class_conditional")

    def test_missing_ba_not_interpolated_in_selected_table(self):
        src=(HERE/"tables/selected_rounds.tex").read_text("utf8")
        for i in [0,1,2,3,4,5,10,20,30,40,48]:
            row=self.rows[i]
            match=next((line for line in src.splitlines() if line.startswith("$"+str(i)+"$ & ")),None)
            self.assertIsNotNone(match)
            self.assertIn(f"{float(row['I']):.5f}",match)
            self.assertIn(f"{float(row['mean_target_realized_jsd']):.6f}",match)
            if not row["validation_balanced_accuracy"]:
                self.assertIn("---",match)
            else:
                self.assertIn(f"{100*float(row['validation_balanced_accuracy']):.2f}",match)

    def test_svg_observations_and_full_49_geometry(self):
        ns={"s":"http://www.w3.org/2000/svg"}
        a=ET.parse(HERE/"figures/trajectory_geometry.svg").getroot()
        b=ET.parse(HERE/"figures/observed_validation_ba.svg").getroot()
        for key,mx,panel in [("I",.28,110),("train_within_class_jsd",.014,700)]:
            line=a.find(".//s:polyline[@id='"+key+"']",ns)
            self.assertIsNotNone(line)
            coords=line.attrib["points"].split()
            self.assertEqual(len(coords),49)
            for xy,row in zip(coords,self.rows):
                x,y=map(float,xy.split(","))
                self.assertAlmostEqual(x,panel+460*int(row["round"])/48,delta=.006)
                self.assertAlmostEqual(y,360-230*float(row[key])/mx,delta=.006)
        circles=b.findall(".//s:circle",ns)
        obs=[r for r in self.rows if r["validation_balanced_accuracy"]]
        self.assertEqual(len(circles),25)
        for mark,row in zip(circles,obs):
            self.assertAlmostEqual(float(mark.attrib["cx"]),112+850*int(row["round"])/48,delta=.006)
            self.assertAlmostEqual(float(mark.attrib["cy"]),355-260*float(row["validation_balanced_accuracy"]),delta=.006)

    def test_latex_sources_have_unique_figure_table_labels(self):
        targets=[
            ("figures/trajectory_geometry.tex","fig:ch4-target-center-trajectory-geometry"),
            ("figures/observed_validation_ba.tex","fig:ch4-target-center-trajectory-validation"),
            ("tables/three_cases.tex","tab:ch4-center-trajectory-three-cases"),
            ("tables/selected_rounds.tex","tab:ch4-center-trajectory-selected"),
        ]
        for name,label in targets:
            src=(HERE/name).read_text("utf8")
            self.assertEqual(src.count("\\label{"+label+"}"),1)
        points=(HERE/"figures/observed_validation_ba.tex").read_text("utf8")
        self.assertEqual(points.count("\\fill["),25)


    def test_chapter45_and_chapter46_link_registered_sources_exactly_once(self):
        chapter=(HERE.parents[2]/"tex/chapter4.tex").read_text("utf8")
        self.assertEqual(chapter.count(
            r"\section{ارزیابی سیاست یکنواختِ بدون نویز و مقایسه معیارها}"),1)
        self.assertEqual(chapter.count(
            r"\section{هدایت گشت به مراکز هدف و تحلیل مسیر معیارهای بازنمایی}"),1)
        for folder,names in (
            ("guided_informative_uniform_mixture",[
                "tables/main_results.tex", "tables/centers_endpoints.tex",
                "tables/center_distances.tex", "tables/classwise_dispersion.tex",
                "figures/geometry.tex", "figures/balanced_accuracy.tex",
            ]),
            ("target_center_guidance_trajectory",[
                "tables/selected_rounds.tex", "figures/trajectory_geometry.tex",
                "figures/observed_validation_ba.tex",
            ]),
        ):
            for name in names:
                path="./code/experiments/"+folder+"/"+name
                self.assertEqual(chapter.count("\\input{"+path+"}"),1)
                self.assertTrue((HERE.parent/folder/name).is_file())


if __name__=="__main__":
    unittest.main()
