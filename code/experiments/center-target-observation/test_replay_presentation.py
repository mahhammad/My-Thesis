"""Evidence-only checks for newly replayed and balanced E-CH4-06 presentation."""
import json
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT=Path(__file__).resolve().parent

class BalancedReplayPresentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=json.loads((ROOT/"results/metrics_full.json").read_text("utf-8"))
        cls.cases=cls.report["replayed_fixed_targets"]["cases"]

    def test_49_rounds_and_44_legacy_aggregation_differences(self):
        self.assertEqual(len(self.cases),49)
        self.assertEqual([c["round"] for c in self.cases],list(range(49)))
        self.assertEqual(sum(bool(c["historical_validation_observed"]) for c in self.cases),25)
        self.assertEqual(sum(not c["historical_validation_observed"] for c in self.cases),24)
        mismatches=[c for c in self.cases
                    if not c["historical_R_JS_aggregation_diagnostic"]["matches_balanced"]]
        self.assertEqual(len(mismatches),44)
        for c in mismatches:
            self.assertTrue(c["historical_R_JS_aggregation_diagnostic"]["matches_unweighted"])

    def test_selected_table_uses_v2_balanced_rjs(self):
        src=(ROOT/"tables/selected_rounds_balanced.tex").read_text("utf-8")
        for n in (0,1,2,3,4,5,10,20,30,40,48):
            r=self.cases[n]
            line=next(x for x in src.splitlines() if x.startswith("$"+str(n)+"$ & "))
            self.assertIn(f"{r['train']['R_JS']:.5f}",line)
            self.assertIn(f"{r['train']['I']:.5f}",line)
            if not r["historical_validation_observed"]:
                self.assertIn("---",line)

    def test_balanced_svg_and_ba_provenance(self):
        ns={"s":"http://www.w3.org/2000/svg"}
        root=ET.parse(ROOT/"figures/trajectory_geometry_balanced.svg").getroot()
        line=root.find(".//s:polyline[@id='train_within_class_jsd']",ns)
        self.assertIsNotNone(line)
        coords=line.attrib["points"].split()
        self.assertEqual(len(coords),49)
        for pt,c in zip(coords,self.cases):
            x,y=map(float,pt.split(","))
            self.assertAlmostEqual(x,700+460*c["round"]/48,delta=.006)
            self.assertAlmostEqual(y,360-230*c["train"]["R_JS"]/.014,delta=.006)
        ba=ET.parse(ROOT/"figures/validation_ba_replayed_49.svg").getroot()
        data=[v for v in ba.findall(".//s:circle",ns) if float(v.attrib["cy"])<360]
        self.assertEqual(len(data),49)
        self.assertEqual(sum(v.attrib["fill"]=="white" for v in data),24)
        self.assertEqual(sum(v.attrib["fill"]!="white" for v in data),25)

if __name__=="__main__":
    unittest.main()
