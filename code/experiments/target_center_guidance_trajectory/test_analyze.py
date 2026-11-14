"""Regression checks for archived data, metric formulas, selection and continuity."""
from __future__ import annotations
import copy
import json
import unittest
from pathlib import Path
from analyze import analyze, geometry, validate_history, select_cases

HERE = Path(__file__).resolve().parent

class ComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads((HERE/'results/history_00_48.json').read_text(encoding='utf-8'))

    def test_merged_rounds_and_target_continuity(self):
        history=validate_history(self.data)
        self.assertEqual(len(history), 49)
        self.assertEqual(history[0]['round'], 0)
        self.assertEqual(history[-1]['round'], 48)
        self.assertEqual(history[5]['round'], 5)

    def test_first_round_uses_realized_not_target_center(self):
        b,first,peak=select_cases(self.data)
        self.assertEqual(first['round'],0)
        self.assertNotEqual(first['centers'], first['targets_used_to_construct_policy'])
        self.assertAlmostEqual(first['validation']['balanced_accuracy'], 0.7260596546310832)
        self.assertAlmostEqual(first['geometry']['I'], 0.26872891360274676)

    def test_best_round_is_chosen_only_among_measured_validation(self):
        _,_,peak=select_cases(self.data)
        self.assertEqual(peak['round'],4)
        self.assertAlmostEqual(peak['validation']['balanced_accuracy'],0.8383045525902669)
        self.assertEqual(peak['centers'],self.data['history'][4]['realized_centers'])

    def test_uniform_center_information_and_validation(self):
        base=select_cases(self.data)[0]
        self.assertAlmostEqual(base['geometry']['I'],0.01817675859745771,places=12)
        self.assertAlmostEqual(base['validation']['balanced_accuracy'],0.848639455782313)
        self.assertEqual(base['protocol'],'uniform')

    def test_center_geometry_metrics_obey_probabilities(self):
        v=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
        g=geometry(v)
        self.assertAlmostEqual(g['I'],__import__('math').log(3))
        self.assertEqual(g['T'],1.)
        self.assertEqual(g['BT'],1.)
        self.assertAlmostEqual(g['min_center_sqrt_jsd'],__import__('math').sqrt(__import__('math').log(2)))

    def test_reject_broken_checkpoint_chain(self):
        bad=copy.deepcopy(self.data)
        bad['history'][5]['targets'][0][0]+=0.05
        with self.assertRaisesRegex(ValueError,'Invalid|discontinuity'):
            validate_history(bad)

if __name__=='__main__':
    unittest.main(verbosity=2)
