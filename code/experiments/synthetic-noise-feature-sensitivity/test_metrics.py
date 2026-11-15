import json
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from engine import ExactMaskedWalk
from runner import initial_logits
from metrics import report_graph,by_hypothesis,metric_coverage


def small_graph():
    rng=np.random.default_rng(205)
    tx=rng.normal(size=(6,3)).astype(np.float64)
    vx=rng.normal(size=(3,3)).astype(np.float64)
    ty=np.array([0,1,2,0,1,2],dtype=np.int64)
    vy=np.array([0,1,2],dtype=np.int64)
    return ExactMaskedWalk(tx,ty,np.ones(3)*.5,horizon=2,query_x=vx),ty,vy


class MetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_canonical_metrics_and_heldout_label_separation(self):
        g,ty,vy=small_graph()
        for family in ('shared','class'):
            z=initial_logits(3,family)
            train_hyp,val_hyp=by_hypothesis(g,z,family)
            self.assertEqual(train_hyp.shape,(6,3,3))
            self.assertEqual(val_hyp.shape,(3,3,3))
            report=report_graph(g,z,family,ty,vy,data_hashes={'fixture':True},subset_id='fixture')
            other=report_graph(g,z,family,ty,np.roll(vy,1),data_hashes={'fixture':True},subset_id='fixture')
            self.assertEqual(report['centers'],other['centers'])
            self.assertEqual(report['train']['balanced_accuracy'],other['train']['balanced_accuracy'])
            self.assertEqual(report['validation']['per_sample']['scores_all_candidates'],
                             other['validation']['per_sample']['scores_all_candidates'])
            self.assertFalse(report['test_accessed'])
            self.assertEqual(report['center_source'],'train_only')
            self.assertEqual(report['metric_contract'],'chapter4-common-evaluation-contract-v2')
            for split in ('train','validation'):
                for key in ('R_KL','R_JS','R_sqrt_JS','balanced_accuracy','ordinary_accuracy',
                            'margin_quantiles','per_class_recall','confusion_true_rows_pred_cols'):
                    self.assertIn(key,report[split])
            self.assertIn('I',report['train']['center_geometry'])
            self.assertEqual(report['coverage']['test']['status'],'not_applicable')
            self.assertEqual(report['coverage']['train.per_class_recall']['status'],'measured')
            self.assertNotIn('validation.I',report['coverage'])

    def test_contract_coverage_unavailable_is_not_zero(self):
        contract={'metric_groups':[{'group':'density','ids':['R_KL','R_JS'],'center_only':False},
                                   {'group':'experiment_specific','ids':[]}]}
        c=metric_coverage({'R_KL':None,'R_JS':0.0},{'R_KL':None,'R_JS':0.4},contract)
        self.assertEqual(c['train.R_KL']['status'],'undefined_or_infinite')
        self.assertEqual(c['train.R_JS']['status'],'measured')
        self.assertEqual(c['test']['status'],'not_applicable')

if __name__=='__main__':unittest.main()
