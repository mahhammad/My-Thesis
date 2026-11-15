"""Negative-only tests for read-only phase2 reducer; no optimizer, no Test."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from phase2_aggregate import aggregate,predict
from phase2_protocol import grid,MAX_FITS

class AggregateGates(unittest.TestCase):
    def test_missing_or_incomplete_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):aggregate(Path(d)/"not_created")
            with self.assertRaises(FileNotFoundError):aggregate(Path(d))
            (Path(d)/"manifest.json").write_text('{"fit_count":11,"test_accessed":false,"runs":[]}')
            with self.assertRaises(ValueError):aggregate(d)

    def test_class_candidate_prediction_does_not_accept_validation_labels(self):
        arrays={
          "validation_by_hypothesis":np.array([
             [[.95,.05],[.95,.05]],
             [[.05,.95],[.05,.95]],
          ]),
          "centers":np.array([[.8,.2],[.2,.8]])
        }
        pred,scores=predict(arrays)
        np.testing.assert_array_equal(pred,[0,1])
        self.assertEqual(scores.shape,(2,2))
        self.assertEqual(len(grid()),MAX_FITS)

    def test_tie_is_lowest_index(self):
        arrays={"validation_by_hypothesis":np.full((2,2,2),.5),
                "centers":np.full((2,2),.5)}
        pred,_=predict(arrays)
        np.testing.assert_array_equal(pred,[0,0])

if __name__=="__main__":unittest.main()
