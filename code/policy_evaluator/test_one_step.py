import sys, unittest
from pathlib import Path
import numpy as np
from walks import ReferenceGraph
from one_step import one_step_q
sys.path.insert(0,str(Path(__file__).parent.parent/'data_generator'))
from generate import finite_one_step, load_config

class OneStepQTests(unittest.TestCase):
    def test_hand_calculated_and_missing(self):
        x=np.array([[0.,np.nan],[1.,5.],[2.,np.nan],[3.,6.]],float)
        y=np.array([0,1,0,1])
        g=ReferenceGraph(x,y,np.ones(2))
        Q=one_step_q(g)
        self.assertEqual(Q.shape,(4,2,2))
        self.assertTrue(np.isnan(Q[0,1]).all())
        w=np.exp(-np.array([1.,4.,9.]));w/=w.sum()
        np.testing.assert_allclose(Q[0,0],[w[1],w[0]+w[2]])
        q=one_step_q(g, x[[1]],train_indices=np.array([1]), origin_indices=np.array([3]))
        self.assertTrue(np.isnan(q[0,1]).all())  # only x[3] had feature 1

    def test_against_generator_finite_q(self):
        config=Path(__file__).parent.parent/'data_generator/config.json'
        spec=load_config(config)
        rng=np.random.default_rng(19)
        x=rng.normal(size=(30,5)); x[rng.random(x.shape)<.2]=np.nan
        y=np.arange(30)%3
        g=ReferenceGraph(x,y,spec['gamma'])
        actual=one_step_q(g,batch_size=7)
        expected=finite_one_step(spec,x,x,y,train_queries=True)
        np.testing.assert_allclose(actual,expected,atol=1e-13,equal_nan=True)
        queries=x[:5]+.1
        actual=one_step_q(g,queries, batch_size=2)
        expected=finite_one_step(spec,queries,x,y)
        np.testing.assert_allclose(actual,expected,atol=1e-13,equal_nan=True)

    def test_no_query_label_used_and_origin_exclusion(self):
        x=np.array([[0.],[1.],[2.],[3.]])
        g=ReferenceGraph(x,np.array([0,1,1,2]),np.array([1.]))
        q=one_step_q(g,x[[1]],train_indices=np.array([1]),origin_indices=np.array([0]))
        w=np.exp(-np.array([1.,4.])) # destinations 2 and 3
        np.testing.assert_allclose(q[0,0],[0.,w[0]/w.sum(),w[1]/w.sum()])
        self.assertAlmostEqual(q[0,0].sum(),1.)

if __name__=='__main__': unittest.main()
