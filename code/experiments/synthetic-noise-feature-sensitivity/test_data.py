import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (SUBSETS, EXPECTED_COUNTS, Mother, NoiseSpec, column_mapping,
                  fixture_mother, generate_raw_noise, generate_noise_masks,
                  derive_all, derived_manifest, fingerprint, save_npz_once,
                  validate_mother, load_mother)


def fixture():
    rng = np.random.default_rng(17)
    tx = rng.normal(size=(9, 5)).astype(np.float64)
    vx = rng.normal(size=(6, 5)).astype(np.float64)
    tx[0, 2] = np.nan
    vx[1, 4] = np.nan
    return fixture_mother(tx, np.array([0, 1, 2]*3, dtype=np.int64),
                          vx, np.array([0, 1, 2]*2, dtype=np.int64))


def spec():
    return NoiseSpec(raw_seed=418, mask_seed=419, std=(1., 1.1, .8),
                     gamma=(.9, .8, .7), missing_rate=(0., .2, .35))


class DataTests(unittest.TestCase):
    def test_exact_subsets(self):
        self.assertEqual(len(SUBSETS), 18)
        self.assertEqual([sum(len(noise)==k for _, noise in SUBSETS) for k in range(6)],
                         list(EXPECTED_COUNTS))
        self.assertEqual(column_mapping((2, 4)), (0, 1, 2, 3, 4))
        self.assertEqual(len({s for s, _ in SUBSETS}), 18)

    def test_noise_is_raw_and_mask_has_separate_rng(self):
        s=spec()
        raw=generate_raw_noise(s, 9, 6, fixture=True)
        masks=generate_noise_masks(s, 9, 6, fixture=True)
        self.assertTrue(all(a.dtype==np.float64 and np.isfinite(a).all() for a in raw))
        self.assertTrue(all(m.dtype==np.bool_ for m in masks))
        saved=[r.copy() for r in raw]
        ds=derive_all(fixture(), *raw, *masks, old_gamma=(.8,1.2,.5,1.,.6), spec=s)
        self.assertTrue(all(np.array_equal(a,b) for a,b in zip(raw,saved)))
        self.assertEqual(len(ds),18)
        self.assertEqual([a.shape for a in raw],[(9,3),(6,3)])
        self.assertEqual([a.shape for a in masks],[(9,3),(6,3)])
        self.assertTrue(all(np.array_equal(a,b) for a,b in zip(raw,generate_raw_noise(s,9,6,fixture=True))))
        self.assertTrue(all(np.array_equal(a,b) for a,b in zip(masks,generate_noise_masks(s,9,6,fixture=True))))

    def test_derive_invariance(self):
        m=fixture();s=spec()
        raw=generate_raw_noise(s,9,6,fixture=True)
        mask=generate_noise_masks(s,9,6,fixture=True)
        ds=derive_all(m,*raw,*mask,old_gamma=(.8,1.2,.5,1.,.6),spec=s)
        for d in ds:
            self.assertEqual(tuple(d.reference_ids),tuple(range(9)))
            self.assertTrue(np.array_equal(d.train_y,m.train_y))
            self.assertTrue(np.array_equal(d.validation_y,m.validation_y))
            self.assertEqual(d.train_x.shape,(9,3+len(d.noise_indices)))
            self.assertEqual(d.validation_x.shape,(6,3+len(d.noise_indices)))
            for j,idx in enumerate(d.feature_mapping):
                if idx<5:
                    self.assertTrue(np.array_equal(d.train_x[:,j],m.train_x[:,idx],equal_nan=True))
                    self.assertTrue(np.array_equal(d.validation_x[:,j],m.validation_x[:,idx],equal_nan=True))
                else:
                    self.assertTrue(np.array_equal(d.train_mask[:,j],mask[0][:,idx-5]))
                    self.assertTrue(np.array_equal(d.validation_mask[:,j],mask[1][:,idx-5]))
            self.assertEqual(d.hashes['test_accessed'],False)
            self.assertEqual(fingerprint(d.train_x),d.hashes['train_x_sha256'])
        d=ds[5]
        self.assertEqual(d.subset_id,'N2-01')
        self.assertEqual(d.feature_mapping,(0,1,2,3,4))
        self.assertTrue(np.array_equal(d.train_x,m.train_x,equal_nan=True))
        self.assertTrue(np.array_equal(d.validation_x,m.validation_x,equal_nan=True))
        self.assertEqual(d.gamma.tolist(),[.8,1.2,.5,1.,.6])
        self.assertEqual(len(derived_manifest(ds)['datasets']),18)

    def test_reject_missing_approval_and_bad_arrays(self):
        s=spec()
        with self.assertRaises(PermissionError):
            generate_raw_noise(s,300,90)
        with self.assertRaises(PermissionError):
            generate_noise_masks(s,300,90)
        with self.assertRaises(PermissionError):
            generate_raw_noise(s,300,90,fixture=True)
        with self.assertRaises(ValueError):
            NoiseSpec(1,1,(1.,)*3,(.6,)*3,(0.,)*3).validate()
        m=fixture()
        raw=generate_raw_noise(s,9,6,fixture=True)
        mask=generate_noise_masks(s,9,6,fixture=True)
        with self.assertRaises(ValueError):
            derive_all(m,raw[0].astype('float32'),raw[1],*mask,old_gamma=(1.,)*5,spec=s)
        with self.assertRaises(ValueError):
            derive_all(m,*raw,mask[0].astype('uint8'),mask[1],old_gamma=(1.,)*5,spec=s)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fixture.npz'
            with self.assertRaises(PermissionError):
                save_npz_once(path,{'data':raw[0]})
            digest=save_npz_once(path,{'data':raw[0]},fixture=True)
            self.assertEqual(len(digest),64)
            with self.assertRaises(FileExistsError):
                save_npz_once(path,{'data':raw[0]},fixture=True)

    def test_frozen_mother_read_only_if_repository_fixture_present(self):
        if not (Path(__file__).resolve().parents[1]/'objective-start-diagnostics').exists():
            self.skipTest('Full repository checkout required for frozen mother')
        mother=load_mother()
        self.assertEqual(mother.train_x.shape,(300,5))
        self.assertEqual(mother.validation_x.shape,(90,5))
        self.assertFalse(mother.train_x.flags.writeable)
        self.assertEqual(len(mother.source_sha),64)


if __name__=='__main__': unittest.main()
