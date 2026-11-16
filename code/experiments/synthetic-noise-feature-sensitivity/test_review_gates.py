"""Independent small-fixture negative gates for QA G1/G2, never research data."""
from pathlib import Path
from hashlib import sha256
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (HERE, NoiseSpec, generate_raw_noise, generate_noise_masks,
                  save_npz_once, spec_sha)


def spec():
    return NoiseSpec(418, 419, (1.,1.1,.8),(.9,.8,.7),(0.,.2,.35),
                     'independent_zero_mean_gaussian')


def fake_approval(s, stage):
    return {'issue':90, 'stage':stage, 'source_sha':'1'*40,
            'approved_spec_sha256':spec_sha(s),
            'owner_approval_url':
                'https://github.com/mahhammad/My-Thesis/issues/90#issuecomment-123'}


ENV={'E_CH4_14_DATA_AUTHORIZED':'true','GITHUB_ACTIONS':'true',
     'GITHUB_REF_NAME':'experiment/ch4-14-noise-sensitivity-01',
     'GITHUB_WORKFLOW':'E-CH4-14 authorized data generation','GITHUB_SHA':'1'*40}


class ReviewGateTests(unittest.TestCase):
    def test_stage_mismatch_and_false_approval(self):
        s=spec()
        with patch.dict(os.environ,ENV):
            with self.assertRaises(PermissionError):
                generate_raw_noise(s,300,90,approval=fake_approval(s,'noise_masks'))
            with self.assertRaises(PermissionError):
                generate_noise_masks(s,300,90,approval=fake_approval(s,'raw_noise'))
            with self.assertRaises(PermissionError):
                generate_raw_noise(s,300,90,approval={
                    **fake_approval(s,'raw_noise'),'approved_spec_sha256':'0'*64})
            with self.assertRaises(PermissionError):
                generate_noise_masks(s,300,90,approval={
                    **fake_approval(s,'noise_masks'),'source_sha':'2'*40})

    def test_size_lock_and_bool_rejection_before_real_rng(self):
        s=spec()
        with patch.dict(os.environ,ENV):
            for func,stage in ((generate_raw_noise,'raw_noise'),
                               (generate_noise_masks,'noise_masks')):
                for n,m in ((299,90),(300,89),(301,90),(300,91)):
                    with self.subTest(func=func.__name__,rows=(n,m)):
                        with self.assertRaisesRegex(PermissionError,'300/Validation=90'):
                            func(s,n,m,approval=fake_approval(s,stage))
                with self.assertRaises(ValueError):
                    func(s,True,90,approval=fake_approval(s,stage))

    def test_fixture_storage_is_temporary_small_stage_scoped(self):
        s=spec()
        raw=generate_raw_noise(s,9,6,fixture=True)
        arrays={'train_noise':raw[0], 'validation_noise':raw[1]}
        with tempfile.TemporaryDirectory(prefix='e14-fixture-') as tmp:
            path=Path(tmp)/'new_noise_seed20261102.npz'
            with self.assertRaises(ValueError):
                save_npz_once(path,arrays,stage='raw_noise',schema='wrong',fixture=True)
            with self.assertRaises(ValueError):
                save_npz_once(path,arrays,stage='noise_masks',
                              schema='e-ch4-14-noise-masks-v1',fixture=True)
            with self.assertRaises(PermissionError):
                save_npz_once(path,{'train_noise':np.zeros((300,3),dtype='float64'),
                                    'validation_noise':np.zeros((90,3),dtype='float64')},
                              stage='raw_noise',schema='e-ch4-14-raw-noise-v1',fixture=True)
            self.assertFalse(path.exists())
            with self.assertRaises(ValueError):
                bad={'train_noise':raw[0].copy(),'validation_noise':raw[1]}
                bad['train_noise'][0,0]=np.nan
                save_npz_once(path,bad,stage='raw_noise',
                              schema='e-ch4-14-raw-noise-v1',fixture=True)
            self.assertFalse(path.exists())
            self.assertEqual(len(save_npz_once(path,arrays,stage='raw_noise',
                         schema='e-ch4-14-raw-noise-v1',fixture=True)),64)
        real=HERE/'inputs/noise_raw/new_noise_seed20261102.npz'
        # The production experiment may now exist. A rejected fixture call
        # must preserve the exact preexisting bytes rather than require absence.
        before=sha256(real.read_bytes()).hexdigest() if real.exists() else None
        with self.assertRaises(PermissionError):
            save_npz_once(real,arrays,stage='raw_noise',
                          schema='e-ch4-14-raw-noise-v1',fixture=True)
        after=sha256(real.read_bytes()).hexdigest() if real.exists() else None
        self.assertEqual(after,before)

    def test_derived_filename_must_match_noise_dimension(self):
        arrays={'train_x':np.zeros((9,3),dtype='float64'),
                'train_y':np.array([0,1,2]*3,dtype='int64'),
                'validation_x':np.zeros((6,3),dtype='float64'),
                'validation_y':np.array([0,1,2]*2,dtype='int64')}
        with tempfile.TemporaryDirectory(prefix='e14-fixture-') as tmp:
            with self.assertRaisesRegex(ValueError,'filename/dimension'):
                save_npz_once(Path(tmp)/'N5-01.npz',arrays,stage='derived',
                              schema='e-ch4-14-derived-data-v1',fixture=True)
            with self.assertRaisesRegex(ValueError,'Unrecognized derived'):
                save_npz_once(Path(tmp)/'N9-99.npz',arrays,stage='derived',
                              schema='e-ch4-14-derived-data-v1',fixture=True)
            with self.assertRaisesRegex(ValueError,'Filename'):
                save_npz_once(Path(tmp)/'other.npz',
                              {'train_noise':np.zeros((9,3),dtype='float64'),
                               'validation_noise':np.zeros((6,3),dtype='float64')},
                              stage='raw_noise',schema='e-ch4-14-raw-noise-v1',
                              fixture=True)

    def test_production_persistence_rejects_wrong_stage_path(self):
        s=spec()
        arrays={'train_noise':np.zeros((300,3),dtype='float64'),
                'validation_noise':np.zeros((90,3),dtype='float64')}
        target=HERE/'inputs/noise_raw/new_noise_seed20261102.npz'
        original=sha256(target.read_bytes()).hexdigest() if target.exists() else None
        with patch.dict(os.environ,ENV):
            with self.assertRaises(PermissionError):
                save_npz_once(target,arrays,stage='raw_noise',
                              schema='e-ch4-14-raw-noise-v1',
                              approval=fake_approval(s,'noise_masks'),spec=s)
            with tempfile.TemporaryDirectory(prefix='e14-fixture-') as tmp:
                with self.assertRaises(PermissionError):
                    save_npz_once(Path(tmp)/'new_noise_seed20261102.npz',
                                  arrays,stage='raw_noise',
                                  schema='e-ch4-14-raw-noise-v1',
                                  approval=fake_approval(s,'raw_noise'),spec=s)
        current=sha256(target.read_bytes()).hexdigest() if target.exists() else None
        self.assertEqual(current,original)


if __name__=='__main__':
    unittest.main()
