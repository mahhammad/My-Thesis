"""E-CH4-14 read-only mother, staged noise assets, and 18 paired projections.

Functions do not write research artifacts. Production noise/mask generation and
persistence remain behind a separate, owner-approved data gate (Issue #90).
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FROZEN = ROOT / 'code/experiments/objective-start-diagnostics/inputs/phase2_train_validation'
CONFIG = ROOT / 'code/data_generator/config.json'
INFORMATIVE = (0, 1, 3)
OLD_NOISE = (2, 4)
NEW_NOISE = (5, 6, 7)
SEED = 20261102
SUBSETS = (
    ('N0-01', ()),
    ('N1-01', (2,)), ('N1-02', (4,)), ('N1-03', (5,)), ('N1-04', (6,)),
    ('N2-01', (2, 4)), ('N2-02', (2, 5)), ('N2-03', (4, 6)), ('N2-04', (5, 7)),
    ('N3-01', (2, 4, 5)), ('N3-02', (2, 6, 7)),
    ('N3-03', (4, 5, 7)), ('N3-04', (5, 6, 7)),
    ('N4-01', (2, 4, 5, 6)), ('N4-02', (2, 4, 6, 7)),
    ('N4-03', (2, 5, 6, 7)), ('N4-04', (4, 5, 6, 7)),
    ('N5-01', (2, 4, 5, 6, 7)),
)
EXPECTED_COUNTS = (1, 4, 4, 4, 4, 1)


def fingerprint(array):
    """Byte-for-byte compatible with benchmark_v1.snapshot_pipeline.array_fingerprint."""
    a = np.ascontiguousarray(np.asarray(array))
    header = json.dumps({'dtype': a.dtype.str, 'shape': list(a.shape)},
                        sort_keys=True).encode('utf8')
    return sha256(header + b'\0' + a.tobytes()).hexdigest()


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _exact(a, shape, dtype, name):
    if not isinstance(a, np.ndarray) or a.shape != shape or a.dtype != dtype:
        raise ValueError(f'{name}: expected shape {shape} dtype {dtype}')
    return a


def validate_mother(tx, ty, vx, vy, *, full=True):
    n, m = (300, 90) if full else (len(tx), len(vx))
    _exact(tx, (n, 5), np.dtype('float64'), 'train_x')
    _exact(vx, (m, 5), np.dtype('float64'), 'validation_x')
    _exact(ty, (n,), np.dtype('int64'), 'train_y')
    _exact(vy, (m,), np.dtype('int64'), 'validation_y')
    if np.isinf(tx).any() or np.isinf(vx).any():
        raise ValueError('Old features may only use NaN for missingness')
    if np.any((ty < 0) | (ty > 2)) or np.any((vy < 0) | (vy > 2)):
        raise ValueError('Expected labels 0..2')
    if not np.array_equal(np.unique(ty), np.arange(3)):
        raise ValueError('Missing Train class')
    return True


@dataclass(frozen=True)
class Mother:
    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    source_sha: str


def fixture_mother(tx, ty, vx, vy):
    """Only for small synthetic unit tests: never a production data loader."""
    if len(tx) > 12 or len(vx) > 8:
        raise ValueError('Fixture size cap')
    validate_mother(tx, ty, vx, vy, full=False)
    return Mother(*[np.array(a, copy=True) for a in (tx, ty, vx, vy)], 'FIXTURE_ONLY')


def load_mother():
    """Read only the single approved frozen Train/Validation NPZ, checking all SHA fields."""
    manifest_path = FROZEN / 'manifest.json'
    m = json.loads(manifest_path.read_text(encoding='utf8'))
    if (m.get('schema') != 'e-ch4-13-phase2-frozen-train-validation-v1'
            or m.get('test_accessed') is not False
            or m.get('train_validation_only') is not True
            or m.get('seed_set') != [20261102, 20261103, 20261104]):
        raise ValueError('Frozen manifest contract mismatch')
    if file_sha(CONFIG) != m['config_sha256']:
        raise ValueError('Frozen generator config hash mismatch')
    rec = m['records']['20261102']
    if rec['file'] != 'ds20261102_train_val.npz' or rec['test_accessed'] is not False:
        raise ValueError('Frozen mother record mismatch')
    path = FROZEN / rec['file']
    if file_sha(path) != rec['file_sha256']:
        raise ValueError('Frozen mother byte SHA mismatch')
    with np.load(path, allow_pickle=False) as f:
        if set(f.files) != {'train_x', 'train_y', 'validation_x', 'validation_y'}:
            raise ValueError('Unexpected frozen NPZ keys (possible Test leakage)')
        tx, ty, vx, vy = (f[k].copy() for k in ('train_x', 'train_y', 'validation_x', 'validation_y'))
    validate_mother(tx, ty, vx, vy)
    checks = {'train_x_sha256': fingerprint(tx), 'train_y_sha256': fingerprint(ty),
              'validation_x_sha256': fingerprint(vx), 'validation_y_sha256': fingerprint(vy),
              'mask_train_sha256': fingerprint(np.isfinite(tx)),
              'mask_validation_sha256': fingerprint(np.isfinite(vx))}
    if any(checks[k] != rec[k] for k in checks):
        raise ValueError('Frozen mother array SHA mismatch')
    if rec['n_train'] != 300 or rec['n_validation'] != 90 or rec['reference_ids'] != '0..299':
        raise ValueError('Frozen row ID mismatch')
    for a in (tx, ty, vx, vy):
        a.flags.writeable = False
    return Mother(tx, ty, vx, vy, rec['file_sha256'])


@dataclass(frozen=True)
class NoiseSpec:
    raw_seed: int
    mask_seed: int
    std: tuple[float, float, float]
    gamma: tuple[float, float, float]
    missing_rate: tuple[float, float, float]
    distribution: str = 'independent_zero_mean_gaussian'

    def validate(self):
        if (type(self.raw_seed) is not int or type(self.mask_seed) is not int
                or self.raw_seed < 0 or self.mask_seed < 0
                or self.raw_seed == self.mask_seed):
            raise ValueError('Raw and mask RNG seeds must be distinct nonnegative integers')
        if self.distribution != 'independent_zero_mean_gaussian':
            raise ValueError('Unsupported noise distribution: requires explicit research review')
        for name, values, low, high in [('std', self.std, 0, np.inf),
                                        ('gamma', self.gamma, 0, np.inf),
                                        ('missing_rate', self.missing_rate, -1e-12, 1)]:
            arr = np.asarray(values)
            if arr.shape != (3,) or not np.isfinite(arr).all() or not (arr > low).all() or not (arr <= high).all():
                raise ValueError('Invalid '+name)
        if any(v < 0 for v in self.missing_rate):
            raise ValueError('Negative missing_rate')
        return self


def _generation_gate(spec, *, fixture, train_rows, validation_rows, approval):
    spec.validate()
    if fixture:
        if train_rows > 12 or validation_rows > 8:
            raise PermissionError('Fixtures only; real Train/Validation generation prohibited')
    elif not (isinstance(approval, dict)
              and approval.get('issue') == 90
              and approval.get('stage') in ('raw_noise', 'noise_masks')
              and approval.get('approved_spec_sha256') == spec_sha(spec)
              and isinstance(approval.get('owner_approval_url'), str)
              and approval['owner_approval_url'].startswith('https://github.com/mahhammad/My-Thesis/issues/90#issuecomment-')):
        raise PermissionError('Owner-approved E14 data stage required (Issue #90)')


def spec_sha(spec):
    from dataclasses import asdict
    return sha256(json.dumps(asdict(spec), sort_keys=True,
                             separators=(',', ':')).encode()).hexdigest()


def generate_raw_noise(spec, train_rows, validation_rows, *, fixture=False, approval=None):
    """Pure memory generator; independent columns, no NaN, never applies masks."""
    _generation_gate(spec, fixture=fixture, train_rows=train_rows,
                     validation_rows=validation_rows, approval=approval)
    if min(train_rows, validation_rows) <= 0:
        raise ValueError('Nonpositive split size')
    rng = np.random.default_rng(spec.raw_seed)
    sd = np.asarray(spec.std, dtype=np.float64)
    tx = (rng.standard_normal((train_rows, 3)) * sd).astype(np.float64)
    vx = (rng.standard_normal((validation_rows, 3)) * sd).astype(np.float64)
    if not np.isfinite(tx).all() or not np.isfinite(vx).all():
        raise FloatingPointError('Nonfinite raw noise')
    return tx, vx


def generate_noise_masks(spec, train_rows, validation_rows, *, fixture=False, approval=None):
    """Independent RNG stream; true means observed; raw arrays remain untouched."""
    _generation_gate(spec, fixture=fixture, train_rows=train_rows,
                     validation_rows=validation_rows, approval=approval)
    if min(train_rows, validation_rows) <= 0:
        raise ValueError('Nonpositive split size')
    rng = np.random.default_rng(spec.mask_seed)
    observed = 1.0 - np.asarray(spec.missing_rate, dtype=np.float64)
    return (rng.random((train_rows, 3)) < observed,
            rng.random((validation_rows, 3)) < observed)


def column_mapping(noise_indices):
    noise_indices = tuple(noise_indices)
    if len(set(noise_indices)) != len(noise_indices) or any(i not in OLD_NOISE+NEW_NOISE for i in noise_indices):
        raise ValueError('Invalid or repeated noise feature')
    return tuple(sorted(INFORMATIVE + noise_indices))


@dataclass(frozen=True)
class Derived:
    subset_id: str
    noise_indices: tuple[int, ...]
    feature_mapping: tuple[int, ...]
    gamma: np.ndarray
    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    train_mask: np.ndarray
    validation_mask: np.ndarray
    reference_ids: np.ndarray
    hashes: dict


def derive_all(mother, raw_train, raw_validation, mask_train, mask_validation,
               *, old_gamma, spec):
    """Deterministic, no random draws or persistence. Preserves every old NaN bit."""
    spec.validate()
    n, m = len(mother.train_y), len(mother.validation_y)
    validate_mother(mother.train_x, mother.train_y, mother.validation_x,
                    mother.validation_y, full=(n == 300 and m == 90))
    _exact(raw_train, (n, 3), np.dtype('float64'), 'raw_train')
    _exact(raw_validation, (m, 3), np.dtype('float64'), 'raw_validation')
    _exact(mask_train, (n, 3), np.dtype('bool'), 'mask_train')
    _exact(mask_validation, (m, 3), np.dtype('bool'), 'mask_validation')
    if not np.isfinite(raw_train).all() or not np.isfinite(raw_validation).all():
        raise ValueError('Raw new noise must be unmasked and finite')
    old_g = np.asarray(old_gamma, dtype=np.float64)
    if old_g.shape != (5,) or not np.isfinite(old_g).all() or (old_g <= 0).any():
        raise ValueError('Old gamma shape/value mismatch')
    g = np.concatenate((old_g, np.asarray(spec.gamma, dtype=np.float64)))
    def combine(x, raw, mask):
        new = np.array(raw, copy=True)
        new[~mask] = np.nan
        return np.concatenate((x, new), axis=1)
    tx = combine(mother.train_x, raw_train, mask_train)
    vx = combine(mother.validation_x, raw_validation, mask_validation)
    results = []
    for subset_id, noise in SUBSETS:
        mapping = column_mapping(noise)
        a, b = tx[:, mapping].copy(), vx[:, mapping].copy()
        am, bm = np.isfinite(a), np.isfinite(b)
        if not np.array_equal(a[:, [mapping.index(i) for i in INFORMATIVE]],
                              mother.train_x[:, INFORMATIVE], equal_nan=True):
            raise AssertionError('Informative Train columns drifted')
        if not np.array_equal(b[:, [mapping.index(i) for i in INFORMATIVE]],
                              mother.validation_x[:, INFORMATIVE], equal_nan=True):
            raise AssertionError('Informative Validation columns drifted')
        ids = np.arange(n, dtype=np.int64)
        gamma = g[list(mapping)].copy()
        hashes = dict(source_mother_sha256=mother.source_sha, raw_seed=spec.raw_seed,
                      mask_seed=spec.mask_seed, spec_sha256=spec_sha(spec),
                      raw_train_sha256=fingerprint(raw_train), raw_validation_sha256=fingerprint(raw_validation),
                      new_train_mask_sha256=fingerprint(mask_train),
                      new_validation_mask_sha256=fingerprint(mask_validation),
                      train_x_sha256=fingerprint(a), validation_x_sha256=fingerprint(b),
                      train_y_sha256=fingerprint(mother.train_y),
                      validation_y_sha256=fingerprint(mother.validation_y),
                      train_mask_sha256=fingerprint(am), validation_mask_sha256=fingerprint(bm),
                      gamma_sha256=fingerprint(gamma), test_accessed=False)
        results.append(Derived(subset_id, noise, mapping, gamma, a, mother.train_y.copy(),
                               b, mother.validation_y.copy(), am, bm, ids, hashes))
    return tuple(results)


def derived_manifest(datasets):
    if len(datasets) != 18 or tuple(d.subset_id for d in datasets) != tuple(s[0] for s in SUBSETS):
        raise ValueError('Exactly 18 ordered datasets required')
    return {'schema': 'e-ch4-14-derived-v1', 'test_accessed': False,
            'independent_dataset_seeds': 1, 'n_train': len(datasets[0].train_y),
            'n_validation': len(datasets[0].validation_y),
            'datasets': [dict(subset_id=d.subset_id, k=len(d.noise_indices),
                              noise_indices=list(d.noise_indices), feature_mapping=list(d.feature_mapping),
                              d=len(d.feature_mapping), gamma=d.gamma.tolist(),
                              reference_ids=d.reference_ids.tolist(), hashes=d.hashes)
                         for d in datasets]}


def save_npz_once(path, arrays, *, fixture=False, approval=None):
    """Exclusive-create asset writer for a separately authorized future data stage."""
    if not fixture and not (isinstance(approval, dict) and approval.get('issue') == 90
                              and approval.get('owner_approval_url','').startswith(
                                  'https://github.com/mahhammad/My-Thesis/issues/90#issuecomment-')
                              and approval.get('stage') in ('raw_noise','noise_masks','derived')):
        raise PermissionError('Data persistence needs separate Issue #90 authorization')
    if not isinstance(arrays, dict) or not arrays or any('test' in k.lower() for k in arrays):
        raise ValueError('Unexpected data keys')
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    with path.open('xb') as f:
        np.savez_compressed(f, **arrays)
    return file_sha(path)


assert tuple(sum(len(x[1]) == k for x in SUBSETS) for k in range(6)) == EXPECTED_COUNTS
assert column_mapping((2, 4)) == (0, 1, 2, 3, 4)
