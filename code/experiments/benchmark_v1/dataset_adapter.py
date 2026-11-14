"""Deterministic Train/Validation loader for the IMC pipeline.

Matches the first two SeedSequence children of data_generator.generate().
Does NOT create or read the historical Test split or any oracle labels.
The private helper formerly embedded in backfill_saved_scores.py is retained
unchanged there for replay backward compatibility.
"""
import numpy as np


def dataset_train_val(spec):
    """Return dict train/validation -> (X, label), no Test materialization."""
    children = np.random.SeedSequence(spec["seed"]).spawn(3)
    out = {}
    for name, child in zip(("train", "validation"), children[:2]):
        rng = np.random.default_rng(child)
        n = spec["sizes"][name]
        labels = rng.choice(len(spec["pi"]), size=n, p=spec["pi"])
        x = np.empty((n, len(spec["mu"][0])))
        for c in range(len(spec["pi"])):
            rows = np.flatnonzero(labels == c)
            if len(rows):
                x[rows] = rng.multivariate_normal(
                    spec["mu"][c], spec["cov"][c], size=len(rows)
                )
        mask = rng.random(x.shape) >= spec["missing"][None, :]
        x[~mask] = np.nan
        out[name] = (x, labels)
    return out
