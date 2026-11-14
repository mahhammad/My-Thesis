"""JSON-configured correlated Gaussian dataset with exact Bayes and one-step oracles.

Only NumPy is required. This is a data generator, not a random-walk trainer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = logits - np.max(logits, axis=axis, keepdims=True)
    w = np.exp(z)
    return w / w.sum(axis=axis, keepdims=True)


def vector(value, d: int, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.ndim == 0:
        a = np.full(d, a.item())
    if a.shape != (d,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite scalar or length-{d} array")
    return a


def load_config(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    pi = np.asarray(cfg["class_priors"], float)
    mu = np.asarray(cfg["means"], float)
    sd = np.asarray(cfg["stds"], float)
    if mu.ndim != 2 or mu.shape[0] < 2 or mu.shape[1] < 1:
        raise ValueError("means must have shape (K>=2, D>=1)")
    k, d = mu.shape
    corr = np.asarray(cfg["correlation"], float)
    if (pi.shape != (k,) or sd.shape != mu.shape or corr.shape != (d, d)
        or not all(np.isfinite(a).all() for a in (pi,mu,sd,corr))):
        raise ValueError("invalid parameter shape or non-finite value")
    if np.any(pi <= 0) or not np.isclose(pi.sum(),1.0,atol=1e-10):
        raise ValueError("class_priors must be positive and sum to one")
    if np.any(sd <= 0):
        raise ValueError("stds must be positive")
    if not np.allclose(corr,corr.T,atol=1e-12) or not np.allclose(np.diag(corr),1,atol=1e-12):
        raise ValueError("correlation must be symmetric with unit diagonal")
    if np.linalg.eigvalsh(corr).min() <= 1e-10:
        raise ValueError("correlation must be strictly positive definite")
    miss = vector(cfg["missing_percent"],d,"missing_percent")
    gamma = vector(cfg["kernel_gamma"],d,"kernel_gamma")
    if np.any((miss < 0) | (miss > 100)) or np.any(gamma <= 0):
        raise ValueError("missing_percent must be in [0,100]; kernel_gamma must be positive")
    sizes = cfg["split_sizes"]
    if set(sizes) != {"train","validation","test"} or any(type(n) is not int or n < 1 for n in sizes.values()):
        raise ValueError("split_sizes must contain three strictly positive integers")
    seed = cfg["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    cov = sd[:,:,None] * corr[None,:,:] * sd[:,None,:]
    # Symmetric KL between one-dimensional class-conditional Gaussians.
    importance = np.zeros(d)
    for c in range(k):
        for r in range(c+1,k):
            dm2 = (mu[c]-mu[r])**2
            skl = 0.25*((sd[c]**2+dm2)/sd[r]**2
                       +(sd[r]**2+dm2)/sd[c]**2-2)
            importance += pi[c]*pi[r]*skl
    noise = [a for a in range(d) if np.allclose(mu[:,a],mu[0,a],rtol=0,atol=1e-12)
             and np.allclose(sd[:,a],sd[0,a],rtol=0,atol=1e-12)]
    return dict(config=cfg,pi=pi,mu=mu,sd=sd,corr=corr,cov=cov,
                missing=miss/100,gamma=gamma,seed=seed,sizes=sizes,
                importance=importance,noise=noise)


def bayes_posterior(spec: dict, x: np.ndarray) -> np.ndarray:
    """Exact P(Y=c | all observed features), including correlations and MCAR."""
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[1] != spec["mu"].shape[1] or np.isinf(x).any():
        raise ValueError("x shape invalid (use NaN for missing values)")
    n,k = len(x),len(spec["pi"])
    logp = np.broadcast_to(np.log(spec["pi"]),(n,k)).copy()
    observed = np.isfinite(x)
    for pattern in np.unique(observed,axis=0):
        rows = np.flatnonzero(np.all(observed==pattern,axis=1))
        idx = np.flatnonzero(pattern)
        if not len(idx):
            continue
        for c in range(k):
            cov = spec["cov"][c][np.ix_(idx,idx)]
            chol = np.linalg.cholesky(cov)
            residual = (x[np.ix_(rows,idx)] - spec["mu"][c,idx]).T
            whitened = np.linalg.solve(chol,residual)
            logp[rows,c] -= (0.5*np.sum(whitened**2,axis=0)
                             + np.log(np.diag(chol)).sum()
                             + 0.5*len(idx)*np.log(2*np.pi))
    return softmax(logp)


def population_one_step(spec: dict, x: np.ndarray) -> np.ndarray:
    """Exact population label Q(x,a,c) under single-feature Gaussian kernel.

    Cross-feature correlations do NOT affect this one-feature marginal integral.
    Missing query features produce NaN (not fake uniform probabilities).
    """
    n,d=x.shape
    denominator=1+2*spec["gamma"][None,:]*spec["sd"]**2
    safe=np.where(np.isfinite(x),x,0)
    logg=(-0.5*np.log(denominator)[None,:,:]
          -spec["gamma"][None,None,:]
          *(safe[:,None,:]-spec["mu"][None,:,:])**2/denominator[None,:,:])
    logw=np.log(spec["pi"])[None,:,None]+logg
    out=softmax(logw,axis=1).transpose(0,2,1).copy()
    out[~np.isfinite(x)]=np.nan
    return out


def finite_one_step(spec: dict, queries: np.ndarray, reference_x: np.ndarray,
                    reference_y: np.ndarray, train_queries: bool = False,
                    batch_size: int = 128) -> np.ndarray:
    """Exact finite-reference one-step LABEL distribution (no walk sampling).

    Reference labels are *training only*. Exclude self for training queries.
    Undefined feature transitions are NaN; multi-step origin exclusion belongs
    to the later walker, not this single-step oracle.
    """
    n,d = queries.shape
    r=len(reference_y)
    if reference_x.shape!=(r,d) or (train_queries and n!=r):
        raise ValueError("reference dimensions mismatch")
    yhot=np.eye(len(spec["pi"]))[reference_y]
    out=np.full((n,d,len(spec["pi"])),np.nan)
    for a in range(d):
        eligible=np.isfinite(reference_x[:,a])
        if not eligible.any():
            continue
        for first in range(0,n,batch_size):
            last=min(n,first+batch_size)
            rows=np.arange(first,last)
            present=np.isfinite(queries[rows,a])
            rows=rows[present]
            if not len(rows):
                continue
            logw=-spec["gamma"][a]*(queries[rows,a,None]-reference_x[None,:,a])**2
            logw[:,~eligible]=-np.inf
            if train_queries:
                logw[np.arange(len(rows)),rows]=-np.inf
            good=np.any(np.isfinite(logw),axis=1)
            if not good.any():
                continue
            stable=logw[good]-np.max(logw[good],axis=1,keepdims=True)
            w=np.exp(stable)
            w/=w.sum(axis=1,keepdims=True)
            out[rows[good],a,:]=w@yhot
    return out


def generate(spec: dict, outdir: str | Path, finite_q: bool = True) -> dict:
    outdir=Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    children=np.random.SeedSequence(spec["seed"]).spawn(3)
    splits={}
    for name,child in zip(("train","validation","test"),children):
        rng=np.random.default_rng(child)
        n=spec["sizes"][name]
        y=rng.choice(len(spec["pi"]),size=n,p=spec["pi"])
        x=np.empty((n,spec["mu"].shape[1]))
        for c in range(len(spec["pi"])):
            idx=np.flatnonzero(y==c)
            if len(idx):
                x[idx]=rng.multivariate_normal(spec["mu"][c],spec["cov"][c],size=len(idx))
        mask=rng.random(x.shape)>=spec["missing"][None,:]
        x[~mask]=np.nan
        splits[name]=(x,mask,y)
        np.savez_compressed(outdir/f"{name}.npz",x=x,mask=mask,y=y)
        np.savez_compressed(outdir/f"oracle_{name}.npz",
                            posterior=bayes_posterior(spec,x),
                            q_population=population_one_step(spec,x))
    ref_x,_,ref_y=splits["train"]
    if finite_q:
        for name,(x,_,_) in splits.items():
            q=finite_one_step(spec,x,ref_x,ref_y,train_queries=(name=="train"))
            np.savez_compressed(outdir/f"finite_q_{name}.npz",q_finite=q)
    meta=dict(spec["config"])
    meta["marginal_importance_symKL"]=spec["importance"].tolist()
    meta["marginal_importance_rank_zero_based"]=np.argsort(-spec["importance"],kind="stable").tolist()
    meta["marginal_noise_features_zero_based"]=spec["noise"]
    meta["feature_indices_zero_based"]=True
    meta["q_population"]="kernel-weighted one-step DESTINATION LABEL distribution, not Bayes posterior"
    meta["finite_q_reference"]="train only, excluding self for training queries"
    meta["missing_mechanism"]="MCAR, per feature, independent of labels and values"
    (outdir/"spec.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return splits


def main() -> None:
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config",type=Path,default=Path(__file__).with_name("config.json"))
    cli.add_argument("--out",type=Path,default=Path("generated_data"))
    cli.add_argument("--skip-finite-q",action="store_true")
    args=cli.parse_args()
    spec=load_config(args.config)
    data=generate(spec,args.out,finite_q=not args.skip_finite_q)
    print("Output:",args.out)
    print("Marginal feature scores:",np.round(spec["importance"],5).tolist())
    print("Importance rank (0-based):",np.argsort(-spec["importance"],kind="stable").tolist())
    print("Marginal noise features:",spec["noise"])
    for name,(x,mask,y) in data.items():
        posterior=bayes_posterior(spec,x)
        print(f"{name}: n={len(y)}, observed={mask.mean():.3f}, oracle_Bayes_accuracy={(posterior.argmax(axis=1)==y).mean():.3f}")


if __name__=="__main__":
    main()
