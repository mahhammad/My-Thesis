"""Canonical metric suite across benchmark-v1 experiments.
True-class representations use the corresponding policy hypothesis.
"""
import torch
from engine import DT,jsd,balanced

def kl_forward(p,q):
    return torch.where(p>0,p*(torch.log(p.clamp_min(1e-300))-torch.log(q.clamp_min(1e-300))),
                       torch.zeros_like(p)).sum(-1)

def summarize(out,labels,tau=.005,center_is_class_mean=True):
    """Class-balanced R_sqrt_JS, R_JS and R_KL, also for held-out queries.

    Entropy identity R_KL = H(V)-E[H(Y)] holds only when V is the
    arithmetic mean of these same class-member representations.
    For Validation, V comes from Train: report the direct KL but not
    the invalid entropy identity.
    """
    y=torch.as_tensor(labels,dtype=torch.long);k=out.centers.shape[0]
    pred=out.scores.argmax(1);correct=(pred==y).to(DT)
    own=out.correct_representations;C=out.centers;ctrue=C[y]
    diff=jsd(own,ctrue);sroot=torch.sqrt(diff.clamp_min(0))
    kval=kl_forward(own,ctrue)
    confusion=torch.stack([torch.bincount(pred[y==c],minlength=k) for c in range(k)])
    pairs=torch.stack([jsd(C[c],C[r]) for c in range(k) for r in range(c+1,k)])
    H=lambda v:-(v*torch.log(v.clamp_min(1e-300))).sum(-1)
    margins=out.margin
    return {
        "counts":[int((y==c).sum()) for c in range(k)],
        "balanced_accuracy":float(balanced(correct,y,k).detach()),
        "ordinary_accuracy":float(correct.mean().detach()),
        "balanced_hard_margin_error":float(balanced((margins<=0).to(DT),y,k).detach()),
        "balanced_sigmoid_margin_loss":float(balanced(torch.sigmoid(-margins/tau),y,k).detach()),
        "margin_quantiles":{str(p):float(torch.quantile(margins.detach(),p).item())
                            for p in [0.,.1,.25,.5,.75,.9,1.]},
        "confusion_true_rows_pred_cols":confusion.detach().tolist(),
        "R_JS":float(balanced(diff,y,k).detach()),
        "R_sqrt_JS":float(balanced(sroot,y,k).detach()),
        "R_KL":float(balanced(kval,y,k).detach()),
        "R_KL_entropy_identity":(float((H(C)-torch.stack([H(own[y==c]).mean() for c in range(k)])).mean().detach())
                                 if center_is_class_mean else None),
        "R_KL_entropy_identity_applicable":bool(center_is_class_mean),
        "mean_true_class_similarity":float((1-balanced(diff,y,k)/torch.log(torch.tensor(2.,dtype=DT))).detach()),
        "I":float((H(C.mean(0))-H(C).mean()).detach()),
        "min_pairwise_sqrt_JS":float(pairs.min().sqrt().detach()),
        "mean_pairwise_JS":float(pairs.mean().detach()),
        "centers":C.detach().tolist(),
        "per_class":[{"class":c,"accuracy":float(correct[y==c].mean().detach()),
                      "R_JS":float(diff[y==c].mean().detach()),
                      "R_sqrt_JS":float(sroot[y==c].mean().detach()),
                      "R_KL":float(kval[y==c].mean().detach())} for c in range(k)],
        "predictions":pred.detach().tolist(),
        "margins":margins.detach().tolist()
    }
