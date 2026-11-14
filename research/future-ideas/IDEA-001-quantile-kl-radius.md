# IDEA-001 — Quantile KL-radius for post-training outlier analysis

**Status:** Research hypothesis; not validated.  
**Date:** 2026-11-10  
**Scope:** Future work only; no changes to thesis experiments or chapter text.

## Motivation
After an initial graph/random-walk classifier is trained, each labeled sample has an estimated neighborhood-label probability vector \(P_i\). A small set of unusual samples may lie far from the corresponding class center \(V_c\). A coverage-versus-radius curve could help diagnose these samples and motivate a *separately validated* reweighting or data-quality review step. Large accuracy improvements are a hypothesis, not an observed result.

## Definitions
Let \(\mathcal I_c\) denote training samples of class \(c\), \(n_c=|\mathcal I_c|\), and \(P_i\in\Delta^{K-1}\). For a fixed class center (arithmetic mean),
\[
V_c=\frac{1}{n_c}\sum_{i\in\mathcal I_c}P_i.
\]
Define forward KL distances (direction matters):
\[
d_i=D_{\mathrm{KL}}(P_i\|V_c)=\sum_j P_{ij}\log(P_{ij}/V_{cj}).
\]
Use the usual \(0\log(0/a)=0\) convention; if \(P_{ij}>0\) and \(V_{cj}=0\), the divergence is infinite. With the arithmetic mean of the same samples, this last case cannot arise on the training subset.

For \(0<\alpha\le1\), sort \(d_i\) within class and define the **fixed-center quantile radius**
\[
R_{c,\alpha}=d_{(\lceil\alpha n_c\rceil)}.
\]
The coverage curve \(\alpha\mapsto R_{c,\alpha}\) is nondecreasing and stepwise for finite data. Flag candidates with \(d_i>R_{c,\alpha}\); ties may yield fewer than \((1-\alpha)n_c\) flagged samples. This is an outlier *screen*, not proof of corrupted labels.

An alternative, computationally harder **optimized quantile radius** is
\[
C_{c,\alpha}=\min_{r\in\Delta^{K-1}}\operatorname{Quantile}_{\alpha}
\{D_{\mathrm{KL}}(P_i\|r):i\in\mathcal I_c\}.
\]
Then \(C_{c,\alpha}\le R_{c,\alpha}\); the optimizing center may change with \(\alpha\), so covered subsets need not be nested. At \(\alpha=1\), \(C_{c,1}\) is the KL information radius \(\min_r\max_iD_{\mathrm{KL}}(P_i\|r)\), distinct from mean KL dispersion and from mutual information.

## Relation to existing thesis metrics
For fixed \(V_c\), the mean of \(d_i\) is a KL-based within-class dispersion measure. The full quantile curve adds tail information that the mean loses. Keep existing density/separation/margin metrics intact; do not rename or silently replace the thesis's established \(R\), \(I\), or JS-based metrics.

## Proposed experiment (not executed)
1. Freeze a baseline model, data split, seeds, and prediction protocol; retain untouched Test data.
2. Compute training-only class centers and per-sample KL distances. Plot per-class and pooled (clearly labeled) coverage curves, with 90%, 95%, and 99% thresholds. Record the denominator and tie handling.
3. Inspect whether high-distance samples correlate with misclassification, missing features, label noise, or valid minority patterns; check class imbalance.
4. Compare baseline versus training-only interventions: flag-for-review, downweight, or remove high-distance samples. Refit centers/model as appropriate; tune thresholds only using Train/Validation.
5. Report Train/Validation/Test balanced accuracy, per-class recall, calibration, class counts, retained fraction, seeds, uncertainty intervals, and computational cost. Never use Test to select a threshold or filter data.
6. Include controls: random removal at the same rate; distance based on an alternative center; different quantiles; repeated seeds. Beware of circularity if the same fitted model both identifies and evaluates outliers—consider out-of-fold scores.
7. If no improvement, retain the metric as a descriptive diagnostic rather than claiming a learning benefit.

## Open questions
- Does a sharp elbow reliably distinguish anomalous from difficult-but-valid samples?
- How sensitive are distances to smoothing, class size, feature missingness, and policy changes?
- Should class-specific quantiles replace a pooled threshold?
- Is robust refitting preferable to deletion?
- Does the optimized-center curve add enough value to justify its cost?

## Acceptance criteria for future investigation
Reproducible script and seed manifest; raw distance vectors and sample IDs; coverage plots; fixed/optimized-center definitions kept distinct; predeclared validation protocol; balanced accuracy and uncertainty on untouched Test. No claims of improved generalization without independent evidence.


## Clarification from the idea's originator — center-only outlier handling (2026-11-10)

**This is a clarification of the original research intent, not a correction of the earlier author's work.** The first formulation was broader because the distinction had not yet been communicated explicitly. For the proposed primary experiment, the intended intervention is **robust estimation of class centers**, **not deletion of outlier samples from the dataset or graph**.

**Crucial structural reason:** An atypical training sample can need its *own specialized feature-selection/random-walk policy*. If it is deleted from training or the sample-feature graph, we lose the opportunity to discover that policy, and may also remove meaningful graph nodes and transitions. A sample being far from the *current* class center does not show that the sample is inherently uninformative, mislabeled, or unlearnable.

Accordingly, preserve **all** training samples and their graph participation, including candidate outliers, during learning and evaluation. The intervention is restricted to **their contribution to constructing the class center**. For a fixed set of training neighborhood distributions \(P_i\), consider

\[
\widetilde V_c=
\frac{\sum_{i\in\mathcal I_c}w_iP_i}
{\sum_{i\in\mathcal I_c}w_i},
\qquad w_i\ge0,\quad
\sum_{i\in\mathcal I_c}w_i>0.
\]

Here \(w_i=1\) reproduces the ordinary center; setting \(w_i=0\) for sufficiently distant samples **removes them only from the center average**; soft weights \(0<w_i<1\) give an alternative. In every case those samples remain in the learning population and graph. The initial KL quantile-radius curve provides candidate weights or thresholds, rather than serving as a data-deletion rule.

**Primary comparison to test:** with the *same retained dataset and clearly specified policy/inference protocol*, compare ordinary centers against center-only trimming/reweighting at predeclared quantiles such as 90%, 95%, and 99%. Select thresholds using Train/Validation only; preserve an untouched Test set for final evaluation. Report class-balanced metrics, per-class recall, minority-pattern effects, retained weight mass per class, and sensitivity across independent seeds. Also investigate whether alternating policy updates with center updates changes conclusions; because policies and centers can interact, fixed-policy and joint-refitting experiments must be distinguished. Avoid circular assessment of distances using the same fitted representations (e.g. validate with out-of-fold scores where feasible). Numerical support, including possible accuracy gains, remains **an open hypothesis**.

**Interpretation of the preceding proposal:** references above to removing or downweighting samples should *for this central idea* be read as removing or downweighting **their contribution to class-center estimation**, not eliminating samples, policies, or graph connectivity. Whole-sample deletion, if ever studied, is only a separate ablation/control and is **not** the proposed main method. No experiments have been run or current thesis results altered by this clarification.
