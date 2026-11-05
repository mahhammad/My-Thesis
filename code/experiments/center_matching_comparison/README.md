# Chapter 4 — Center-matching comparison (observation, not policy training)

This directory preserves an **auditable comparison of three exact checkpoints** from the center-target matching experiment and the full diagnostic trajectory of **49 rounds (0–48)**. It does not change the working `policy_evaluator` code or implement any reward/gradient learning.

## Three cases

1. **Uniform baseline:** a single class-independent random-walk representation for each query, with centers fitted from the training representations of that same policy.
2. **First guided execution (`round=0`, first execution):** target centers are near one-hot / maximally separated; optimize a local feature distribution against each origin's target with exact finite-reference one-step Q and JSD, then run the **full multi-step walk**. Use the **realized** training centers to score validation queries; **do not update targets/centers before this evaluation**.
3. **Peak observed guided validation (`round=4`, fifth execution):** this checkpoint has the largest **measured** guided validation balanced accuracy among 25 measured rounds. Its policy targets were updated in *preceding* iterations; evaluation uses **realized round-4 centers**, before the round-4 target update.

For guided classification, a separate class-hypothesis walk is run for every candidate class; no validation labels are used to select features. The uniform baseline uses one class-independent walk per query. **The validation accuracies are not an identical-protocol comparison**, and the peak was selected on validation (selection bias). Training accuracy in guided mode uses the known origin label to construct the policy and is an **oracle-conditioned diagnostic**, not a deployable train score.

## Files

- [`REPORT_FA.md`](REPORT_FA.md): Persian comparison, side-by-side center matrices, all recorded key metrics, interpretation and limitations.
- [`results/history_00_48.json`](results/history_00_48.json): 49 rounds of targets, realized centers, before-update validation metrics when available, plus the exact saved uniform baseline. The first five rounds were obtained from the existing committed `code/experiments/center_matching_observation/results/center_matching_round1.json`. Rounds 5–48 are the saved verified continuation started at round 5 from `next_targets` of round 4. The verification rerun reproduced round-4 realized centers exactly.
- [`results/three_cases.json`](results/three_cases.json) and [`results/three_cases.csv`](results/three_cases.csv): extracted auditable cases and recomputed center-only metrics.
- [`results/trajectory_metrics.csv`](results/trajectory_metrics.csv): full 49-round geometry. **Missing validation values mean not measured**, not zero or interpolated.
- [`figures/information_vs_accuracy.svg`](figures/information_vs_accuracy.svg): only measured points of I against balanced validation accuracy.
- [`figures/center_matrices.svg`](figures/center_matrices.svg): heatmap of the three **realized** center matrices.
- `analyze.py`: regenerates the comparison and all artifacts from saved JSON (standard library only).
- `reproduce.py`: regenerates synthetic data if missing, copies the five-round original checkpoint, then uses the existing `observe.py --resume` to rerun rounds 5 onward with the original parameters. It writes to a **separate output directory**.
- `test_analyze.py`: numerical/structural regression tests.

## Re-analyze saved data

Run from repository root:

```bash
python code/experiments/center_matching_comparison/analyze.py
python -m unittest discover -s code/experiments/center_matching_comparison -p 'test_*.py' -v
```

No NumPy is needed for **analysis**. Replaying the original experiment requires NumPy:

```bash
python code/experiments/center_matching_comparison/reproduce.py --rounds 49
```

`reproduce.py` starts at **round 5** from the original committed checkpoint rather than reinitializing near one-hot centers. It will perform additional Monte Carlo and convex-optimization work and can take substantial time. It writes to `generated_data/center_matching_replay`, leaving original archives untouched. Do not use `--force` unless explicitly replacing that replay output.

## Data and key definitions

Source configuration: `code/data_generator/config.json`; **300 training / 90 validation / 120 test**, 3 classes, 5 features, data seed `20261102`. Policy `beta=0.4`, `lambda=0.8`, `max_steps=12`, `walks_per_sample=100`, local JSD projection solver `optimizer_steps=45`, seed `20261104`, target smoothing `eta=0.5`, initial `epsilon=0.08`. Query labels are not used when constructing validation walks; test is not involved in selecting the peak.

All class-center metrics (`I`, `T`, `TNL`, `BT`, `BTNL`, pairwise JSD, min sqrt-JSD) are recomputed from **realized** class-center matrices with **uniform weights over classes** and natural logarithms. `train_within_class_jsd` is preserved from the original sampled training representations, not recomputed from centers alone. Metrics that need unavailable per-sample representations, such as normalized NLL, are **not invented**.

**Interpretation:** self-consistency (`target` approaching `realized`) can improve while between-center information and eventually held-out classification collapse; one-seed observational behavior is not a convergence theorem or a general relation between I and accuracy.

## Expanded geometric, likelihood and per-sample metrics

The detailed analysis now tracks **all center-only metrics for 49 rounds** and **all sample-dependent metrics for the three selected checkpoints** (uniform, round 0, round 4). Exact-seed replayed per-sample distributions are stored in `results/representations_*.json`. Run:

```bash
python code/experiments/center_matching_comparison/detailed_metrics.py
python -m unittest discover -s code/experiments/center_matching_comparison -p 'test_*.py' -v
```

Additional generated outputs are:

- `results/center_geometry_49.{csv,json}`: class-center `I`, `T`, `TNL`, `BT`, `BTNL`, entropy, pairwise JSD, sqrt-JSD (mean/minimum) and normalized JSD similarity for *every* round. `json` includes all pairwise matrices.
- `results/detailed_three_cases.{csv,json}`: balanced training/validation KL (`R`), within-class JSD, within-class **mean** sqrt-JSD, mean JS similarity, center log-likelihood and cross-entropy, raw KL NLL, normalized KL NLL, JSD-similarity NLL, classwise diagnostic quantities, Chapter-3 training `J_margin`, and soft margins. The class-conditional validation loss uses a distinct candidate representation per class. `E sqrt(JSD)` must not be confused with `sqrt(E JSD)`.
- `results/representations_uniform.json`, `results/representations_guided_round0.json`, `results/representations_guided_round4.json`: saved training and validation representations for fully independent metric recomputation without repeating Monte Carlo walks.

The training identity `R_c = H(V_c) - mean_{i in c} H(P_i)` and the **fitted-center** expected log-likelihood identity `mean_i E_{P_i}[log V_c] = -H(V_c)` are regression-tested. Center likelihood, within-class KL concentration, and classification accuracy are **distinct** concepts. Only per-center metrics are available for all 49 saved rounds; sample metrics for other rounds are intentionally omitted rather than fabricated. The expanded experiment and the Chapter 4 LaTeX write-up do **not** modify Chapter 3's theoretical definitions; the additional JS criteria may be formalized there in a subsequent pass.
