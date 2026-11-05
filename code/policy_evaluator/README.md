# Fixed-policy baseline evaluator

This package evaluates **fixed, untrained feature-selection policies** on the synthetic datasets from `code/data_generator`. Building class centers and evaluating with already-built centers are two separate commands. Only NumPy is required.

## Files

- `walks.py`: sample the sample-feature graph with a fixed policy; return a neighborhood **label probability distribution** for each query.
- `centers.py`: compute training neighborhoods, average by **training class**, save centers and their data/policy fingerprints.
- `evaluate.py`: load previously computed centers, compute new query neighborhoods, predict classes, save metrics; **never refit centers**.
- `metrics.py`: Jensen–Shannon similarity, JSD distance, KL and classwise diagnostics.
- `policy_uniform.json`, `policy_guided.json`: reproducible fixed policies, **not learned**.
- `test_evaluator.py`: checks normalization, one-step law, forbidden return to origin, missing-feature masks, held-out label independence and center immutability.

## Run

From the repository root, generate data once:

```bash
python code/data_generator/generate.py \
  --config code/data_generator/config.json --out generated_data
```

Build centers with **training data and one fixed policy**:

```bash
python code/policy_evaluator/centers.py \
  --train generated_data/train.npz \
  --policy code/policy_evaluator/policy_uniform.json \
  --out generated_data/centers_uniform.npz
```

Evaluate **without recalculating centers**:

```bash
python code/policy_evaluator/evaluate.py \
  --reference generated_data/train.npz \
  --query generated_data/test.npz \
  --centers generated_data/centers_uniform.npz \
  --policy code/policy_evaluator/policy_uniform.json \
  --out generated_data/test_uniform
```

To compare the fixed feature-informed policy, use `policy_guided.json` **in both commands** and a separate center/output filename. You can evaluate validation or unlabeled query files in the second command; unlabeled queries produce predictions without classification accuracy.

Run tests from the root:

```bash
python -m unittest discover -s code/policy_evaluator -p 'test_*.py' -v
```

## Fixed-policy semantics and important distinctions

- `logits` has shape `(D+1, D)`: rows `0..D-1` select the next feature given the preceding feature, and row `D` is the *start* row. Eligible features are those present at the current node with a valid destination. Logits are softmax-normalized **only over eligible features**.
- Vertical transition weights are `exp(-kernel_gamma[a] * (x_a - z_a)**2)`, matching the generator's single-feature kernel. The *current* reference node is always disallowed, and the original training sample is disallowed at every later step. For an unseen query, the origin is not a reference node.
- Each walk's length is geometric with parameter `beta` but capped at `max_steps`; the cap is a computational approximation. An individual walk of length `L` contributes the **single normalized** vector `R(W) = sum_{t=1}^L lambda^(t-1) e_(label at step t) / sum_{t=1}^L lambda^(t-1)`. Independent walks are averaged. We do not concatenate all observations and normalize once across different walk lengths.
- There is no parameter training, gradient update or post-hoc center update. Policy-specific centers are the arithmetic means of neighborhood label distributions within each **training** class. Results will depend on the policy and on the finite Monte Carlo sample.
- `centers.npz` stores both the centers and SHA-256 fingerprints of the fixed policy and training reference; evaluation rejects mismatches.
- Query labels only enter `metrics.report`; deleting or changing test labels does **not** change query neighborhoods, predicted labels or stored centers. The reference set still needs **training** labels to aggregate visited nodes into neighborhood-label distributions.
- If a query has no permissible transition, evaluation **raises an error** rather than fabricating a prior distribution. Current implementation precomputes a dense `D × N_train × N_train` weight tensor and targets small/medium research datasets.

**Metrics**: accuracy, balanced accuracy, confusion matrix, per-class recall, winning and true-class margins, within-class JSD/KL, between-center JSD and minimum sqrt-JSD separation, and information among class centers. For held-out metrics involving true labels, the labels are used **only after predictions**. Center geometry on held-out reports remains that of the already-stored training centers.

## First sanity check, not a thesis conclusion

For the seed and configuration checked in `data_generator/config.json` (300 train / 90 validation / 120 test, 3 classes, 5 features), fixed-policy test accuracy was:

| Fixed policy | Test accuracy | Balanced accuracy |
|---|---:|---:|
| Uniform | 78.33% | 78.64% |
| Feature-informed | 79.17% | 79.27% |

Only **one additional test example** is correct with the feature-informed policy. This policy was specified using knowledge of which synthetic features were informative, so it is an illustrative hand-designed baseline, not evidence of successful policy learning. The observed Bayes classifier accuracy on the very same 120 test labels was 85.83%; the mean maximum true posterior (conditional expected Bayes accuracy) was 85.42%. These finite-sample numbers are not claims of generalization or statistical significance.

No `generated_data`, center artifacts or test results are committed: they are produced locally on demand.
