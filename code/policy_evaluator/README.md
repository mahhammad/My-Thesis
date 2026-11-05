# Fixed-policy baseline evaluator

This package evaluates **fixed, untrained feature-selection policies** on the synthetic datasets from `code/data_generator`. Building class centers and evaluating with already-built centers are two separate commands. Only NumPy is required.

## Files

- `walks.py`: sample the sample-feature graph with a fixed policy; return a neighborhood **label probability distribution** for each query.
- `centers.py`: compute training neighborhoods, average by **training class**, save centers and their data/policy fingerprints.
- `evaluate.py`: load previously computed centers, compute new query neighborhoods, predict classes, save metrics; **never refit centers**.
- `metrics.py`: reusable weighted Chapter-3 metrics and selectable KL / sqrt-JSD / JS-similarity decision rules.
- `one_step.py`: exact finite-reference one-step Q for every sample and feature, independent of Monte Carlo and policy logits.
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

**Metrics**: use `--metric kl`, `--metric sqrt_jsd`, or `--metric js_similarity` (the default) in evaluation. Use `--class-weights w0 w1 ...` to specify class importance; uniform class weights are the default. The library exports `class_centers(neighborhoods, labels, sample_weights=...)`, `center_metrics(centers, class_weights=...)` and `report(neighborhoods, centers, labels, metric=..., class_weights=..., sample_weights=...)`. It includes T, TNL, BT, BTNL, I, KL-based J_margin, unnormalized/normalized NLL and smooth JSD margin losses. KL is mathematically exact: unsupported positive mass produces infinity rather than arbitrary smoothing. Undefined diagnostics are returned as null in reports. The Cantelli interpretation of J_margin requires training representations and their own fitted centers.

**Exact Q**: `graph.one_step_q()` computes shape `(N_train,D,K)` on all training nodes with self-exclusion; `graph.one_step_q(query_x)` computes Q for unseen queries; `graph.one_step_q(query_x,train_indices=indices,origin_indices=origins)` also forbids returns to a specified start node. Each `Q[i,a,:]` is the expected next *label* distribution after selecting feature `a` at sample `i`. Missing features or no valid destinations are `NaN`; query labels are never used.

```bash
python code/policy_evaluator/one_step.py --reference generated_data/train.npz --query generated_data/train.npz --train-queries --policy code/policy_evaluator/policy_uniform.json --out generated_data/q_one_step_train.npz
python code/policy_evaluator/evaluate.py --reference generated_data/train.npz --query generated_data/validation.npz --centers generated_data/centers_uniform.npz --policy code/policy_evaluator/policy_uniform.json --metric kl --out generated_data/validation_uniform_kl
```

**Parallel walk semantics**: for a single query origin, the Monte Carlo engine evolves all sampled walks in vectorized NumPy batches. Geometric walk lengths are drawn using `beta`, capped at `max_steps`, and each walk is independently normalized using `lambda` before averaging. Kernel weights are computed in bounded query batches. The hard cap changes the untruncated theoretical length distribution: increase it and verify stability when approximating the infinite-horizon model. For held-out metrics involving true labels, the labels are used **only after predictions**. Center geometry on held-out reports remains that of the already-stored training centers.

## First sanity check, not a thesis conclusion

For the seed and configuration checked in `data_generator/config.json` (300 train / 90 validation / 120 test, 3 classes, 5 features), fixed-policy test accuracy was:

| Fixed policy | Test accuracy | Balanced accuracy |
|---|---:|---:|
| Uniform | 78.33% | 78.64% |
| Feature-informed | 79.17% | 79.27% |

Only **one additional test example** is correct with the feature-informed policy. This policy was specified using knowledge of which synthetic features were informative, so it is an illustrative hand-designed baseline, not evidence of successful policy learning. The observed Bayes classifier accuracy on the very same 120 test labels was 85.83%; the mean maximum true posterior (conditional expected Bayes accuracy) was 85.42%. These finite-sample numbers are not claims of generalization or statistical significance.

Large generated datasets and temporary center files are produced locally. Small reproducible experiment outputs are versioned under `code/experiments/`; they must not be globally ignored.
