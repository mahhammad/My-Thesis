# Center-target matching under actual multi-step random walks (observation only)

This experiment tests a **diagnostic iteration**, not a deployable learned policy. It never updates the core `policy_evaluator` files.

At round `t`, let `V_target[t]` be a set of K class-target probability vectors. For every training-origin walk, the origin class is known. At **every visited state**, compute the finite-reference one-step label distribution Q (excluding the current node and the original training origin), and find a valid feature distribution `pi` minimizing `JSD(Q.T @ pi, V_target[t][origin_class])`. The local solver is a batched conditional-gradient (Frank–Wolfe) algorithm and is approximate; its reported duality gap should be checked.

These policies drive **multi-step** geometric random walks. Each trajectory is independently weighted by `lambda`, normalized to unit mass, and averaged, yielding training representations P. Realized centers are their *arithmetic means by training label*: `V_real[t][c] = mean(P[i] for y[i]==c)`. Target centers are **not** replaced by the realized centers; instead the next target is `V_target[t+1] = (1-eta)*V_target[t] + eta*V_real[t]`.

Initial target centers are near one-hot: `(1-epsilon)*I + epsilon/K`. They are *targets*, not necessarily attainable centers.

Validation is evaluated **without using validation labels in policies**: for each candidate class c, simulate a separate trajectory optimized towards target c. Compare that representation with the **realized training center c**, and choose the smallest JSD. Validation labels are only used after predictions. Training metrics that use the *known origin class* are reported as **oracle-conditioned diagnostics**, not comparable to a deployable classifier's accuracy. The uniform-policy validation baseline uses a single class-independent walk and is therefore not an identical inference protocol.

## Reproduce from the repository root

Requires `numpy` (no SciPy):

```bash
python code/data_generator/generate.py --config code/data_generator/config.json --out generated_data
python code/experiments/center_matching_observation/observe.py \
  --train generated_data/train.npz \
  --validation generated_data/validation.npz \
  --policy code/policy_evaluator/policy_uniform.json \
  --rounds 5 --eta 0.5 --epsilon 0.08 --walks 100 \
  --optimizer-steps 45 --seed 20261104 \
  --out code/experiments/center_matching_observation/results/center_matching_round1.json
python -m unittest discover -s code/experiments/center_matching_observation -p 'test_*.py' -v
```

Baseline policies provide beta=0.4, lambda=0.8 and capped length 12. Per-origin random seeds are held fixed across rounds to improve comparability, not to remove Monte Carlo error. The data generator's seed and dataset specifications are preserved in `code/data_generator/config.json`.

## Interpretation / methodological cautions

- This experiment **does not** optimize global class separation or held-out accuracy. It only matches a fixed target locally and moves targets toward observed centers; monotonic improvement in the classification metrics is not guaranteed.
- Q's label distribution is exact on the **finite training reference set**, conditional on a state, feature and forbidden origin. The *full multi-step neighborhood* is estimated from Monte Carlo trajectories.
- Class-conditional simulation can fit assumed targets for multiple hypotheses; an apparently large training separation can be artificial and does **not** imply generalization.
- Report target mismatch (`mean_target_realized_jsd`), movement (`target_update_l1`), realized center information (`center_information`), minimum inter-center sqrt-JSD distance (`min_center_js_distance`), training within-class JSD, balanced validation accuracy under class hypotheses and projection dual gaps. Non-monotone results are expected.
- A few rounds on a single seed cannot establish convergence. Further study needs more seeds, walks, better local solver tolerance and stopping criteria.
