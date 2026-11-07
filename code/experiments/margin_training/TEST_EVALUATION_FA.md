# Retrospective held-out test evaluation (November 2026)

These are the **already-trained policies**, not policies retrained after opening the test set. The sigmoid margin was the sole training objective, with 50 Adam steps at lr=0.12. The generator seed is 20261102.

| Policy (300 reference samples) | Train BA | Validation BA | Test BA |
|---|---:|---:|---:|
| Uniform | 83.65% | 96.77% | 76.42% |
| Sigmoid tau=0.002 | 87.21% | 96.60% | 78.37% |
| Sigmoid tau=0.005 | 86.52% | 97.28% | 78.37% |
| Sigmoid tau=0.020 | 81.02% | 90.14% | 82.39% |
| Generative Bayes oracle (not our walk) | 84.32% | 97.45% | 83.88% |

The reported accuracy is **balanced** across three classes. Ordinary test accuracy is 76.67%, 79.17%, 79.17%, and 81.67% respectively. Test scores were not used for training or to select tau, and **must not be retroactively used as hyperparameter-selection evidence**. In particular, the best validation tau (0.005) differs from the best observed test tau (0.020). All of this is a **single synthetic split**, not evidence of statistically robust ranking.

For three 72-reference subset seeds (same training split), mean test balanced accuracy is 74.00% for uniform, compared with 76.21%, 76.72%, and 75.94% for respective tau 0.002, 0.005, 0.020. These are not independent replications.

Note a protocol difference: training data are evaluated with the walk's no-return-to-origin exclusion, whereas held-out validation/test queries are not nodes in the reference graph. Consequently, train/validation/test differences cannot be naively interpreted as a classical overfitting gap. True Bayes classifier accuracy is also unusually high on validation.

Files:
- `results/full300_policy_snapshots.json`: trained full-300 Z matrices, serialized to ten decimal places.
- `results/heldout_test_comparison.json`: precise scores.
- `evaluate_test_snapshot.py`: reproduce full-300 train/val/test metrics from the policies above.

The test set has now been examined; further model changes motivated by this result require a *new* untouched test split for an unbiased final evaluation.
