# CH4-EXP-INV-01 — B01 first 18-case block scientific QA

**Decision: COMPUTATION PASS / SCIENTIFIC PROTOCOL REVISE before B02.**

**Repository source:** chapter4@de02fdda91e23db177de4429b3e1a9693c046d83
**Code commit:** c882208a4438ef4b343c0ac062cf3dd1142ca6dc
**Raw data commit:** f8739623b405905d8f5c83497903179ffee827c0
**CI SUCCESS:** https://github.com/mahhammad/My-Thesis/actions/runs/42977067592

## Inputs, coverage and reproducibility

Only B01 was executed: n=150, exactly 50 samples per class, one synthetic data generator (seed 20261102), reference-selection seed 20261201, witness-policy seed 20261211, and independent random-start seed 20261221+100*c for class c. Model: existing StreamGraph, float64 exact walk expectations, beta=0.4, lambda=0.8, H=12. Threshold: per-class center JSD <= 1e-8. No Monte Carlo or Test usage; hidden teacher policy never supplied to an inverse solver. All three class targets are known reachable (saved SHA256 hashes and self-JSD=0).

Three class targets x two start policies (neutral or independent random) x three solvers (Q-only, Exact-Adam, Q+Bootstrap-GN): **18 solver cases** but only **three distinct class-target problems** on **one data generator**. Thus these are not 18 independent statistical replications. GN does not use true class labels of unseen queries; this is an experimental class-conditional inverse problem, not a classification benchmark.

Stored source files: results/B01/manifest.json (written before solving), trace.json (all steps, including failures), curve.csv (JSD versus cost), summary.json, compact_results.csv, and ../execution_ledger.json (all 108 planning statuses). The frozen proposed plan_108.json remains unchanged; the separate ledger is authoritative for actual completed work.

## Success at the official threshold, and actual measured cost

| Solver | Threshold successes of 6 starts x classes | Mean best JSD | Maximum best JSD | Total labeled solver seconds |
|---|---:|---:|---:|---:|
| Q-only | 0/6 | 8.32658e-4 | 2.38271e-3 | 6.53 |
| Exact-Adam | 3/6 | 1.36664e-8 | 3.58522e-8 | 31.60 |
| Q+Bootstrap-GN | 6/6 | 5.68141e-9 | 9.72185e-9 | 11.14 including Q warm-up |

**Total actual numerical job wall clock: 44.024 seconds; peak overall RSS 441.14 MiB**, well below the guarded 840-second and 1430-MiB soft limits. Solver totals above must NOT be added to obtain job time: the Q component is included again in the labeled Q+GN runtime. Monitoring overhead also includes one exact-center forward after each Q-only step; this is an instrumentation cost not incurred by original historical Q-only.

| Class | Initial logits | Best Q JSD | Best Adam JSD (steps/reason) | Best Q+GN JSD (GN corrections) |
|---:|---|---:|---|---|
| 0 | neutral | 1.0169e-4 | 3.5078e-9 (78 / tolerance) | 6.5059e-9 (7) |
| 0 | random | 9.1701e-5 | 1.2133e-8 (77 / plateau) | 3.3751e-9 (7) |
| 1 | neutral | 2.9083e-5 | 3.5852e-8 (29 / plateau) | 4.7042e-9 (12) |
| 1 | random | 5.1131e-5 | 7.8207e-9 (47 / tolerance) | 4.2785e-9 (12) |
| 2 | neutral | 2.3396e-3 | 9.5951e-9 (133 / tolerance) | 5.5030e-9 (12) |
| 2 | random | 2.3827e-3 | 1.3089e-8 (150 / cap) | 9.7219e-9 (12) |

All GN cases passed the threshold on their first normal trajectory, so no natural restart or fallback was required in B01. Their correctness had been independently exercised in the previous preflight with finite differences, including forced fallback and neutral restart on a small graph. Random initialization of the Q-only and Adam update equations was checked on two steps and agreed to numerical precision (max saved logits discrepancy = 0).

## Important scientific confounder: stopping rule, NOT the exact engine

The B01 run applied an automatic plateau rule from P2: after at least 25 steps and a patience window of 18, stop if the best exact center JSD barely improves. This is **not a neutral rule for all three solvers**.

**Q-only:** all six Q runs stopped at step **25**, even though they optimize a distinct local-Q surrogate. The exact multi-step JSD is the evaluation criterion but not Q's internal objective; stopping Q on lack of JSD progress can suppress additional surrogate optimization and changes the quality of the policy handed to GN. In several cases best and final exact JSD differed substantially. Thus the observed 0/6 must NOT be interpreted as an ultimate Q-only failure or as a uniform cap100 experiment.

**Exact-Adam:** two failures stopped from plateau *before* exhausting cap150: class-1 neutral at step 29 (best 3.585e-8) and class-0 random at step 77 (best 1.213e-8). One failure reached the actual cap150: class-2 random (best 1.309e-8). All three best errors were close to the official 1e-8 threshold. The records show that **cap/stop choice matters**; they cannot establish that the optimizer lacks a solution or that another start would fail.

**Recommended scientific revision:** before running B02, remove plateau as an automatic termination condition (record it as a diagnosis only), and halt only on the exact 1e-8 success threshold, the *pre-registered* iteration cap, an actual numerical failure, or a hard resource bound. Alternatively, if the supervisory team wants a genuine plateau stop, define it for Q using its own surrogate and for Adam using its exact loss/gradient, then freeze and test those rules before further experiments. No posthoc retuning of threshold is permitted. Preserve B01 raw outputs unchanged, and rerun a separate versioned B01 only after approving revised stop rules. Do not speculate numerical outcomes from a hypothetical longer run.

**Evaluation decision:** computational trace integrity PASS and memory/cost PASS; evidential sufficiency LIMITED for the general algorithm comparison; vote REVISE before executing the 90 still-unrun cases. A successful witness center is not a uniqueness guarantee, and exact reconstruction is not evidence of better classification BA/generalization.

## Next priority

The user's next research priority is the full 108-case experiment, but it is executed in stage gates: B01 is saved (18/108), B02-B06 remain **NOT STARTED (90/108)** pending the revised stopping-rule contract and independent QA. Data from the 72-sample historical D09 remains in its archive and must not be inserted into Chapter 4. No LaTeX, chapter4, main, old experiment sources, or Test files were modified by this block.
