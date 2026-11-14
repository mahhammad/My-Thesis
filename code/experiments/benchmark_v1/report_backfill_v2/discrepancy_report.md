# [RESEARCHER-03][BACKFILL] CH4-METRICS-BACKFILL-01 — discrepancy and coverage report

**Author:** Researcher No. 3 / پژوهشگر شماره ۳ · **date:** 2026-11-11  
**Immutable reference:** `chapter4@de02fdda91e23db177de4429b3e1a9693c046d83`  
**Delivery branch:** `experiment/ch4-metrics-backfill-v2` · **contract:** `chapter4-common-evaluation-contract-v2`  
**Scope:** 76 historical Benchmark v1 checkpoint records. Zero optimizer steps; zero walk inference replays; zero Test access. Original ZIPs and `chapter4/main/tex` unchanged. This report is **not independent QA**.

## A. Manifest / provenance results

- `coverage_manifest.json`: exactly **76 unique run_id** with 55 metric status/reason entries each, versions and source-member SHA. `coverage_manifest.csv`: exactly **76 data rows** and 129 columns; `coverage_summary.csv` enumerates status counts per metric. Status vocabulary: `measured/not_measured/not_applicable/undefined_or_infinite/not_verified`. Null/NA is never replaced by zero.
- Parsed `master_runs_76.csv` and `posthoc_score_backfill_76_compact.csv`, each containing 76 source rows. Verified family distribution **shared 25, class 25, node 13, node_class 13** and **50 numeric Validation BA / 26 undefined query-start**.
- Direct original ZIP retrieval and verification covered **36 / 76** records:
  - `extra_25_12runs.zip` (**12**), bytewise SHA-256 `061a00923ae69ac2f8b923947bd9cdea9d4f3dba1d1771063bcb3ecffc0201e3`.
  - `size_25_24runs_seeds20261103_20261104.zip` (**24**), SHA-256 `fdb862462e237e5094204ff17a5bc1a79a0f1d17f3a9b9d90ce967dba8b11fdd`.
  - For **all 36**, computed each raw JSON SHA-256 and confirmed equality to **both** its archive manifest entry and the master CSV member digest; matched family, objective, steps, data seed, `test_used=false`, valid 3×3 centers.
  - `core_50_16runs.zip` (**16**) and `size_25_24runs_seed20261102.zip` (**24**) are present in the GitHub tree / historical source registry, but the current GitHub binary retrieval returned **empty data**. The separate Library copy could not be materialized. **These 40 were NOT reverified from raw bytes**; their historical scalar metrics remain version-labelled CSV evidence, not regenerated center metrics. Do not infer missing centers from the summary.
- `verified_sources_36.json`: exact 36 verified center matrices, original checkpoint summaries, member digests and provenance. `center_only_metrics.csv`: **36 numerical center-only rows** with `T,BT,TNL,BTNL,I`, all pairwise JS/sqrtJS and similarity matrices, per-class nearest sqrtJS, min/mean pair distances. **40 rows are deliberately absent**, not zero.

## B. Numeric checks and discrepancies

**No positive discrepancy detected among the 36 accessible center records.** Recomputed center `I`, `min_pairwise_sqrt_JS`, and `mean_pairwise_JS` agree with the original final Train summary to observed floating-point exact agreement in the JavaScript computation (`max |Δ|=0` in these three comparisons). Independently comparing original raw values to master CSV yielded maximal discrepancies `|ΔI|=9.03e-17`, `|ΔR_JS|=9.72e-17`, `|ΔBA|=1.12e-16`, fully attributable to roundoff. **Not checked:** T/BT/TNL/BTNL for the unavailable 40, all unscanned per-sample arrays, and full 76-run v2 congruence. The `center_only_rebuild.py` script calls the **canonical** Chapter-3 NumPy `center_metrics` from `code/policy_evaluator/metrics.py` after checking both ZIP and member SHA. It is **dry-run by default**, reads no Test, and performs no walk inference/training; it can regenerate all 76 center-only values when all four archives are locally accessible.

**KL support certificate on 36, strictly limited scope:** the minimum entry across **every class center in these 36 raw verified checkpoints** is **0.024847449072930662**, greater than zero and far larger than the legacy clipping floor `1e-300`. Thus no valid candidate representation `P` can have `P_j>0,V_{c,j}=0` against *these saved center matrices*; the particular `+∞` versus clipped-KL **support mismatch cannot arise for these 36 centers** (Train or defined Validation). This is a **center-only mathematical certificate**, not an empirical enumeration of all candidate `P`, nor a precise re-evaluation of the KL sums. The historical numeric `R_KL` is retained, tagged `legacy_clipped_numerically_support_compatible_not_replayed`; exact-support KL candidate scores and NLL are not newly measured. The **remaining 40** lack a verified center-support certificate and are `not_verified` under primary v2 KL.

**Tiny independently executed expression-equivalent fixture:** using NumPy 2.3.5 / Torch 2.10.0+cpu, `KL((1,0)||(0,1))` is `+∞` in exact contract while legacy floor produces **690.7755278982137**; all-invalid masked softmax produces `[.5,.5]`; explicit valid mask produces `[1,0]`; positive-mass invalid row would leak choice while zero-mass row contributes zero. Tie/near-tie fixtures verify that `argmax`, `isclose(atol=1e-12)` and `margin<=0` are **not equivalent**, as documented in `boundary_fixture_results.json`. The fixture completed in **0.00534 s**, peak whole-process RSS **253.31 MiB**; it **reimplemented the checked expressions**, it was *not* execution of the imported repo unit test. A real import-based regression suite is provided as `test_backfill_boundaries.py` but has **NOT** been run in CI in this branch. No engine patch was applied.

## C. Important metric coverage counts

| Metric group | measured | not_verified | not_measured | not_applicable |
|---|---:|---:|---:|---:|
| each center-only metric `T/BT/TNL/BTNL/I/pairwise...` | 36 | 40 | 0 | 0 |
| Train `R_JS`, Train `R_sqrt_JS` (legacy-v1 numbers) | 76 | 0 | 0 | 0 |
| Train **primary** `R_KL` (verified center support) | 36 | 40 | 0 | 0 |
| Train `R_KL_legacy_clipped` (reported) | 76 | 0 | 0 | 0 |
| Train BA | 76 | 0 | 0 | 0 |
| Train ordinary accuracy / confusion / margin quantiles / sigmoid | 36 | 40 | 0 | 0 |
| Train JS-NLL / posthoc Softplus(τ=.005) / historical KL-NLL | 76 | 0 | 0 | 0 |
| Train exact/near `tie_count` | 0 | 0 | 76 | 0 |
| Validation BA | 50 | 0 | 0 | 26 |
| Validation `R_JS/R_sqrt_JS/R_KL` | 30 | 20 | 0 | 26 |
| Validation JS-NLL / posthoc historical KL-NLL | 50 | 0 | 0 | 26 |
| Validation ordinary accuracy/margins | 30 | 20 | 0 | 26 |
| Validation exact/near `tie_count` | 0 | 0 | 50 | 26 |

`measured` encompasses a **number actually recorded in the historical CSV**; it does **not imply we independently retrieved/checksummed that original member**. Per-run `source_evidence` disambiguates the 36 SHA-verified raw records from 40 old CSV-only records. Historical scores and `NLL_KL_normalized_historical` are **not** the `η,ε`-smoothed stable KL score from Chapter 3. All Softplus values imported from `posthoc_score_backfill_76_compact.csv` use **posthoc τ=.005**, which must not be confused with a checkpoint trained with τ=.002 or .020.

**Other prohibitions:** 50-step and 25-step budgets are separately labelled, size-study subsamples of one data seed are correlated rather than independent repetitions; Oracle Bayes posterior geometry is distinct from walk-neighborhood geometry; node/node_class Validation has no defined unseen-query start and must remain `not_applicable`; historical Test cannot select models.

## D. BLOCKER / EDITORIAL / FUTURE / next gate

- **BLOCKER for claiming 76 fully regenerated v2 center reports:** 40 archival members were not actually retrieved in this session. Remedy: independent QA executes **only** `center_only_rebuild.py --execute` from the source-locked checkout with all four local ZIPs, verifies 76/76 source SHA + reference formula matches, and adds the derived 40 without overwriting historical sources. This is **center-only** and needs **no inference**. The current 36-row CSV must **not** be presented as a 76-row measured center set.
- **EDITORIAL:** maintain historical-v1 versus center-v2 versions, legacy clipped versus support-aware KL, `near_tie` versus exact tie, `hard-margin error` versus `1-BA`, posthoc τ and objective τ, and explicit null/not-applicable states in §4.12. No LaTeX modified.
- **FUTURE, not authorized by this report:** after independent QA decide whether to permit bounded inference-only replay to measure per-sample exact KL/near ties; the current mission used **0 of 2** allowable inference replays and **0 of 180 seconds** replay budget. No new training, Test usage, or algorithm ranking.
- **Mask proposal only:** on future authorized patch, make all-invalid masked policy rows explicitly zero mass and fail fast if reachable with positive state mass; keep current graph validity and normalization unchanged until compatibility tests pass.

**Recommended handoff:** `PARTIAL_WITH_REASONS`, with 76/76 manifest records and 36/76 SHA-verified center-only updates. Independent QA should first secure byte access to the remaining two historical ZIPs and execute checksum-guarded **center-only** backfill; no automatic broad inference replay is scientifically justified.

**RESEARCHER-03 — پژوهشگر شماره ۳.**
