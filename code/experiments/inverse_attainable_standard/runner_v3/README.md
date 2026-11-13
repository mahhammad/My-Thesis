# R05 §4.10 — runner_v3 / historical attested execution and evidence contract

**Status as of 2026-11-13:** completed historical execution of B02–B06 (90/90 new runs); the raw results, source pins, owner-root provenance, audit logs, and selected-policy replays were committed to `researcher/05-chapter4@d9610fa3559b044fd88ac2719a4e8d9de01adbd5`. Numeric results remain **limited by three independent generator seeds** and are not a license to launch additional solver runs.

**Preserve provenance:** historical pure V2 remains frozen in `b02_gate_candidate/`; corrective G17 V3 remains non-admitting in `b02_gate_v3/`. The separate `runner_v3/` was used for the owner-authorized historical solves. This README describes completed evidence, **not** a general reusable safe-to-run certificate.

- `data_contract.py` authenticates an out-of-checkout owner GitHub Issue root, pinned plan/Train snapshot/targets and sources; all n300 run-row `reference_seed=null` denotes no subsampling of full Train. Nominal block seed remains provenance only.
- `run_one.py` selects one frozen (block,class,initialization,method), emits exclusive START and TERMINAL audit files outside worktree, and executes captured source bytes in memory under resource/time guards.
- The owner-root record is authenticated by GitHub author/association; it is **not** an independent third-party cryptographic signature. Source-loader/at-use/TOCTOU assurances from the author workflow are **not independent adversarial execution QA**.
- B01-v2's 18 accepted runs are separate historical observations; B02–B06's 90 outcomes are published with `PROVISIONAL_PENDING_FINAL_QA` immutable raw status, plus later review evidence. Never rewrite original raw, trace, witness or audit files to change historical status.
- One invalid B02 preflight attempt `0f545f2ca40c4171853fa09043f24219` failed closed before graph construction and is explicitly excluded from the 108-run denominator.

## Recorded timing fields: important historical semantics

The original `run_one.py` writes `exclusive_solver_wall_s = time.perf_counter() - p2.START` when producing a final record. This **includes more than an exclusive optimizer phase**: exact monitoring and runner record overhead can be inside that timer. `inclusive_solver_wall_s` instead comes from method-specific instrumented elapsed time for Q/Adam, and for Q+GN it explicitly adds a Q warm start plus GN and optional restart elapsed times. The two fields are **not guaranteed nested or directly comparable**. Indeed the frozen 90-run archive has `exclusive_solver_wall_s > inclusive_solver_wall_s` in 60/90 cases (30 Q, 30 Adam); this is explained by the different timing scopes, not evidence of negative runtime.

For historical raw, retain the original field names **unchanged** but label comparisons as `recorded_timing_mixed_scope_not_speedup`. Do not derive independent full-job wall from sums of per-method fields and do not claim GN speedup relative to Adam. The upstream `inclusive` field also overlaps warm Q cost across methods. Peak RSS is per recorded process; n300 maximum was 545.05859375 MiB vs historical limit 1536 MiB.

## Original trace state: important historical semantics

All 90 new `trace.json` files still have top-level `status="in_progress"` even though their nested run stage is `"complete"`, archived `summary.json` has `"COMPLETED"`, and the matching START/TERMINAL audit record ends with `"COMPLETED"`. The top-level trace flag is a **stale serialization phase**, not the final outcome. Do not overwrite it: verify run completion by matching summary, nested trace stages, archived pinned inputs, and the SHA-256-linked terminal audit.

The read-only post-handoff regression `qa_final/verify_trace_contract.py` checks **90/90** full trace best/final JSD and **90/90 START/TERMINAL SHA-256 pairs**. It ran successfully on Python 3.11 normal and `-O` modes in [GitHub Actions #43440602775](https://github.com/mahhammad/My-Thesis/actions/runs/43440602775). This is **AUTHOR_REMEDIATION_WITH_SELF_VALIDATION**, not fresh independent graph replay or third-party security attestation.

**No new solving is authorized by this README.** All future graph/optimizer runs require an explicit bounded owner scope, read-at-use source pin audit, immutable snapshots, no Test selection, new attempt IDs, strict resource caps and a separate versioned execution/QA record.
