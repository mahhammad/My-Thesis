# RESEARCHER-05 — B02 pure verifier implementation evidence (candidate)

**Mission:** `CH4-R05-B02-VERIFIER-IMPLEMENT-01`. No graph, optimizer, witness forward, Jacobian, resource probe, Validation/Test access, PR or merge. **B02–B06 remain BLOCKED.**

## Baseline and provenance

- Historical pinned snapshot: `05acde39b76b49e0b55337301251594be0ad6d92`; plan: `968675cde8783adf69db13efd7de0fb3aab703c5`; chapter4: `de02fdda91e23db177de4429b3e1a9693c046d83`.
- Candidate SHA-256 of byte-exact `b02_preflight/manifest.json`: **`5007534aaee94817e28168c5adcf1194e6ba546a84486497b7f8566e70657b60`**. Computed from the UTF-8 GitHub source (ASCII text); **independent auditor approval is pending**. Do not confuse it with Git blob `c94ebda1d3a7aed0585024d9125f9fc42bc5a021`.
- Source file SHA-256 and historical binary digests are documented as **non-authorizing candidates** in `trusted_anchor_candidate.json`. This file is kept separately from historical `manifest.json`, but it lives in an ordinary git branch; **it is not itself a protected external trust root**.
- Git source is limited to new files under `b02_gate_candidate/`. The old snapshot, raw targets, plan, `tamper_deep_verification.json`, historical `__pycache__`, source engine and LaTeX were not changed.

## G00–G20 mapping: code → pure tests

| Controls | Implemented check | Test evidence |
|---|---|---|
| G00–G04 | pinned refs/trusted manifest SHA, metadata/magic/size/layout, 4 raw hashes | `test_no_production_trust_resign`, `test_header_length`, `test_metadata_endianness`, `test_array_hash_layer` |
| G05–G07 | raw mask domain, mask↔finite, bitwise canonical qNaN | `test_mask_nonbinary`, `test_mask_semantics`, `test_noncanonical_nan` |
| G08–G11 | y domain, exact class histogram, ordered IDs, observed counts | `test_labels`, `test_histogram`, `test_reference_ids`, `test_observed_counts` |
| G12–G14 | target size/hash, strict nonnegative simplex with absolute tolerance, bitwise JSON↔binary | `test_target_length`, `test_target_negative_sum1`, `test_target_mass_strict`, `test_target_json_binary_bitwise` |
| G15–G17 | 5 source digests, module resolution *without import*, runtime/seeds, 18 exact planned contexts | `test_source_hash_drift`, `test_import_resolution_shadow`, `test_runtime_drift`, `test_seed_drift`, `test_plan_duplicate_run`, `test_plan_state` |
| G18–G20 | scope deny, no graph stub, `python -O` rejection, immutable verified copy/TOCTOU | `test_valid_semantic_bundle_but_denied`, `test_unsupported_scope_deny`, `test_python_optimized_rejects_before_io`, `test_toctou_immutable_copy` |
| Audit | append-only attempt ID + external trust digest + error code; no overwrite | `test_audit_append_only_and_failures_have_attempt_ids`; every negative fixture in `check_code` is independently recorded in test scratch |

**Additional code QA:** `test_no_assert_used_in_candidate` disallows plain `assert` statements in the verifier.

## Actual bounded execution (local, not CI)

Final test command used on Linux with pure synthetic Train-shaped fixtures:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout --kill-after=2s 120s \
  prlimit --as=1073741824 --rss=1073741824 \
  python -m unittest -v test_pure_gate
```

**Final result: 28/28 PASS, ~1.512 seconds unittest wall, exit status 0.** The command enforces a 120-second outer wall and 1-GiB address-space/RSS limit. Two prior corrected runs also achieved 28/28 PASS; the initial exploratory run had one faulty test-fixture expectation, not a verified production bug, which was repaired before final evidence. No CI was triggered for this task, and no B02 solver data were evaluated. Synthetic tests do **not** validate the production binary on the historical runtime.

## Remaining independent gates

1. Independent approval of candidate `trusted_manifest_sha256` as a **protected external root**; immutable checkout identity cannot be derived from caller-supplied ref strings alone.
2. Independent QA of source code, error codes, negative fixtures, import-resolution stability, `python -O`, path safety and rerun idempotence.
3. Separate authorization and integration in a future runner **before the first graph/optimizer operation**; candidate always denies `SOLVE` and `RESOURCE_PROBE` and has no grant mechanism.
4. Independent witness-only Train forwards and later bounded class-147 resource probes, if separately authorized. No inference about their correctness/cost is made here.

**Status: `READY_FOR_INDEPENDENT_GATE_QA` as a non-admitting candidate only.**
