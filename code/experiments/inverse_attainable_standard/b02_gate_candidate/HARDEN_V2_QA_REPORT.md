# [RESEARCHER-05][B02-HARDEN-V2] — G15/G20/preflight audit hardening evidence

**Mission:** `CH4-R05-B02-VERIFIER-HARDEN-V2-01`.  
**Frozen V1 parent:** `1dda91c073783fd16be7a607ccbfe407f523f075`.  
**Historical B02 snapshot:** `05acde39b76b49e0b55337301251594be0ad6d92`.  
**Branch:** `research/ch4-r05-b02-gate-hardening-v2`.  
**Scope:** only `code/experiments/inverse_attainable_standard/b02_gate_candidate/`; no graph, model, optimizer, forward, Jacobian, solver, resource probe, Test, CI, PR, merge, historical raw, or tex change.

## 1. Independent-audit findings addressed in candidate V2

| Issue | V1 gap | Candidate V2 repair | Remaining blocker |
|---|---|---|---|
| **G15 effective import** | `PathFinder.find_spec(sys.path)` misses `sys.meta_path` hooks and `sys.modules` | `importlib.util.find_spec` observes effective origin; rejects preloaded `run/stream_graph/generate`, verifies `SourceFileLoader` class, `loader.name/path/origin`, and SHA-256 of source bytes read via loader and OS | **INTEGRATION_REQUIRED / NOT_YET_ATTESTED_AT_USE**: no protected loader/import binding at actual first graph use, import-hook side effects, Python bytecode caches and changes after the check are not ruled out |
| **G20 immutable/pre-use** | `Bundle` frozen dataclass contains mutable mappings; prior test changed file only *after* inspect | `VerifiedB02Inputs` contains deeply immutable data/targets and source byte tuples copied **before** hash/return (same bytes checked), source path/digest pins and import environment observation. `check_preuse_stub` rechecks pinned source digests, effective import origin, `sys.path`, `sys.meta_path` and `sys.modules` | **INTEGRATION_REQUIRED**: stub deliberately raises this code even on clean state. No real runner/first-graph integration or atomic code consumption; post-preuse mutation cannot be excluded here |
| **Audit completeness** | trust/path/JSON/loader exceptions before `audit(...)` had no record | `audit_preflight_attempt` emits `<uuid>.start.json` **before** trust/bundle parsing, then `<uuid>.terminal.json` with failure stage, code, optional trusted digest, and zero graph/optimizer. Root must be private, owned, non-symlink, not in source/historical tree; `O_EXCL|O_NOFOLLOW` on dir FD, file+dir fsync; no overwrite | Only root accepted as externally protected can be used. If audit root unavailable, **DENY** without claiming any audit write. Interrupted processes may leave a `STARTED` file without terminal record; this is an incomplete auditable attempt, not an authorization |

## 2. Test execution — actual local results on the exact GitHub code blobs

**Git blob source matches of the executed files:** `pure_gate.py` = `de276385f680ad206b24dee32b442cadf514ed9d`; `audit_candidate.py` = `a1ff114674c7d1689aaab2a3457abeebb17740c7`; `test_pure_gate.py` = `fd227953ab8aebbcc6ae855f1180d134591cf84c`. These are **Git blob SHA-1 identifiers**, not SHA-256 raw-file fingerprints. Independently of any historical full300 files, the local tool comparison hashed the actual UTF-8 bytes.

```sh
# From b02_gate_candidate/ (on Linux):
timeout --kill-after=2s 90s prlimit --as=1073741824 --rss=1073741824 \
  python -m unittest -v test_pure_gate

# Explicit optimized-mode negatives (not the full positive suite):
timeout --kill-after=2s 30s prlimit --as=1073741824 --rss=1073741824 \
  python -O -m unittest -v \
    test_pure_gate.PureGateTests.test_python_optimized_rejects_before_io \
    test_pure_gate.PureGateTests.test_audit_python_O_starts_then_rejects

# `python -OO` direct no-IO inspect: E_RUNTIME_OPTIMIZED / G19.
```

| Test execution | Outcome | Bounded wall (GNU time) | Peak RSS | Exit |
|---|---|---:|---:|---:|
| **Final 52-test suite** | **52/52 PASS**, no errors/failures | **3.87 s** | **101,260 KiB** | 0 |
| `python -O` two explicit negative tests | **2/2 PASS** | **2.54 s** | **100,320 KiB** | 0 |
| `python -OO` direct negative | `E_RUNTIME_OPTIMIZED G19` | **0.82 s** | **92,028 KiB** | 0 |

Earlier exploratory, corrected and pre-final pure suites were also within budget (48/48 and 49/49 PASS). The **aggregate** bounded test-command wall was below 120 seconds, all observed peaks below 1 GiB; a `-OO -c` shell-quoting typo in one exploratory launcher failed *before running candidate code* and was corrected before final results. Linux `RLIMIT_AS` is enforced via `prlimit`; reported observed peak RSS comes from GNU time, not inference. **This local environment uses Python 3.13.5**, whereas historical production manifest pins 3.11.18, NumPy 2.4.6, Torch 2.15.0+cpu; historical-runtime execution **NOT_CHECKED**. No NumPy or Torch imported by gate.

### Explicit adversarial cases added beyond V1

- G15: PathFinder would pass but `MetaPathFinder` returns shadow origin; preloaded `sys.modules['run']`; forged loader with matching origin but wrong loader class; pinned loader origin with altered `get_data` source bytes.
- G20: mutable `Bundle` mapping after inspection; mapping replaced *during inspection* after source hash (immutable checked-byte pin remains correct); threaded source-file swap between verification and stub pre-use; newly inserted `sys.modules['run']`; drift in `sys.meta_path`.
- Audit: absent trust (including missing flag), malformed JSON, duplicate trust key, invalid schema, trust inside checkout, trust symlink (leaf and ancestor), nonexistent snapshot, checkout/snapshot symlink, wrong commit ref, injected loader exception, unavailable/symlink output root, duplicate `O_EXCL` record, simultaneous distinct UUID records, repeated/parallel preflight failures, `python -O` produces a START plus terminal optimized-reject record, and CLI return code 2 (no grant).
- V1 semantic-resigned-trust negative fixtures remain: binary magic/size, four raw array hashes, mask=2, mask/finite mismatch, noncanonical NaN bits, label counts, IDs, observed counts, target positivity and sum with absolute-only tolerance, bitwise JSON↔binary, source hash/plan/seed, explicit deny and stop.

No actual graph/optimizer constructor exists in these fixtures. All audit results store `graph_constructed=0`, `optimizer_steps=0`. **Test-only** re-signed trust anchors are isolated from the proposed production digest.

## 3. G00–G20 status matrix — independently reviewable, not self-certification

`PASS` below means **author local toy/static checks passed**. Additional `INTEGRATION_REQUIRED` or `NOT_CHECKED` limits that verdict to this candidate; it does **not** certify B02 runtime admission.

| ID | Validation / coverage | Evidence in `test_pure_gate.py` | Current verdict |
|---|---|---|---|
| G00 | pinned refs, trust SHA candidate, path safety | production-trust tamper; wrong ref / symlink / root cases | **PASS** for rejection; protected external trust/checkout **INTEGRATION_REQUIRED** |
| G01 | manifest schema, duplicate JSON, invalid constants | `test_schema`, trust JSON negatives | **PASS** |
| G02 | Train size, magic and layout | `test_header_length` | **PASS** |
| G03 | fixed little-endian / C metadata | `test_metadata_endianness` | **PASS** |
| G04 | Train snapshot plus X/y/mask/IDs SHA | `test_array_hash_layer`, resigned fixture | **PASS** |
| G05 | byte mask strictly 0/1 | `test_mask_nonbinary` | **PASS** |
| G06 | observed ↔ finite consistency | `test_mask_semantics` | **PASS** |
| G07 | NaN canonical bit pattern | `test_noncanonical_nan` | **PASS** |
| G08 | class label domain | `test_labels` | **PASS** |
| G09 | class histogram exactly 72/147/81 | `test_histogram` | **PASS** |
| G10 | ordered Train IDs 0..299 | `test_reference_ids` | **PASS** |
| G11 | observed per-feature counts | `test_observed_counts` | **PASS** |
| G12 | target binary size, file SHA and class SHA | `test_target_length`, checksum checks | **PASS** |
| G13 | target finite, nonnegative, sum≈1, rtol=0 | `test_target_negative_sum1`, `test_target_mass_strict` | **PASS** |
| G14 | target JSON↔BIN byte equality | `test_target_json_binary_bitwise` | **PASS** |
| G15 | source hashes, effective import origin, loader class/source data, unloaded modules | `test_source_hash_drift`, `test_import_resolution_shadow`, four new G15 adversarial cases | **PASS** local rejection / **INTEGRATION_REQUIRED** for imported-code attestation and side effects |
| G16 | seeds, exact walk metadata and runtime versions | `test_runtime_drift` | **PASS** synthetic; historical interpreter/runtime **NOT_CHECKED** |
| G17 | 18 planned contexts, init seeds, NOT_RUN and run-ID uniqueness | `test_plan_duplicate_run`, `test_plan_state`, `test_seed_drift` | **PASS** |
| G18 | zero grant, both scopes rejected | `test_valid_semantic_bundle_but_denied`, `test_unsupported_scope_deny`, CLI nonzero | **PASS** candidate; runtime integration not present |
| G19 | explicit optimized rejection, no security assert | `test_python_optimized_rejects_before_io`, `test_audit_python_O_starts_then_rejects`, direct `-OO` | **PASS** under tested Python 3.13.5; pinned historical 3.11.18 **NOT_CHECKED** |
| G20 | deep immutable data, exact checked source pin, preuse drift and concurrency | `test_toctou_immutable_copy` plus six G20 targeted cases | **PASS** stub / **INTEGRATION_REQUIRED** for first-graph at-use contract |
| Audit | append-only start/terminal, including missing trust and loader errors; root safety | 14+ audit cases, `-O`, concurrency and `O_EXCL` | **PASS** local; protected external output root **NOT_CHECKED** |

## 4. External dependencies / blockers

1. **Protected root-of-trust missing:** `trusted_anchor_candidate.json` is a repository-local *candidate*, not a management-approved external trust object. `manifest_sha256=5007534aaee94817e28168c5adcf1194e6ba546a84486497b7f8566e70657b60` independently matches historical manifest bytes but does **not** authorize B02.
2. **Effective imported-code attestation at use missing:** custom import finders can execute arbitrary hook code; Python may consume cached bytecode, and import graph/solver runtime is not linked to immutable verified bytes or pinned loader. This stage remains `NOT_YET_ATTESTED_AT_USE` and requires a separately authorized integration QA.
3. **Full TOCTOU at first graph not proven:** pre-use stub checks drift and always denies. Future runner must consume only the checked immutable data and code bytes and bind actual module execution without re-opening paths. No race-free admission has been demonstrated.
4. **Protected audit-output root approval missing:** local private scratch directory was safe for toy tests; real external root ownership, retention and crash recovery must be specified independently. An unavailable root rejects without a fictitious audit record; interruption after START can leave a visible incomplete attempt.
5. **No historical-runtime parity:** no tests under the recorded Python/NumPy/Torch versions, no byte replay of the real n300 snapshot in this mission, no witness-forward, no F0/Q1/A1/J1/M1 resource measurements and no runner integration.

**Verdict for the code candidate only:** `READY_FOR_INDEPENDENT_GATE_REQA`. This is **not** G15/G20 production acceptance and **never** grants `SOLVE` or `RESOURCE_PROBE`. **B02–B06=BLOCKED**. No PR, merge, LaTeX or historical-file mutation permitted.
