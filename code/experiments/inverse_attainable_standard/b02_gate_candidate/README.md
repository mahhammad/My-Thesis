# B02 gate candidate — RESEARCHER-05 (HARDEN V2)

**CH4-R05-B02-VERIFIER-HARDEN-V2-01** (based on v1 implementation `CH4-R05-B02-VERIFIER-IMPLEMENT-01`). 21 controls G00–G20, synthetic-byte fixtures and hardened negative preflight tests. **Never authorizes `SOLVE` or `RESOURCE_PROBE`**. `B02–B06` remain blocked. This candidate is **not integrated with any model or runner** and carries no scientific witness/solver certification.

Source: `experiment/ch4-inv01-b02-preflight@05acde39b76b49e0b55337301251594be0ad6d92`; Plan: `experiment/ch4-inv01-standard-plan@968675cde8783adf69db13efd7de0fb3aab703c5`.

`pure_gate.py` imports **only Python standard library**, no NumPy, PyTorch, `generate`, `run`, `stream_graph`, or optimizer. `inspect_b02_bundle(...)` is a non-admitting pure semantic inspector. `verify_b02_inputs(..., requested_scope)` always raises `DENY_NO_AUTHORIZATION` for correctly verified artifacts and rejects invalid inputs earlier. It never returns permission to construct a graph. The `audit_candidate.py` CLI writes an append-only `STARTED` record **before untrusted reads and parsing**, plus a separate terminal failure record, to a **private, owned, external audit root** (`O_EXCL`, `O_NOFOLLOW`, file/dir `fsync`). No writable/symlink audit root means deny with no claim of recording. `audit(bundle,trust,...)` remains a one-record inner-check helper for already-loaded synthetic tests. No legacy artifacts are written, and `tamper_deep_verification.json` remains unchanged.

## Invariants and limitations

- A trust anchor is external, protected, SHA-256-based and independent of `manifest.json` and the checkout. `trusted_anchor_candidate.json` is **only a suggested value list** for independent review, **NOT** a trusted root or an authority to run probes/solvers. `manifest_sha256` is a byte-level **candidate**; Git blob SHA is not the same format. Run-time `TrustedSpec` must be instantiated from a separately approved root, not directly from this candidate file in the checkout.
- The `read_pinned_bundle` helper accepts fixed SHA **labels** and verifies byte hashes; it cannot by itself attest Git checkout identity. Production CI must do a protected SHA-pinned Git-object checkout/attestation before calling this module. Source paths must be inside the checkout, free of symlinks, and match verified digests. `importlib.util.find_spec` observes effective `sys.meta_path` (not just `PathFinder`), checks the pinned `SourceFileLoader` origin and its `get_data` source bytes, and rejects already-loaded `run`, `stream_graph`, `generate` in `sys.modules`. **This is not an attestation of actual imported bytecode**: a custom finder can have side effects, Python bytecode caches may exist, and the import state may change after inspection. Future integration must strictly bind/attest loader and code at real import/use, protect state and prohibit re-opens. Until then G15 is `INTEGRATION_REQUIRED`.
- All safety predicates use explicit `GateError` (no `assert` statements). Optimized Python (`-O`, `-OO`) returns `E_RUNTIME_OPTIMIZED` before any IO/model import. Schema must match all fixed type/shape/binary/semantic/source/plan/runtime checks. The returned `VerifiedB02Inputs` contains immutable nested tuples, immutable source **bytes captured at validation time**, pinned digests and the observed import-state fingerprint; it never exposes the mutable `Bundle` mappings. `check_preuse_stub` detects source-file and module/import-environment swaps then **always** rejects clean state with `INTEGRATION_REQUIRED`. It is deliberately not wired to a graph/runner and cannot guarantee the final TOCTOU boundary on its own. Future runner must consume verified immutable inputs only (not reload mutable files) and enforce pre-use at the first graph boundary.
- Runtime versions require external QA; historical snapshot states Python 3.11.18, NumPy 2.4.6, Torch 2.15.0+cpu. The gate reads installed package **metadata**, never imports those packages. Version drift is `E_RUNTIME_VERSION`, not silent acceptance.
- Future **independent** witness test (not run): isolated Train-only `teacher_seed=20261211`, logits hash and exactly 3 no-grad `center_train` forwards, compare 3 target rows byte-for-byte; **never deliver witness logits** to downstream solver. The teacher seed is public, not cryptographically secret.
- Future **independently authorized** resource probe (not run): largest class `c=1,m=147`, one graph construction and five bounded operations F0 baseline exact forward, Q1 one Q-gradient (no update), A1 one exact-JSD gradient (no update), J1 two reverse passes for full 2×30 Jacobian (no update), M1 exact monitor-forward; outer hard timeout and RSS cap, stage-specific append-only log. Neither historical resource envelope nor caps are approved for B02.

## Test protocol

Only synthetic **Train-shaped** bytes, never original Validation/Test or generator output. Each negative semantic fixture uses a **test-only re-signed trust root**, deliberately separate from the production candidate to test the intended semantic stage; with unchanged production trust root any tampering must fail at `E_MANIFEST_TRUST` or a pinned byte hash. The stub asserts `graph_constructed=optimizer_steps=0`, and no graph code is imported or created.

A future sandbox CI invocation may use:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout --kill-after=2s 120s \
  prlimit --as=1073741824 --rss=1073741824 \
  python -m unittest -v test_pure_gate
```

**Important:** this wrapper enforces a 120-second wall and 1-GiB address-space/RSS limit (Linux). Tests are light, but it does **not** constitute any graph benchmark. If these resource guards are unavailable or cannot be guaranteed, **do not run**.

A separate QA is required to approve the candidate trusted root, source code, explicit `python -O` test, rerun-safe auditing and integration boundary. **No PR/merge, graph, solver, Test, LaTeX or release permission is implied.**

## Harden V2 execution and remaining gates

- **Local synthetic-only tests:** 52/52 PASS on Python 3.13.5; final bounded run 3.87s wall, peak process RSS 101,260 KiB (measured with GNU time). Baseline v1 28 tests retained, with 24 new adversarial tests. The local interpreter is **not** historical Python 3.11.18, so the historical runtime has **NOT_CHECKED** executable parity.
- **Optimized interpreter negative:** 2 selected unit tests PASS under `python -O`, and direct `python -OO` probe yields `E_RUNTIME_OPTIMIZED G19`; no model import. No full suite is intended under `-O` because optimized mode must reject all positive inspections.
- **G15:** adversarial `MetaPathFinder`, `sys.modules`, custom loader with good origin and malicious `get_data` all rejected; actual loaded-code attestation remains `INTEGRATION_REQUIRED`.
- **G20:** swap of source file or import state between verification and stub use rejected; mutation of the mutable Bundle during validation does not poison verified pinned bytes. Stub always raises `INTEGRATION_REQUIRED` on clean pre-use. No first-graph integration or protection against future mutation after the stub yet.
- **Preflight audit:** missing/malformed/duplicate trust JSON, nested symlink trust, checkout symlink, missing source, invalid refs and injected loader errors produce append-only start + terminal attempt records. Parallel attempts use separate UUIDs, duplicate destination fails with O_EXCL. Unavailable private root refuses operation and makes **no audit-success claim**. No externally approved audit root has been provided.
- `trusted_anchor_candidate.json` is an **unapproved suggestion in the repository**; SHA-256 manifest candidate `5007534aaee94817e28168c5adcf1194e6ba546a84486497b7f8566e70657b60` is a digest, NOT an admission grant. Production trust and git-object checkout attestation remain outside this repo candidate.

Full G00–G20 actual-state matrix, negative fixture names and further blockers: `HARDEN_V2_QA_REPORT.md`. **B02–B06 remain BLOCKED; no CI, graph, solver, resource probe, Test read, PR or merge.**
