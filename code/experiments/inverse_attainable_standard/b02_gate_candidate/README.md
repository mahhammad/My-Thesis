# B02 gate candidate — RESEARCHER-05

**CH4-R05-B02-VERIFIER-IMPLEMENT-01.** 21 static controls, G00–G20, isolated lightweight synthetic-byte fixtures. **Never authorizes `SOLVE` or `RESOURCE_PROBE`**. `B02–B06` remain blocked. This candidate is **not integrated with any model or runner** and carries no scientific witness/solver certification.

Source: `experiment/ch4-inv01-b02-preflight@05acde39b76b49e0b55337301251594be0ad6d92`; Plan: `experiment/ch4-inv01-standard-plan@968675cde8783adf69db13efd7de0fb3aab703c5`.

`pure_gate.py` imports **only Python standard library**, no NumPy, PyTorch, `generate`, `run`, `stream_graph`, or optimizer. `inspect_b02_bundle(...)` is a non-admitting pure semantic inspector. `verify_b02_inputs(..., requested_scope)` always raises `DENY_NO_AUTHORIZATION` for correctly verified artifacts and rejects invalid inputs earlier. It never returns permission to construct a graph. The `audit_candidate.py` entrypoint writes only append-only, per-attempt records to a **new, explicitly separate** output path. No legacy artifacts are written, and `tamper_deep_verification.json` remains unchanged.

## Invariants and limitations

- A trust anchor is external, protected, SHA-256-based and independent of `manifest.json` and the checkout. `trusted_anchor_candidate.json` is **only a suggested value list** for independent review, **NOT** a trusted root or an authority to run probes/solvers. `manifest_sha256` is a byte-level **candidate**; Git blob SHA is not the same format. Run-time `TrustedSpec` must be instantiated from a separately approved root, not directly from this candidate file in the checkout.
- The `read_pinned_bundle` helper accepts fixed SHA **labels** and verifies byte hashes; it cannot by itself attest Git checkout identity. Production CI must do a protected SHA-pinned Git-object checkout/attestation before calling this module. Source paths must be inside the checkout, free of symlinks, and match verified digests. `PathFinder.find_spec` only resolves import locations **without importing** generator/model code. Import order and `sys.modules` must remain stable after verification; a future audited runner must prevent bypass/TOCTOU.
- All safety predicates use explicit `GateError` (no `assert` statements). Optimized Python (`-O`, `-OO`) returns `E_RUNTIME_OPTIMIZED` before any IO/model import. Schema must match all fixed type/shape/binary/semantic/source/plan/runtime checks. The data returned by the internal inspector are immutable tuple snapshots. Caller must not re-open mutable files after checking.
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
