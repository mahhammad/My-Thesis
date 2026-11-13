# R02 §4.11 — Gate A: frozen-V2 tiny smoke (2026-11-13)

**Mission:** `R02-SEC411-SELF-MANAGED-GATE-A-01` · **status:** `PASS_TOY_SMOKE_ONLY` · **Scientific:** `D10/D11=SCIENTIFIC_HOLD`.

Purpose: confirm exact Git blob identities for V2 and both old test helpers, then recheck two independent-auditor counterexamples and masked tiny walks. The code fails closed for n>6, changes no existing source, and never reads Train/Validation/Test. This is **not** an independently designed second Bellman DP, not a certified interval bound, not data identity, and not a full300 resource probe.

## Reproduction in checked-out branch

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python code/experiments/bayes_center_audit_pack/sec411_gate_a/sec411_gate_a_smoke.py \
  --out /tmp/sec411_gate_a_smoke.json
```

No train data or additional Python dependencies beyond NumPy are required for these toy fixtures. Requires `git` available for `git hash-object`. The smoke runner imports `bound_repair_candidate/test_numerical_v2.py::independent_enumeration`; it is a recheck of the **existing** independent-path test oracle, so its independence is only with respect to the candidate Bellman recurrence, not independent human development.

Locked source anchors:
- `exact_origin_excluded_bound.py` Git blob `7fe3ae82f10b3120054854f82ee3916c55605f30`.
- `test_numerical_v2.py` Git blob `af68374c8876a434950a674c4435c426bdb0f509`.
- `test_exact_origin_excluded_bound.py` Git blob `abd4668bd1e5986473b1f5c579f3f84bf08afe36`.

## Recorded result and limitations

The associated immutable JSON `sec411_gate_a_smoke_results.json` records 4 tiny fixtures: 2 n3 (H2), n6 masked H1, n6 masked H3. **42 fixed-policy origin checks**; maximum observed interval violation `2.220446049250313e-16`, well below the **toy-only** threshold `1e-10`. Historical return-kernel counterexample expected 0.36; cancellation counterexample center expected 0.3833333333338. Python 3.13.5 / NumPy 2.3.5, ~0.0404 s and peak RSS ~97.8 MiB on the recorded machine (not extrapolatable).

**Pending before any full-data run:** source/data identity SHA freeze (B02 != D10 target), origin and mask agreement with original engine, H1/checkpoint/BLAS supervisor, independent error-bound method, independent QA. The V2 function itself remains immutable.

For process decisions and future scopes, see [Issue #58](https://github.com/mahhammad/My-Thesis/issues/58). This README documents only reproducible tiny technical evidence; research management instructions belong in the Issue, not in this folder.
