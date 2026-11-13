# B02 G17 versioned corrective pure verifier (R05-SEC410-ENDTOEND-01)

**Historical source:** `research/ch4-r05-b02-gate-hardening-v2@6eb78e22945002848ec21d0ae7efdef35b524369`.
**Do not modify historical V2:** `b02_gate_candidate/`, historical canonical B02 snapshot and `standard_plan/plan_108.json` remain immutable.

## Root cause and semantic fix

The 300-reference B02 snapshot uses **all Train rows**, X shape (300,5), y and mask of length 300 and `ids=0..299`, with no reference subsampling. `b02_preflight/build_snapshot.py` lines 128–149 calls `generate(...)["train"]` and uses `np.arange(300)` for IDs without a choice step. The historical plan has `blocks[].reference_seed=20261201` and manifest `seed.reference=20261201` as provenance/nominal block metadata, **but `runs[].reference_seed=null` on all 54 n300 rows**: this denotes that **no reference sampling occurred**. The historic V2 `G17` incorrectly demanded `20261201` from each B02 run row, rejecting all 18 canonical B02 rows. Replacing null with 20261201 would falsify run-level semantics and change a frozen plan.

`pure_gate.py` here is an exact copy of the V2 pure verifier **except one G17 run-row expectation changed from `reference_seed=20261201` to `reference_seed=None`**; the B02 block-level provenance and manifest label remain 20261201. This is a versioned source change, **not** a revision of the previous independent verdict. The new `test_pure_gate.py` adjusts the synthetic B02 run-row fixture to actual plan semantics, adds a positive full-Train/IDs regression and a negative false sampling-seed test. `audit_candidate.py` remains a copy of V2 in this folder. All tests are non-numerical pure fixtures, no graph or solver.

**Critical:** `verify_b02_inputs` still **always denies**; `check_preuse_stub` still **always raises `INTEGRATION_REQUIRED`**. This module **is not an admission layer**. To run a solver, a *new* externally trusted, separately tested at-use runner implementation is required. No `DENY` exception may be converted into `ALLOW` as a shortcut.

**Preflight status at source commit:** `SOURCE_PATCH_PREPARED`; execution of exact versioned Python candidate / tests must be proven by captured CI/local logs, and trust/TOCTOU/runner integration remains `NOT_READY`. This README does **not** claim PASS.
