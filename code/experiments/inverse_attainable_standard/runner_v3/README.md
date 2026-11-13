# R05 §4.10 — runner_v3 / work-in-progress trust contract

**Historical pure V2 untouched:** `b02_gate_candidate/`; **G17 V3 pure remains DENY** at `b02_gate_v3/`. This *separate* runner is permitted by owner mandate `R05-SEC410-ENDTOEND-01`, but **must not run until internal gates pass**.

- `data_contract.py`: stdlib-only out-of-checkout root import via authenticated owner Issue comment; SHA and git blob pinning for manifest, Train reference, target and graph source; immutable data; G17 full Train has `reference_seed=null`. The historical nominal n300 block seed is metadata only.
- `run_one.py`: one (block, class, initialization, solver) context; owner-authorized external trust root, append-only START/TERMINAL audit root outside checkout, in-memory code execution from verified captures (no disk import/pyc for graph modules), numerical wall/RSS watchdog, fail-closed, versioned attempt outputs and bounded stop.
- The trusted root in Issue #60 is a **GitHub owner-controlled external record**, not a cryptographic signing key or stronger guarantee than GitHub owner/authenticated API. Its author/association and edit timestamps are verified. The external private audit directory lives in the runner temporary area.
- For B02–B06, only the frozen standard plan IDs are accepted. Results become `PROVISIONAL_PENDING_FINAL_QA` until the one final independent audit. Any remaining integration flaw must be recorded; no silent promotion of source V2 to PASS.
- `run_one.py` is **experimental and not yet tested on graph/solver**. This README deliberately does not claim `READY_FOR_SOLVE`, resource availability or data-derived scientific success.
- Original B01-v2 raw, PR #50, `chapter4` and historical pinned source trees remain immutable.

Internal sequence: contract pure tests → owner-root pin → guarded first-graph/probe → bounded B03/B05 microbatch → n300 probe and B02/B04/B06 → raw/metrics/tables/figures → §4.10 text → final independent QA.
