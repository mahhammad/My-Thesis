# E-CH4-13 — independent QA follow-up: versioned derived fixes, zero training

Request: Issue #80, independent verdict ACCEPT_PHASE2_WITH_FIXES, comment #6889701832.

Purpose: generate **only** new derivation products under `derived/phase2_qa_fixes_v2/` and `figures/phase2_repro_v2/` using existing immutable 12-fit raw and strict source SHA checks. No change to prior figures the owner is reviewing. No optimizer, Test, raw overwrite, LaTeX or PR merge.

- Frozen audit delivery: `d836d72beec4cc9dd47467c7e39755e4c0c8cbad`
- Original training SHA: `cdb45644c245e04a276eb2a419e4d64f71c6c18a`
- Original successful Actions run: `43670524935`, artifact `14144445918`
- Post-hoc source: `phase2_derived_reproduce_v2.py`
- Gate: exactly 12 corrected coverage metadata files, 11 reproducible SVGs, 2 plotting data CSVs, 2 provenance JSONs (27 files total)
- Generated files must pass `--check` for byte-level reconstruction.

This file is an opt-in **nontraining** workflow marker; after completion, the derived correction code is the canonical source of chart values, not the originally hand-crafted SVGs. The independent auditor must recheck this new v2 material rather than treating the researcher declaration as acceptance.
