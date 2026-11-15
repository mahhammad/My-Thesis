# E-CH4-13 / 2026-11-15 — authorized archival checkpoint

**Purpose:** Run the no-training archive workflow once on the already-successful [official phase2 Actions run #43670524935](https://github.com/mahhammad/My-Thesis/actions/runs/43670524935). This is **NOT** an optimizer trigger or an extension of the 12-fit cap.

- Training source SHA: `cdb45644c245e04a276eb2a419e4d64f71c6c18a`.
- Archive artifact ID: `14144445918`.
- Original archive ZIP SHA256: `26e7a94e36edeae2a3153b76b40cf282c30b550a3fac359558a20d11e72c3e3e`.
- Validation: 12 full raw models, 312 optimizer state checkpoints, 12 immutable charge markers, full metric/source SHA, exact 25-update traces, no Test.
- No Test, no new training, no edit to historical pilot, no LaTeX, no PR #79 merge. Outcome remains provisional until independent scientific QA.

The GitHub Actions archival workflow is restricted to this single explicit marker in the commit message and fails closed if `results/phase2_300_25` already exists. It copies original bytes rather than fitting models.
