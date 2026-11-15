# E-CH4-13 owner-selected figure layout (Figures 1–8 only)

Owner-requested display: exact overlapping x for Sigmoid and Softplus **within each seed** (no horizontal dodge), two distinct shapes/colors, separate panels for policy families shared and class. In each panel **each method gets three horizontal dashed descriptive lines**: minimum, arithmetic mean and maximum **over its three dataset seeds**. Mean heavier dashed, extrema lighter fine-dashed. Each panel shares the same y scale, seeds are named in full. These lines must NOT be interpreted as confidence intervals, fitted trends, or a population inference.

New graphs: original 1–8 report figure slots, with 1 = absolute Validation BA of two losses, 2 = absolute Train I of two losses, 3–8 = three R variants by Train and Validation. **Figures 9–11 unchanged**. Existing v2 images and raw evidence are immutable.

Output in figures/phase2_repro_v3/ (8 SVG) plus derived/phase2_figures_1_8_v3_manifest.json. Source: phase2_figures_1_8_v3.py using the stored 12-fit raw and 60 independent checksum checks, without Test/optimizer. This trigger authorizes only the visualization-derived workflow, not fitting or LaTeX edits.

Do not merge PR #79; await independent re-QA and owner's chart review.
