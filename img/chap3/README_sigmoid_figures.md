# Sigmoid margin figures (Chapter 3)

The **formula-only black-and-white** PDFs are included in `tex/chapter3.tex`:

- `sigmoid_loss_formula_bw.pdf` (loss `ell_tau(m)`)
- `sigmoid_sensitivity_formula_bw.pdf` (normalized sensitivity `4 tau |ell'_tau(m)|`)

The corresponding PNGs are stored for convenient review. The original **English-axis black-and-white** variants (`sigmoid_loss_bw.*` and `sigmoid_sensitivity_bw.*`) are kept unchanged for comparison. All graphs have four line patterns for `tau = 0.02, 0.05, 0.10, 0.20` and are title-free; Persian captions appear only in LaTeX.

Regenerate the formula-only thesis figures from the repository root with:

```bash
python code/figures/plot_sigmoid_margin_figures.py --output img/chap3
```

Requires Python, NumPy, and Matplotlib. The original two English-labeled versions were preserved from earlier plotting output; this script regenerates the new formula-only versions.
