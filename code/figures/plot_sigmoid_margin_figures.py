"""Reproducibly plot the chapter-3 sigmoid margin loss and normalized sensitivity.

Generates black-and-white, title-free figures with different linestyles for
four values of tau. The math-only variant is the one included in the thesis.
Requires numpy and matplotlib.

Usage:
    python code/figures/plot_sigmoid_margin_figures.py --output img/chap3
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TAUS = (0.02, 0.05, 0.10, 0.20)
LINESTYLES = ('-', '--', '-.', ':')


def _loss(m, tau):
    return 1 / (1 + np.exp(m / tau))


def _normalized_sensitivity(m, tau):
    # 4*tau*|d loss/dm| = sech^2(m/(2*tau))
    return 1 / np.cosh(m / (2 * tau)) ** 2


def generate(output):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans',
                         'mathtext.fontset': 'dejavusans',
                         'font.size': 11, 'axes.linewidth': 0.85,
                         'pdf.fonttype': 42})
    m = np.linspace(-0.35, 0.35, 1401)
    specs = (
        ('sigmoid_loss_formula_bw', _loss,
         r'$\ell_{\tau}(m)=\dfrac{1}{1+e^{m/\tau}}$'),
        ('sigmoid_sensitivity_formula_bw', _normalized_sensitivity,
         r'$4\tau\,|\ell_{\tau}^{\prime}(m)|=\mathrm{sech}^{2}\!\left(\dfrac{m}{2\tau}\right)$'),
    )
    for filename, fn, ylabel in specs:
        fig, ax = plt.subplots(figsize=(8.4, 5.0))
        for tau, style in zip(TAUS, LINESTYLES):
            ax.plot(m, fn(m, tau), color='black', linewidth=1.7,
                    linestyle=style, label=rf'$\tau={tau:.2f}$')
        ax.axvline(0, linewidth=0.8, linestyle=(0, (3, 4)), color='0.52')
        ax.set_xlabel(r'$m$', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13, labelpad=12)
        ax.set_xlim(-0.35, 0.35)
        ax.set_ylim(-0.015, 1.015)
        ax.set_xticks(np.arange(-0.3, 0.31, 0.1))
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(alpha=0.14, color='0.5')
        ax.legend(loc='upper right', frameon=True, edgecolor='0.5', fontsize=10)
        fig.tight_layout(pad=1.2)
        fig.savefig(output / (filename + '.png'), dpi=230, bbox_inches='tight', facecolor='white')
        fig.savefig(output / (filename + '.pdf'), bbox_inches='tight', facecolor='white')
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('img/chap3'))
    args = parser.parse_args()
    generate(args.output)
