"""Plot comparison of all QH optimization variants solved at the same res."""

import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from desc.io import load
from desc.grid import LinearGrid
from desc.integrals import surface_averages
from desc.plotting import *
from qs_error import plot_qs_error_fsa
from desc.backend import print_backend_info

print_backend_info()

plt.rcParams.update(
    {
        "font.size": 22,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "legend.fontsize": 16,
        "figure.titlesize": 22,
    }
)

folder = "../extra-images"
eqp = load("./equilibria/same-res-solves/Poincare_LM12_N8_tight_solve.h5")
eq_qh = load("./equilibria/same-res-solves/L&P_2022_LM12_N8_tight_solve.h5")
eq_qh2 = load("./equilibria/same-res-solves/LCFS_(unnorm_f_qs)_LM12_N8_tight_solve.h5")
eq_qh3 = load("./equilibria/same-res-solves/LCFS_LM12_N8_tight_solve.h5")
eqs = [eqp, eq_qh, eq_qh2, eq_qh3]
labels = [
    "Poincaré",
    "L&P 2022",
    "LCFS (unnorm. f_qs)",
    "LCFS",
]
colors = ["red", "blue", "green", "magenta"]

fig, ax = plt.subplots(3, 4, figsize=(24, 15))
# bottom row mimics what plot_surfaces does when it makes its own axes
for a in ax[2]:
    a.set_aspect("equal")
for a in ax[2]:
    a.sharex(ax[2, 0])
    a.sharey(ax[2, 0])


for i, (eq, label) in enumerate(zip(eqs, labels)):
    known = set(fig.axes)
    plot_boozer_surface(eq, ax=ax[1, i])
    ax[1, i].set_title(label)
    for cax in set(fig.axes) - known:
        cax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

for eq, label, color in zip(eqs, labels, colors):
    plot_qs_error_fsa(eq, rho=10, ax=ax[0, 0], label=label, color=color)
for eq, label, color in zip(eqs, labels, colors):
    lw = 3 if eq in eqs[:2] else 1
    plot_1d(eq, "iota", label=label, color=color, ax=ax[0, 1], lw=lw)

for eq, label, color in zip(eqs, labels, colors):
    lw = 3 if eq in eqs[:2] else 1
    plot_fsa(
        eq, "|F|_normalized", label=label, color=color, ax=ax[0, 2], log=True, lw=lw
    )

rho = np.linspace(0.05, 1, 40)
for eq, label, color in zip(eqs, labels, colors):
    lw = 3 if eq in eqs[:2] else 1
    grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
    data = eq.compute(["|J|", "sqrt(g)"], grid=grid)
    J = surface_averages(grid, data["|J|"], sqrt_g=data["sqrt(g)"], expand_out=False)
    vol = eq.compute(["<|B|>_vol", "<|grad(|B|^2)|/2mu0>_vol"])
    J = J * vol["<|B|>_vol"] / vol["<|grad(|B|^2)|/2mu0>_vol"]
    ax[0, 3].semilogy(rho, J, label=label, color=color, lw=lw)
ax[0, 2].set_ylabel(
    r"$\langle |\mathbf{F}| \rangle / \langle |\nabla |B|^2 / (2\mu_0)| \rangle_{vol}$"
)
ax[0, 3].set_xlabel(r"$\rho$")
ax[0, 3].set_ylabel(
    r"$\langle |B| \rangle_{vol} \langle |\mathbf{J}| \rangle"
    r" / \langle |\nabla |B|^2 / (2\mu_0)| \rangle_{vol}$"
)


for axi in ax[0]:
    if axi.get_legend() is not None:
        axi.get_legend().remove()
# at least up to 1e-2 so the log axis shows more than one decade tick
ax[0, 0].set_ylim(top=max(ax[0, 0].get_ylim()[1], 1e-2))

levels = np.logspace(-6, -2, 30)
exps = np.arange(np.log10(levels[0]), np.log10(levels[-1]) + 1).astype(int)
for i, (eq, label) in enumerate(zip(eqs, labels)):
    plot_section(
        eq,
        "|F|_normalized",
        ax=ax[2, i],
        log=True,
        levels=levels,
        phi=np.array([0.0]),
    )
    ax[2, i].set_title(label)
    cbar = ax[2, i].collections[0].colorbar
    cbar.set_ticks(10.0**exps, labels=[f"$10^{{{e}}}$" for e in exps])


fig.canvas.draw()
for axi, tag in zip([*ax[0], *ax[1:, 0]], ["a)", "b)", "c)", "d)", "e)", "f)"]):
    x0 = axi.yaxis.get_tightbbox(fig.canvas.get_renderer()).x0
    x = axi.transAxes.inverted().transform((x0, 0))[0] - 0.04
    axi.text(x, 0.5, tag, transform=axi.transAxes, ha="right", va="center", fontsize=22)

# shared single-line legend above the top row, clear of the tall y-labels
fig.set_layout_engine("none")  # freeze tight layout so it can't move the axes
renderer = fig.canvas.get_renderer()
top = max(
    max(axi.get_tightbbox(renderer).y1, axi.yaxis.label.get_window_extent(renderer).y1)
    for axi in ax[0]
)
fig.legend(
    *ax[0, 1].get_legend_handles_labels(),
    loc="lower center",
    bbox_to_anchor=(0.5, fig.transFigure.inverted().transform((0, top))[1]),
    ncol=len(eqs),
    frameon=False,
)

fig.savefig(f"{folder}/compare-qh-results-same-res.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{folder}/compare-qh-results-same-res.png", dpi=300, bbox_inches="tight")

fig, ax = plt.subplots(
    2, 2, figsize=(12, 12), sharex=True, sharey=True, subplot_kw=dict(aspect="equal")
)
plot_comparison(
    eqs,
    labels=labels,
    rho=np.array([1.0]),
    theta=0,
    color=colors,
    phi=np.linspace(0, np.pi / eqp.NFP, 4),
    lw=[3, 3, 1, 1],
    legend=False,
    ax=ax,
)
ax[1, 1].legend(*ax[0, 0].get_legend_handles_labels(), loc="lower right")
fig.savefig(f"{folder}/compare-qh-surfaces-same-res.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{folder}/compare-qh-surfaces-same-res.png", dpi=300, bbox_inches="tight")


for eq, label in zip(eqs, labels):
    d = eq.compute(
        ["V", "R0", "a", "<|B|>_vol", "<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol"]
    )
    _, qs = plot_qs_error_fsa(eq, rho=np.array([1.0]))
    print(
        f"{label:20s} V = {d['V']:8.3f}  R0/a = {d['R0'] / d['a']:6.3f}  "
        f"<|B|> = {d['<|B|>_vol']:7.4f}  "
        f"<|F|>/<|grad(p_B)|> = "
        f"{d['<|F|>_vol'] / d['<|grad(|B|^2)|/2mu0>_vol']:.3e}  "
        f"f_qs(rho=1) = {qs[-1]:.3e}"
    )
