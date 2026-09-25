import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from desc.io import load
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.continuation import solve_continuation_automatic
from desc.optimize import Optimizer
from desc.grid import LinearGrid
from desc.objectives import (
    ObjectiveFunction,
    ForceBalance,
    FixCurrent,
    FixSectionLambda,
    FixSectionR,
    FixSectionZ,
    FixBoundaryR,
    FixBoundaryZ,
    FixPressure,
    FixPsi,
    FixIota,
    QuasisymmetryTwoTerm,
    AspectRatio,
    Elongation,
    Volume,
    get_fixed_xsection_constraints,
    get_fixed_boundary_constraints,
)
from desc.examples import get
from desc.plotting import *
from desc.geometry import ZernikeRZToroidalSection, FourierRZToroidalSurface
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
eqp = load("./results/same-res-solves/poincare_LM12_N8_tight_solve.h5")
eq_qh = load(
    "./results/same-res-solves/landreman_paul_precise_QH_output_LM12_N8_tight_solve.h5"
)
eq_qh2 = load(
    "./results/same-res-solves/equivalent_precise_QH_output_LM12_N8_tight_solve.h5"
)
eq_qh3 = load(
    "./results/same-res-solves/normalized_qs_precise_QH_output_LM12_N8_tight_solve.h5"
)
eqs = [eqp, eq_qh, eq_qh2, eq_qh3]
labels = [
    "Poincaré",
    "L&P 2022",
    "LCFS (unnorm. f_qs)",
    "LCFS",
]
colors = ["red", "blue", "green", "magenta"]

fig, ax = plt.subplots(2, 4, figsize=(24, 10))
# bottom row mimics what plot_surfaces does when it makes its own axes
for a in ax[1]:
    a.set_aspect("equal")
for a in ax[1]:
    a.sharex(ax[1, 0])
    a.sharey(ax[1, 0])


for eq, label, color in zip(eqs, labels, colors):
    lw = 3 if eq in eqs[:2] else 1
    plot_1d(eq, "current", label=label, color=color, ax=ax[0, 0], lw=lw)
for eq, label, color in zip(eqs, labels, colors):
    lw = 3 if eq in eqs[:2] else 1
    plot_1d(eq, "iota", label=label, color=color, ax=ax[0, 1], lw=lw)

for eq, label, color in zip(eqs, labels, colors):
    lw = 3 if eq in eqs[:2] else 1
    plot_1d(
        eq, "|F|_normalized", label=label, color=color, ax=ax[0, 2], log=True, lw=lw
    )
for eq, label, color in zip(eqs, labels, colors):
    lw = 3 if eq in eqs[:2] else 1
    plot_1d(eq, "|J|", label=label, color=color, ax=ax[0, 3], log=False, lw=lw)


# leave the bottom 40% of each top-row panel free for the legend
for axi in ax[0, 1:3]:
    lo, hi = axi.get_ylim()
    if axi.get_yscale() == "log":
        lo, hi = np.log10(lo), np.log10(hi)
        axi.set_ylim(10 ** (hi - (hi - lo) / 0.6), 10**hi)
    else:
        axi.set_ylim(hi - (hi - lo) / 0.6, hi)
    axi.legend(loc="lower center", framealpha=1.0)

ax[0, 3].legend(loc="upper left", framealpha=1.0)

levels = np.logspace(-6, -2, 30)
exps = np.arange(np.log10(levels[0]), np.log10(levels[-1]) + 1).astype(int)
for i, (eq, label) in enumerate(zip(eqs, labels)):
    plot_section(
        eq,
        "|F|_normalized",
        ax=ax[1, i],
        log=True,
        levels=levels,
        phi=np.array([0.0]),
    )
    ax[1, i].set_title(label)
    cbar = ax[1, i].collections[0].colorbar
    cbar.set_ticks(10.0**exps, labels=[f"$10^{{{e}}}$" for e in exps])

fig.savefig(
    f"{folder}/compare-qh-results-same-res.png",
    dpi=500,
    bbox_inches="tight",
)

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
fig.savefig(
    f"{folder}/compare-qh-surfaces-same-res.png",
    dpi=500,
    bbox_inches="tight",
)


for eq, label in zip(eqs, labels):
    d = eq.compute(
        ["V", "R0", "a", "<|B|>_vol", "<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol"]
    )
    _, _, qs = plot_qs_error(
        eq,
        helicity=(1, eq.NFP),
        rho=np.array([1.0]),
        fT=False,
        fB=False,
        return_data=True,
    )
    print(
        f"{label:20s} V = {d['V']:8.3f}  R0/a = {d['R0'] / d['a']:6.3f}  "
        f"<|B|> = {d['<|B|>_vol']:7.4f}  "
        f"<|F|>/<|grad(p_B)|> = "
        f"{d['<|F|>_vol'] / d['<|grad(|B|^2)|/2mu0>_vol']:.3e}  "
        f"f_qs(rho=1) = {np.asarray(qs['f_C']).ravel()[-1]:.3e}"
    )
