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

L, M, N = 12, 12, 6
name = "wqs3-war3-wvol0"
folder = "../draft-images"
eqfam = load(f"./results/Scan-QH-Paper-tight/eqfam-L{L}M{M}N{N}-{name}.h5")

fig, ax = plt.subplots(2, 3, figsize=(18, 12))
# bottom row mimics what plot_surfaces does when it makes its own axes
for a in ax[1]:
    a.set_aspect("equal")
for a in ax[1, 1:]:
    a.sharex(ax[1, 0])
    a.sharey(ax[1, 0])

known = set(fig.axes)
plot_boozer_surface(eqfam[-1], ax=ax[0, 0])
for cax in set(fig.axes) - known:
    cax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
plot_qs_error(
    eqfam[0],
    helicity=(1, eqfam[-1].NFP),
    log=True,
    ax=ax[0, 1],
    color=["blue"],
    labels=[r"$f_{qs}$ initial"],
    rho=10,
    fT=False,
    fB=False,
)
plot_qs_error(
    eqfam[-1],
    helicity=(1, eqfam[-1].NFP),
    log=True,
    ax=ax[0, 1],
    color=["red"],
    labels=[r"$f_{qs}$ optimized"],
    rho=10,
    fT=False,
    fB=False,
)
plot_1d(eqfam[0], "iota", label="initial", color="blue", ax=ax[0, 2])
plot_1d(eqfam[-1], "iota", label="optimized", color="red", ax=ax[0, 2])
plot_comparison(
    [eqfam[0], eqfam[-1]],
    labels=["initial", "optimized"],
    phi=3,
    ax=ax[1],
    legend=False,
)
ax[1, -1].legend(*ax[1, 0].get_legend_handles_labels(), loc="lower right")
ax[0, 1].set_ylim(1e-3, 1e0)

tagged = [*ax[0], ax[1, 0]]
tags = [
    axi.text(0, 0.5, tag, transform=axi.transAxes, ha="right", va="center", fontsize=22)
    for axi, tag in zip(tagged, ["a)", "b)", "c)", "d)"])
]
# bottom row height is locked by its equal aspect ratio, so shrink the figure until
# the top row, which takes whatever is left, matches it. tags move with the layout
for _ in range(20):
    fig.canvas.draw()
    for axi, t in zip(tagged, tags):
        x0 = axi.yaxis.get_tightbbox(fig.canvas.get_renderer()).x0
        t.set_x(axi.transAxes.inverted().transform((x0, 0))[0] - 0.04)
    dh = ax[0, 0].get_window_extent().height - ax[1, 0].get_window_extent().height
    if abs(dh) < 1:
        break
    w, h = fig.get_size_inches()
    fig.set_size_inches(w, h - dh / fig.dpi)
fig.savefig(
    f"{folder}/post-plots-L{L}M{M}N{N}-{name}.png",
    dpi=300,
    bbox_inches="tight",
)
