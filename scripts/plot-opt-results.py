import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../"))

import numpy as np
import matplotlib.pyplot as plt

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
        "font.size": 16,
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

plot_boozer_surface(eqfam[-1], ax=ax[0, 0])
plot_qs_error(
    eqfam[0],
    helicity=(1, eqfam[-1].NFP),
    log=True,
    ax=ax[0, 1],
    marker=["o"] * 2,
    labels=[r"$f_B$ initial", r"$f_C$ initial"],
    rho=10,
    fT=False,
)
plot_qs_error(
    eqfam[-1],
    helicity=(1, eqfam[-1].NFP),
    log=True,
    ax=ax[0, 1],
    marker=["x"] * 2,
    labels=[r"$f_B$ optimized", r"$f_C$ optimized"],
    rho=10,
    fT=False,
)
plot_1d(eqfam[-1], "iota", ax=ax[0, 2])
plot_comparison(
    [eqfam[0], eqfam[-1]],
    labels=["initial", "optimized"],
    phi=3,
    ax=ax[1],
    legend=False,
)
for axi in ax[1]:
    axi.legend(*ax[1, 0].get_legend_handles_labels())

for axi, tag in zip(ax[0], ["a)", "b)", "c)"]):
    axi.text(0.5, -0.2, tag, transform=axi.transAxes, ha="center", fontsize=22)
ax[1, 1].text(0.5, -0.2, "d)", transform=ax[1, 1].transAxes, ha="center", fontsize=22)
fig.savefig(
    f"{folder}/post-plots-L{L}M{M}N{N}-{name}.png",
    dpi=300,
    bbox_inches="tight",
)
