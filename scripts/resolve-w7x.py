import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../"))

from desc import set_device

# set_device("gpu")

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

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "legend.fontsize": 16,
        "figure.titlesize": 22,
    }
)

print_backend_info()


def set_poincare_equilibrium(eq):
    eq_poincare = Equilibrium(
        xsection=eq.get_surface_at(zeta=0),
        pressure=eq.pressure,
        iota=eq.iota,
        Psi=eq.Psi,  # flux (in Webers) within the last closed flux surface
        NFP=eq.NFP,  # number of field periods
        L=eq.L,  # radial spectral resolution
        M=eq.M,  # poloidal spectral resolution
        N=eq.N,  # toroidal spectral resolution
        L_grid=eq.L_grid,  # real space radial resolution, slightly oversampled
        M_grid=eq.M_grid,  # real space poloidal resolution, slightly oversampled
        N_grid=eq.N_grid,  # real space toroidal resolution
        sym=eq.sym,  # explicitly enforce stellarator symmetry
        spectral_indexing=eq._spectral_indexing,
    )

    eq_poincare.change_resolution(eq.L, eq.M, eq.N)
    eq_poincare.axis = eq_poincare.get_axis()
    eq_poincare.surface = eq_poincare.get_surface_at(rho=1)
    return eq_poincare


savedir = "../draft-images"
os.makedirs(savedir, exist_ok=True)

eq0 = get("W7-X")

try:
    eq_poin = load("poincare-resolve-w7x.h5")
except FileNotFoundError:
    eq_poin = eq0.copy()
    constraints = get_fixed_xsection_constraints(eq=eq_poin, fix_lambda=False)
    objective = ObjectiveFunction(ForceBalance(eq_poin))

    eq_poin.solve(
        verbose=3,
        objective=objective,
        constraints=constraints,
        maxiter=100,
        ftol=1e-3,
    )
    eq_poin.surface = eq_poin.get_surface_at(rho=1)
    eq_poin.save("poincare-resolve-w7x.h5")


try:
    eq3 = load("poincare-resolve-w7x-solved.h5")
except FileNotFoundError:
    eq3 = eq_poin.copy()
    eq3.solve(verbose=3, maxiter=100, ftol=1e-3, gtol=0, xtol=0)
    eq3.save("poincare-resolve-w7x-solved.h5")

fig, ax = plt.subplots(3, 3, figsize=(18, 18))
for arow in ax:
    for a in arow:
        a.set_aspect("equal")
for arow in ax:
    for a in arow:
        a.sharex(ax[1, 0])
        a.sharey(ax[1, 0])
levels = np.logspace(-6, -1, 30)
plot_section(eq0, "|F|_normalized", log=True, levels=levels, phi=3, ax=ax[0])
plot_section(eq_poin, "|F|_normalized", log=True, levels=levels, phi=3, ax=ax[1])
plot_comparison(
    [eq0, eq_poin],
    labels=["original", "after"],
    theta=0,
    rho=1,
    phi=3,
    color=["black", "purple"],
    ax=ax[1],
    legend=False,
)
for axi in ax[1]:
    axi.legend(*ax[1, 0].get_legend_handles_labels())
plot_section(eq3, "|F|_normalized", log=True, levels=levels, phi=3, ax=ax[2])
for a, tag in zip(ax[:, 0], ["a)", "b)", "c)"]):
    a.text(-0.3, 0.5, tag, transform=a.transAxes, va="center", ha="center", fontsize=22)
# drop the per-axes colorbars plot_section makes, use a single centered one instead
for a in fig.axes:
    cbar = getattr(a, "_colorbar", None)
    if cbar is not None:
        cbar.remove()
for a in ax.flatten():
    a.set_axes_locator(None)
fig.canvas.draw()
fig.set_layout_engine("none")  # freeze the tight layout, cax positions are absolute
exps = np.arange(np.log10(levels[0]), np.log10(levels[-1]) + 1).astype(int)
box = [a.get_position() for a in ax.flatten()]
y0, y1 = min(b.y0 for b in box), max(b.y1 for b in box)
height = 0.33 * (y1 - y0)
cax = fig.add_axes(
    [max(b.x1 for b in box) + 0.01, 0.5 * (y0 + y1 - height), 0.012, height]
)
cbar = fig.colorbar(ax[0, 0].collections[0], cax=cax)
cbar.set_ticks(10.0**exps, labels=[f"$10^{{{e}}}$" for e in exps])
phi = np.linspace(0, 2 * np.pi / eq0.NFP, 3, endpoint=False)
for axi in ax:
    for i in range(3):
        axi[i].set_title(
            "$\\phi \\cdot N_{{FP}}/2\\pi = {:.3f}$".format(
                eq0.NFP * phi[i] / (2 * np.pi)
            )
        )
fig.suptitle(
    "$|\\mathbf{J} \\times \\mathbf{B} - \\nabla p|/\\langle "
    + "|\\nabla |B|^{2}/(2\\mu_0)| \\rangle_{vol}$",
    y=1.01,
)
plt.savefig(f"{savedir}/w7x-force-e-1-e-6-v2.png", dpi=500, bbox_inches="tight")

f1 = (
    eq0.compute("<|F|>_vol")["<|F|>_vol"]
    / eq0.compute("<|grad(|B|^2)|/2mu0>_vol")["<|grad(|B|^2)|/2mu0>_vol"]
)
f2 = (
    eq_poin.compute("<|F|>_vol")["<|F|>_vol"]
    / eq_poin.compute("<|grad(|B|^2)|/2mu0>_vol")["<|grad(|B|^2)|/2mu0>_vol"]
)
f3 = (
    eq3.compute("<|F|>_vol")["<|F|>_vol"]
    / eq3.compute("<|grad(|B|^2)|/2mu0>_vol")["<|grad(|B|^2)|/2mu0>_vol"]
)
print(f"Force error LCFS: {f1:.4e}")
print(f"Force error Poincare: {f2:.4e}")
print(f"Force error Poincare and LCFS: {f3:.4e}")

f1 = eq0.compute("W")["W"]
f2 = eq_poin.compute("W")["W"]
f3 = eq3.compute("W")["W"]
print(f"Energy LCFS: {f1:.4e}")
print(f"Energy Poincare: {f2:.4e}")
print(f"Energy Poincare and LCFS: {f3:.4e}")

V1 = eq0.compute("V")["V"]
V2 = eq_poin.compute("V")["V"]
V3 = eq3.compute("V")["V"]
print(f"Volume LCFS: {V1:.4e}")
print(f"Volume Poincare: {V2:.4e}")
print(f"Volume Poincare and LCFS: {V3:.4e}")
print(f"Volume change : {(V1 - V2)/V1 * 100}%")

dense = LinearGrid(rho=np.array([1.0]), M=256, N=64, NFP=eq0.NFP)
data0 = eq0.compute(["R", "Z"], grid=dense, basis="rpz")
datap = eq_poin.compute(["R", "Z"], grid=dense, basis="rpz")
a0 = eq0.compute("a", grid=dense)["a"]

zeta = dense.nodes[:, 2]
dmin = []
for z in np.unique(zeta):
    m = zeta == z
    dR = data0["R"][m][:, None] - datap["R"][m][None, :]
    dZ = data0["Z"][m][:, None] - datap["Z"][m][None, :]
    dmin.append(np.sqrt(dR**2 + dZ**2).min(axis=1))
dmin = np.concatenate(dmin)
print(f"Minor radius : {a0} m")
print(f"Max LCFS deviation: {dmin.max():.4e} m, {dmin.max() / a0:.3%} of minor radius")
print(
    f"Mean LCFS deviation: {dmin.mean():.4e} m, {dmin.mean() / a0:.3%} of minor radius"
)
