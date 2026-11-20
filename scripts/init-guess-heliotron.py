"""Plots and optimizations related to HELIOTRON case."""

import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../"))

from desc import set_device

set_device("gpu")

import numpy as np
import matplotlib.pyplot as plt

from desc.io import load
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from desc.objectives import (
    ObjectiveFunction,
    ForceBalance,
    get_fixed_xsection_constraints,
)
from desc.plotting import *
from desc.backend import print_backend_info

plt.rcParams.update(
    {
        "font.size": 22,
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


eq = load("./equilibria/HELIOTRON_output.h5")[-1]
savedir = "../figures"
os.makedirs(savedir, exist_ok=True)

eq_poin0 = set_poincare_equilibrium(eq)
eq0 = eq.copy()
eq0.axis = eq0.surface.get_axis()
eq0.set_initial_guess()

try:
    eq_poin = load("./equilibria/poincare_heliotron.h5")
except:
    eq_poin = set_poincare_equilibrium(eq)  # zeta=0 surface will be fixed
    for N in range(1, eq.N + 1):
        print(f"\n\nSolving for N={N}...\n\n")
        eq_poin.change_resolution(N=N, N_grid=2 * N)

        constraints = get_fixed_xsection_constraints(eq=eq_poin, fix_lambda=True)
        objective = ObjectiveFunction(ForceBalance(eq_poin))

        eq_poin.solve(
            verbose=3,
            objective=objective,
            constraints=constraints,
            maxiter=75,
            ftol=1e-3,
        )

    eq_poin.surface = eq_poin.get_surface_at(rho=1)
    eq_poin.save("./equilibria/poincare_heliotron.h5")

phi = np.linspace(0, np.pi / eq.NFP, 3, endpoint=True)
fig, ax = plt.subplots(2, 3, figsize=(18, 12))
for a in ax[0]:
    a.set_aspect("equal")
for a in ax[1]:
    a.set_aspect("equal")
for arow in ax:
    for a in arow:
        a.sharex(ax[1, 0])
        a.sharey(ax[1, 0])
plot_comparison(
    eqs=[eq0, eq_poin0],
    phi=phi,
    theta=6,
    rho=5,
    labels=["LCFS", "Poincare"],
    color=["k", "r"],
    lw=[2, 1],
    ax=ax[0],
    legend=False,
)
plot_comparison(
    eqs=[eq, eq_poin],
    rho=5,
    phi=phi,
    labels=["LCFS", "Poincare"],
    ls=["-", "--"],
    lw=[3, 1],
    color=["k", "r"],
    ax=ax[1],
    legend=False,
)
for axi in ax.flatten():
    axi.legend(*ax[1, 0].get_legend_handles_labels())
for a, tag in zip(ax[:, 0], ["a)", "b)"]):
    a.text(
        -0.26, 0.5, tag, transform=a.transAxes, va="center", ha="center", fontsize=22
    )
fig.savefig(f"{savedir}/heliotron-init-guess.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{savedir}/heliotron-init-guess.png", dpi=300, bbox_inches="tight")

fig, ax = plt.subplots(2, 3, figsize=(18, 12))
for a in ax[:, 1:].flatten():
    a.set_aspect("equal")
    a.sharex(ax[1, 1])
    a.sharey(ax[1, 1])
levels = np.logspace(-6, -2, 30)
phi2 = np.array([0, np.pi / eq.NFP])
plot_section(eq, "|F|_normalized", log=True, levels=levels, phi=phi2, ax=ax[0, 1:])
plot_section(eq_poin, "|F|_normalized", log=True, levels=levels, phi=phi2, ax=ax[1, 1:])

# surface (rho) and cross-section (zeta) averages and maxima of the normalized error
rho = np.linspace(0.05, 1, 40)
for eq_i, name, c in zip([eq, eq_poin], ["LCFS", "Poincaré"], ["k", "r"]):
    zeta = np.linspace(0, 2 * np.pi / eq_i.NFP, 193)
    theta = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    grid = LinearGrid(rho=rho, theta=theta, zeta=zeta, NFP=eq_i.NFP, sym=False)
    data = eq_i.compute(["|F|", "sqrt(g)"], grid=grid)
    norm = eq_i.compute("<|grad(|B|^2)|/2mu0>_vol")["<|grad(|B|^2)|/2mu0>_vol"]
    F = grid.meshgrid_reshape(data["|F|"] / norm, "rtz")
    w = grid.meshgrid_reshape(np.abs(data["sqrt(g)"]), "rtz")
    zN = zeta * eq_i.NFP / (2 * np.pi)
    for a, x, axes in zip(ax[:, 0], [rho, zN], [(1, 2), (0, 1)]):
        avg = (F * w).sum(axis=axes) / w.sum(axis=axes)
        a.semilogy(x, avg, color=c, lw=3, label=f"{name} avg")
        a.semilogy(x, F.max(axis=axes), color=c, lw=2, ls="--", label=f"{name} max")
ax[0, 0].set_xlabel(r"$\rho$")
ax[1, 0].set_xlabel(r"$\phi \cdot N_{FP}/2\pi$")
for a in ax[:, 0]:
    a.set_ylabel(r"$|\mathbf{F}| / \langle |\nabla |B|^2 / (2\mu_0)| \rangle_{vol}$")
for a in ax[:, 0]:
    # leave the bottom 30% of the log range free for the legend
    lo, hi = np.log10(a.get_ylim())
    a.set_ylim(10 ** (hi - (hi - lo) / 0.7), 10**hi)
    a.legend(fontsize=13, framealpha=1.0, ncol=2, loc="lower center")

for a, tag, x in zip(
    [ax[0, 0], ax[1, 0], ax[0, 1], ax[1, 1]], ["a)", "b)", "c)", "d)"], [-0.4] * 4
):
    a.text(x, 0.5, tag, transform=a.transAxes, va="center", ha="center", fontsize=22)
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
height = 0.5 * (y1 - y0)
cax = fig.add_axes(
    [max(b.x1 for b in box) + 0.01, 0.5 * (y0 + y1 - height), 0.012, height]
)
cbar = fig.colorbar(ax[0, 1].collections[0], cax=cax)
cbar.set_ticks(10.0**exps, labels=[f"$10^{{{e}}}$" for e in exps])
for axi in ax:
    for i in range(2):
        axi[i + 1].set_title(
            "$\\phi \\cdot N_{{FP}}/2\\pi = {:.3f}$".format(
                eq.NFP * phi2[i] / (2 * np.pi)
            )
        )
cbar.set_label(
    r"$|\mathbf{F}| / \langle |\nabla |B|^2 / (2\mu_0)| \rangle_{vol}$"
)
fig.savefig(f"{savedir}/heliotron-force-e-2-e-6.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{savedir}/heliotron-force-e-2-e-6.png", dpi=300, bbox_inches="tight")

f1 = (
    eq.compute("<|F|>_vol")["<|F|>_vol"]
    / eq.compute("<|grad(|B|^2)|/2mu0>_vol")["<|grad(|B|^2)|/2mu0>_vol"]
)
f2 = (
    eq_poin.compute("<|F|>_vol")["<|F|>_vol"]
    / eq_poin.compute("<|grad(|B|^2)|/2mu0>_vol")["<|grad(|B|^2)|/2mu0>_vol"]
)
print(f"Force error eq_lcfs: {f1:.4e}")
print(f"Force error eq_poin: {f2:.4e}")
