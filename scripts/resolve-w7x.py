"""Plots and optimizations related to W7-x like case."""

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


savedir = "../figures"
os.makedirs(savedir, exist_ok=True)

eq0 = load("./equilibria/W7-X_output.h5")[-1]

try:
    eq_poin = load("./equilibria/poincare-resolve-w7x.h5")
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
    eq_poin.save("./equilibria/poincare-resolve-w7x.h5")


try:
    eq3 = load("./equilibria/poincare-resolve-w7x-solved.h5")
except FileNotFoundError:
    eq3 = eq_poin.copy()
    eq3.solve(verbose=3, maxiter=100, ftol=1e-3, gtol=0, xtol=0)
    eq3.save("./equilibria/poincare-resolve-w7x-solved.h5")

try:
    eq4 = load("./equilibria/poincare-resolve-w7x-cold-solved.h5")
except FileNotFoundError:
    eq4 = eq_poin.copy()
    eq4.axis = eq4.surface.get_axis()
    eq4.set_initial_guess()
    eq4.solve(verbose=3, maxiter=1000, ftol=1e-3, gtol=0, xtol=0)
    eq4.save("./equilibria/poincare-resolve-w7x-cold-solved.h5")

fig, ax = plt.subplots(3, 3, figsize=(18, 18))
for arow in ax:
    for a in arow:
        a.set_aspect("equal")
for arow in ax:
    for a in arow:
        a.sharex(ax[1, 0])
        a.sharey(ax[1, 0])
levels = np.logspace(-6, -1, 30)
plot_section(
    eq0,
    "|F|_normalized",
    log=True,
    levels=levels,
    phi=np.linspace(0, np.pi / eq0.NFP, 3, endpoint=True),
    ax=ax[0],
)
plot_section(
    eq_poin,
    "|F|_normalized",
    log=True,
    levels=levels,
    phi=np.linspace(0, np.pi / eq0.NFP, 3, endpoint=True),
    ax=ax[1],
)
plot_comparison(
    [eq0, eq_poin],
    labels=["Original LCFS", "Poincaré Solved LCFS"],
    theta=0,
    rho=1,
    phi=np.linspace(0, np.pi / eq0.NFP, 3, endpoint=True),
    color=["black", "purple"],
    ax=ax[1],
    legend=False,
)
ax[1, 2].legend(*ax[1, 0].get_legend_handles_labels())
compare_kwargs = dict(
    labels=["Poincaré", "LCFS warm", "LCFS cold"],
    theta=6,
    rho=6,
    phi=np.linspace(0, np.pi / eq0.NFP, 3, endpoint=True),
    color=["black", "red", "blue"],
    lw=[3, 2, 1],
    ls=["-", "--", ":"],
    legend=False,
)
_, _, cdata = plot_comparison(
    [eq_poin, eq3, eq4], ax=ax[2], return_data=True, **compare_kwargs
)
ax[2][2].legend(*ax[2, 0].get_legend_handles_labels())

# zoom on the magnetic axis of each cross-section, placed in an empty corner
Rc, Zc = cdata["rho_R_coords"], cdata["rho_Z_coords"]
corners = [[0.03, 0.03, 0.34, 0.34], [0.03, 0.03, 0.34, 0.34], [0.63, 0.03, 0.34, 0.34]]
w = 0.5e-3  # half width of the zoomed region, 1 mm across
axins = []
for corner, a in zip(corners, ax[2]):
    axi = a.inset_axes(corner)
    axi.set_aspect("equal")
    axins.append(axi)
# rho of the surface that falls inside the zoomed region, the axis being at rho=0
plot_comparison(
    [eq_poin, eq3, eq4], ax=axins, **{**compare_kwargs, "rho": np.array([5e-4])}
)
for i, (a, axi) in enumerate(zip(ax[2], axins)):
    axi.set_xlim(Rc[0][0, 0, i] - w, Rc[0][0, 0, i] + w)
    axi.set_ylim(Zc[0][0, 0, i] - w, Zc[0][0, 0, i] + w)
    axi.set(title="", xlabel="", ylabel="", xticks=[], yticks=[])
    axi.text(
        0.5,
        0.03,
        f"{2e3 * w:.0f} mm",
        transform=axi.transAxes,
        va="bottom",
        ha="center",
        fontsize=12,
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1),
    )
    a.indicate_inset_zoom(axi, edgecolor="gray")
for a, tag in zip(ax[:, 0], ["a)", "b)", "c)"]):
    a.text(-0.4, 0.5, tag, transform=a.transAxes, va="center", ha="center", fontsize=22)
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
# only top 2 row use colorbar
cax = fig.add_axes(
    [max(b.x1 for b in box) + 0.01, 0.7 * (y0 + y1 - height), 0.012, height]
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
plt.savefig(f"{savedir}/w7x-force-e-1-e-6-v3.png", dpi=500, bbox_inches="tight")

eq_all = [eq0, eq_poin, eq3, eq4]
eq_names = ["LCFS", "Poincare", "Poincare and LCFS warm", "Poincare and LCFS cold"]
eq_data = {}
for eq, name in zip(eq_all, eq_names):
    f1 = (
        eq.compute("<|F|>_vol")["<|F|>_vol"]
        / eq.compute("<|grad(|B|^2)|/2mu0>_vol")["<|grad(|B|^2)|/2mu0>_vol"]
    )
    print(f"Force error {name}: {f1:.4e}")

    w1 = eq.compute("W")["W"]
    print(f"Energy {name}: {w1:.4e}")

    V1 = eq.compute("V")["V"]
    print(f"Volume {name}: {V1:.4e}")
    eq_data[name] = {"F": f1, "W": w1, "V": V1}

print(
    f"Volume change : {(eq_data["LCFS"]["V"] - eq_data["Poincare"]["V"])/eq_data["LCFS"]["V"] * 100}%"
)

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

axis_grid = LinearGrid(
    rho=np.array([0.0]),
    theta=np.array([0.0]),
    zeta=np.linspace(0, 2 * np.pi / eq0.NFP, 256, endpoint=False),
    NFP=eq0.NFP,
)
datap = eq_poin.compute(["R", "Z"], grid=axis_grid, basis="rpz")
for eq, name in zip([eq3, eq4], eq_names[2:]):
    data = eq.compute(["R", "Z"], grid=axis_grid, basis="rpz")
    daxis = np.hypot(datap["R"] - data["R"], datap["Z"] - data["Z"])
    print(
        f"Max axis deviation {name}: {daxis.max():.4e} m, "
        f"{daxis.max() / a0:.3%} of minor radius"
    )
