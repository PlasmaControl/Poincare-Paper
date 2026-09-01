import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../"))

from desc import set_device

set_device("gpu")

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


eq = get("W7-X")
savedir = "../draft-images"
os.makedirs(savedir, exist_ok=True)

eq_poin0 = set_poincare_equilibrium(eq)
eq0 = eq.copy()
eq0.axis = eq0.surface.get_axis()
eq0.set_initial_guess()

try:
    eq_poin = load("poincare_w7x.h5")
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
            maxiter=250,
            ftol=1e-3,
        )

    eq_poin.surface = eq_poin.get_surface_at(rho=1)
    eq_poin.save("poincare_w7x.h5")

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
    phi=3,
    theta=6,
    rho=5,
    color=["k", "r"],
    lw=[2, 1],
    ax=ax[0],
    legend=False,
)
plot_comparison(
    eqs=[eq, eq_poin],
    rho=5,
    phi=3,
    labels=["LCFS", "Poincare"],
    ls=["-", "--"],
    lw=[3, 1],
    color=["k", "r"],
    ax=ax[1],
    legend=False,
)
fig.suptitle("Final Solution")
fig.savefig(f"{savedir}/w7x-init-guess.png", dpi=400)

fig, ax = plt.subplots(2, 3, figsize=(18, 12))
for a in ax[0]:
    a.set_aspect("equal")
for a in ax[1]:
    a.set_aspect("equal")
for arow in ax:
    for a in arow:
        a.sharex(ax[1, 0])
        a.sharey(ax[1, 0])
levels = np.logspace(-6, -2, 30)
plot_section(eq_poin, "|F|_normalized", log=True, levels=levels, phi=3, ax=ax[0])
plot_section(eq, "|F|_normalized", log=True, levels=levels, phi=3, ax=ax[1])
plt.savefig(f"{savedir}/w7x-force-init.png", dpi=500)

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
