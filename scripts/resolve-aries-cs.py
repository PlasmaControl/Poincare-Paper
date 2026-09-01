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


savedir = "../draft-images"
os.makedirs(savedir, exist_ok=True)

eq0 = get("ARIES-CS")

try:
    eq_poin = load("poincare-resolve-aries-cs.h5")
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
    eq_poin.save("poincare-resolve-aries-cs.h5")


try:
    eq3 = load("poincare-resolve-aries-cs-solved.h5")
except FileNotFoundError:
    eq3 = eq_poin.copy()
    eq3.solve(verbose=3, maxiter=100, ftol=1e-3, gtol=0, xtol=0)
    eq3.save("poincare-resolve-aries-cs-solved.h5")

fig, ax = plt.subplots(3, 3, figsize=(18, 18))
for arow in ax:
    for a in arow:
        a.set_aspect("equal")
for arow in ax:
    for a in arow:
        a.sharex(ax[1, 0])
        a.sharey(ax[1, 0])
levels = np.logspace(-6, 0, 30)
plot_section(eq0, "|F|_normalized", log=True, levels=levels, phi=3, ax=ax[0])
plot_section(eq_poin, "|F|_normalized", log=True, levels=levels, phi=3, ax=ax[1])
plot_section(eq3, "|F|_normalized", log=True, levels=levels, phi=3, ax=ax[2])
plt.savefig(f"{savedir}/aries-cs-force.png", dpi=500)

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
