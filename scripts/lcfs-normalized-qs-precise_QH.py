"""Example script for recreating the "precise QH" configuration of Landreman and Paul.

Note that this resembles their optimization process in SIMSOPT, but the final optimized
equilibrium is slightly different from their VMEC solution.
"""

import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../"))

from desc import set_device

set_device("gpu")

import numpy as np

from desc.continuation import solve_continuation_automatic
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.io import load
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.integrals.surface_integral import surface_averages
from desc.objectives import (
    AspectRatio,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixPressure,
    FixPsi,
    ForceBalance,
    GenericObjective,
    ObjectiveFunction,
    ObjectiveFromUser,
    QuasisymmetryTwoTerm,
)
from desc.optimize import Optimizer


def make_qs(eq, grid, helicity, weight):
    """QS objective for the requested metric."""

    # f_C divided by <B>^3 of this iterate, so a weaker field buys nothing
    def two_term_hat(grid, data):
        Bbar = surface_averages(grid, q=data["|B|"], sqrt_g=data["sqrt(g)"])
        return data["f_C"] / Bbar**3

    return ObjectiveFromUser(
        fun=two_term_hat,
        thing=eq,
        grid=grid,
        compute_kwargs={"helicity": helicity},
        normalize=False,
        weight=weight,
        name="QS two-term hat",
    )


try:
    eq = load(f"./results/lcfs-initial-QH-L8M8N8.h5")
except FileNotFoundError:
    # create initial equilibrium
    surf = FourierRZToroidalSurface(
        R_lmn=[1, 0.125, 0.3],
        Z_lmn=[-0.125, -0.3],
        modes_R=[[0, 0], [1, 0], [0, 1]],
        modes_Z=[[-1, 0], [0, -1]],
        NFP=4,
    )
    eq = Equilibrium(M=8, N=8, Psi=0.04, surface=surf)
    eq = solve_continuation_automatic(eq, objective="force", verbose=3)[-1]

eqfam = EquilibriaFamily(eq)

# optimize in steps
grid = LinearGrid(M=eq.M, N=eq.N, NFP=eq.NFP, rho=np.array([0.6, 0.8, 1.0]), sym=True)

B0 = eq.compute("<|B|>_vol")["<|B|>_vol"]

for n in range(1, eq.M + 1):
    print("\n==================================")
    print("Optimizing boundary modes M,N <= {}".format(n))
    print("====================================")
    objective = ObjectiveFunction(
        (
            make_qs(eqfam[-1], grid, (1, eq.NFP), weight=1),
            GenericObjective(
                "<|B|>_vol", thing=eqfam[-1], target=B0, weight=1, normalize=False
            ),
            AspectRatio(eq=eqfam[-1], target=8, weight=1e1, normalize=False),
        ),
    )
    R_modes = np.vstack(
        (
            [0, 0, 0],
            eq.surface.R_basis.modes[
                np.max(np.abs(eq.surface.R_basis.modes), 1) > n, :
            ],
        )
    )
    Z_modes = eq.surface.Z_basis.modes[
        np.max(np.abs(eq.surface.Z_basis.modes), 1) > n, :
    ]
    constraints = (
        ForceBalance(eq=eqfam[-1]),  # J x B - grad(p) = 0
        FixBoundaryR(eq=eqfam[-1], modes=R_modes),
        FixBoundaryZ(eq=eqfam[-1], modes=Z_modes),
        FixPressure(eq=eqfam[-1]),
        FixCurrent(eq=eqfam[-1]),
        FixPsi(eq=eqfam[-1]),
    )
    optimizer = Optimizer("proximal-lsq-exact")
    eq_new, out = eqfam[-1].optimize(
        objective=objective,
        constraints=constraints,
        optimizer=optimizer,
        maxiter=50,
        verbose=3,
        copy=True,
        options={
            "initial_trust_radius": 0.5,
            "perturb_options": {"verbose": 0},
            "solve_options": {"verbose": 0, "ftol": 1e-3, "maxiter": 250},
        },
    )
    eqfam.append(eq_new)

eqfam.save("./results/normalized_qs_precise_QH_output.h5")
