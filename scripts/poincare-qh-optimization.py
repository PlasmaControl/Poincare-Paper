"""QH optimization using Poincare xsection variables."""

import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../"))

from desc import set_device

set_device("gpu")

import numpy as np
import matplotlib.pyplot as plt

from desc.io import load
from desc.equilibrium import EquilibriaFamily
from desc.integrals.surface_integral import surface_averages
from desc.optimize import Optimizer
from desc.grid import LinearGrid
from desc.objectives import (
    ObjectiveFunction,
    ForceBalance,
    FixCurrent,
    FixSectionR,
    FixSectionZ,
    FixPressure,
    FixPsi,
    GenericObjective,
    ObjectiveFromUser,
    AspectRatio,
    Volume,
    get_fixed_xsection_constraints,
)
from desc.plotting import plot_qs_error
from desc.backend import print_backend_info

print_backend_info()

L, M, N = 12, 12, 6
w_qs = 3
w_ar = 3
w_vol = 0
w_B = 1
folder = "./results"
name = f"wqs{w_qs}-war{w_ar}-wvol{w_vol}-wb{w_B}"
os.makedirs(folder, exist_ok=True)

try:
    eq = load(f"poincare-QH-initial-L{L}M{M}N{N}.h5")
    eq.xsection = eq.get_surface_at(zeta=0)
    eq.surface = eq.get_surface_at(rho=1)
except FileNotFoundError:
    # get the initial unoptimized equilibrium
    eq = load("init_precise_QH.h5")
    eq.change_resolution(L=L, M=M, N=N, L_grid=2 * L, M_grid=2 * M, N_grid=2 * N)
    eq.solve(maxiter=500, verbose=3, ftol=1e-3)
    eq.xsection = eq.get_surface_at(zeta=0)
    eq.surface = eq.get_surface_at(rho=1)
    constraints = get_fixed_xsection_constraints(eq=eq, fix_lambda=True)
    objective = ObjectiveFunction(ForceBalance(eq))

    # before optimization make sure that the initial equilibrium
    # is in force balance in terms of poincare constraints
    eq.solve(
        verbose=3,
        objective=objective,
        constraints=constraints,
        maxiter=1000,
        ftol=1e-3,
    )
    eq.xsection = eq.get_surface_at(zeta=0)
    eq.surface = eq.get_surface_at(rho=1)
    eq.save(f"poincare-initial-QH-L{L}M{M}N{N}.h5")

eqfam = EquilibriaFamily(eq)
V = eq.compute("V")["V"]
B0 = eq.compute("<|B|>_vol")["<|B|>_vol"]

rho_qs = np.array([0.6, 0.8, 1.0])
# grids for the QS objective
grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=rho_qs, sym=True)


def make_qs(eq, weight):
    """QS objective with normalization."""
    helicity = (1, eq.NFP)

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


# use initial qs error as a weight-factor
probe = ObjectiveFunction(make_qs(eq, 1.0))
probe.build(verbose=0)
norm0 = float(np.linalg.norm(np.asarray(probe.compute_scaled_error(probe.x(eq)))))
print(f"dim_f = {probe.dim_f}, initial norm = {norm0:.4e}")


def run_step(n, eqfam, ftol=1e-2, **kwargs):
    eq = eqfam[-1]
    objs = (make_qs(eq, w_qs / norm0),)
    if w_ar > 0:
        objs += (AspectRatio(eq=eq, target=8, weight=w_ar, normalize=False),)
    if w_vol > 0:
        # we can actually just rescale the equilibrium at the end to get the right
        # volume, adding volume makes the optimization harder (to find proper weights)
        objs += (
            Volume(eq=eq, bounds=(0.9 * V, 1.1 * V), weight=w_vol, normalize=False),
        )
    if w_B > 0:
        # residual is in Tesla, GenericObjective does not normalize
        objs += (
            GenericObjective(
                "<|B|>_vol", thing=eq, target=B0, weight=w_B, normalize=False
            ),
        )
    objective = ObjectiveFunction(objs, deriv_mode="batched")
    # modes to fix
    bc_surf = eq.xsection
    R_order = np.maximum(
        bc_surf.R_basis.modes[:, 0], np.abs(bc_surf.R_basis.modes[:, 1])
    )
    Z_order = np.maximum(
        bc_surf.Z_basis.modes[:, 0], np.abs(bc_surf.Z_basis.modes[:, 1])
    )
    R_modes = bc_surf.R_basis.modes[R_order > n, :]
    Z_modes = bc_surf.Z_basis.modes[Z_order > n, :]

    constraints = (
        ForceBalance(eq=eq),
        FixSectionR(eq=eq, modes=R_modes),
        FixSectionZ(eq=eq, modes=Z_modes),
        FixPressure(eq=eq),
        FixCurrent(eq=eq),
        FixPsi(eq=eq),
    )

    anchored = w_B > 0 or w_vol > 0
    tr_method = "qr" if (anchored and w_vol == 0) else "svd"
    optimizer = Optimizer("proximal-lsq-exact")
    eq_new, _ = eq.optimize(
        objective=objective,
        constraints=constraints,
        optimizer=optimizer,
        maxiter=50,
        verbose=3,
        ftol=ftol,
        xtol=1e-6,
        gtol=1e-6,
        copy=True,
        x_scale="ess",
        options={
            "perturb_options": {"verbose": 0},
            "solve_options": {"verbose": 0, "ftol": 1e-3, "maxiter": 150},
            # with bounds, we can get singular Jacobians
            "tr_method": tr_method,
            **kwargs,
        },
    )
    # to make sure the surfaces are updated properly
    eq_new.xsection = eq_new.get_surface_at(zeta=0)
    eq_new.surface = eq_new.get_surface_at(rho=1)
    eqfam.append(eq_new)
    return eqfam


def report(eq, tag):
    """Scale free QS error, so runs at different field strengths are comparable."""
    fig, _, data = plot_qs_error(
        eq, rho=rho_qs, helicity=(1, eq.NFP), fT=False, return_data=True
    )
    plt.close(fig)
    d = eq.compute(["V", "R0", "a", "<|B|>_vol"])
    print(
        f"{tag:9s} V = {d['V']:8.3f}  R0/a = {d['R0'] / d['a']:6.3f}  "
        f"<|B|> = {d['<|B|>_vol']:7.4f}  "
        f"f_B = {np.array2string(data['f_B'], precision=4)}  "
        f"f_C = {np.array2string(data['f_C'], precision=4)}"
    )


for n in range(2, M + 1):
    print(f"\n===== optimizing section modes max(l,|m|) <= {n} =====\n")
    eqfam = run_step(n, eqfam, ftol=1e-3)

eqfam.save(f"{folder}/poincare-QH-optimized-{name}.h5")

for i, eqi in enumerate(eqfam):
    report(eqi, f"step {i}")
