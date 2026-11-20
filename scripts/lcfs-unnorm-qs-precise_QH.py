"""QH optimization using LCFS variables but no QS normalization."""

import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../"))

from desc import set_device

set_device("gpu")

import time

import numpy as np

from desc.continuation import solve_continuation_automatic
from desc.io import load
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.optimize import Optimizer
from desc.objectives import (
    ObjectiveFunction,
    ForceBalance,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixPressure,
    FixPsi,
    GenericObjective,
    AspectRatio,
    Volume,
)
from qs_error import make_qs, print_run_info, report

print_run_info()

L, M, N = 8, 8, 8
w_qs = 3
w_ar = 3
w_vol = 0
w_B = 1
folder = "./equilibria"
os.makedirs(folder, exist_ok=True)

try:
    eq = load(f"./equilibria/lcfs-initial-QH-L{L}M{M}N{N}.h5")
except FileNotFoundError:
    # create initial equilibrium
    surf = FourierRZToroidalSurface(
        R_lmn=[1, 0.125, 0.3],
        Z_lmn=[-0.125, -0.3],
        modes_R=[[0, 0], [1, 0], [0, 1]],
        modes_Z=[[-1, 0], [0, -1]],
        NFP=4,
    )
    eq = Equilibrium(L=L, M=M, N=N, Psi=0.04, surface=surf)
    eq = solve_continuation_automatic(eq, objective="force", verbose=3)[-1]
    eq.save(f"./equilibria/lcfs-initial-QH-L{L}M{M}N{N}.h5")

eqfam = EquilibriaFamily(eq)
V = eq.compute("V")["V"]
B0 = eq.compute("<|B|>_vol")["<|B|>_vol"]

# use initial qs error as a weight-factor
probe = ObjectiveFunction(make_qs(eq, 1.0, normalized=False))
probe.build(verbose=0)
norm0 = float(np.linalg.norm(np.asarray(probe.compute_scaled_error(probe.x(eq)))))
print(f"dim_f = {probe.dim_f}, initial norm = {norm0:.4e}")


def run_step(n, eqfam):
    eq = eqfam[-1]
    objs = (make_qs(eq, w_qs / norm0, normalized=False),)
    if w_ar > 0:
        objs += (AspectRatio(eq=eq, target=8, weight=w_ar, normalize=False),)
    if w_vol > 0:
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
    objective = ObjectiveFunction(objs)
    # modes to fix, R00 is the average major radius
    modes = eq.surface.R_basis.modes
    R_modes = np.vstack(([0, 0, 0], modes[np.max(np.abs(modes), 1) > n, :]))
    modes = eq.surface.Z_basis.modes
    Z_modes = modes[np.max(np.abs(modes), 1) > n, :]

    constraints = (
        ForceBalance(eq=eq),
        FixBoundaryR(eq=eq, modes=R_modes),
        FixBoundaryZ(eq=eq, modes=Z_modes),
        FixPressure(eq=eq),
        FixCurrent(eq=eq),
        FixPsi(eq=eq),
    )
    optimizer = Optimizer("proximal-lsq-exact")
    eq_new, _ = eq.optimize(
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
    return eqfam


t_total = time.perf_counter()
for n in range(1, M + 1):
    print(f"\n===== optimizing boundary modes max(|m|,|n|) <= {n} =====\n")
    t_step = time.perf_counter()
    eqfam = run_step(n, eqfam)
    print(f"Step {n} wall time: {time.perf_counter() - t_step:.1f} s")
print(f"\nTotal optimization wall time: {time.perf_counter() - t_total:.1f} s")

eqfam.save(f"{folder}/QH_LCFS_(unnorm_f_qs)_output.h5")

for i, eqi in enumerate(eqfam):
    report(eqi, f"step {i}")
