"""Test: W7-X rho/zeta force error including the LCFS re-solve of the Poincaré boundary."""

import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../"))

from desc import set_device

set_device("gpu")

import numpy as np
import matplotlib.pyplot as plt

from desc.io import load
from desc.grid import LinearGrid
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

folder = "../figures-test"
out = "/tmp/claude-1000/-CODES-DESC/3572c642-6bb3-4656-b651-cf4b3d132545/scratchpad"
eqs = {
    "LCFS": load("./equilibria/W7-X_output.h5")[-1],
    "Poincaré": load("./equilibria/poincare-resolve-w7x.h5"),
    "LCFS warm": load("./equilibria/poincare-resolve-w7x-solved.h5"),
}
colors = {"LCFS": "k", "Poincaré": "r", "LCFS warm": "b"}

fig, ax = plt.subplots(2, 1, figsize=(9, 12))
rho = np.linspace(0.05, 1, 40)
for name, eq_i in eqs.items():
    c = colors[name]
    zeta = np.linspace(0, 2 * np.pi / eq_i.NFP, 193)
    theta = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    grid = LinearGrid(rho=rho, theta=theta, zeta=zeta, NFP=eq_i.NFP, sym=False)
    data = eq_i.compute(["|F|", "sqrt(g)"], grid=grid)
    norm = eq_i.compute("<|grad(|B|^2)|/2mu0>_vol")["<|grad(|B|^2)|/2mu0>_vol"]
    F = grid.meshgrid_reshape(data["|F|"] / norm, "rtz")
    w = grid.meshgrid_reshape(np.abs(data["sqrt(g)"]), "rtz")
    zN = zeta * eq_i.NFP / (2 * np.pi)
    for a, x, axes in zip(ax, [rho, zN], [(1, 2), (0, 1)]):
        avg = (F * w).sum(axis=axes) / w.sum(axis=axes)
        a.semilogy(x, avg, color=c, lw=3, label=f"{name} avg")
        a.semilogy(x, F.max(axis=axes), color=c, lw=2, ls="--", label=f"{name} max")
    za = (F * w).sum(axis=(0, 1)) / w.sum(axis=(0, 1))
    ra = (F * w).sum(axis=(1, 2)) / w.sum(axis=(1, 2))
    print(
        f"{name:10s} vol avg = {(F * w).sum() / w.sum():.3e}  "
        f"zeta=0: {za[0]:.2e}  zeta=0.5: {za[96]:.2e}  ratio {za[0] / za[96]:.1f}  "
        f"rho=0.5: {ra[19]:.2e}  rho=1: {ra[-1]:.2e}"
    )
ax[0].set_xlabel(r"$\rho$")
ax[1].set_xlabel(r"$\phi \cdot N_{FP}/2\pi$")
for a in ax:
    a.set_ylabel(r"$|\mathbf{F}| / \langle |\nabla |B|^2 / (2\mu_0)| \rangle_{vol}$")
    lo, hi = np.log10(a.get_ylim())
    a.set_ylim(10 ** (hi - (hi - lo) / 0.7), 10**hi)
    a.legend(fontsize=13, framealpha=1.0, ncol=3, loc="lower center")
fig.tight_layout()
fig.savefig(f"{folder}/w7x-resolve-test.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{out}/w7x-resolve-test.png", dpi=60, bbox_inches="tight")
