import sys
import os
import warnings

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../"))

import numpy as np
import matplotlib.pyplot as plt

from desc.equilibrium import Equilibrium
from desc.objectives import BoundaryRSelfConsistency, SectionRSelfConsistency

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 12,
    }
)

savedir = "../draft-images"
os.makedirs(savedir, exist_ok=True)

NFP = 5
Ms = [4, 8, 12, 16]
Ns = np.arange(0, 9)
Nc = np.linspace(0, 8, 400)


def measured(M, N, sym):
    """Rows that DESC actually puts in A for the two boundary conditions, L=M."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq = Equilibrium(L=M, M=M, N=N, NFP=NFP, sym=sym, spectral_indexing="ansi")
        section = SectionRSelfConsistency(eq)
        boundary = BoundaryRSelfConsistency(eq)
        section.build(verbose=0)
        boundary.build(verbose=0)
    return section.dim_f, boundary.dim_f, eq.R_basis.num_modes


def formula(M, N, sym):
    P = (M + 2) ** 2 / 4
    Q = M * (M + 2) / 4
    if sym:
        return P * np.ones_like(N, dtype=float), 2 * M * N + M + N + 1.0
    return (P + Q) * np.ones_like(N, dtype=float), (2 * M + 1) * (2 * N + 1.0)


def crossover(M, sym):
    if sym:
        return M**2 / (4 * (2 * M + 1))
    return (M + 1) * (M + 2) / (4 * (2 * M + 1)) - 0.5


fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharex=True)
colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.8, len(Ms)))

err = 0
for ax, sym in zip(axes, [True, False]):
    for M, c in zip(Ms, colors):
        Np, Nb = formula(M, Nc, sym)
        ax.plot(Nc, Np, "--", color=c, lw=1.5)
        ax.plot(Nc, Nb, "--", color=c, lw=1.5)

        mp, mb = np.array([measured(M, n, sym)[:2] for n in Ns]).T
        fp, fb = formula(M, Ns, sym)
        err = max(err, np.abs(mp - fp).max(), np.abs(mb - fb).max())
        ax.plot(Ns, mp, "o", color=c, mfc="none", ms=9, mew=2, label=f"$M={M}$")
        ax.plot(Ns, mb, "s", color=c, mfc="none", ms=9, mew=2)

        Nx = crossover(M, sym)
        ax.plot(Nx, formula(M, Nx, sym)[0], "*", color=c, ms=20)
        ax.axvline(Nx, color=c, ls=":", lw=1)
        print(f"sym={sym} M={M}: crossover at N = {Nx:.3f}")

    ax.set_yscale("log")
    ax.set_xlabel("$N$")
    ax.set_title("stellarator symmetric" if sym else "no symmetry")
    ax.grid(alpha=0.3)

axes[0].set_ylabel("number of constraints on $R$")
handles = [
    plt.Line2D([], [], ls="--", color="gray", label="formula"),
    plt.Line2D([], [], ls="", marker="o", mfc="none", color="gray", label="Poincaré"),
    plt.Line2D([], [], ls="", marker="s", mfc="none", color="gray", label="LCFS"),
    plt.Line2D([], [], ls="", marker="*", color="gray", label="crossover"),
]
axes[0].legend(loc="upper left", ncol=2)
axes[1].legend(handles=handles, loc="upper left")
fig.suptitle("Linear constraints on $R$ for $L=M$, markers measured in DESC")
fig.tight_layout()
plt.savefig(f"{savedir}/constraint-counts.png", dpi=300, bbox_inches="tight")

print(f"\nmax |formula - DESC| over all M, N: {err}")
for M in Ms:
    for sym in [True, False]:
        mp, mb, nc = measured(M, 6, sym)
        print(f"sym={sym} M={M} N=6: poincare {mp}, lcfs {mb}, coefficients {nc}")
