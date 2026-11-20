"""Two-term QS objective and error, normalized as in Eq. 3.6 of the paper."""

import os
import subprocess

import numpy as np

from desc.backend import print_backend_info
from desc.grid import LinearGrid
from desc.integrals import surface_averages
from desc.objectives import ObjectiveFromUser, QuasisymmetryTwoTerm

RHO_QS = np.array([0.6, 0.8, 1.0])


def print_run_info():
    """Print branch and commit of the DESC and paper repos, then the JAX backend."""
    here = os.path.dirname(os.path.abspath(__file__))
    for name, path in [("DESC", os.path.join(here, "../..")), ("Paper", here)]:
        git = ["git", "-C", path]
        branch = subprocess.run(
            git + ["rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        commit = subprocess.run(
            git + ["log", "-1", "--format=%h %ci %s"], capture_output=True, text=True
        ).stdout.strip()
        print(f"{name} repo: branch {branch}, commit {commit}")
    print_backend_info()


def report(eq, tag):
    """Print global quantities and the QS error at the objective surfaces."""
    _, f_qs = plot_qs_error_fsa(eq, rho=RHO_QS)
    d = eq.compute(["V", "R0", "a", "<|B|>_vol", "<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol"])
    print(
        f"{tag:9s} V = {d['V']:8.3f}  R0/a = {d['R0'] / d['a']:6.3f}  "
        f"<|B|> = {d['<|B|>_vol']:7.4f}  "
        f"<|F|>_n = {d['<|F|>_vol'] / d['<|grad(|B|^2)|/2mu0>_vol']:.3e}  "
        f"f_qs = {np.array2string(f_qs, precision=4)}"
    )


def qs_grid(eq):
    """Grid the QS objective is evaluated on, shared by all optimizations."""
    return LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=RHO_QS, sym=True)


def make_qs(eq, weight=1.0, normalized=True):
    """Two-term QS objective on ``qs_grid(eq)``.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to optimize.
    weight : float
        Objective weight.
    normalized : bool
        If True, divide f_C by the flux-surface averaged |B|^3 of the current
        iterate (Eq. 3.6), otherwise use the unnormalized DESC two-term metric.

    Returns
    -------
    objective : _Objective
        QS objective.

    """
    grid = qs_grid(eq)
    helicity = (1, eq.NFP)
    if not normalized:
        return QuasisymmetryTwoTerm(
            eq=eq, helicity=helicity, grid=grid, normalize=False, weight=weight
        )

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


def plot_qs_error_fsa(
    eq, rho=10, ax=None, label=None, color=None, lw=1, marker="o", log=True
):
    """Flux-surface averaged |f_C| / <|B|>^3, with <|B|> the flux-surface average.

    Same averaging as ``desc.plotting.plot_qs_error`` but normalized per surface by
    the flux-surface averaged field, matching the objective used in the
    optimizations, instead of the volume-averaged field.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to evaluate.
    rho : int or array-like
        Surfaces to evaluate on. An int gives that many equispaced surfaces in (0, 1].
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Nothing is plotted if None.
    label, color, lw, marker : optional
        Line properties.
    log : bool
        Use a log y-axis.

    Returns
    -------
    rho : ndarray
        Flux surface labels.
    f_qs : ndarray
        Normalized two-term QS error on each surface.

    """
    if np.isscalar(rho) and rho > 1:
        rho = np.linspace(1, 0, num=int(rho), endpoint=False)
    rho = np.sort(np.atleast_1d(rho))
    grid = LinearGrid(M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP, rho=rho, sym=False)
    data = eq.compute(["f_C", "|B|", "sqrt(g)"], grid=grid, helicity=(1, eq.NFP))
    fsa = lambda x: surface_averages(grid, x, sqrt_g=data["sqrt(g)"], expand_out=False)
    f_qs = np.asarray(fsa(np.abs(data["f_C"])) / fsa(data["|B|"]) ** 3)
    if ax is not None:
        plot = ax.semilogy if log else ax.plot
        plot(rho, f_qs, color=color, lw=lw, marker=marker, label=label)
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$f_{qs}$")
    return rho, f_qs
