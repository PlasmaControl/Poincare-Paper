import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../"))

from desc import set_device

set_device("gpu")

from desc.io import load
from desc.backend import print_backend_info

print_backend_info()

eqp = load(f"./results/Scan-QH-Paper-tight/eqfam-L12M12N6-wqs3-war3-wvol0.h5")[-1]
eq_qh = load(f"./results/landreman_paul_precise_QH_output.h5")
eq_qh2 = load(f"./results/equivalent_precise_QH_output.h5")[-1]
eq_qh3 = load(f"./results/normalized_qs_precise_QH_output.h5")[-1]
eqs = [eqp, eq_qh, eq_qh2, eq_qh3]
labels = [
    "Poincare",
    "L&P_2022",
    "LCFS_(unnorm.f_qs)",
    "LCFS",
]

eqs_same_res = eqs.copy()
for eq, label in zip(eqs_same_res, labels):
    eq.change_resolution(L=12, M=12, N=8, L_grid=24, M_grid=24, N_grid=16)
    eq.solve(verbose=3, maxiter=1000, ftol=1e-4, gtol=1e-8, xtol=1e-8)
    eq.save(f"./results/same-res-solves/{label}_LM12_N8_tight_solve.h5")
