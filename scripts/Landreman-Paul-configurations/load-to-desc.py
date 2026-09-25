"""Load and create a DESC equilibrium of Landreman&Paul 2022 QH configuration."""

import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../../"))

from desc import set_device

set_device("gpu")

from desc.backend import print_backend_info
from desc.vmec import VMECIO

print_backend_info()

file = "./new_QH/wout_20210704-01-063_nfp4_QH_A8_weight_2.00e+00_rel_step_1.00e-05_forward_B1.nc"
eq = VMECIO.load(file, L=8, M=8, N=8, profile="current")

# loaded eq is just a fit, solve to get actual equilibrium
eq.solve(maxiter=1000, ftol=1e-6, gtol=1e-8, xtol=1e-8, verbose=3)
eq.save(f"../results/landreman_paul_precise_QH_output.h5")
