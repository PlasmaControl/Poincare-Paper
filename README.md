This repo consists of the scripts that are used to obtain the optimization results in ``Applications of Poincaré Boundary Condition for 3D Ideal MHD Equilibrium and Optimization`` as well as the scripts to generate the figures. The exact versions of the dependencies used for the results are in ``dependencies.txt``. This work uses ``DESC`` stellarator optimization suite with additional changes. To be able run the scripts, you can use,
```
cd DESC/
git checkout poincare-bc-paper
```

``scripts/`` include all the scripts that are used to conduct optimizations and to plot the results. This folder also includes the printed stdout files in ```outs/``, and the resulting equilibria files in ``equilibria/``. ``Landreman-Paul-configurations/`` folder includes the script to load and solve the precise QH example from that paper, the data is obtained from the Zenodo page of the paper.
