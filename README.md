# Usage

The scripts can be put into three main parts: Gillespie, plotting, and plot analysis. They are as follows.

## Gillespie
These are for the simulations of downstream molecules in signalling, used for verification of the calculations numerical methods employed.
1. 'settings.py','params.py', 'formulae.py', 'trajectory_analysis.py', 'trajectory_plotting.py', 'trajectory_simulate.py' are used for the actual simulations. I borrowed these from https://github.com/uoftbiophysics/cellsignalling (June 2020 versions) and added minor modifications to running the non-pleiotropic model.
NOTE: 'params.py' now saves the variables into the 'gill_var.npz' file so that I can remember which ones were used. An unintended consequence is that this file is save everytime a script imports 'params.py'.
2. sd
