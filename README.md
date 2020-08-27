# Usage

The scripts can be put into three main parts: Gillespie, plotting, and plot analysis. They are as follows.

## Gillespie
These are for the simulations of downstream molecules in signalling, used for verification of the calculations numerical methods employed.
1. `settings.py`,`params.py`, `formulae.py`, `trajectory_analysis.py`, `trajectory_plotting.py`, `trajectory_simulate.py` are used for the actual simulations. I borrowed these from https://github.com/uoftbiophysics/cellsignalling (June 2020 versions) and added minor modifications to running the non-pleiotropic model.
**NOTE**: `params.py` now saves the variables into the `gill_var.npz` file so that I can remember which ones were used. An unintended consequence is that this file is save everytime a script imports `params.py`.
2. `run_gill.sh` is a bash script to run these the simulation and save into a date directory.

## Plotting
These are mainly used to save the arrays used for plotting. They also create plots for a first look.
1. ``
2. `steady_plotting.py` plots the CRLB heat plots in 2D snippets of (x,y,z) space: the (horizontal,vertical) axis are determined alphabetically. This script saves the normal approx and analytic CRLB as `err_norm.npy` and `err_analytic.npy`, respectively.
3. `plot_newvar.py` does the same but in the variables used in the pleiotropy paper.
4.
