# Usage

The scripts can be put into three main parts: Gillespie, plotting, and plot analysis. They are as follows.

## Gillespie
These are for the simulations of downstream molecules in signalling, used for verification of the calculations numerical methods employed.
1. `settings.py`,`params.py`, `formulae.py`, `trajectory_analysis.py`, `trajectory_plotting.py`, `trajectory_simulate.py` are used for the actual simulations. I borrowed these from https://github.com/uoftbiophysics/cellsignalling (June 2020 versions) and added minor modifications to running the non-pleiotropic model.
**NOTE:** `params.py` now saves the variables into the `gill_var.npz` file so that I can remember which ones were used. An unintended consequence is that this file is save everytime a script imports `params.py`.
2. `run_gill.sh` is a bash script to run these the simulation and save into a date directory.

## Plotting
These are mainly used to save the arrays used for plotting. They also create plots for a first look.
1. `numeric.py` contains all the functions for the distributions, CRLB etc. **Note:** I called the CRLB functions `Fisher` even though they calculate the inverse, a convention I stuck to throughout.
2. `steady_plotting.py` plots the CRLB heat plots in 2D snippets of (x,y,z) space: the (horizontal,vertical) axis are determined alphabetically. This script saves the normal approx and analytic CRLB as `err_norm.npy` and `err_analytic.npy`, respectively.
3. `plot_newvar.py` does the same but in the variables used in the pleiotropy paper.
4. `run_plot.sh` is a bash script to run these plots. It creates a staging directory so that I can run many plots simultaneously, and saves the outputs in a time-stamped directory. `run_new_plot.sh` is the same, for plotting in the other variables.
5. `dynamical_plotting_server.py` is the script for non-steady plots based on analytic expression, parallelized due to slowness of serial computations. The array is saved as a `.npy` file. **Note:**when importing onto a server, make sure to bring the auxiliary files the scripts import: `settings`, `params`, `formulae`.

## Analysis
1. After the steady state arrays are saved, `./plotting/plot_analysis.py` is used for analysis. Same with `plot_newvar_analysis.py`.
2. `dynamical_plotting2.py` is for non-steady normal approx plots which are less computationally intensive and can be run locally. It imports the
analytic `.npy` file and also saves the approx in the same file format.
3. `dynamical_plotting.py` is used for analysis by loading the already generated arrays. Make sure to check that the loaded files are right.

In general, I think there a some redundancies in the files and functions within files, which could be made more efficient with more general functions.

Mathematica note books are also provided.
