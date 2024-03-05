# Plotting Cp statistics from full-scale measurements and LES
This repository contains scripts and helper functions for plotting full-scale and LES Cp statistics together, using the plotly library. 

## Data sources
The experimental results are input in [util/SN_exp_params.py](util/SN_exp_params.py) and [util/Cal650_exp_params.py](util/Cal650_exp_params.py), for the Space Needle and 650 California data, respectively. Each file contains a dictionary of format:
```
deployment: {results_file, motes : {name, location}}
```
Where the results file is a .mat that has been generated using the relevant Cp_calc_*.m function for either [the Space Needle](github.com/jhochs/Cp_measurements_MATLAB/Cp_calc_SN_xlsinput.m) or [650 California](github.com/jhochs/Cp_measurements_MATLAB/Cp_calc_650Cal.m).

The path to the LES results are specified towards the beginning of the main scripts [SN_LES_FS_comparison_dCp.py](SN_LES_FS_comparison_dCp.py) and [Cal650_LES_FS_comparison_dCp.py](Cal650_LES_FS_comparison_dCp.py) and are also assumed to be in .mat files exported by [LES_results.m](github.com/jhochs/CharLES_MATLAB/LES_results.m).

## Changes in code
You must change the directories to match what they are on your machine. As stated in the section above, these are in the paths to LES results in the scripts and to the experimental results in the *_exp_params.py files.  