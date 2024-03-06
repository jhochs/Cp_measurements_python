import numpy as np
import pandas as pd
import plotly.graph_objects as go

import util.common_funcs as common
import util.SN_funcs as SN
import util.SN_exp_params as exp_params

pd.options.mode.chained_assignment = None  # default='warn'

'''
To generate .mats:
LES results - run this script LESResults.m
FS results - run the script Cp_calc_xlsinput.m
'''

#====================================================================================
# SETTINGS
# LES_results_paths = ['/Users/jackhochschild/Dropbox/School/Wind_Engineering/CFD/CharLES/SN/Results/SN_RL4_5_x0_5stresses/',
#                      '/Users/jackhochschild/Dropbox/School/Wind_Engineering/CFD/CharLES/SN/Results/SN_RL4_5_x4stresses/',
#                      '/Users/jackhochschild/Dropbox/School/Wind_Engineering/CFD/CharLES/SN/Results/SN_RL4_5_x4stresses_x2Lint/']
# LES_Iu = [0.046, 0.084, 0.1]  # LES turbulence intensity as measured by rooftop anemometer

LES_results_paths = ['/Users/jackhochschild/Dropbox/School/Wind_Engineering/CFD/CharLES/SN/Results/SN_smooth_RL4_5_x0_5stresses/',
                     '/Users/jackhochschild/Dropbox/School/Wind_Engineering/CFD/CharLES/SN/Results/SN_smooth_RL4_5/',
                     '/Users/jackhochschild/Dropbox/School/Wind_Engineering/CFD/CharLES/SN/Results/SN_smooth_RL4_5_x4stresses_x2Lint/']
LES_Iu = [0.051, 0.087, 0.120]  # LES turbulence intensity as measured by rooftop anemometer

minU = 6  # windspeed after correction, i.e. freestream threshold
dCpmin_maxRange = 0.4 # exclude datapoints where range between measurements from different sensors on mote exceeds this value
dCprms_maxRange = 0.04
exclude_third_outlier = True
exclude_lowWS = True
exclude_highRangeStats = True

cbound_percentiles = [10, 90]  # used for both binning and regular plot colorbar
n_bins = 3  # if plotting by binning turbulence intensity

# Plot options
roofs = ['sloped', 'flat']
stats = ['dCprms', 'dCpmin']
color_choice = 'TurbIntensity' # None | 'TurbIntensity' | 'Lux' | 'eta' ; etc.
cmap_choice = 'Hot'
plot_meas = True
plot_LES = True
show_region_divides = True
save_path = '../Plots/Space_Needle/LES_FS_dCp.html'
# for more plot control, uncomment/comment function calls on lines 98-105

#====================================================================================
# IMPORT LES RESULTS AND EXTRACT STATS ALONG PERIMETER
Cp_flat_perim = []
Cp_sloped_perim = []
for path in LES_results_paths:
  Cp_sloped_perim.append(SN.get_perimeter_results(path + 'sloped_roof_Cpstats.mat', 'sloped_roof'))
  Cp_flat_perim.append(SN.get_perimeter_results(path + 'flat_roof_Cpstats.mat', 'flat_roof'))
  
#====================================================================================
# IMPORT MEAS RESULTS
meas_data = SN.prepare_meas_data(exp_params.FS, exp_params.WS_correction, exp_params.WDir_correction, minU, dCprms_maxRange, dCpmin_maxRange, exclude_third_outlier)

# Initial filtering:
if exclude_lowWS:
  meas_data = meas_data[meas_data['Above WS threshold'] == 1]
if exclude_highRangeStats:
  meas_data = meas_data[meas_data['Above Cp range threshold'] == 0]

# Calculate gust factor:
for i in [1, 2, 3]:
    meas_data['g_' + str(i)] = np.abs(meas_data['dCpmin_' + str(i)]) / meas_data['dCprms_' + str(i)] 

# For exporting all data in a single csv:
# meas_data.drop(columns=['i', 'outlier_idx', 'Above WS threshold', 'Above Cp range threshold'], inplace=True)
# meas_data.rename(columns={'Sensors online':'Sensors_online', 'Degrees from WDir':'Degrees_from_WDir'}, inplace=True)
# meas_data.to_csv('~/Dropbox/School/Wind_Engineering/Sensor_Network/Code/Data/SN_Cpstats_all_WScorrected.csv')
# exit()

# Convert range to [0,180]:
# meas_data['Degrees from WDir'] = np.abs(((meas_data['Degrees from WDir'] + 180) % 360) - 180)

meas_data_binned = SN.bin_by(meas_data)

if color_choice is not None:
  cbounds = [np.percentile(meas_data[color_choice], cbound_percentiles[0]), np.percentile(meas_data[color_choice], cbound_percentiles[1])]
  cbounds[1] = 0.13 # hardcoded since otherwise some datapoints appear too white
  bins = np.linspace(cbounds[0], cbounds[1], n_bins+1)
  meas_data_Iu_binned = SN.bin_by(meas_data, grouping=color_choice, bins=bins)
else:
  cbounds = []

#====================================================================================
# PLOT MEASUREMENTS
  
fig = common.init_subplots(len(stats), len(roofs))
legend_label, cbar_label = common.labels(color_choice)

if plot_meas:
  SN.plot_meas_points(fig, meas_data, roofs, stats, color=color_choice, cmap=cmap_choice, cbounds=cbounds)
  
  plot_mean = True
  plot_CI = True
  # SN.plot_meas_mean_CI(fig, meas_data_binned, roofs, stats, [plot_mean, plot_CI])

  # SN.plot_meas_intervals(fig, meas_data_Iu_binned, roofs, stats, color_choice, cmap_choice, cbounds, [plot_mean, plot_CI])
  # common.bins_legend(fig, bins, cmap_choice, cbounds, legend_label)

#====================================================================================
# PLOT LES
  
if plot_LES:
  for i in range(len(LES_Iu)):
    # Plot EACH LES run with the following lines (number of list elements = number of lines):
    line_styles = ['solid', 'solid']
    line_widths = [4, 2]
    if len(cbounds) == 0:
      line_colors = ['white', 'black']
    else:
      line_colors = ['white', common.get_color(cmap_choice, cbounds, LES_Iu[i])]

    for j in range(len(stats)):
      for k in range(len(line_styles)):
        fig.add_trace(go.Scatter(x=Cp_sloped_perim[i]['Degrees'],
                                y=Cp_sloped_perim[i][stats[j]],
                                mode='lines',
                                line_color=line_colors[k],
                                line_width=line_widths[k],
                                line=dict(dash=line_styles[k]),
                                showlegend=False, 
                                hoverinfo='none'),
                    row=1+j, col=1)

        # If applicable, plot flat roof as well:
        if len(roofs)==2:
          fig.add_trace(go.Scatter(x=Cp_flat_perim[i]['Degrees'],
                                  y=Cp_flat_perim[i][stats[j]],
                                  mode='lines',
                                  line_color=line_colors[k],
                                  line_width=line_widths[k],
                                  line=dict(dash=line_styles[k]),
                                  showlegend=False, 
                                  hoverinfo='none'),
                      row=1+j, col=2)


#====================================================================================
# FINE TUNING FIGURE

SN.set_figure_size(fig, len(stats), len(roofs))

if show_region_divides:
  # For showing delineation between flow regions
  SN.add_regions(fig, len(stats))

SN.set_xyranges(fig, stats, roofs, meas_data)

# Add axis lines:
fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

common.add_axis_labels(fig, stats, roofs, 'Degrees along perimeter, from wind direction')

if color_choice is not None:
  # Add colorbar:
  fig.update_traces(marker=dict(colorbar={'title': {'text' : cbar_label, 'font': {'size': 24, 'family': 'Arial'}}}, colorbar_x=0.46), row=1, col=1)

fig.write_html(save_path)
