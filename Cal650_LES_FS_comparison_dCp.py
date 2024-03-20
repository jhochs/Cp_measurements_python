import numpy as np
import pandas as pd
import plotly.graph_objects as go

import util.common_funcs as common
import util.Cal650_funcs as Cal650
import util.Cal650_exp_params as exp_params
import util.aggregation_funcs as agg_fx

pd.options.mode.chained_assignment = None  # default='warn'

#====================================================================================
# SETTINGS

minU = 5.7
dCpmin_maxRange = 0.2 # exclude datapoints where range between measurements from different sensors on mote exceeds this value
dCprms_maxRange = 0.03 
exclude_third_outlier = True  # doesn't apply for tethered motes
exclude_lowWS = True
exclude_highRangeStats = False  # doesn't apply for tethered motes
WDir_range = [248, 266]
# Note that LES WDir range is [255, 258] and Iu = [0.68, 0.73]

cbound_percentiles = [3, 97]  # used for both binning and regular plot colorbar
n_bins = 3  # if plotting by binning turbulence intensity

agg_Iu_range = [0.68, 0.73] # for aggregation only
agg_ranges_method = 'MC_ranges' # MC_ranges | LES_ranges

LES_results_paths = ['/Users/jackhochschild/Dropbox/School/Wind_Engineering/CFD/CharLES/650Cal/270deg/650Cal_RL1_5/']
wind_angles = [270]
wind_angles_corr = [256]

# LES turbulence intensity as measured by rooftop anemometer:
LES_Iu = [0.71]

# Plot options
types = ['tethered']  # ['tethered', 'onboard']
stats = ['dCprms', 'dCpmin']  # ['dCprms', 'dCp_skewness', 'dCp_kurtosis', 'dCpmin', 'g']
color_choice = None  # None | 'TurbIntensity_x' | 'Lux' | 'eta' ; etc.
cmap_choice =  'oranges' # 'Haline' | 'thermal'
plot_meas = True
plot_agg_meas = True
plot_LES = True
plot_LES_uncertainty = True
save_path = '../Plots/650Cal/LES_FS_dCp_agg.html'
# for more plot control, uncomment/comment function calls on lines 106-116

#====================================================================================
# IMPORT LES RESULTS
LES_side = []
LES_top = []
for i in range(len(LES_results_paths)):
    side_mat = LES_results_paths[i] + 'parapet_side_Cpstats.mat'

    if plot_LES_uncertainty:
      LES_side.append(Cal650.get_perimeter_results_with_ranges(side_mat, 'side', wind_angles[i]))
    else:
      LES_side.append(Cal650.get_perimeter_results(side_mat, 'side', wind_angles[i]))

    if len(types)==2:
      top_mat = LES_results_paths[i] + 'parapet_top_Cpstats.mat'  
      if plot_LES_uncertainty:
        LES_top.append(Cal650.get_perimeter_results_with_ranges(top_mat, 'top', wind_angles[i]))
      else:
        LES_top.append(Cal650.get_perimeter_results(top_mat, 'top', wind_angles[i]))   

#====================================================================================
# IMPORT & PROCESS MEASUREMENT RESULTS
meas_data = Cal650.prepare_meas_data(exp_params.FS, exp_params.WS_correction, exp_params.WDir_correction, minU, dCprms_maxRange, dCpmin_maxRange, exclude_third_outlier)

# Exclude any obvious outliers (since no outlier/range filtering for tethered):
for i in [1, 2, 3]:
  meas_data.loc[meas_data['dCprms_' + str(i)] > 2, meas_data.filter(regex='_' + str(i), axis=1).columns] = np.nan

# Initial filtering:
if exclude_lowWS:
  meas_data = meas_data[meas_data['Above WS threshold'] == 1]
if exclude_highRangeStats:
  meas_data = meas_data[meas_data['Above Cp range threshold'] == 0]

# Filter by wind direction
meas_data = meas_data[np.logical_and(meas_data['WDiravg'] > WDir_range[0], meas_data['WDiravg'] < WDir_range[1])]

# Calculate gust factor:
for i in [1, 2, 3]:
  meas_data['g_' + str(i)] = np.abs(meas_data['dCpmin_' + str(i)]) / meas_data['dCprms_' + str(i)] 

# For exporting all data in a single csv:
# meas_data.drop(columns=['i', 'outlier_idx', 'Above WS threshold', 'Above Cp range threshold', 'dCprms_range', 'dCpmin_range', 'conv_dCpmin_1', 'conv_dCpmin_2', 'conv_dCpmin_3'], inplace=True)
# meas_data.rename(columns={'Sensors online':'Sensors_online'}, inplace=True)
# meas_data.to_csv('~/Desktop/SF_building_data/SFbldg_Cpstats_all_WScorrected.csv')
# exit()

meas_data_binned = Cal650.bin_by(meas_data)

if color_choice is not None:
  cbounds = [np.percentile(meas_data[color_choice], cbound_percentiles[0]), np.percentile(meas_data[color_choice], cbound_percentiles[1])]
  bins = np.linspace(cbounds[0], cbounds[1], n_bins+1)
  meas_data_Iu_binned = Cal650.bin_by(meas_data, grouping=color_choice, bins=bins)
else:
  cbounds = []

if plot_agg_meas:
  df_long = Cal650.wide_to_long(meas_data[meas_data['Type'] == 'tethered'])
  df_agg = agg_fx.perform_aggregation(df_long, agg_Iu_range, 6, agg_ranges_method)

#====================================================================================
# PLOT MEASUREMENTS

fig = common.init_subplots(len(stats), len(types))
legend_label, cbar_label = common.labels(color_choice)

if plot_meas:
  if plot_agg_meas:
    agg_fx.plot_agg_meas_points(fig, df_agg, stats, color='N_windows', cmap=cmap_choice)
  else:
    Cal650.plot_meas_points(fig, meas_data, types, stats, color=color_choice, cmap=cmap_choice, cbounds=cbounds)
  # plot_mean = True
  # plot_CI = True
  # Cal650.plot_meas_mean_CI(fig, meas_data_binned, types, stats, [plot_mean, plot_CI])

  # Cal650.plot_meas_intervals(fig, meas_data_Iu_binned, types, stats, color_choice, cmap_choice, cbounds, [plot_mean, plot_CI])
  # common.bins_legend(fig, bins, cmap_choice, cbounds, legend_label)

#====================================================================================
# PLOT LES

if plot_LES:
  for i in range(len(LES_Iu)): # LES runs
    # Plot EACH LES run with the following lines (number of list elements = number of lines):
    line_styles = ['solid']
    line_widths = [3]
    if len(cbounds) == 0:
      line_colors = ['black']
    else:
      line_colors = [common.get_color(cmap_choice, cbounds, LES_Iu[i])]

    for j in range(len(stats)):
      for k in range(len(line_styles)):
        # Plot parapet side:
        x_data = LES_side[i]['from_NW']
        y_data = LES_side[i][stats[j]]

        fig.add_trace(go.Scatter(x=x_data,
                                y=y_data,
                                mode='lines',
                                line_color=line_colors[k],
                                line_width=line_widths[k],
                                line=dict(dash=line_styles[k]),
                                showlegend=False, 
                                hoverinfo='none'),
                    row=1+j, col=1)
        
        if plot_LES_uncertainty:
          if stats[j] + '_deltaplus' not in LES_side[i].columns:
            raise ValueError('Specified LES data does not report uncertainties')
          
          errplus = LES_side[i][stats[j] + '_deltaplus']
          errminus = LES_side[i][stats[j] + '_deltaminus']
          common.plot_CIs(fig, 1+j, 1, x_data, y_data, errplus, errminus, color=line_colors[k])
        
        # If applicable, plot parapet top:
        if len(types)==2:
          x_data = LES_top[i]['from_NW']
          y_data = LES_top[i][stats[j]]

          fig.add_trace(go.Scatter(x=x_data,
                                  y=y_data,
                                  mode='lines',
                                  line_color=line_colors[k],
                                  line_width=line_widths[k],
                                  line=dict(dash=line_styles[k]),
                                  showlegend=False, 
                                  hoverinfo='none'),
                      row=1+j, col=2)
  
#====================================================================================
# FINE TUNING FIGURE

Cal650.set_figure_size(fig, len(stats), len(types))
Cal650.add_corners(fig, len(stats), len(types))
Cal650.set_yranges(fig, stats, types, meas_data)
  
# Add axis lines:
fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

common.add_axis_labels(fig, stats, types, 'Position [m]')

# Add colorbar:
if color_choice is not None:
  fig.update_traces(marker=dict(colorbar={'title': {'text' : cbar_label, 'font': {'size': 24, 'family': 'Arial'}}, 'len':0.5, 'y':0.77}), row=1, col=1) #, colorbar_x=0.46
if plot_agg_meas is True:
  fig.update_traces(marker=dict(colorbar={'title': {'text' : 'N<sub>windows</sub>', 'font': {'size': 24, 'family': 'Arial'}}}), row=1, col=1)


config = {
  'toImageButtonOptions': {
    'format': 'png', # one of png, svg, jpeg, webp
    'scale':4 # Multiply title/legend/axis/canvas sizes by this factor
  }
}
fig.write_html(save_path, config=config)