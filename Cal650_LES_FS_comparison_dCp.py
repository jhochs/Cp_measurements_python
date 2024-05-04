import os
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
WDir_range = [246, 266]
# WDir_range = [246, 277]
# WDir_range = np.array([[255, 265], [265, 275]])
# Note that 270deg LES WDir range is [255, 258] and Iu = [0.68, 0.73]

cbound_percentiles = [3, 97]  # used for both binning and regular plot colorbar
n_bins = 3  # if plotting by binning turbulence intensity

agg_Iu_range = [0.68, 0.74] # for aggregation only
agg_ranges_method = 'MC_ranges' # MC_ranges | LES_ranges

LES_results_paths = ['/Users/jackhochschild/Dropbox/School/Wind_Engineering/CFD/CharLES/650Cal/270deg/650Cal_RL1_5/']
# LES_results_paths = ['/Users/jackhochschild/Dropbox/School/Wind_Engineering/CFD/CharLES/650Cal/270deg/650Cal_RL1_5/',
#                      '/Users/jackhochschild/Dropbox/School/Wind_Engineering/CFD/CharLES/650Cal/280deg/650Cal_280deg_RL1_5/']
wind_angles = [270]
wind_angles_corr = [256, 267]

# LES turbulence intensity as measured by rooftop anemometer:
LES_Iu = [0.71] #[0.71, 0.57]

# Plot options
types = ['tethered']  # ['tethered', 'onboard']
stats = ['dCprms', 'dCpmin']  # ['dCprms', 'dCp_skewness', 'dCp_kurtosis', 'dCpmin', 'g']
color_choice = 'TurbIntensity_x' # None | 'TurbIntensity_x' | 'WDiravg' | 'Lux' | 'eta' ; etc.
cmap_choice = 'Haline' # 'Haline' | 'thermal' | 'oranges'
plot_meas = True
plot_agg_meas = True
LES_plot_type = 'curve_and_range' # None | 'single_curve' | 'curve_and_range' | '10min_curves'
save_path = '../Plots/650Cal/LES_FS_dCp_agg_270.html'
# for more plot control, uncomment/comment function calls on lines 106-116

#====================================================================================
# IMPORT LES RESULTS
LES_side = []
LES_top = []

if LES_plot_type == '10min_curves':
  wind_angles_corr = []
  LES_Iu = []

for i in range(len(LES_results_paths)):
    side_mat = LES_results_paths[i] + 'parapet_side_Cpstats.mat'
    top_mat = LES_results_paths[i] + 'parapet_top_Cpstats.mat'

    if LES_plot_type == 'single_curve':
      LES_side.append(Cal650.get_perimeter_results(side_mat, 'side', wind_angles[i]))
      if len(types) == 2:
        LES_top.append(Cal650.get_perimeter_results(top_mat, 'top', wind_angles[i]))
    elif LES_plot_type == 'curve_and_range':
      LES_side.append(Cal650.get_perimeter_results_with_ranges(side_mat, 'side', wind_angles[i]))
      if len(types) == 2:
        LES_top.append(Cal650.get_perimeter_results_with_ranges(top_mat, 'top', wind_angles[i]))
    elif LES_plot_type == '10min_curves':
      mats = os.listdir(LES_results_paths[i] + 'probes_10min')
      for mat in mats:
          side_mat = LES_results_paths[i] + 'probes_10min/' + mat
          LES_side.append(Cal650.get_perimeter_results(side_mat, 'side', wind_angles[i]))
          LES_Iu.append(common.loadmat(side_mat)['probes']['Iu'])
          wind_angles_corr.append(common.loadmat(side_mat)['probes']['WDir'])

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
if isinstance(WDir_range, list):
  meas_data = meas_data[np.logical_and(meas_data['WDiravg'] > WDir_range[0], meas_data['WDiravg'] < WDir_range[1])]
  WDir_range = np.array(WDir_range)
else:
  meas_data = meas_data[np.logical_and(meas_data['WDiravg'] > WDir_range[0,0], meas_data['WDiravg'] < WDir_range[WDir_range.shape[0]-1,1])]

# Calculate gust factor:
for i in [1, 2, 3]:
  meas_data['g_' + str(i)] = np.abs(meas_data['dCpmin_' + str(i)]) / meas_data['dCprms_' + str(i)] 

# For exporting all data in a single csv:
# meas_data.drop(columns=['i', 'outlier_idx', 'Above WS threshold', 'Above Cp range threshold', 'dCprms_range', 'dCpmin_range', 'conv_dCpmin_1', 'conv_dCpmin_2', 'conv_dCpmin_3'], inplace=True)
# meas_data.rename(columns={'Sensors online':'Sensors_online'}, inplace=True)
# meas_data.to_csv('~/Desktop/SF_building_data/SFbldg_Cpstats_all_WScorrected.csv')
# exit()

meas_data_binned = Cal650.bin_by(meas_data)


if plot_agg_meas:
  df_long = Cal650.wide_to_long(meas_data[meas_data['Type'] == 'tethered'])
  df_agg = agg_fx.perform_aggregation(df_long, agg_Iu_range, 6, agg_ranges_method)
  df_agg['T'] = df_agg['N_windows'] * 10

if plot_agg_meas:
  cbounds = [df_agg['T'].min(), df_agg['T'].max()]
elif color_choice is not None:
  cbounds = [np.percentile(meas_data[color_choice], cbound_percentiles[0]), np.percentile(meas_data[color_choice], cbound_percentiles[1])]
  bins = np.linspace(cbounds[0], cbounds[1], n_bins+1)
  meas_data_Iu_binned = Cal650.bin_by(meas_data, grouping=color_choice, bins=bins)
else:
  cbounds = []

#====================================================================================
# PLOT MEASUREMENTS

fig = common.init_subplots(len(stats), len(types))
legend_label, cbar_label = common.labels(color_choice)

if plot_meas:
  if plot_agg_meas:
    agg_fx.plot_agg_meas_points(fig, df_agg, stats, color='T', cmap=cmap_choice, cbounds=cbounds)
  else:
    if LES_plot_type == '10min_curves':
      fill_ranges = {
                    'TurbIntensity_x' : [np.min(LES_Iu)-0.01, np.max(LES_Iu)+0.01],
                    'WDiravg': [np.min(wind_angles_corr)-1, np.max(wind_angles_corr)+1]
      }
    else:
      fill_ranges = None
    Cal650.plot_meas_points(fig, meas_data, types, stats, color=color_choice, cmap=cmap_choice, cbounds=cbounds, fill_ranges=fill_ranges, WDir_ranges=WDir_range)
  # plot_mean = True
  # plot_CI = True
  # Cal650.plot_meas_mean_CI(fig, meas_data_binned, types, stats, [plot_mean, plot_CI])

  # Cal650.plot_meas_intervals(fig, meas_data_Iu_binned, types, stats, color_choice, cmap_choice, cbounds, [plot_mean, plot_CI])
  # common.bins_legend(fig, bins, cmap_choice, cbounds, legend_label)

#====================================================================================
# PLOT LES

if LES_plot_type is not None:
  for i in range(len(LES_Iu)): # LES runs
    # Plot EACH LES run with the following lines (number of list elements = number of lines):
    if LES_Iu[i] > 0.6:
      line_styles = ['solid']
    else:
      line_styles = ['dot']
    line_widths = [1.5] # was [3]
    if plot_agg_meas:
      line_colors = [common.get_color(cmap_choice, cbounds, 60)]
    elif color_choice == 'TurbIntensity' or color_choice == 'TurbIntensity_x':
      line_colors = [common.get_color(cmap_choice, cbounds, LES_Iu[i])]
    elif color_choice == 'WDiravg':
      line_colors = [common.get_color(cmap_choice, cbounds, wind_angles_corr[i])]
    else:
      line_colors = ['black']

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
        
        if LES_plot_type == 'curve_and_range':
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
  fig.update_traces(marker=dict(colorbar={'title': {'text' : cbar_label, 'font': {'size': 24, 'family': 'Arial'}}, 'len':0.4, 'y':0.83}), row=1, col=1) #, colorbar_x=0.46
if plot_agg_meas is True:
  fig.update_traces(marker=dict(colorbar={'title': {'text' : 'T [min]', 'font': {'size': 24, 'family': 'Arial'}}, 'len':0.7, 'y':0.7}), row=1, col=1)


config = {
  'toImageButtonOptions': {
    'format': 'png', # one of png, svg, jpeg, webp
    'scale':4 # Multiply title/legend/axis/canvas sizes by this factor
  }
}
fig.write_html(save_path, config=config)