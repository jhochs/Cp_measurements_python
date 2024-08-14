import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colorbar import ColorbarBase

import util.common_funcs as common
import util.Cal650_funcs as Cal650
import util.Cal650_exp_params as exp_params

pd.options.mode.chained_assignment = None  # default='warn'

#====================================================================================
# SETTINGS

minU = 5.7
dCpmin_maxRange = 0.2 # exclude datapoints where range between measurements from different sensors on mote exceeds this value
dCprms_maxRange = 0.03 
exclude_lowWS = True
WDir_range = [255, 268]

LES_results_paths = ['/Users/jackhochschild/Dropbox/School/Wind_Engineering/CFD/CharLES/650Cal/270deg/650Cal_RL1_5/',
                     '/Users/jackhochschild/Dropbox/School/Wind_Engineering/CFD/CharLES/650Cal/280deg/650Cal_280deg_RL1_5/']
wind_angles = [270, 280]
LES_Iu = [] #[0.71, 0.57] # leave as empty list if using 10min .mats
LES_WDir = []

types = ['tethered']
plot_LES_10min = True
plot_LES_uncertainty = False

#====================================================================================
# IMPORT LES RESULTS
LES_side = []
for i in range(len(LES_results_paths)):
    if plot_LES_10min:
       mats = os.listdir(LES_results_paths[i] + 'probes_10min/')
       for mat in mats:
          side_mat = LES_results_paths[i] + 'probes_10min/' + mat
          LES_side.append(Cal650.get_perimeter_results(side_mat, 'side', wind_angles[i]))
          LES_WDir.append(common.loadmat(side_mat)['probes']['WDir'])
          LES_Iu.append(common.loadmat(side_mat)['probes']['Iu'])
    else:
        side_mat = LES_results_paths[i] + 'parapet_side_Cpstats.mat'
        if plot_LES_uncertainty:
            LES_side.append(Cal650.get_perimeter_results_with_ranges(side_mat, 'side', wind_angles[i]))
        else:
            LES_side.append(Cal650.get_perimeter_results(side_mat, 'side', wind_angles[i]))

#====================================================================================
# IMPORT & PROCESS MEASUREMENT RESULTS
meas_data = Cal650.prepare_meas_data(exp_params.FS, exp_params.WS_correction, exp_params.WDir_correction, minU, dCprms_maxRange, dCpmin_maxRange, False)

# Exclude any obvious outliers (since no outlier/range filtering for tethered):
for i in [1, 2, 3]:
  meas_data.loc[meas_data['dCprms_' + str(i)] > 2, meas_data.filter(regex='_' + str(i), axis=1).columns] = np.nan

# Initial filtering:
if exclude_lowWS:
  meas_data = meas_data[meas_data['Above WS threshold'] == 1]

# Filter by wind direction
meas_data = meas_data[np.logical_and(meas_data['WDiravg'] > WDir_range[0], meas_data['WDiravg'] < WDir_range[1])]

# Tethered only
meas_data = meas_data[meas_data['Type'] == 'tethered']
df = Cal650.wide_to_long(meas_data)

#====================================================================================
# PLOT MEASUREMENTS

# Convert position to face (N,E,S,W)
face_edges = exp_params.edge_length / 100 * np.arange(0, 5)
face_bins = range(4)
face_names = ['North', 'East', 'South', 'West']
colors = ['red', 'green', 'blue', 'orange']

face_dict = dict(zip(face_bins, face_names))
color_dict = dict(zip(face_bins, colors))
df['Face id'] = df['Position'].apply(lambda x: common.find_idx(x, face_edges))
df['Face name'] = df['Face id'].map(face_dict)
df['Face color'] = df['Face id'].map(color_dict)

stats = ['dCprms', 'dCpskew', 'dCpkurt', 'dCpmin']
PCC = np.zeros((len(stats), len(face_names)))
labels = [r"$C_{p,rms}$", r"$C'_{p,skew}$", r"$C'_{p,kurt}$", r"$C'_{p,min}$"]
fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(12, 8.5), gridspec_kw={"left" : 0.06, "right" : 0.97, "top" : 0.96, "bottom" : 0.07, "wspace": 0.3})
xaxis_choice = 'TurbIntensity_x' #'WDiravg'
xlim = [df[xaxis_choice].min(), df[xaxis_choice].max()]
for i in range(len(stats)):
    if i in [0, 1, 3]:  # for Cprms, dCpskew, dCpmin, just take min and max:
        ylim = [df[stats[i]].min(), df[stats[i]].max()]
    elif i == 2:  # for dCpkurt, take the max to be the 98% percentile
        ylim = [df[stats[i]].min(), np.percentile(df[stats[i]], 98)]
    for j in range(len(face_names)):
        dff = df[df['Face name'] == face_names[j]]
        
        # ax[i,j].scatter(dff[xaxis_choice], dff[stats[i]], c=dff['Face color'], s=0.5*np.ones((dff.shape[0], 1)), alpha=0.3)
        ax[i,j].scatter(dff[xaxis_choice], dff[stats[i]], c='black', s=0.5*np.ones((dff.shape[0], 1)), alpha=0.3)
        
        ax[i,j].set_xlim(xlim)
        ax[i,j].set_ylim(ylim)

        x_fit = np.arange(np.min(dff[xaxis_choice]), np.max(dff[xaxis_choice]), 0.001)
        fit = np.polyfit(dff[xaxis_choice], dff[stats[i]], 1)
        y_fit = np.polyval(fit, x_fit)
        # ax[i,j].plot(x_fit, y_fit, '-', color=colors[j], linewidth=1.5)
        ax[i,j].plot(x_fit, y_fit, '-', color='black', linewidth=1.5)
        PCC[i,j] = np.corrcoef(dff[xaxis_choice], y=dff[stats[i]])[0,1]

        if i==0:
            ax[i,j].set_title(face_names[j], fontsize=15)
        if i==3:
            ax[i,j].set_xlabel(r'$I_u$', fontsize=14)
            # ax[i,j].set_xlabel(r'$\theta_{rooftop} [°]$', fontsize=14)
        if j==0:
            ax[i,j].set_ylabel(labels[i], fontsize=14)

#====================================================================================
# PLOT LES

# Move all LES dataframes into a single:
for i in range(len(LES_side)):
    cur = LES_side[i]
    cur['Iu'] = LES_Iu[i]
    cur['WDir'] = LES_WDir[i]
    if i==0:
        df_LES = cur
    else:
        df_LES = pd.concat((df_LES, cur))

# Filter out LES probes that are not near a measurement location:
meas_positions = df['Position'].unique()
df_LES['Near measurement position'] = df_LES['from_NW'].apply(lambda x: Cal650.within_range(x, meas_positions, 0.5))
df_LES = df_LES[df_LES['Near measurement position']]

# Convert position to face (N,E,S,W)
df_LES['Face id'] = df_LES['from_NW'].apply(lambda x: common.find_idx(x, face_edges))
df_LES['Face name'] = df_LES['Face id'].map(face_dict)
LES_stats = ['dCprms', 'dCp_skewness', 'dCp_kurtosis', 'dCpmin']
PCC_LES = np.zeros((len(LES_stats), len(face_names)))
for i in range(len(LES_stats)):
    for j in range(len(face_names)):
        dff = df_LES[df_LES['Face name'] == face_names[j]]
        
        ax[i,j].scatter(dff['Iu'], dff[LES_stats[i]], c='red', s=0.5*np.ones((dff.shape[0], 1)), alpha=0.3)

        x_fit = np.arange(np.min(dff['Iu']), np.max(dff['Iu']), 0.001)
        fit = np.polyfit(dff['Iu'], dff[LES_stats[i]], 1)
        y_fit = np.polyval(fit, x_fit[:len(x_fit)//2])
        ax[i,j].plot(x_fit[:len(x_fit)//2], y_fit, '-', color='red', linewidth=1.5)
        PCC_LES[i,j] = np.corrcoef(dff['Iu'], y=dff[LES_stats[i]])[0,1]

# plt.show()
# plt.savefig("../Plots/650Cal/LES_FS_dCp_vs_Iu_by_face.png", dpi=300)
plt.savefig("/Users/jackhochschild/Desktop/LES_FS_dCp_vs_Iu_by_face.png", dpi=300)

print('----------------------------------------------------------------')
print('Meas PCC:')
for i in range(len(stats)):
    print('%s: %.2f | %.2f | %.2f | %.2f' %(stats[i], PCC[i,0], PCC[i,1], PCC[i,2], PCC[i,3]))
print('LES PCC:')
for i in range(len(stats)):
    print('%s: %.2f | %.2f | %.2f | %.2f' %(stats[i], PCC_LES[i,0], PCC_LES[i,1], PCC_LES[i,2], PCC_LES[i,3]))
print('----------------------------------------------------------------')
