import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

import util.common_funcs as common
import util.SN_funcs as SN
import util.SN_exp_params as exp_params

pd.options.mode.chained_assignment = None  # default='warn'

#====================================================================================
# SETTINGS
minU = 6 # windspeed after correction, i.e. freestream threshold
dCpmin_maxRange = 0.4 # exclude datapoints where range between measurements from different sensors on mote exceeds this value
dCprms_maxRange = 0.04
exclude_third_outlier = True
exclude_lowWS = True
exclude_highRangeStats = True

#====================================================================================
# IMPORT FS RESULTS
FS_data = SN.prepare_FS_data(exp_params.FS, exp_params.WS_correction,exp_params.WDir_correction, minU, dCprms_maxRange, dCpmin_maxRange, exclude_third_outlier)

# Calculate gust factor:
for i in [1, 2, 3]:
    FS_data['g_' + str(i)] = FS_data['dCpmin_' + str(i)]/ FS_data['dCprms_' + str(i)] 

# Filter:
df = FS_data[FS_data['Roof'] == 'sloped']
df = df[df['Above WS threshold'] == 1]
df = df[df['Above Cp range threshold'] == 0]
df = df[['Mote', 'Degrees from WDir', 'TurbIntensity', 'eta', 'dCprms_1', 'dCprms_2', 'dCprms_3', 'g_1', 'g_2', 'g_3', 'dCp_skewness_1', 'dCp_skewness_2', 'dCp_skewness_3', 'dCp_kurtosis_1', 'dCp_kurtosis_2', 'dCp_kurtosis_3']]
# df = df[['Mote', 'Degrees from WDir', 'TurbIntensity_x', 'TurbIntensity_z', 'Lux', 'Lwx', 'dCprms_1', 'dCprms_2', 'dCprms_3', 'dCpmin_1', 'dCpmin_2', 'dCpmin_3']]

# Reshape to get rid of _1, _2, _3:
df.reset_index(drop=True, inplace=True)
df['id'] = df.index
df = pd.wide_to_long(df, ['dCprms_', 'g_', 'dCp_skewness_', 'dCp_kurtosis_'], i='id', j='Sensor')
df.rename(columns={'dCprms_':'dCprms', 'g_':'g', 'dCp_skewness_':'dCp_skewness', 'dCp_kurtosis_':'dCp_kurtosis'}, inplace=True)
df.dropna(axis=0, inplace=True)

# Convert range to [0,180]:
df['Degrees from WDir'] = np.abs(((df['Degrees from WDir'] + 180) % 360) - 180)

# df_Gaussian = df[np.logical_and(np.abs(df['dCp_skewness']) < 0.5, df['dCp_kurtosis'] < 3.5)]
# print(df_Gaussian.shape[0] / df.shape[0] * 100)

# Group positions by flow region:
# Windward separation 0-->105 deg
# Leeward separation 130-->180 deg
# Attachment 105-->130  X165-->180 degX
df['Region'] = 'None'
df.loc[df['Degrees from WDir'] < 45, 'Region'] = 'Windward separation'
df.loc[np.logical_and(df['Degrees from WDir'] >= 45, df['Degrees from WDir'] < 125), 'Region'] = 'Attachment'
df.loc[df['Degrees from WDir'] >= 160, 'Region'] = 'Cylinder wake'
# df.loc[df['Degrees from WDir'] >= 165, 'Region'] = 'Attachment'

#====================================================================================
# IMPORT LES RESULTS
LES_results_paths = ['/Users/jackhochschild/Dropbox/School/Wind_Engineering/CFD/CharLES/SN/Results/SN_RL4_5_x0_5stresses/',
                     '/Users/jackhochschild/Dropbox/School/Wind_Engineering/CFD/CharLES/SN/Results/SN_RL4_5_x4stresses/',
                     '/Users/jackhochschild/Dropbox/School/Wind_Engineering/CFD/CharLES/SN/Results/SN_RL4_5_x4stresses_x2Lint/']
LES_Iu = [0.046, 0.084, 0.1]  # LES turbulence intensity as measured by rooftop anemometer

Cp_sloped_perim = []
for path in LES_results_paths:
    Cp_sloped_perim.append(SN.get_perimeter_results(path + 'sloped_roof_Cpstats.mat', 'sloped_roof'))

# Separate into regions:
for i in range(len(LES_results_paths)):
    # Convert range to [0,180]:
    Cp_sloped_perim[i]['Degrees'] = np.abs(((Cp_sloped_perim[i]['Degrees'] + 180) % 360) - 180)

    Cp_sloped_perim[i]['Region'] = 'None'
    Cp_sloped_perim[i].loc[Cp_sloped_perim[i]['Degrees'] < 45, 'Region'] = 'Windward separation'
    Cp_sloped_perim[i].loc[np.logical_and(Cp_sloped_perim[i]['Degrees'] >= 45, Cp_sloped_perim[i]['Degrees'] < 125), 'Region'] = 'Attachment'
    Cp_sloped_perim[i].loc[Cp_sloped_perim[i]['Degrees'] >= 160, 'Region'] = 'Cylinder wake'

#====================================================================================
# PLOT
'''
[counts, bin_edges] = np.histogram(df['g'])
print(bin_edges)
bin_center = bin_edges[:-1] + np.diff(bin_edges) / 2
plt.plot(bin_center, counts, 'k+')
ax = plt.gca()
ax.set_yscale('log')
'''
cm = plt.cm.get_cmap('brg')
marker_size = 3
'''
# Skewness and Kurtosis vs. Peak factor plot:
plt.rcParams['figure.figsize'] = [11, 5]
fig, ax = plt.subplots(1, 2)
# ax[0].plot(df['g'], df['dCp_skewness'], 'k+', markersize=2) 
ax[0].scatter(df['g'], df['dCp_skewness'], s=marker_size, c=df['Degrees from WDir'], marker='.', cmap=cm)
ax[0].set_xlabel(r'$g$', fontsize='14')
ax[0].set_ylabel(r"$\mathrm{Skew}[C'_p]$", fontsize='14')
ax[0].tick_params(axis='both', which='major', labelsize=14)

# ax[1].plot(df['g'], df['dCp_kurtosis'], 'k+', markersize=2)
sc = ax[1].scatter(df['g'], df['dCp_kurtosis'], s=marker_size, c=df['Degrees from WDir'], marker='.', cmap=cm)
ax[1].set_xlabel(r'$g$', fontsize='14')
ax[1].set_ylabel(r"$\mathrm{Kurt}[C'_p]$", fontsize='14')
ax[1].tick_params(axis='both', which='major', labelsize=14)

fig.subplots_adjust(right=0.8, wspace=0.3)
cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
cb = fig.colorbar(sc, cax=cbar_ax)
cb.ax.set_title('Degrees from\nwind direction\n[˚]')
cb.ax.tick_params(labelsize=13)
fig.savefig('../Plots/Space_Needle/FS_Plots/Skew_Kurt_vs_g_coloredbyWDir.png')
'''
# Skewness vs. kurtosis plot:
plt.rcParams['figure.figsize'] = [7, 5]
fig = plt.figure()
sc = plt.scatter(df['dCp_skewness'], df['dCp_kurtosis'], s=marker_size, c=df['Degrees from WDir'], marker='.', cmap=cm)
# sc = plt.scatter(df['dCp_skewness'], df['dCp_kurtosis'], s=marker_size, c=df['dCprms'], marker='.', cmap=cm)
# plt.plot([-0.5, -0.5, 0.5, 0.5, -0.5], [0, 3.5, 3.5, 0, 0], '--', color='darkgray')
plt.fill_between([-0.5, 0.5], [3.5, 3.5], alpha=0.2, color='black')

sigma2 = np.arange(0, 0.3, 0.001)
skew_logn = np.multiply(np.sqrt(np.exp(sigma2)-1), (np.exp(sigma2)+2))
kurt_logn = 3 + np.multiply((np.exp(sigma2)-1), (np.exp(3*sigma2)+3*np.exp(2*sigma2)+6*np.exp(sigma2)+6))
plt.plot(skew_logn, kurt_logn, 'k--')
plt.plot(-skew_logn, kurt_logn, 'k--')

plt.xlabel(r"$C'_{p,skew}$", fontsize='14')
plt.ylabel(r"$C'_{p,kurt}$", fontsize='14')
plt.tick_params(axis='both', which='major', labelsize=14)
cb = plt.colorbar(sc)
cb.ax.tick_params(labelsize=13)
cb.ax.set_title('Degrees from\nwind direction\n[˚]')
# cb.ax.set_title("C'_{p,rms}")
plt.subplots_adjust(top=0.85, bottom=0.15, right=1)
ax = plt.gca()
ax.set_xlim([-2, 2])
ax.set_ylim([1.5, 11])
fig.savefig('../Plots/Space_Needle/FS_Plots/Skew_vs_kurt_coloredbyWDir.png')


# Cprms vs. skewness, colored by region:
plt.rcParams['figure.figsize'] = [7, 5]
fig = plt.figure()
sc = plt.scatter(df['dCprms'], df['dCp_skewness'], s=marker_size, c=df['Degrees from WDir'], marker='.', cmap=cm)

sigma2 = np.arange(0, 0.3, 0.001)
fac = 0.5
Cprms_logn = np.sqrt(sigma2)
skew_logn = np.multiply(np.sqrt(np.exp(sigma2)-1), (np.exp(sigma2)+2))
plt.plot(Cprms_logn, skew_logn, 'k--')

plt.xlabel(r"$C_{p,rms}$", fontsize='14')
plt.ylabel(r"$\mathrm{Skew}[C'_p]$", fontsize='14')
plt.tick_params(axis='both', which='major', labelsize=14)
cb = plt.colorbar(sc)
cb.ax.tick_params(labelsize=13)
cb.ax.set_title('Degrees from\nwind direction\n[˚]')
plt.subplots_adjust(top=0.85, bottom=0.15, right=1)
ax = plt.gca()
# ax.set_xlim([-2, 2])
# ax.set_ylim([1.5, 11])
# plt.show()
fig.savefig('../Plots/Space_Needle/FS_Plots/Skew_vs_Cprms_coloredbyWDir.png')


# Cprms vs. Iu plot, colored by region:
plt.rcParams['figure.figsize'] = [10, 4.8]

regions = df['Region'].unique().tolist()
regions.remove('None')

fig, ax = plt.subplots(1, 2) #len(regions))
colors = ['green', 'blue']
# colors = ['red', 'green', 'blue']
x_fit = [0, 0.25]

regions = ['Windward separation', 'Cylinder wake']
for i in range(len(regions)):
    # Plot measurement data:
    # TI_cur = df.loc[df['Region'] == regions[i], 'TurbIntensity']
    eta_cur = df.loc[df['Region'] == regions[i], 'eta']
    Cprms_cur = df.loc[df['Region'] == regions[i], 'dCprms']

    ax[i].plot(eta_cur, Cprms_cur, '.', color=colors[i], markersize=5)
    ax[i].tick_params(axis='both', which='major', labelsize=14)
    # ax[i].set_xlabel(r'$I_u$', fontsize='14')
    ax[i].set_xlabel(r'$\eta$', fontsize='14')
    ax[i].title.set_text(regions[i])

    # Plot LES data:
    '''for j in range(len(LES_results_paths)):
        Cprms_LES = Cp_sloped_perim[j].loc[Cp_sloped_perim[j]['Region'] == regions[i], 'Cprms']
        TI_LES = LES_Iu[j] * np.ones(Cprms_LES.shape)

        ax[i].plot(TI_LES, Cprms_LES, '+', color='black', markersize=5)'''

    # Regression:
    # fit = np.polyfit(TI_cur, Cprms_cur, 1)
    fit = np.polyfit(eta_cur, Cprms_cur, 1)
    y_fit = np.polyval(fit, x_fit)
    ax[i].plot(x_fit, y_fit, '-', linewidth=2, color=colors[i])

    # Calculate and add correlation coefficient
    # coeff = np.corrcoef(TI_cur, y=Cprms_cur)
    coeff = np.corrcoef(eta_cur, y=Cprms_cur)
    ax[i].text(0.03, 0.48, ('Slope = %.2f\nPCC = %.2f' %(fit[0], coeff[0,1])), fontsize=14)

    ax[i].set_xlim([0.02, 0.25])
    ax[i].set_ylim([0.05, 0.55])

ax[0].set_ylabel(r"$C_{p,rms}$", fontsize='14')
# fig.savefig('../Plots/Space_Needle/Cprms_vs_Iu_colored_by_region_withLES.png')
fig.savefig('../Plots/Space_Needle/FS_plots/Cprms_vs_eta_colored_by_region.png')
'''
# Cprms vs. Iu plot for WM18 and CM17 only:
motes = ['WM4', 'CM17', 'WM18', 'WM8']
colors = ['red', 'blue', 'green', 'green']
plt.rcParams['figure.figsize'] = [14, 5]
fig, ax = plt.subplots(1, len(motes))
x_fit = [0.02, 0.25]

for i in range(len(motes)):
    TI_cur = df.loc[df['Mote'] == motes[i], 'TurbIntensity']
    Cprms_cur = df.loc[df['Mote'] == motes[i], 'dCprms']

    # Plot data:
    ax[i].plot(TI_cur, Cprms_cur, '+', color=colors[i], markersize=3)
    ax[i].tick_params(axis='both', which='major', labelsize=14)
    ax[i].set_xlabel(r'$I_u$', fontsize='14')
    ax[i].title.set_text(motes[i])

    # Regression:
    fit = np.polyfit(TI_cur, Cprms_cur, 1)
    y_fit = np.polyval(fit, x_fit)
    ax[i].plot(x_fit, y_fit, '-', linewidth=2, color=colors[i])

    # Calculate and add correlation coefficient
    coeff = np.corrcoef(TI_cur, y=Cprms_cur)
    ax[i].text(0.02, 0.43, ('Slope = %.2f\nPCC = %.2f' %(fit[0], coeff[0,1])), fontsize=14)

    ax[i].set_xlim([0, 0.27])
    ax[i].set_ylim([0.02, 0.5])

ax[0].set_ylabel(r"$C_{p,rms}$", fontsize='14')
fig.savefig('../Plots/Space_Needle/FS_Plots/Cprms_vs_Iu_by_mote.png')
'''