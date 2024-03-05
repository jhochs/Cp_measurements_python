import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylustrator

import util.common_funcs as common
import util.Cal650_exp_params as FS_params

pd.options.mode.chained_assignment = None  # default='warn'

def coords_from_perim(pos, length, corners_x, corners_y, offset):
    idx = np.floor(np.divide(pos, length)).astype(int)
    remainder = pos % length

    cur_vec = [corners_x[idx+1] - corners_x[idx], corners_y[idx+1] - corners_y[idx]]
    cur_unit_vec = np.divide(cur_vec, L)  # [x,y] vector that points along the direction of the edge on which this point lies

    x = corners_x[idx] + cur_unit_vec[0] * remainder 
    y = corners_y[idx] + cur_unit_vec[1] * remainder

    if offset:
        offset_distance = 1.2
        if abs(cur_unit_vec[0]) > abs(cur_unit_vec[1]):  # this edge is along x, so offset in y
            y = y + offset_distance * cur_unit_vec[0]
        else:
            x = x - offset_distance * cur_unit_vec[1]

    return x, y
    

# Import and filter, keeping only necessary columns
FS_data = common.prepare_FS_data(FS_params.FS, 1.83, 0, 5, 0.03, 0.2, True)
FS_data = FS_data[FS_data['Above WS threshold'] == 1]
FS_data = FS_data[['Mote', 'Position', 'Type', 'i']]
FS_data = FS_data.groupby('Mote', as_index=False).agg({'Position' : 'first', 'Type' : 'first', 'i' : 'first'})
FS_data['Mote'] = FS_data['Mote'].apply(lambda x: x[:-2]).values

# Edge coordinates
L = FS_params.edge_length / 100
corners_x = [0, L, L, 0, 0, L]
corners_y = [L, L, 0, 0, L, L]

# Plot
pylustrator.start()
# First plot edges
fig = plt.figure(figsize=(8, 8))
plt.plot(corners_x, corners_y, '-', linewidth=25, color='lightgray', solid_capstyle='butt')

# Plot onboard motes

for i in [0, 1]:
    df = FS_data[np.logical_and(FS_data['Type'] == 'onboard', FS_data['i'] == i)]
    # df['Position'] = df['Position'].apply(lambda x: x[0]).values  # index doesn't matter for onboard motes, just use first
    # df[['x', 'y']] = df['Position'].apply(lambda x: coords_from_perim(x[0], L, corners_x, corners_y))
    for row_i, row in df.iterrows():
        x, y = coords_from_perim(row['Position'][0], L, corners_x, corners_y, False)
        plt.plot(x, y, 'k+', markersize=13)
        plt.text(x, y, row['Mote'], color='black', fontsize='x-large')

# Plot tethered motes
styles = ['bx', 'rx']
colors = ['blue', 'red']
for i in [0, 1]:
    df = FS_data[np.logical_and(FS_data['Type'] == 'tethered', FS_data['i'] == i)]
    for row_i, row in df.iterrows():
        for j in [0, 1, 2]:
            if row['Position'][j] > 0 and row['Position'][j] < 1000:
                x, y = coords_from_perim(row['Position'][j], L, corners_x, corners_y, True)
                plt.plot(x, y, styles[i], markersize=8)
                plt.text(x, y, row['Mote'], color=colors[i], fontsize='x-large')

ax = plt.gca()
ax.set_ylim([-0.05 * L, 1.05 * L])
ax.set_xlim([-0.05 * L, 1.05 * L])
plt.axis('off')
plt.show()
# fig.savefig('../650Cal/Plots/Mote_locations.png')

