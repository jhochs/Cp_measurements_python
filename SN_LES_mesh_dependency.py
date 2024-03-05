import numpy as np
import pandas as pd
import scipy.io as sio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import util.common_funcs as common
import util.SN_funcs as SN

pd.options.mode.chained_assignment = None  # default='warn'

LES_results_path = '/Users/jackhochschild/Dropbox/School/Wind_Engineering/CFD/CharLES/SN/Results/'

mesh_names = ['SN_RL4_5_x4stresses', 'SN_RL5_x4stresses', 'SN_RL6_x4stresses']
colors = ['black', 'blue', 'red', 'green', 'orange']
line_type = ['solid', 'solid', 'solid', 'solid']

roofs = ['sloped_roof']
stats = ['dCprms', 'dCpmin']
save_path = '../Plots/Space_Needle/LES_mesh_dependency.html'

#====================================================================================
# PLOT

fig = common.init_subplots(len(stats), len(roofs))

for i in range(len(mesh_names)):
  for j in range(len(roofs)):
    Cp_perim = SN.get_perimeter_results(LES_results_path + mesh_names[i] + '/' + roofs[j] + '_Cpstats.mat', roofs[j])    

    if j==0:
      show_in_legend = True

    for k in range(len(stats)):
      fig.add_trace(go.Scatter(x=Cp_perim['Degrees'],
                            y=Cp_perim[stats[k]],
                            mode='lines',
                            line_color=colors[i],
                            marker_line_width=1,
                            line=dict(dash=line_type[i]),
                            name=mesh_names[i],
                            showlegend=show_in_legend,
                            hoverinfo='none'),
                row=k+1, col=j+1)
      show_in_legend = False

SN.set_figure_size(fig, len(stats), len(roofs))

# Add axis lines:
fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

common.add_axis_labels(fig, stats, roofs, 'Degrees along perimeter, from wind direction')

fig.write_html(save_path)