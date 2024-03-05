import numpy as np
import pandas as pd
import scipy.io as sio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import util.common_funcs as common
import util.Cal650_funcs as Cal650
import util.Cal650_exp_params as exp_params

pd.options.mode.chained_assignment = None  # default='warn'

LES_results_path = '/Users/jackhochschild/Dropbox/School/Wind_Engineering/CFD/CharLES/650Cal/270deg/'

wind_angle = 270
mesh_names = ['650Cal_RL0_5', '650Cal_RL1_5', '650Cal_RL2_5']
colors = ['black', 'blue', 'red', 'green', 'orange']
line_type = ['solid', 'solid', 'solid', 'solid']

roofs = ['side'] # ['side', 'top']
stats = ['dCprms', 'dCpmin']
save_path = '../Plots/650Cal/LES_mesh_dependency.html'

# #====================================================================================
# # IMPORT LES RESULTS
# LES_side = []
# LES_top = []
# for mesh in mesh_names:
#     side_mat = LES_results_path + mesh + '/' + 'parapet_side_Cpstats.mat'
#     LES_side.append(Cal650.get_perimeter_results(side_mat, 'side', wind_angle))

#     top_mat = LES_results_path + mesh + '/' + 'parapet_top_Cpstats.mat'
#     LES_top.append(Cal650.get_perimeter_results(top_mat, 'top', wind_angle))

#====================================================================================
# PLOT

fig = common.init_subplots(len(stats), len(roofs))

for i in range(len(mesh_names)):
  for j in range(len(roofs)):
    Cp_perim = Cal650.get_perimeter_results(LES_results_path + mesh_names[i] + '/parapet_' + roofs[j] + '_Cpstats.mat', roofs[j], wind_angle)    

    if j==0:
      show_in_legend = True

    for k in range(len(stats)):
      fig.add_trace(go.Scatter(x=Cp_perim['from_NW'],
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

Cal650.set_figure_size(fig, len(stats), len(roofs))

# Add axis lines:
fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

common.add_axis_labels(fig, stats, roofs, 'Position [m]')

fig.write_html(save_path)