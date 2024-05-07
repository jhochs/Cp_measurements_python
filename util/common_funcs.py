import scipy.io as sio
import pandas as pd
import numpy as np
from plotly.express.colors import sample_colorscale
from plotly.subplots import make_subplots
import plotly.graph_objects as go

symbols = ['circle', 'diamond', 'square', 'star', 'triangle-down', 'x', 'cross', 'pentagon', 'triangle-up', 'triangle-left', 'triangle-right', 'hexagram', 'star-triangle-up', 'star-triangle-down', 'diamond-tall', 'circle', 'diamond', 'square', 'star', 'triangle-down', 'x', 'cross']
symbols_2D = np.array([['circle', 'circle-open'], ['diamond', 'diamond-open'], ['square', 'square-open'], ['pentagon', 'pentagon-open']])
colors = ['black', 'navy', 'blue', 'cyan', 'aqua', 'limegreen', 'green', 'darkgreen', 'plum', 'purple', 'red', 'crimson', 'magenta', 'coral', 'brown']

def check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = todict(dict[key])
    return dict


def todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = todict(elem)
        else:
            dict[strg] = elem
    return dict


def loadmat(filename):
    '''
    this function should be called instead of direct scipy.io .loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return check_keys(data)

def get_color(cmap_choice, cbounds, value):
    # Sample the colorscale, returning the min/max if out of range
    frac = (value - cbounds[0])/(cbounds[1] - cbounds[0])
    if frac < 0:
        frac = 0.0
    if frac > 1:
        frac = 1.0
    return sample_colorscale(cmap_choice, frac)[0]

def third_outlier_on_row(input, mult_ratio, divide_ratio): 
    if np.sum(np.isnan(input)) != 0:
        # Only two values, cannot detect outlier
        return np.nan
    
    min_val = np.min(np.abs(input))
    ranges = np.array([np.abs(input[1] - input[2]), np.abs(input[0] - input[2]), np.abs(input[0] - input[1])])
    if np.max(ranges) >= mult_ratio * np.min(ranges) and np.max(ranges) / min_val >= divide_ratio:
        # There is an outlier
        return np.argmin(ranges)
    else:
        # No outlier
        return np.nan

def third_outlier_detection(df_in, cols):
    mult_ratio = 3
    divide_ratio = 0.3

    df = df_in[cols]
    df['arr'] = df[cols].values.tolist()
    df['arr'] = df['arr'].apply(np.array)
    df['outlier_idx'] = df['arr'].apply(lambda x: third_outlier_on_row(x, mult_ratio, divide_ratio))

    df_in['outlier_idx'] = df['outlier_idx']
    return df_in

def find_idx(x, bins):
    # Find the idx of which bin x belongs in, return nan if outside the range
    diff = x-bins

    # If diff is all + or -, we are outside the range:
    if sum(diff < 0)==len(diff) or sum(diff > 0)==len(diff):
        return np.nan
    else:
        diff[diff < 0] = np.nan
        return np.argmin(diff)-1

def nanptp(x):
    # np.ptp, ignoring nan
    x = x[~np.isnan(x)]
    return np.ptp(x)

def deltaplus(x):
    # max from mean, ignoring nan
    x = x[~np.isnan(x)]
    return np.abs(np.max(x)-np.mean(x))

def deltaminus(x):
    # min from mean, ignoring nan
    x = x[~np.isnan(x)]
    return np.abs(np.min(x)-np.mean(x))

def gumbel_min(x):
    # See MATLAB function cookMayneGumbel.
    # 10 <= len(x) < 16 are corrected for having <160 mins acquisition.
    # There is no statistical treatment for len(x) > 16

    # Settings:
    min_elements = 16
    window_len = 10 # minutes

    # Remove NaNs:
    x = x[~np.isnan(x)]

    if len(x)<min_elements:
        return np.nan
    else:
        # Use the Gumbel equations (see Kasperski 2003)
        Vc = np.std(x) / np.abs(np.mean(x))
        c_ad = 1 + 0.636 * Vc
        alpha = np.max([160/(len(x)*window_len), 1])
        return np.mean(x) * (c_ad + np.sqrt(6)/np.pi * np.log(alpha) * Vc)

def gumbel_min_6(x):
    # See MATLAB function cookMayneGumbel.
    # 6 <= len(x) < 16 are corrected for having <160 mins acquisition.
    # There is no statistical treatment for len(x) > 16

    # Settings:
    min_elements = 6
    window_len = 10 # minutes

    # Remove NaNs:
    x = x[~np.isnan(x)]

    if len(x)<min_elements:
        return np.nan
    else:
        # Use the Gumbel equations (see Kasperski 2003)
        Vc = np.std(x) / np.abs(np.mean(x))
        c_ad = 1 + 0.636 * Vc
        alpha = np.max([160/(len(x)*window_len), 1])
        return np.mean(x) * (c_ad + np.sqrt(6)/np.pi * np.log(alpha) * Vc)
    
def sort_xyerr(x, y, errplus, errminus):
    # Convert lists to arrays:
    x = np.array(x)
    y = np.array(y)
    errplus = np.array(errplus)
    errminus = np.array(errminus)

    # Remove NaN
    mask = np.logical_and(~np.isnan(x), ~np.isnan(y))
    y = y[mask]
    x = x[mask]
    errplus = errplus[mask]
    errminus = errminus[mask]

    # Sort by increasing position:
    I = np.argsort(x, axis=0)
    x = np.take_along_axis(np.array(x), I, axis=0)
    y = np.take_along_axis(np.array(y), I, axis=0)
    errplus = np.take_along_axis(np.array(errplus), I, axis=0)
    errminus = np.take_along_axis(np.array(errminus), I, axis=0)

    return x, y, errplus, errminus

def init_subplots(rows, cols):
    one_plot = {"type": "scatter"}
    row = [one_plot for _ in range(cols)]
    all = [row[:] for _ in range(rows)]

    fig = make_subplots(
        rows=rows, cols=cols,
        vertical_spacing=0.1,
        specs=all
        )
    
    return fig

def labels(color_choice):
    if color_choice == 'TurbIntensity' or color_choice == 'TurbIntensity_x':
        legend_label = 'I_u'
        cbar_label = 'I<sub>u</sub>'
    elif color_choice == 'WDiravg':
        legend_label = 'θ<sub>anem</sub> [°]'
        cbar_label = 'θ<sub>anem</sub> [°]'
    elif color_choice == 'eta':
        legend_label = 'η'
        cbar_label = 'η'
    elif color_choice == 'Lux':
        legend_label = 'L_{u,x}'
        cbar_label = 'L<sub>u,x</sub> [m]'
    else:
        legend_label = color_choice
        cbar_label = color_choice
    
    return legend_label, cbar_label

def plot_means(fig, row, col, x, y, **kwargs):
    if 'color' not in kwargs:
        kwargs['color'] = 'black'
    if 'mode' not in kwargs:
        kwargs['mode'] = 'markers+lines'
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode=kwargs['mode'],
        line=dict(color=kwargs['color']),
        marker_color=kwargs['color'],
        marker = dict(
        size=6,
        showscale=False,
        line_width=0.5),
        showlegend=False),
        row=row, col=col)

def plot_CIs(fig, row, col, x, y, errplus, errminus, **kwargs):
    y_upper = y + errplus
    y_lower = y - errminus
    if 'color' not in kwargs:  
        kwargs['color'] = 'black'
    fig.add_trace(go.Scatter(
        x=np.concatenate((x, x[::-1])),
        y=np.concatenate((y_upper, y_lower[::-1])),
        mode='lines',
        line=dict(color=kwargs['color']),
        fill='toself',
        fillcolor=kwargs['color'],
        opacity=0.2,
        hoverinfo='skip',
        marker=dict(showscale=False),
        showlegend=False),
        row=row, col=col)

def bins_legend(fig, bins, cmap_choice, cbounds, legend_label):
    # Adds dummy data to the plot so that the legend can be populated

    for i in np.flip(np.arange(len(bins)-1)):
        midpoint = bins[i] + (bins[i+1] - bins[i])/2
        color_cur = get_color(cmap_choice, cbounds, midpoint)

        fig.add_trace(go.Scatter(
            x=[np.nan, np.nan],
            y=[0, 0],
            name=('%.2f < ' %bins[i]) + legend_label + (' < %.2f' %bins[i+1]),
            mode='lines',
            line=dict(color=color_cur, width=4)
        ), row=1, col=1)


def add_axis_labels(fig, rows, cols, xlabel):
    label_dict = {'dCprms' : "C<sub>p, rms</sub>", 
                  'dCpmin' : "C'<sub>p, min</sub>", 
                  'dCp_skewness' : "C'<sub>p, skew</sub>",
                  'dCp_kurtosis' : "C'<sub>p, kurt</sub>",
                  'g' : '|g|'}
    for col in range(len(cols)):
        for row in range(len(rows)):
            fig.update_yaxes(title=dict(text=label_dict[rows[row]], font_size=20), row=row+1, col=col+1)
        fig.update_xaxes(title=dict(text=xlabel, font_size=18), row=len(rows), col=col+1)

def ellipse(center, major_axis, minor_axis, theta):
    # Center of the ellipse
    h, k = center
    
    # Semi-major and semi-minor axes
    a = major_axis / 2
    b = minor_axis / 2
    
    # Parametric equation of the ellipse
    t = np.linspace(0, 2*np.pi, 200)
    x = h + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
    y = k + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
    
    return x, y