import numpy as np
import pandas as pd
import re
import plotly.graph_objects as go

import util.common_funcs as common
import util.Cal650_exp_params as FS_params

def prepare_meas_data(FS, WS_correction, WDir_correction, minU, dCprms_maxRange, dCpmin_maxRange, exclude_third_outlier):
    '''
    Import the full-scale measurement data according to the JSON FS (see FS_exp_params.py)

    Applies correction to windspeed and Cp/dCp measurements, using WS_correction, the 
    multiplicative scale factor to convert rooftop anemometer measurements to freestream
    and WDir_correction, the change in wind direction relative to the freestream.

    Adds columns to the dataframe 'Above WS threshold' and 'Above Cp range threshold'
    which refers to wheter each row has windspeed > minU (input parameter) and range(dCpmin),
    range(dCprms) > dCpmin_maxRange, dCprms_maxRange. These new columns can later be used
    to filter the data when plotting.

    Returns a single dataframe with measurements from all deployments.
    '''
    for i in FS.keys():
        new_data = pd.read_csv(FS[i]['file'])
        new_data['i'] = i
        new_data[['Position', 'Type']] = pd.json_normalize(new_data['Mote'].map(FS[i]['motes']))
        new_data['Mote'] = new_data['Mote'].apply(lambda x: x + '_' + str(i))  # since motes can change position between deployments, this keeps them separate when binning

        # Append to one array for all measurement periods:
        if i==0:
            FS_data = new_data
        else:
            FS_data = pd.concat([FS_data, new_data])

    # Correct Cp for windspeed and wind direction:
    FS_data['WDiravg'] = FS_data['WDiravg'] + WDir_correction
    Cp_cols = [col for col in FS_data.columns if re.search('^Cpmean|dCpmean|Cprms|dCprms|Cpmin|dCpmin', col)]
    if isinstance(WS_correction, np.ndarray):
        for i in range(WS_correction.shape[1]-1):
            mask = np.logical_and(FS_data['WDiravg'] >= WS_correction[i,0], FS_data['WDiravg'] < WS_correction[i,1])
            FS_data.loc[mask, Cp_cols] = FS_data.loc[mask, Cp_cols] / (WS_correction[i,2] ** 2)
    else:
        FS_data[Cp_cols] = FS_data[Cp_cols] / (WS_correction ** 2)

    # Count how many sensors worked for each measurement:
    FS_data['Sensors online'] = 3 - np.sum(np.isnan(FS_data.filter(regex='^dCprms', axis=1).to_numpy()), axis=1)

    # If one of the sensors has Cpmin is outlying, mark it
    if exclude_third_outlier:
        FS_data = common.third_outlier_detection(FS_data, ['dCpmin_1', 'dCpmin_2', 'dCpmin_3'])
        FS_data.loc[FS_data['Type'] == 'tethered', 'outlier_idx'] = np.nan  # ensure this doesn't apply to tethered sensors, since it makes sense that their statistics will vary more than the threshold
        for i in [1,2,3]:
            FS_data.loc[FS_data['outlier_idx'] == i-1, FS_data.filter(regex='_' + str(i), axis=1).columns] = np.nan

    # Mark measurements where different sensors measured significantly different stats
    # or where freestream windspeed < threshold:
    FS_data[['Above WS threshold', 'Above Cp range threshold']] = 0
    FS_data.loc[FS_data['WSavg'] > minU, 'Above WS threshold'] = 1

    dCpmin = FS_data.filter(regex='^dCpmin', axis=1).to_numpy()
    FS_data['dCpmin_range'] = np.abs(np.nanmax(dCpmin, axis=1) - np.nanmin(dCpmin, axis=1))
    FS_data.loc[FS_data['Sensors online'] == 0, 'dCpmin_range'] = np.nan

    dCprms = FS_data.filter(regex='^dCprms', axis=1).to_numpy()
    FS_data['dCprms_range'] = np.nanmax(dCprms, axis=1) - np.nanmin(dCprms, axis=1) # this creates a runtime warning, ignore since the next line handles this
    FS_data.loc[FS_data['Sensors online'] == 0, 'dCprms_range'] = np.nan

    FS_data.loc[np.logical_or(FS_data['dCpmin_range'] > dCpmin_maxRange, FS_data['dCprms_range'] > dCprms_maxRange), 'Above Cp range threshold'] = 1
    FS_data.loc[FS_data['Type'] == 'tethered', 'Above Cp range threshold'] = np.nan
    
    # Replicate position entry for onboard motes to be [1,3]
    FS_data.loc[FS_data['Type'] == 'onboard', 'Position'] = FS_data.loc[FS_data['Type'] == 'onboard', 'Position'].apply(lambda x: np.tile(x, 3))

    # Convert position to meters:
    FS_data['Position'] = FS_data['Position'].apply(lambda x: np.divide(x, 100))

    return FS_data

def bin_by(df, **kwargs):
    if 'grouping' in kwargs:
        group_cols = ['Mote', 'bin_idx']
        bins = kwargs['bins']

        # Add bin index column:
        df['bin_idx'] = df[kwargs['grouping']].apply(lambda x: common.find_idx(x, kwargs['bins']))

        # Drop those outside the range:
        df.dropna(subset='bin_idx', inplace=True)
        df['bin_idx'] = df['bin_idx'].astype(int)
    else:
        group_cols = ['Mote']

    # Separate by type of sensor:
    df_tethered = df[df['Type'] == 'tethered']
    df_onboard = df[df['Type'] == 'onboard']

    # Aggregate the three sensors for onboard motes:
    for i in [1, 2, 3]:
        cols = ['bin_idx', 'Mote', 'Position', 'Type', 'dCpmin_' + str(i), 'dCpmin_noEV_' + str(i), 'dCprms_' + str(i), 'dCp_skewness_' + str(i), 'dCp_kurtosis_' + str(i), 'g_' + str(i)]
        if 'grouping' not in kwargs:
            cols.remove('bin_idx')
        
        df_cur = df_onboard[cols]
        df_cur.rename(columns={'dCpmin_' + str(i) : 'dCpmin', 
                               'dCpmin_noEV_' + str(i) : 'dCpmin_noEV', 
                               'dCprms_' + str(i) : 'dCprms',
                               'dCp_skewness_' + str(i) : 'dCp_skewness', 
                               'dCp_kurtosis_' + str(i) : 'dCp_kurtosis', 
                               'g_' + str(i) : 'g'}, inplace=True)
        if i==1:
            df_onboard_all = df_cur
        else:
            df_onboard_all = pd.concat((df_onboard_all, df_cur))
    
    df = pd.concat((df_onboard_all, df_tethered))
    df.reset_index(inplace=True, drop=True)

    # Aggregate and calculate statistics:
    df_binned = df.groupby(group_cols).agg({'Position':'first',
                                            'Type':'first',
                                            # 'dCpmin' : ['mean', 'std', common.deltaplus, common.deltaminus],
                                            # 'dCpmin_1' : ['mean', 'std', common.deltaplus, common.deltaminus],
                                            # 'dCpmin_2' : ['mean', 'std', common.deltaplus, common.deltaminus],
                                            # 'dCpmin_3' : ['mean', 'std', common.deltaplus, common.deltaminus],
                                            'dCpmin_noEV' : common.gumbel_min,
                                            'dCpmin_noEV_1' : common.gumbel_min,
                                            'dCpmin_noEV_2' : common.gumbel_min,
                                            'dCpmin_noEV_3' : common.gumbel_min,
                                            'dCprms' : ['mean', 'std', common.deltaplus, common.deltaminus],
                                            'dCprms_1' : ['mean', 'std', common.deltaplus, common.deltaminus],
                                            'dCprms_2' : ['mean', 'std', common.deltaplus, common.deltaminus],
                                            'dCprms_3' : ['mean', 'std', common.deltaplus, common.deltaminus],
                                            'dCp_skewness' : ['mean', 'std', common.deltaplus, common.deltaminus],
                                            'dCp_skewness_1' : ['mean', 'std', common.deltaplus, common.deltaminus],
                                            'dCp_skewness_2' : ['mean', 'std', common.deltaplus, common.deltaminus],
                                            'dCp_skewness_3' : ['mean', 'std', common.deltaplus, common.deltaminus],
                                            'dCp_kurtosis' : ['mean', 'std', common.deltaplus, common.deltaminus],
                                            'dCp_kurtosis_1' : ['mean', 'std', common.deltaplus, common.deltaminus],
                                            'dCp_kurtosis_2' : ['mean', 'std', common.deltaplus, common.deltaminus],
                                            'dCp_kurtosis_3' : ['mean', 'std', common.deltaplus, common.deltaminus],
                                            'g' : ['mean', 'std', common.deltaplus, common.deltaminus],
                                            'g_1' : ['mean', 'std', common.deltaplus, common.deltaminus],
                                            'g_2' : ['mean', 'std', common.deltaplus, common.deltaminus],
                                            'g_3' : ['mean', 'std', common.deltaplus, common.deltaminus]}).reset_index()

    # Collapse multi index:
    df_binned.columns = df_binned.columns.map('_'.join)

    # If using Cook and Mayne method:
    df_binned.rename(columns={'dCpmin_noEV_gumbel_min' : 'dCpmin_mean'}, inplace=True)
    df_binned.rename(columns={'dCpmin_noEV_1_gumbel_min' : 'dCpmin_1_mean'}, inplace=True)
    df_binned.rename(columns={'dCpmin_noEV_2_gumbel_min' : 'dCpmin_2_mean'}, inplace=True)
    df_binned.rename(columns={'dCpmin_noEV_3_gumbel_min' : 'dCpmin_3_mean'}, inplace=True)
    df_binned[['dCpmin_deltaplus', 'dCpmin_1_deltaplus', 'dCpmin_2_deltaplus', 'dCpmin_3_deltaplus', 'dCpmin_deltaminus', 'dCpmin_1_deltaminus', 'dCpmin_2_deltaminus', 'dCpmin_3_deltaminus']] = 0

    if 'grouping' in kwargs:
        df_binned.rename(columns={'Mote_':'Mote', 'bin_idx_':'bin_idx', 'Position_first':'Position', 'Type_first':'Type'}, inplace=True)
        
        # Add TurbIntensity/etc. column
        df_binned[kwargs['grouping'] + '_midpoint'] = df_binned['bin_idx'].apply(lambda x: bins[x] + (bins[x+1] - bins[x])/2)
    else:
        df_binned.rename(columns={'Mote_':'Mote', 'Position_first':'Position', 'Type_first':'Type'}, inplace=True)

    return df_binned

def wide_to_long(df):
    df = df[['Position', 
             'Type', 
             'Uavg', 
             'WDiravg', 
             'TurbIntensity_x',
             'dCprms_1', 'dCprms_2', 'dCprms_3', 
             'dCpmin_1', 'dCpmin_2', 'dCpmin_3', 
             'dCpmin_noEV_1', 'dCpmin_noEV_2', 'dCpmin_noEV_3', 
             'dCp_skewness_1', 'dCp_skewness_2', 'dCp_skewness_3', 
             'dCp_kurtosis_1', 'dCp_kurtosis_2', 'dCp_kurtosis_3',
             'dCp_Tint_1', 'dCp_Tint_2', 'dCp_Tint_3']]

    df[['Position_1', 'Position_2', 'Position_3']] = df['Position'].apply(pd.Series)
    df.drop(columns='Position', inplace=True)

    # Get rid of _1,2,3:
    df.reset_index(drop=True, inplace=True)
    df['id'] = df.index
    df = pd.wide_to_long(df, ['Position_', 
                              'dCprms_', 
                              'dCpmin_', 
                              'dCpmin_noEV_', 
                              'dCp_skewness_', 
                              'dCp_kurtosis_', 
                              'dCp_Tint_'], i='id', j='Sensor')

    df.rename(columns={'Position_':'Position', 
                       'dCprms_':'dCprms', 
                       'dCpmin_':'dCpmin', 
                       'dCpmin_noEV_':'dCpmin_noEV', 
                       'dCp_skewness_':'dCpskew', 
                       'dCp_kurtosis_':'dCpkurt', 
                       'dCp_Tint_':'dCp_Tint'}, inplace=True)
    df.dropna(axis=0, inplace=True)

    # Remove NaN:
    df.dropna(subset='dCprms', inplace=True)

    return df

def remove_nan_position(df):
    df[['Position_1', 'Position_2', 'Position_3']] = df['Position'].apply(pd.Series)

    for i in [1, 2, 3]:
        cols = [col for col in df.columns if '_' + str(i) in col]
        df.loc[np.isnan(df['Position_' + str(i)]), cols] = np.nan

    return df

def process_probes(Cp, perim_file):
    # Load files with perimeter coordinate (measured from the northwest corner))
    perim = pd.read_csv(perim_file, names=['X', 'Y', 'Z', 'from_NW'])

    # Invert LES scaling:
    perim['from_NW'] = perim['from_NW'] * 200

    # Round and combine the dataframes:
    Cp[['X', 'Y', 'Z']] = Cp[['X', 'Y', 'Z']].round(decimals=4)
    perim = perim.round(decimals=4)
    Cp_perim = pd.merge(Cp, perim, on=['X', 'Y', 'Z'])

    Cp_perim.sort_values(by=['from_NW'], inplace=True)  # sort by degrees
    Cp_perim.reset_index(drop=True, inplace=True)  # reset the index

    return Cp_perim

def get_perimeter_results(file, roof, wind_angle):
    struct = common.loadmat(file)

    if np.size(struct['probes']['Cpstats'], 1) == 6:
        # Skewness and kurtosis included in LES results
        df = pd.DataFrame(np.hstack((struct['probes']['Cpstats'], struct['probes']['coords'])), columns=['Cpmean', 'Cpmax', 'Cpmin', 'dCprms', 'dCp_skewness', 'dCp_kurtosis', 'X', 'Y', 'Z'])
    else:
        # Skewness and kurtosis not included
        df = pd.DataFrame(np.hstack((struct['probes']['Cpstats'], struct['probes']['coords'])), columns=['Cpmean', 'Cpmax', 'Cpmin', 'dCprms', 'X', 'Y', 'Z'])

    df['dCpmin'] = df['Cpmin'] - df['Cpmean']
    df['g'] = np.abs(df['dCpmin'] / df['dCprms'])

    coord_idx_path = 'util/perim_coords/parapet-' + roof + '_perimcoords_' + str(wind_angle) + '.csv'
    return process_probes(df, coord_idx_path)

def get_perimeter_results_with_ranges(file, roof, wind_angle):
    struct = common.loadmat(file)

    df = pd.DataFrame(np.hstack((struct['probes']['Cpstats_agg'], struct['probes']['coords'])), columns=['Cpmean', 'Cpmean_deltaminus', 'Cpmean_deltaplus', 
                                                                                                         'Cpmax', 'Cpmax_deltaminus', 'Cpmax_deltaplus', 
                                                                                                         'Cpmin', 'Cpmin_deltaminus', 'Cpmin_deltaminus',
                                                                                                         'dCprms', 'dCprms_deltaminus', 'dCprms_deltaplus', 
                                                                                                         'dCp_skewness', 'dCp_skewness_deltaminus', 'dCp_skewness_deltaplus', 
                                                                                                         'dCp_kurtosis', 'dCp_kurtosis_deltaminus', 'dCp_kurtosis_deltaplus', 
                                                                                                         'dCpmin', 'dCpmin_deltaminus', 'dCpmin_deltaplus', 
                                                                                                         'g', 'g_deltaminus', 'g_deltaplus', 
                                                                                                         'X', 'Y', 'Z'])

    coord_idx_path = 'util/perim_coords/parapet-' + roof + '_perimcoords_' + str(wind_angle) + '.csv'
    return process_probes(df, coord_idx_path)

def plot_meas_points(fig, meas, types, stats, **kwargs):
    for col in range(len(types)):  # different types tethered/onboard
        for row in range(len(stats)):  # different stats
            df = meas[meas['Type'] == types[col]]
            motes = df['Mote'].unique()
            for i in range(len(motes)): # each mote
                dff = df[df['Mote'] == motes[i]]

                if 'color' in kwargs and kwargs['color'] is not None:
                    color_col = kwargs['color']
                    color = dff[color_col]
                    marker_ranges = kwargs['marker_ranges']
                else:
                    color = 'gray'
                    kwargs['cmap'] = None
                    kwargs['cbounds'] = [None, None]
                
                for j in [1, 2, 3]: # each sensor
                    cur_pos = dff['Position'].iloc[0][j-1]
                    if not np.isnan(cur_pos):
                        inv_mask = pd.Series([True] * len(dff), index=dff.index)
                        if marker_ranges is not None:
                            jitter = [-0.5, 0.5]
                            for k in range(marker_ranges['WDiravg'].shape[0]):
                                # Create a mask that matches the ranges in marker_ranges:
                                WDir_mask = np.logical_and(dff['WDiravg'] > marker_ranges['WDiravg'][k,0],  dff['WDiravg'] <= marker_ranges['WDiravg'][k,1])
                                # Iu_mask = np.logical_and(dff[color_col] > marker_ranges['TurbIntensity_x'][k,0],  dff[color_col] <= marker_ranges['TurbIntensity_x'][k,1])
                                mask = WDir_mask # & Iu_mask
                                dfff = dff[mask]
                                inv_mask = inv_mask & ~mask
                                # mask = pd.Series([True] * len(dfff), index=dfff.index) 
                                # for dfff_col, (min_val, max_val) in fill_ranges.items():
                                #     mask = mask & (dfff[dfff_col] >= min_val) & (dfff[dfff_col] <= max_val)
                                
                                # Plot the filled datapoints:
                                y = dfff[stats[row] + '_' + str(j)]
                                # x = cur_pos + (np.random.rand(np.size(y))-0.5) # jittered values
                                x = (cur_pos + jitter[k]) * np.ones(np.size(y))
                                
                                # Plot points within the ranges:
                                fig.add_trace(go.Scatter(
                                    x=x,
                                    y=y,
                                    mode='markers',
                                    marker_color=color,
                                    marker = dict(
                                        symbol=common.symbols[k+1],
                                        size=5,
                                        colorscale=kwargs['cmap'],
                                        cmin=kwargs['cbounds'][0],
                                        cmax=kwargs['cbounds'][1]),
                                    showlegend=False),
                                row=row+1, col=col+1)

                        # Plot all the data (no fill_range) OR just the unfilled datapoints:
                        dfff = dff[inv_mask] # if there are no marker_ranges, inv_mask is all true
                        y = dfff[stats[row] + '_' + str(j)]
                        # x = cur_pos + (np.random.rand(np.size(y))-0.5) # jittered values
                        x = cur_pos * np.ones(np.size(y))
                        fig.add_trace(go.Scatter(
                            x=x,
                            y=y,
                            mode='markers',
                            marker_color=dfff[color_col],
                            marker = dict(
                                symbol='circle-open',
                                size=3,
                                colorscale=kwargs['cmap'], 
                                cmin=kwargs['cbounds'][0],
                                cmax=kwargs['cbounds'][1]),
                            showlegend=False),
                        row=row+1, col=col+1)
                            

def prepare_data_for_plot(df, sensor_type, stat):
    x_data = []
    y_data = []
    errplus_data = []
    errminus_data = []
    if sensor_type == 'tethered':
        for j in [1, 2, 3]:  # each sensor
            x_data.append(df['Position'].apply(lambda x: x[j-1]).values)
            y_data.append(df[stat + '_' + str(j) + '_mean'].values)
            errplus_data.append(df[stat + '_' + str(j) + '_deltaplus'].values)
            errminus_data.append(df[stat + '_' + str(j) + '_deltaminus'].values)
    else:
        x_data.append(df['Position'].apply(lambda x: x[0]).values)
        y_data.append(df[stat + '_mean'].values)
        errplus_data.append(df[stat + '_deltaplus'].values)
        errminus_data.append(df[stat + '_deltaminus'].values)

    return common.sort_xyerr(x_data, y_data, errplus_data, errminus_data)

def plot_meas_mean_CI(fig, meas, types, stats, choice):
    # Choice is a list of two booleans: [plot_mean, plot_CI]; so [1, 1] plots both mean and CI

    for col in range(len(types)): # different types tethered/onboard
        for row in range(len(stats)): # different stats
            df = meas[meas['Type'] == types[col]]
            x_data, y_data, errplus_data, errminus_data = prepare_data_for_plot(df, types[col], stats[row])

            # Plot means:
            if choice[0]:
                common.plot_means(fig, row+1, col+1, x_data, y_data)
                
            # Plot CIs:
            if choice[1]:
                common.plot_CIs(fig, row+1, col+1, x_data, y_data, errplus_data, errminus_data)
                
def plot_meas_intervals(fig, meas, types, stats, color_choice, cmap_choice, cbounds, choice):
    # Choice is a list of two booleans: [plot_mean, plot_CI]; so [1, 1] plots both mean and CI

    for col in range(len(types)): # different types tethered/onboard
        for row in range(len(stats)): # different stats
            df = meas[meas['Type'] == types[col]]
            bin_centers = df[color_choice + '_midpoint'].unique()
            for i in bin_centers:
                dff = df[df[color_choice + '_midpoint'] == i]
                color_cur = common.get_color(cmap_choice, cbounds, i)

                x_data, y_data, errplus_data, errminus_data = prepare_data_for_plot(dff, types[col], stats[row])

                # Plot means:
                if choice[0]:
                    common.plot_means(fig, row+1, col+1, x_data, y_data, color=color_cur)
                    
                # Plot CIs:
                if choice[1]:
                    common.plot_CIs(fig, row+1, col+1, x_data, y_data, errplus_data, errminus_data, color=color_cur)

def set_figure_size(fig, rows, cols, **kwargs):
    if cols==1:
        width = 1100
    else:
        width = 1700
    
    height = 300*rows

    fig.update_layout(
    autosize=False,
    width=width,
    height=height,
    margin=dict(
        pad=10
    ),
    font=dict(size=17),
    plot_bgcolor='rgba(0,0,0,0)'
)

def add_corners(fig, rows, cols):
    # Plot the corner positions:
    edge_length = FS_params.edge_length / 100 # convert cm to m
    edges_x = [edge_length, edge_length, 
               2*edge_length, 2*edge_length, 
               3*edge_length, 3*edge_length]
    edges_y = [-200, 200, 200, -200, -200, 200]
    for row_idx in range(rows):
        for col_idx in range(cols):
            fig.add_trace(go.Scatter(
            x=edges_x,
            y=edges_y,
            mode='lines',
            marker_color='gray',
            showlegend=False
            ),row=row_idx+1, col=col_idx+1)

            fig.update_xaxes(range=[0, 4*edge_length], row=row_idx+1, col=col_idx+1)

def set_yranges(fig, stats, types, df):
    df = remove_nan_position(df)

    for col in range(len(types)):
        dff = df[df['Type'] == types[col]]
        for row in range(len(stats)):
            dff_stat = dff[[stats[row] + '_1', stats[row] + '_2', stats[row] + '_3']]
            ymin = np.nanmin(dff_stat.values)
            ymax = np.nanmax(dff_stat.values)
            fig.update_yaxes(range=[ymin - 0.05*np.abs(ymin), ymax + 0.05*np.abs(ymax)], row=row+1, col=col+1)

def within_range(value, array, thresh):
    return np.any(np.abs(value - array) < thresh)