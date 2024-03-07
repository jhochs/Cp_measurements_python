import pandas as pd
import numpy as np
import plotly.graph_objects as go

import util.common_funcs as common

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
        try:
            new_data[['Degrees', 'Roof']] = pd.json_normalize(new_data['Mote'].map(FS[i]['motes']))
        except:
            print('Empty data found, proceeding')

        # Append to one array for all measurement periods:
        if i==0:
            FS_data = new_data
        else:
            FS_data = pd.concat([FS_data, new_data])

    # Correct Cp for windspeed and wind direction:
    FS_data['WSavg'] = FS_data['WSavg'] * WS_correction
    FS_data['WDiravg'] = FS_data['WDiravg'] + WDir_correction
    FS_data['Lux'] = FS_data['Lux'] * WS_correction
    # FS_data['eta'] = FS_data['TurbIntensity'] * (FS_data['Lux'] / 37.2) ** 0.15
    FS_data['eta'] = FS_data['eta'] * (WS_correction ** 0.15)
    # FS_data.update(FS_data.filter(regex='^Cp|dCp', axis=1)/ (WS_correction ** 2))  # doesn't work because don't want to scale Cp_skewness and Cp_kurtosis
    FS_data.update(FS_data.filter(regex='^Cpmean|dCpmean', axis=1)/ (WS_correction ** 2)) 
    FS_data.update(FS_data.filter(regex='^Cprms|dCprms', axis=1)/ (WS_correction ** 2)) 
    FS_data.update(FS_data.filter(regex='^Cpmin|dCpmin', axis=1)/ (WS_correction ** 2))

    # If one of the sensors is Cprms is outlying, exclude it
    if exclude_third_outlier:
        FS_data = common.third_outlier_detection(FS_data, ['dCpmin_1', 'dCpmin_2', 'dCpmin_3'])  # detecting by dCpmin found to be more robust than by dCprms 
        for i in [1,2,3]:
            FS_data.loc[FS_data['outlier_idx'] == i-1, FS_data.filter(regex='_' + str(i), axis=1).columns] = np.nan

    # Count how many sensors worked for each measurement:
    FS_data['Sensors online'] = np.sum(FS_data.filter(regex='^SensorMask', axis=1).to_numpy(), axis=1)

    # Mark measurements where different sensors measured significantly different stats
    # or where freestream windspeed < threshold:
    FS_data[['Above WS threshold', 'Above Cp range threshold']] = 0
    FS_data.loc[FS_data['WSavg'] > minU, 'Above WS threshold'] = 1

    dCpmin = FS_data.filter(regex='^dCpmin', axis=1).to_numpy()
    FS_data['dCpmin_range'] = np.abs(np.nanmax(dCpmin, axis=1) - np.nanmin(dCpmin, axis=1)) # this creates a runtime warning, ignore since the next line handles this
    FS_data.loc[FS_data['Sensors online'] == 0, 'dCpmin_range'] = np.nan

    dCprms = FS_data.filter(regex='^dCprms', axis=1).to_numpy()
    FS_data['dCprms_range'] = np.nanmax(dCprms, axis=1) - np.nanmin(dCprms, axis=1) # this creates a runtime warning, ignore since the next line handles this
    FS_data.loc[FS_data['Sensors online'] == 0, 'dCprms_range'] = np.nan

    FS_data.loc[np.logical_or(FS_data['dCpmin_range'] > dCpmin_maxRange, FS_data['dCprms_range'] > dCprms_maxRange), 'Above Cp range threshold'] = 1

    FS_data['Degrees from WDir'] = (FS_data['WDiravg'] - FS_data['Degrees']) % 360
    # FS_data['Degrees from WDir'] = ((FS_data['Degrees from WDir'] + 180) % 360) - 180  # plot [-180,180]

    return FS_data

def extract_perimeter_Cp(all_Cp, perimeter_coords):
    '''
    Select the subset of all_Cp where coordinates match those in `perimeter_coords`,
    and for these points calculate degrees around the perimeter. Sort by degrees
    and reindex.
    '''
    all_Cp[['X', 'Y', 'Z']] = all_Cp[['X', 'Y', 'Z']].round(decimals=4)
    perimeter_coords = perimeter_coords.round(decimals=4)
    Cp_perim = pd.merge(all_Cp, perimeter_coords, on=['X', 'Y', 'Z'])
    Cp_perim['Degrees'] = -np.degrees(np.arctan2(Cp_perim['Z'], Cp_perim['X'])) - 180
    Cp_perim['Degrees'] = Cp_perim['Degrees'] % 360  # wrap to [0, 360]
    Cp_perim.sort_values(by=['Degrees'], inplace=True)  # sort by degrees
    Cp_perim.reset_index(drop=True, inplace=True)  # reset the index

    return Cp_perim

def get_perimeter_results(path, roof):
    struct = common.loadmat(path)
    if np.size(struct['probes']['Cpstats'], 1) == 6:
        # Skewness and kurtosis included in LES results
        Cp = pd.DataFrame(np.hstack((struct['probes']['Cpstats'], struct['probes']['coords'])), columns=['Cpmean', 'Cpmax', 'Cpmin', 'dCprms', 'dCp_skewness', 'dCp_kurtosis', 'X', 'Y', 'Z'])
    else:
        # Skewness and kurtosis not included
        Cp = pd.DataFrame(np.hstack((struct['probes']['Cpstats'], struct['probes']['coords'])), columns=['Cpmean', 'Cpmax', 'Cpmin', 'dCprms', 'X', 'Y', 'Z'])

    Cp['dCpmin'] = Cp['Cpmin'] - Cp['Cpmean']
    Cp['g'] = np.abs(Cp['dCpmin']) / Cp['dCprms']

    # Determine scale:
    if np.mean(struct['probes']['coords'][:,1]) > 100:
        scale = 'FS'
    else:
        scale = 'MS'
    coords_mat = common.loadmat('util/perim_coords/' + roof + '_perimeter_coords_' + scale + '.mat')
    
    if roof == 'sloped_roof':
        perim_coords = pd.DataFrame(coords_mat['sloped_coords'], columns=['X', 'Y', 'Z'])
    else:
        perim_coords = pd.DataFrame(coords_mat['flat_coords'], columns=['X', 'Y', 'Z'])
    
    Cp_perim = extract_perimeter_Cp(Cp, perim_coords)

    return Cp_perim

def get_perimeter_results_moteresolved(path):
    # Sloped roof only
    struct = common.loadmat(path)
    Cp = pd.DataFrame(np.hstack((struct['probes']['Cpstats'], struct['probes']['coords'])), columns=['Cpmean', 'Cpmax', 'Cpmin', 'Cprms', 'dCp_skewness', 'dCp_kurtosis', 'X', 'Y', 'Z'])
    Cp['dCpmin'] = Cp['Cpmin'] - Cp['Cpmean']
    Cp['g'] = Cp['dCpmin'] / Cp['Cprms']

    # Calculate position from coords:
    Cp['Degrees'] = -np.degrees(np.arctan2(Cp['Z'], Cp['X'])) - 180
    Cp['Degrees'] = Cp['Degrees'] % 360  # wrap to [0, 360]
    Cp['Degrees'] = Cp['Degrees'].astype(int)

    # Calculate mean and range for each mote:
    Cp_agg = Cp.groupby(['Degrees']).agg({'Cpmean' : ['mean', common.nanptp],
                                          'Cpmax' : ['mean', common.nanptp],
                                          'Cpmin' : ['mean', common.nanptp],
                                          'dCpmin' : ['mean', common.nanptp],
                                          'dCprms' : ['mean', common.nanptp],
                                          'dCp_skewness' : ['mean', common.nanptp],
                                          'dCp_kurtosis' : ['mean', common.nanptp],
                                          'g' : ['mean', common.nanptp]}).reset_index()
    Cp_agg.columns = Cp_agg.columns.map('_'.join)

    Cp_agg.rename(columns={'Degrees_' : 'Degrees'}, inplace=True)

    return Cp_agg

def bin_by(df, **kwargs):
    # SLOPED ROOF ONLY:
    df = df[df['Roof'] == 'sloped']

    if 'grouping' in kwargs:
        group_cols = ['loc_idx', 'bin_idx']
        bins = kwargs['bins']

        # Add bin index column:
        df['bin_idx'] = df[kwargs['grouping']].apply(lambda x: common.find_idx(x, kwargs['bins']))

        # Drop those outside the range:
        df.dropna(subset='bin_idx', inplace=True)
        df['bin_idx'] = df['bin_idx'].astype(int)
    else:
        group_cols = ['loc_idx']

    # Bin by position:
    locs = np.arange(0,365,10)
    df['loc_idx'] = df['Degrees from WDir'].apply(lambda x: common.find_idx(x, locs))

    # Aggregate sensors [1,2,3] together:
    for i in [1, 2, 3]:
        cols = ['Roof', 'bin_idx', 'loc_idx', 'dCpmin_' + str(i), 'dCpmin_noEV_' + str(i), 'dCprms_' + str(i), 'g_' + str(i), 'dCp_skewness_' + str(i), 'dCp_kurtosis_' + str(i), 'i']
        if 'grouping' not in kwargs:
            cols.remove('bin_idx')
        
        df_cur = df[cols]
        df_cur.rename(columns={'dCpmin_' + str(i):'dCpmin', 
                               'dCpmin_noEV_' + str(i):'dCpmin_noEV', 
                               'dCprms_' + str(i):'dCprms',
                               'g_' + str(i):'g',
                               'dCp_skewness_' + str(i):'dCp_skewness',
                               'dCp_kurtosis_' + str(i):'dCp_kurtosis'}, inplace=True)
        if i==1:
            df_allsensors = df_cur
        else:
            df_allsensors = pd.concat((df_allsensors, df_cur))

    df_allsensors.reset_index(inplace=True, drop=True)

    # Drop empty rows:
    df_allsensors.dropna(subset='dCprms', inplace=True)

    # Aggregate and calculate statistics:
    df_binned = df_allsensors.groupby(group_cols).agg({'Roof' : 'first',
                                                        'dCpmin_noEV' : common.gumbel_min,
                                                        'dCprms' : ['mean', 'std', common.deltaplus, common.deltaminus],
                                                        'dCp_skewness' : ['mean', 'std', common.deltaplus, common.deltaminus],
                                                        'dCp_kurtosis' : ['mean', 'std', common.deltaplus, common.deltaminus],
                                                        'i' : 'size'}).reset_index()

    # Collapse multi index:
    df_binned.columns = df_binned.columns.map('_'.join)

    # If using Cook and Mayne method:
    df_binned.rename(columns={'dCpmin_noEV_gumbel_min' : 'dCpmin_mean'}, inplace=True)
    df_binned[['dCpmin_deltaplus', 'dCpmin_deltaminus']] = 0

    # Recalculate peak factor:
    df_binned['g_mean'] = np.abs(df_binned['dCpmin_mean'] / df_binned['dCprms_mean'])
    df_binned['g_deltaplus'] = np.abs(df_binned['dCpmin_mean'] / df_binned['dCprms_mean'] ** 2) * df_binned['dCprms_deltaplus']
    df_binned['g_deltaminus'] = np.abs(df_binned['dCpmin_mean'] / df_binned['dCprms_mean'] ** 2) * df_binned['dCprms_deltaminus']

    if 'grouping' in kwargs:
        df_binned.rename(columns={'Roof_first':'Roof', 'loc_idx_':'loc_idx', 'bin_idx_':'bin_idx', 'i_size':'Count'}, inplace=True)
        
        # Add TurbIntensity/etc. column
        df_binned[kwargs['grouping'] + '_midpoint'] = df_binned['bin_idx'].apply(lambda x: bins[x] + (bins[x+1] - bins[x])/2)
    else:
        df_binned.rename(columns={'Roof_first':'Roof', 'loc_idx_':'loc_idx', 'i_size':'Count'}, inplace=True)

    # Convert loc_idx back to degrees and sort by this:
    df_binned['Degrees from WDir'] = df_binned['loc_idx'].apply(lambda x: locs[x])
    df_binned.sort_values(by='Degrees from WDir', inplace=True)
    df_binned.reset_index(inplace=True, drop=True)

    return df_binned

def plot_meas_points(fig, meas, roofs, stats, **kwargs):
    for col in range(len(roofs)):
        df = meas[meas['Roof'] == roofs[col]]
        for row in range(len(stats)):  
            # Plot each mote with a different symbol:
            motes = df['Mote'].unique()
            for i in range(len(motes)):
                dff = df[df['Mote'] == motes[i]]

                if 'color' in kwargs and kwargs['color'] is not None:
                    color = dff[kwargs['color']]
                else:
                    color = 'gray'
                    kwargs['cmap'] = None
                    kwargs['cbounds'] = [None, None]
                
                for j in [1, 2, 3]:
                    fig.add_trace(go.Scatter(
                        x=dff['Degrees from WDir'],
                        y=dff[stats[row] + '_' + str(j)],
                        mode='markers',
                        marker_color=color,
                        marker = dict(
                            size=6,
                            colorscale=kwargs['cmap'], 
                            cmin=kwargs['cbounds'][0],
                            cmax=kwargs['cbounds'][1],
                            symbol=common.symbols[i],
                            line_width=0.5
                        ),
                        text=dff['Mote'],
                        customdata=np.stack((dff['AcqStart'], dff['WSavg'], dff['TurbIntensity'], dff['Sensors online'], dff['dCpmin_range'], dff['dCprms_range']), axis=-1),
                        hovertemplate='<br>'.join([
                            'Mote: %{text}',
                            't: %{customdata[0]}',
                            'Mean WS: %{customdata[1]:.1f}',
                            'Turb Intensity: %{customdata[2]:.3f}',
                            'Sensors online: %{customdata[3]}',
                            '&#916;C<sub>p, min</sub> range: %{customdata[4]:.3f}',
                            '&#916;C<sub>p, rms</sub> range: %{customdata[5]:.3f}'
                            ]) + '<extra></extra>',
                        showlegend=False),
                    row=row+1, col=col+1)

def prepare_data_for_plot(df, stat):
    x_data = np.array(df['Degrees from WDir'])
    y_data = np.array(df[stat + '_mean'])
    errminus_data = np.array(df[stat + '_deltaminus'])
    errplus_data = np.array(df[stat + '_deltaplus'])

    # For Cprms, append 0 to end as well:
    if stat == 'dCprms' and x_data[0] == 0:
        x_data = np.append(x_data, 360)
        y_data = np.append(y_data, y_data[0])
        errminus_data = np.append(errminus_data, errminus_data[0])
        errplus_data = np.append(errplus_data, errplus_data[0])
    
    return common.sort_xyerr(x_data, y_data, errplus_data, errminus_data)

def plot_meas_mean_CI(fig, meas, roofs, stats, choice):
    # Choice is a list of two booleans: [plot_mean, plot_CI]; so [1, 1] plots both mean and CI

    for col in range(len(roofs)): # different types tethered/onboard
        for row in range(len(stats)): # different stats
            df = meas[meas['Roof'] == roofs[col]]
            if not df.empty:
                x_data, y_data, errplus_data, errminus_data = prepare_data_for_plot(df, stats[row])

                # Plot means:
                if choice[0]:
                    common.plot_means(fig, row+1, col+1, x_data, y_data)
                    
                # Plot CIs:
                if choice[1]:
                    common.plot_CIs(fig, row+1, col+1, x_data, y_data, errplus_data, errminus_data)
                
def plot_meas_intervals(fig, meas, roofs, stats, color_choice, cmap_choice, cbounds, choice):
    # Choice is a list of two booleans: [plot_mean, plot_CI]; so [1, 1] plots both mean and CI
    
    for col in range(len(roofs)):
        df = meas[meas['Roof'] == roofs[col]]
        for row in range(len(stats)):
            bin_centers = df[color_choice + '_midpoint'].unique()
            for i in bin_centers:
                dff = df[df[color_choice + '_midpoint'] == i]
                color_cur = common.get_color(cmap_choice, cbounds, i)

                x_data, y_data, errplus_data, errminus_data = prepare_data_for_plot(dff, stats[row])

                # Plot means:
                if choice[0]:
                    if stats[row] == 'dCpmin':
                        common.plot_means(fig, row+1, col+1, x_data, y_data, color=color_cur, mode='markers')
                    else:
                        common.plot_means(fig, row+1, col+1, x_data, y_data, color=color_cur)
                    
                # Plot CIs:
                if choice[1]:
                    if stats[row] != 'dCpmin':
                        common.plot_CIs(fig, row+1, col+1, x_data, y_data, errplus_data, errminus_data, color=color_cur)

def set_figure_size(fig, rows, cols):
    if cols==1:
        width = 1400
    else:
        width = 2750
    
    height = 350*rows

    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        margin=dict(
            l=50,
            r=50,
            b=50,
            t=50,
            pad=4
        ),
        font=dict(size=20),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
def add_regions(fig, rows):
    # For showing delineation between flow regions

    for row_idx in range(rows):
        for d in [45, 145]:
            fig.add_trace(go.Scatter(
                    x=[d, d],
                    y=[-200, 200],
                    mode='lines',
                    line=dict(color='gray', width=1),
                    showlegend=False
                ), row=row_idx+1, col=1)
            fig.add_trace(go.Scatter(
                    x=[360-d, 360-d],
                    y=[-200, 200],
                    mode='lines',
                    line=dict(color='gray', width=1),
                    showlegend=False
                ), row=row_idx+1, col=1)
                
def set_xyranges(fig, stats, roofs, df):
    for col in range(len(roofs)):
        dff = df[df['Roof'] == roofs[col]]
        for row in range(len(stats)):
            dff_stat = dff[[stats[row] + '_1', stats[row] + '_2', stats[row] + '_3']]
            ymin = np.nanmin(dff_stat.values)
            ymax = np.nanmax(dff_stat.values)
            fig.update_yaxes(range=[ymin - np.max([0.05*np.abs(ymin), 0.05]), ymax + np.max([0.05*np.abs(ymax), 0.1])], row=row+1, col=col+1)
            fig.update_xaxes(range=[-1,361], row=row+1, col=col+1)
            