import numpy as np
import pandas as pd
import pickle, os
from multiprocessing import cpu_count, Pool
from scipy.stats import linregress

def unchangedTime(df, unchThreshold):
    a=pd.DataFrame(df, columns=['col'])
    a['col'] = a['col'] + 1e-15
    a['ratio'] = ( a['col'].diff() / a['col'] ).abs()
    a['ratio'].iloc[0] = 0
    a['unch'] = 1
    a['unch'][a['ratio'] > unchThreshold] = 0
    a['cumsum'] = a['unch'].cumsum()
    a['dd'] = a['unch'].diff()
    a['dd'].iloc[0] = 0
    a['dd'][a['dd'] > 0] = 0
    a['neg'] = a['dd'] * (a['unch'].cumsum())

    a['neg'].replace(to_replace=0, method='ffill', inplace=True)
    a['cum_unch'] = a['unch'].cumsum() + a['neg']

    return a['cum_unch'].values


def corr_time(df):
    cc_row = [np.corrcoef(range(df.shape[0]), df[col].values.tolist())[0, 1] for col in df.columns]
    return np.array(cc_row)

def generate_window_stat(args_in):
    wdata_list, parameters = args_in

    wdata_tseries = dict(item for item in wdata_list)
    api_list = list(wdata_tseries.keys())

    glb_days, shortTerm_days, longTerm_days= (parameters['global_days'], parameters['shortTerm_days'],
                                               parameters['longTerm_days'] )

    unchanged, unchThreshold = parameters['unchanged'],  parameters['unchThreshold']
    main_features, y_column = parameters['main_features'], parameters['YLABEL']

    min_rows = parameters['min_daysOfData']

    # Include unchanged variable
    unch_features = []
    if unchanged == True:
        unch_features = [col + '_unch' for col in main_features]


    stat_mtd_list =  parameters['window_stat_mtd']
    window_features = []
    for smtd in stat_mtd_list:
        window_features = window_features + [col+'_ST_'+smtd for col in main_features] +\
                          [col+'_LT_'+smtd for col in main_features] + \
                          [col+'_GL_'+smtd for col in main_features]


    fail_features = unch_features
    fail_features = fail_features + window_features
    if len(y_column) > 0:
        fail_features = fail_features + [y_column]

    data_flat_dict = dict()
    eps = 1e-15
    dt = wdata_tseries[api_list[0]].index[0].freq
    stat_mtd_list =  parameters['window_stat_mtd']

    counter_invalid  = 0
    incomplete_wells = {}
    for api in api_list:
        wdata = wdata_tseries[api]

        # Ignore the well if it does not have data for a feature
        col_in_mf = 0
        for col in main_features:
            if col not in wdata.columns:
                if col_in_mf == 0:
                    incomplete_wells[api] = [col]
                    col_in_mf=1
                else:
                    incomplete_wells[api].append(col)


        if col_in_mf==1:
            continue

        d_step = shortTerm_days
        d_step=1
        indx = wdata.index[0::d_step] + glb_days * dt
        if unchanged == True:
            for mf in main_features:
                wdata[ mf + '_unch' ] =  unchangedTime(wdata[mf].values, unchThreshold)

        data_flat = pd.DataFrame( [], columns=fail_features, index=indx[indx<=wdata.index.max()] )


        for datei in data_flat.index:

            row_data = wdata[unch_features].loc[datei].values/365

            d_glb = wdata[main_features].loc[datei - glb_days * dt:datei]
            # d_glb = wdata[main_features].loc[wdata.index[0]:datei]
            # d_glb = wdata[main_features].loc[wdata.index[0]:indx[0]]

            if len(d_glb.dropna())<len(d_glb.dropna()):
                data_flat.loc[datei] = np.full( (len(fail_features),) , np.nan )

            d_1   = wdata[main_features].loc[datei - shortTerm_days * dt:datei]
            d_2   = wdata[main_features].loc[datei - longTerm_days * dt:datei]

            for stat_mtd in stat_mtd_list:
                if stat_mtd=='mean':
                    med_glb = d_glb.mean(axis=0).values
                    med_1 = d_1.mean(axis=0).values
                    med_2 = d_2.mean(axis=0).values
                elif stat_mtd == 'median':
                    med_glb = d_glb.median(axis=0).values
                    med_1 = d_1.median(axis=0).values
                    med_2 = d_2.median(axis=0).values
                elif stat_mtd == 'var':
                    med_glb = d_glb.var(axis=0).values
                    med_1 = d_1.var(axis=0).values
                    med_2 = d_2.var(axis=0).values
                elif stat_mtd == 'AppEntropy':
                    med_glb = cal_ApEn(d_glb)
                    med_1   = cal_ApEn(d_1)
                    med_2   = cal_ApEn(d_2)

                # row_data = np.append(row_data, np.append((med_1+eps)/(med_glb+eps), (med_2+eps)/(med_glb+eps)) )
                row_data = np.append( row_data, np.append(np.append(med_1 , med_2),med_glb) )

            if len(y_column)>0:
                y = wdata[y_column].loc[datei]
                row_data = np.append( row_data, np.asarray([y]) )

            data_flat.loc[datei] = row_data

        data_flat_dict[api] = data_flat

    output = (data_flat_dict, incomplete_wells)

    return output


def approximate_entropy(x, m, r):
    """
    Implements a vectorized Approximate entropy algorithm.
    :return: Approximate entropy
    :return type: float
    """

    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)

    N = x.size
    r *= np.std(x)
    if r < 0:
        raise ValueError("Parameter r must be positive.")
    if N <= m + 1:
        return 0

    def _phi(m):
        x_re = np.array([x[i:i + m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(x_re[:, np.newaxis] - x_re[np.newaxis, :]),
                          axis=2) <= r, axis=0) / (N - m + 1)
        return np.sum(np.log(C)) / (N - m + 1.0)

    return np.abs(_phi(m) - _phi(m + 1))


def cal_ApEn(df):
    output = []
    m = 2
    r = 0.5
    for col in df.columns:
        x = df[col].copy()
        mask = np.isnan(x)
        if len(np.where(mask)[0]) > 0:
            x[mask] = np.mean(x[~mask])
        output.append(approximate_entropy(x, m, r))
    return output

def slidingWindow_stat_series(wdata_tseries, parameters):

    if parameters['parallel_run']==True:
        key_list = list(wdata_tseries.keys())

        cores = cpu_count() #Number of CPU cores on your system
        partitions = cores - 1 #Define as many partitions as you want
        key_split = np.array_split( key_list, partitions )
        rng_split = np.array_split( np.arange(len(key_list)), partitions )
        items = list( wdata_tseries.items() )

        args_in = [(items[rng_split[i][0]:rng_split[i][-1]+1],parameters) for i in range(partitions)]

        # wd_split = {}
        # for i in range(partitions):
        #     wd_split[i] = {api:wdata[api] for api in api_split[i]}

        pool = Pool(cores)
        data = []
        data.append(pool.map(generate_window_stat, args_in))
        pool.close()
        pool.join()

        data_flat_dict = data[0][0][0]
        for i in range(1, len(data[0])):
            data_flat_dict.update(data[0][i][0])


        incomplete_wells = data[0][0][1]
        for i in range(1, len(data[0])):
           incomplete_wells.update(data[0][i][1])

    else:

        args_in = (list(wdata_tseries.items()), parameters)

        data_flat_dict = generate_window_stat(args_in)

    return data_flat_dict, incomplete_wells


def Well_flat_matrix_ML(api_list, data_flat_dict, parameters):
    n_prev_steps = parameters['n_prev_steps']
    dt_back = parameters['deltaT']
    YLABEL = parameters['YLABEL']
    predictors = parameters['predictors']
    forget_thr = parameters['forget_threshold']

    col_list = predictors.copy()
    col_list.append(YLABEL)

    flatMatrix = []

    forget_thr = forget_thr * 24  # in hours

    well_flat_ML = {}
    for api in api_list:
        if len(data_flat_dict[api]) == 0:
            continue

        # Find rows with Missing values (or failure). subsequent data to forget_thr is ignored
        inds_na = np.where(data_flat_dict[api].isna().any(axis=1))[0]
        intervals = inds_na[np.where(np.diff(inds_na) > 1)[0]]

        discontinuity = data_flat_dict[api].index[intervals]
        inds_invalid = np.array([])
        for t in discontinuity:
            t_invalid = np.logical_and((data_flat_dict[api].index - t) > pd.Timedelta('0hour'),
                                       (data_flat_dict[api].index - t) < (forget_thr * pd.Timedelta('1hour')))
            inds_invalid = np.append(inds_invalid, np.where(t_invalid)[0])

        # Add additional features from previous timesteps
        new_df = data_flat_dict[api].loc[:, col_list].copy()
        for i in range(n_prev_steps):
            tstep = (i + 1) * dt_back
            add_col = [col + '[-' + str(tstep) + ']' for col in predictors]
            prev_data = new_df[predictors].values[dt_back * i:, :]
            for j in range(len(predictors)):
                new_df[add_col[j]] = new_df[predictors[j]].shift(tstep)

        new_df.loc[:, 'ML_valid'] = True
        new_df.loc[new_df.index[inds_invalid.astype('int')], 'ML_valid'] = False

        well_flat_ML[api] = new_df.copy()

    predictors0 = predictors
    for i in range(n_prev_steps):
        tstep = (i + 1) * dt_back
        add_col = [col + '[-' + str(tstep) + ']' for col in predictors0]
        predictors = predictors + add_col

    return well_flat_ML, predictors


def Uniform_time_interval(argsin):
    parameters, data = argsin

    start_date = parameters['start_date']
    end_date = parameters['end_date']
    freq = parameters['freq']
    interp_tol = parameters['interp_tol']

    print(parameters['main_features'])
    key_list = list(data.keys())
    # GENERATE TIME SERIES DATA (UNIFORM TIME INTERVAL) FROM CONSOLIDATED DATA
    new_data = dict()
    for key in key_list:
        t1 = max([start_date, data[key].index.min().floor('d')])
        t2 = min([end_date, data[key].index.max().ceil('d')])

        ndf = pd.DataFrame(index=pd.date_range(start=t1, end=t2, freq=freq))
        # for col in data[key].columns:
        for col in parameters['main_features']:
            if col in data[key].columns:
                ndf = pd.merge_asof(ndf, data[key][[col]].dropna(), left_index=True, right_index=True,
                                direction='nearest', tolerance=pd.Timedelta(interp_tol))

        new_data[key] = ndf

    return new_data

def time_resolution(data, parameters):
    partitions = cpu_count()-1
    pool = Pool(partitions)
    key_list = list(data.keys())
    key_split = np.array_split(key_list, partitions)

    data_job = []
    for i in range(partitions):
        data_job.append({})
        for key in key_split[i]:
            data_job[i].update({key: data[key]})

    job_args = [(parameters, data_job[i]) for i in range(partitions)]
    output = []
    output.append( pool.map(Uniform_time_interval, job_args) )
    pool.close()
    pool.join()

    new_data = {}
    for i in range(partitions):
        new_data.update(output[0][i])

    return new_data

def featureEng_fail_data( failure_data, parameters_resolution, parameters_window):

    file_name_unifyT = parameters_resolution['data_dir'] + '\\' + parameters_resolution['RMT'] + '_failureData_unifyT.pkl'

    file_name_SlideWindow = parameters_resolution['data_dir'] + '\\' + parameters_resolution['RMT'] + '_failureDataW_unifyT_SlideWin.pkl'

    parameters_resolution['main_features'] = parameters_window['main_features']

    # Change data resolution
    run_opt = True
    if run_opt == True:
        data_failure_ts = time_resolution(failure_data, parameters_resolution)
        with open(file_name_unifyT, 'wb') as f:
            pickle.dump(data_failure_ts, f)
    else:
        with open(file_name_unifyT, 'rb') as f:
            data_failure_ts = pickle.load(f)


    # Sliding Window Feature Generation
    run_opt = True
    if run_opt == True:
        print('sliding-window calculations...')
        data_failure_slidewin, incomplete_wells = slidingWindow_stat_series(data_failure_ts, parameters_window)
        with open(file_name_SlideWindow, 'wb') as f:
            pickle.dump( [data_failure_slidewin, incomplete_wells] , f )

    elif os.path.exists(file_name_SlideWindow):
        with open(file_name_SlideWindow, 'rb') as f:
            data_failure_slidewin, incomplete_wells = pickle.load(f)
    else:
        data_failure_slidewin = []

    return  data_failure_slidewin, data_failure_ts, incomplete_wells