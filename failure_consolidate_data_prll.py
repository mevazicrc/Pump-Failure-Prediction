import numpy as np
import pandas as pd
import datetime
from multiprocessing import cpu_count, Pool

def initialize_event_dicts(API_LIST, Well_Event_Dict ):
    primary_failure_keywords = {'PUMP': ['ROD PUMP FAILURE', 'PUMP CHANGE','SANDED','ROD PUMP FAILURE: CONVERT TO PCP' ,'WORN/ LOW PROD'],
                                'ROD': ['ROD FAILURE (PART)', 'ROD PART'],
                                'TUBING': ['TUBING LEAK REPAIR', 'TUBING FAILURE (LEAK)'],
                                'POLROD': ['POLISHED ROD FAILURE', 'POLISH ROD']}


    secondary_failure_keywords = {'PUMP': [],
                                 'ROD': [],
                                 'TUBING': [],
                                 'POLROD': []}

    primary_failure_list = []
    secondary_failure_list = []

    for api in API_LIST:
        if api in list(Well_Event_Dict.keys()):
            if Well_Event_Dict[api].shape[0] > 0:
                # Replace None with 'None'
                Well_Event_Dict[api].SECONDARY_FAILURE.fillna('None')

                primary_failure_list.extend(list(Well_Event_Dict[api].PRIMARY_FAILURE.unique()))
                secondary_failure_list.extend(list(Well_Event_Dict[api].SECONDARY_FAILURE.unique()))
                for key in list(primary_failure_keywords.keys()):
                    #                 kInList = [1 if s in list(Well_Event_Dict[api].PRIMARY_FAILURE.unique()) else 0 for s in  primary_failure_keywords[key]]
                    ind_prim = np.where(Well_Event_Dict[api].PRIMARY_FAILURE.isin(primary_failure_keywords[key]))
                    if len(ind_prim[0] ) >0:
                        sf = Well_Event_Dict[api].SECONDARY_FAILURE.iloc[ind_prim[0]]
                        sf_str = [sf_s if type(sf_s)==str else 'Other' for sf_s in sf]
                        secondary_failure_keywords[key].extend(sf_str)

    for key in secondary_failure_keywords:
        secondary_failure_keywords[key] = list(set(secondary_failure_keywords[key]))

    primary_failure_list = set(primary_failure_list)
    secondary_failure_list = set(secondary_failure_list)

    failure_events_stat = {}

    primary_failure_modes = list(primary_failure_keywords.keys())
    for mode_prime in primary_failure_modes:
        failure_events_stat[mode_prime] = \
            {mode_prime + ' [ALL]': pd.DataFrame(columns=['API', 'num_failure', 'event_date'])}

        second_failure_modes = secondary_failure_keywords[mode_prime]
        for mode_second in second_failure_modes:
            failMode = mode_prime + ' [' + mode_second + ']'
            failure_events_stat[mode_prime].update({failMode: pd.DataFrame(columns=['API', 'num_failure', 'event_date'])})

    return failure_events_stat, Well_Event_Dict, primary_failure_keywords, secondary_failure_keywords

def calc_failure_stat(failure_events_stat):

    now = datetime.datetime.now()
    max_year = now.year
    min_year = max_year

    col_names = ['Year']
    for mode1 in list(failure_events_stat.keys()):
        for mode2 in list(failure_events_stat[mode1].keys()):
            col_names.append(mode2)
            minevs = [min(failure_events_stat[mode1][mode2]['event_date'].iloc[ind]).year for ind in
                      range(failure_events_stat[mode1][mode2].shape[0])]

            if len(minevs) > 0:
                min_year = min([min_year, min(minevs)])

    annual_failure_stat = pd.DataFrame(columns=col_names)

    annual_failure_stat['Year'] = np.arange(min_year, max_year+1, 1)
    for mode1 in list(failure_events_stat.keys()):
        for mode2 in list(failure_events_stat[mode1].keys()):
            yrs = list()
            for d in failure_events_stat[mode1][mode2]['event_date']:
                yrs.extend(pd.to_datetime(d).year.values.astype('int').tolist())

            for i in annual_failure_stat.index:
                annual_failure_stat[mode2].iloc[i] = np.sum(np.asarray(yrs) == annual_failure_stat['Year'].iloc[i])

    return annual_failure_stat

def consolidate_failure_dataset(job_args):

    API_LIST, Well_Card_Dict, Well_Analog_Dict,Well_Event_Dict, failure_keyword, failure_events_stat = job_args

    primary_failure_keywords = failure_keyword['primary']
    secondary_failure_keywords = failure_keyword['secondary']

    well_data_cons = {}
    API_LIST_CONS = []

    primary_failure_modes = list(primary_failure_keywords.keys())

    for api in API_LIST:

        if type(Well_Card_Dict[api]) == list or type(Well_Analog_Dict[api]) == list:
            continue

        API_LIST_CONS.append(api)

        df1 = Well_Analog_Dict[api]
        df2 = Well_Card_Dict[api].filter(items=['plot_date', 'card_type', 'SurfaceCardB', 'DownholeCardB',
                                                'POCDownholeCardB'])

        df2.rename(columns={'plot_date': 'Date'}, inplace=True)
        df1['Date'] = pd.to_datetime(df1.index, unit='ns')
        df2['Date'] = pd.to_datetime(df2['Date'], unit='ns')

        well_data_cons[api] = pd.merge(df1, df2, how='outer', on='Date')
        well_data_cons[api] = well_data_cons[api].sort_values(by='Date', ascending=True)

        # Failure Date
        event_datew = pd.to_datetime(Well_Event_Dict[api]['date_ops_start'], unit='ns')

        for mode_prime in primary_failure_modes:
            # PRIMARY FAILURE
            date_list = []
            for fstr in primary_failure_keywords[mode_prime]:
                if fstr in list(Well_Event_Dict[api].PRIMARY_FAILURE):
                    date_list.extend(event_datew[Well_Event_Dict[api]['PRIMARY_FAILURE'] == fstr].tolist())

            om2f = np.zeros((well_data_cons[api].shape[0], 1))
            tm2f, ow2f, tw2f = om2f, om2f, om2f

            failMode = mode_prime + ' [ALL]'
            if len(date_list) > 0:
                failure_events_stat[mode_prime][failMode] = failure_events_stat[mode_prime][failMode].append(
                    {'API': api,'num_failure': len(date_list),'event_date': date_list},  ignore_index=True)

                for date_t in date_list:
                    diffdays = np.asarray([(date_t - well_data_cons[api].iloc[ind]['Date']).days for ind in
                                           range(well_data_cons[api].shape[0])])
                    tm2f[np.logical_and(diffdays > 0, diffdays <= 60)] = 1
                    om2f[np.logical_and(diffdays > 0, diffdays <= 30)] = 1
                    tw2f[np.logical_and(diffdays > 0, diffdays <= 14)] = 1
                    ow2f[np.logical_and(diffdays > 0, diffdays <= 7)] = 1

            well_data_cons[api][failMode + '_twoM2fail'] = tm2f
            well_data_cons[api][failMode + '_oneM2fail'] = om2f
            well_data_cons[api][failMode + '_twoW2fail'] = tw2f
            well_data_cons[api][failMode + '_oneW2fail'] = ow2f

            # SECONDARY FAILURE
            second_failure_modes = secondary_failure_keywords[mode_prime]
            for mode_second in second_failure_modes:
                date_list = []
                if mode_second in list(Well_Event_Dict[api].SECONDARY_FAILURE):
                    date_list.extend(event_datew[Well_Event_Dict[api]['SECONDARY_FAILURE'] == mode_second].tolist())

                om2f = np.zeros((well_data_cons[api].shape[0], 1))
                tm2f, ow2f, tw2f = om2f, om2f, om2f
                failMode = mode_prime + ' [' + mode_second + ']'
                if len(date_list) > 0:
                    failure_events_stat[mode_prime][failMode] = failure_events_stat[mode_prime][failMode].append(
                        {'API': api,
                         'num_failure': len(date_list),
                         'event_date': date_list},
                        ignore_index=True)

                    for date_t in date_list:
                        diffdays = np.asarray([(date_t - well_data_cons[api].iloc[ind]['Date']).days for ind in
                                               range(well_data_cons[api].shape[0])])
                        tm2f[np.logical_and(diffdays > 0, diffdays <= 60)] = 1
                        om2f[np.logical_and(diffdays > 0, diffdays <= 30)] = 1
                        tw2f[np.logical_and(diffdays > 0, diffdays <= 14)] = 1
                        ow2f[np.logical_and(diffdays > 0, diffdays <= 7)] = 1

                well_data_cons[api][failMode + '_twoM2fail'] = tm2f
                well_data_cons[api][failMode + '_oneM2fail'] = om2f
                well_data_cons[api][failMode + '_twoW2fail'] = tw2f
                well_data_cons[api][failMode + '_oneW2fail'] = ow2f

    output = [well_data_cons,failure_events_stat]

    return output


def consolidate_failure_dataset_prll(API_LIST, Well_Card_Dict, Well_Analog_Dict,Well_Event_Dict, run_prll):

    failure_events_stat, Well_Event_Dict, primary_failure_keywords, secondary_failure_keywords = \
        initialize_event_dicts(API_LIST, Well_Event_Dict)

    failure_keyword = {'primary':primary_failure_keywords, 'secondary':secondary_failure_keywords}


    if run_prll:
        # parallelize
        cores = cpu_count()  # Number of CPU cores on your system
        partitions = cores  # Define as many partitions as you want
        API_split = np.array_split(API_LIST, partitions)
        fe = []
        wc = []
        wa = []
        we = []
        fk = []
        for i in range(partitions):
            wc.append({})
            wa.append({})
            we.append({})
            fe.append(failure_events_stat)
            fk.append(failure_keyword)
            for api in API_split[i]:
                wc[i].update({api:Well_Card_Dict[api]})
                wa[i].update({api:Well_Analog_Dict[api]})
                we[i].update({api:Well_Event_Dict[api]})

        pool = Pool(cores)
        output = []
        job_args = [(API_split[i], wc[i], wa[i], we[i], fk[i], fe[i]) for i in range(partitions) ]
        output.append(pool.map(consolidate_failure_dataset, job_args))
        pool.close()
        pool.join()

        well_data_cons = {}
        for i in range(partitions):
            well_data_cons.update(output[0][i][0])
            fes = output[0][i][1]
            if i == 0:
                failure_events_stat = fes
                continue
            for mode1 in list(failure_events_stat.keys()):
                for mode2 in list(failure_events_stat[mode1].keys()):
                    failure_events_stat[mode1][mode2] = failure_events_stat[mode1][mode2].append(fes[mode1][mode2])

        # Sort higher frequency failures
        for mode1 in list(failure_events_stat.keys()):
            for mode2 in list(failure_events_stat[mode1].keys()):
                failure_events_stat[mode1][mode2].sort_values(by=['num_failure'],  ascending=False, inplace = True)
    else:

        job_args = (API_LIST, Well_Card_Dict, Well_Analog_Dict,  Well_Event_Dict,failure_keyword, failure_events_stat)

        output = consolidate_failure_dataset(job_args)
        well_data_cons, failure_events_stat = output

    # CALCULATE FAILURE STATISTICS
    annual_failure_stat = calc_failure_stat(failure_events_stat)

    return well_data_cons, annual_failure_stat, failure_events_stat, primary_failure_keywords, secondary_failure_keywords


