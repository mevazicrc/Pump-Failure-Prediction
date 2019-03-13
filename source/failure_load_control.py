
import numpy as np
import pandas as pd
import pickle

from source.failure_preprocessing import add_card_features_prll


def clean_events_filter_time(event_data,  year_min):
    cols = ['date_ops_end',
            'date_ops_start',
            'cost_authorized',
            'EXPENSE_TYPE',
            'event_type',
            'PRIMARY_FAILURE',
            'SECONDARY_FAILURE',
            'JOB_SUMMARY',
            'FAILURE_DESCRIPTION',
            'fill_top_md',
            'cleanout_md',
            'cleanout_method',
            'FOREIGN_MATERIAL_TYPE',
            'ROOT_CAUSE']

    api_list = list(event_data.keys())

    event_data_new ={}
    all_df =[]
    for api in api_list:
        if len(event_data[api] ) >0:
            if len(event_data[api][ event_data[api]['date_ops_start' ] >year_min ] ) >0:
                if len(all_df )==0:
                    all_df = event_data[api][ event_data[api]['date_ops_start' ] >year_min ].sort_values(by=['date_ops_start'])[cols]
                else:
                    all_df = all_df.append \
                        (event_data[api][ event_data[api]['date_ops_start' ] >year_min ].sort_values
                            (by=['date_ops_start'])[cols])
                event_data_new[api] = event_data[api][ event_data[api]['date_ops_start' ] >year_min ].sort_values(by=['date_ops_start'])[cols]

    return event_data_new

# def clean_filter_events_prll(event_data,  year_min):
#     keys = list(event_data.keys())
#
#     args = zip(keys_split ,year_min)
#
#     pool = Pool(cpu_count())
#     pool.map(clean_filter_events, args)
#
#     pool.close()
#     pool.join()


def get_failure_stat(event_data ,cols):
    keys = list(event_data.keys())
    summ_df =[]

    for api in keys:
        if len(event_data[api] ) >0:
            if len(summ_df )==0:
                summ_df = event_data[api][cols]
            else:
                summ_df = summ_df.append( event_data[api][cols] )

    fail_df = summ_df.groupby(cols).size().sort_values(ascending=False)
    df = pd.DataFrame(fail_df)
    df.reset_index(inplace=True)

    return df


def filter_for_failures(event_data_new, fail_col_dict, word_in_col):
    api_list_fail_filter = list(event_data_new)
    if len(fail_col_dict)>0:
        api_list_fail_filter  = []
        for api in list(event_data_new.keys()):
            api_in =0
            for key1 in fail_col_dict:
                for key2 in fail_col_dict[key1]:
                    if key2 in event_data_new[api][key1].values.tolist():
                        api_in = 1
                        break

            if api_in ==1:
                api_list_fail_filter.append(api)
            else:
                for col in word_in_col:
                    if word_in_col[col] in event_data_new[api][col].values.tolist():
                        api_list_fail_filter.append(api)
                        break

    event_data_new = dict((k, event_data_new[k]) for k in api_list_fail_filter)

    return event_data_new


def segment_well_history(api_list, event_data, FAILURE_DICT, start_date,  last_date):
    well_segmented = {}
    for api in api_list:
        well_segmented[api] = {}
        date_reset = []
        date_events = []
        event_prime_mode = []
        event_second_mode = []

        if isinstance(event_data, pd.DataFrame):
            event_api = event_data[event_data['API14'] == api]
        else:
            event_api=[]
            if api in event_data.keys():
                event_api = event_data[api]


        if len(event_api) > 0:
            for prime_mode in FAILURE_DICT.keys():
                d_events = event_api[event_api['PRIMARY_FAILURE'] == prime_mode]['date_ops_start'].tolist()
                if len(d_events) > 0:
                    date_events.extend(d_events)
                    event_prime_mode.extend \
                        (event_api[event_api['PRIMARY_FAILURE'] == prime_mode]['PRIMARY_FAILURE'].tolist())
                    event_second_mode.extend \
                        (event_api[event_api['PRIMARY_FAILURE'] == prime_mode]['SECONDARY_FAILURE'].tolist())
                    date_reset.extend \
                        (event_api[event_api['PRIMARY_FAILURE'] == prime_mode]['date_ops_end'].tolist())

        no_ev = 1
        if len(date_events) == 0:
            no_ev = 0
            date_reset = [start_date]
            date_events = [last_date]
            event_prime_mode = ['CENSORED']
            event_second_mode = ['CENSORED']

        segment = pd.DataFrame({'PRIMARY_FAILURE': event_prime_mode,
                                'SECONDARY_FAILURE': event_second_mode,
                                'SEGMENT_START': pd.to_datetime(date_reset),
                                'SEGMENT_END': pd.to_datetime(date_events)},
                               columns=['PRIMARY_FAILURE', 'SECONDARY_FAILURE', 'SEGMENT_START', 'SEGMENT_END'])

        if no_ev == 1:
            segment.sort_values(by='SEGMENT_START', inplace=True)

            end_of_last_event = segment['SEGMENT_START'].iloc[-1]

            segment['SEGMENT_START'].iloc[1:] = segment['SEGMENT_START'].iloc[0:-1].values
            segment['SEGMENT_START'].iloc[0] = start_date

            segment = segment.append({'PRIMARY_FAILURE': 'CENSORED',
                                      'SECONDARY_FAILURE': 'CENSORED',
                                      'SEGMENT_START': end_of_last_event,
                                      'SEGMENT_END': last_date}, ignore_index=True)

        well_segmented[api] = segment

    return well_segmented


def filter_wells_from_info( wellInfo, col_filter ):
    cols = col_filter['cols']
    vals = col_filter['vals']

    for i in range(len(cols)):
        if i == 0:
            if type(vals[i]) == list:
                for k in range(len(vals[i])):
                    if k == 0:
                        cond = "( (wellInfo.%s=='%s')" % (cols[i], vals[i][k])
                    else:
                        cond = cond + " | (wellInfo.%s=='%s')" % (cols[i], vals[i][k])
                cond = cond + " )"
            else:
                cond = "(wellInfo.%s=='%s')" % (cols[i], vals[i])

        else:
            if type(vals[i]) == list:
                for k in range(len(vals[i])):
                    if k == 0:
                        cond = cond + " & ( (wellInfo.%s=='%s')" % (cols[i], vals[i][k])
                    else:
                        cond = cond + " | (wellInfo.%s=='%s')" % (cols[i], vals[i][k])
                cond = cond + " )"
            else:
                cond = cond + " & (wellInfo.%s=='%s')" % (cols[i], vals[i])

    return eval(cond)

def get_card_data(wc, api,min_date):

    wc_df = []
    # if api in an_apis:
    if len(wc)>0:
        wc_df = wc.filter(items=['plot_date', 'card_type','SurfaceCardB',
                                                    'DownholeCardB','POCDownholeCardB'])
        wc_df.sort_values(by=['plot_date'],inplace=True)
        wc_df.rename(columns={'plot_date': 'Date'}, inplace=True)
        wc_df.set_index('Date',inplace=True)
#             df1['Date'] = pd.to_datetime(df1.index, unit='ns')

        wc_df = wc_df[wc_df.index>=min_date]
    return wc_df

def get_analog_data(wa, min_date):

    columns_keep = ['Fluid Lvl (meas)', 'Gas Rate', 'Tbg Pressure',
                    'Well Test Oil', 'Pump Size', 'SPM', 'Runtime',
                    'Est Gross Prod', 'Water Rate', 'Well Test Gross',
                    'Idle Time', 'Csg Pressure', 'Firmware Build',
                    'Cycles', 'Stroke Length', 'Comm Pct', 'MinLoad',
                    'PeakLoad', 'Current Motor RPM', 'Differential Pressure',
                    'Gas Flow Rate', 'Orifice Diameter',
                    'Pump Fillage', 'Current Inferred Production', 'Current Percent Run']

    wa_df = []
    if len(wa) > 0:
        wa_df = pd.pivot_table(wa, index='Date',
                               columns='AddressName', values='Value').filter(items=columns_keep)

        wa_df = wa_df[wa_df.index>=min_date]


    return wa_df

def consolidate_failure_data(API_LIST, fname_dict, col_filter, data_dir, min_date):

    info_file = fname_dict['info']
    analog_file = fname_dict['analog']
    card_file = fname_dict['card']

    filename = data_dir + info_file + '.pkl'
    with open(filename, "rb") as f:
        Well_Info_df = pickle.load(f)

    df_init = 0
    for api in API_LIST:
        winfo = Well_Info_df[Well_Info_df['API14'] == api]
        if len(winfo) > 0:
            if df_init == 0:
                wellInfo = pd.DataFrame(winfo, columns=winfo.columns)
                df_init = 1
                continue
            wellInfo = wellInfo.append(winfo)

    # Filter Wells
    cond = filter_wells_from_info(wellInfo, col_filter)

    wellInfo_filtered = wellInfo[cond].filter(items=['API14', 'WELL_AUTO_NAME', 'COMP_TYPE',
                                                     'WELL_STATUS', 'STATUS_DATE', 'MOP', 'RMT_TEAM'])

    wellInfo_filtered.set_index('API14', inplace=True)

    API_LIST = list(wellInfo_filtered.index)

    # ANALOG DATA
    filename = data_dir + analog_file + '.pkl'
    with open(filename, "rb") as f:
        Well_Analogs = pickle.load(f)

    if isinstance(Well_Analogs, list):
        Well_Analog_All = Well_Analogs[0]
        for i in range(1, len(Well_Analogs)):
            Well_Analog_All = Well_Analog_All.append(Well_Analogs[i])
    else:
        Well_Analog_All = Well_Analogs

    # CARD DATA
    filename = data_dir + card_file + '.pkl'
    with open(filename, "rb") as f:
        Well_Cards = pickle.load(f)

    if isinstance(Well_Cards, list):
        Well_Card_All = Well_Cards[0]
        for i in range(1, len(Well_Cards)):
            Well_Card_All = Well_Card_All.append(Well_Cards[i])
    else:
        Well_Card_All = Well_Cards

    Well_Analog_All.set_index(['Date'], inplace=True)

    Well_Card_All.rename(columns={'plot_date': 'Date'}, inplace=True)
    Well_Card_All.set_index('Date', inplace=True)

    well_data_cons = {}
    API_LIST_CONS = []

    for api in API_LIST:

        wa_df = Well_Analog_All[Well_Analog_All['API14'] == api].drop(['API14'], axis=1)
        wa_df = wa_df[wa_df.index >= min_date]

        wc_df = Well_Card_All[Well_Card_All['API14'] == api].filter(
            items=['card_type', 'SurfaceCardB', 'DownholeCardB', 'POCDownholeCardB']).sort_index()
        wc_df = wc_df[wc_df.index >= min_date]

        if type(wc_df) == list or type(wa_df) == list:
            continue

        API_LIST_CONS.append(api)

        well_data_cons[api] = pd.merge(wa_df, wc_df, how='outer', on='Date').sort_values(by='Date', ascending=True)

        #     well_data_cons[api].set_index('Date',inplace=True)

    return well_data_cons


def segment_consolidated_data(apis, data_failure, well_segments):
    segmented_consolidated_data = {}

    api_list = []

    ind = 0
    for api in apis:  # data_failure.keys():
        if api in data_failure.keys():
            if ind == 0:
                segments_info = well_segments[api].copy()
            else:
                segments_info = segments_info.append(well_segments[api], ignore_index=True)

            for i in range(well_segments[api].shape[0]):
                api_list.append(api)

                t1 = well_segments[api]['SEGMENT_START'].iloc[i]
                t2 = well_segments[api]['SEGMENT_END'].iloc[i]

                df = data_failure[api][(data_failure[api].index > t1) & (data_failure[api].index <= t2)]

                segmented_consolidated_data.update({ind: df})

                ind += 1

    segments_info['api'] = api_list

    return segmented_consolidated_data, segments_info


def filterFailures_and_getWellsegments(event_file_dir, year_min_failure, failure_filter_dict, failure_word_in_col,
                                       parameters_seg):
    # Load Event data
    with open(event_file_dir['well_event'], 'rb') as f:
        event_data_list = pickle.load(f)

    event_data_df = event_data_list
    if isinstance(event_data_df,list):
        event_data_df = event_data_list[0]
        for i in range(1, len(event_data_list)):
            event_data_df = event_data_df.append(event_data_list[i])

    # Load Well Info
    with open(event_file_dir['well_info'], "rb") as f:
        Well_Info_df = pickle.load(f)



    event_data_df['date_ops_start'] = pd.to_datetime(event_data_df['date_ops_start'].astype('str'))
    event_data_df['date_ops_end'] = pd.to_datetime(event_data_df['date_ops_end'].astype('str'))

    api_list = event_data_df['API14'].unique().tolist()
    event_data = {}
    for api in api_list:
        event_data[api] = event_data_df[event_data_df['API14'] == api]


    # Filter for time-of-failure
    event_data_tf = clean_events_filter_time(event_data, year_min_failure)

    # Filter wells with Specific Failure Types
    event_data_filtered = filter_for_failures(event_data_tf, failure_filter_dict,
                                                                   failure_word_in_col)

    api_list_fails = list(event_data_filtered.keys())
    if len(failure_filter_dict)==0:
        print('No filter pass. all wells selected')
        api_list_fails = Well_Info_df[Well_Info_df['WELL_STATUS'].isin(['ACTIVE'])]['API14'].tolist()

    # EVENT CAUSE SUMMARY
    event_summ_df = get_failure_stat(event_data_filtered, ['event_type', 'PRIMARY_FAILURE', 'SECONDARY_FAILURE'])

    # Failure root cause Summary
    PRIME_FAIL_MODES = ['ROD PUMP FAILURE', 'TUBING FAILURE (LEAK)', 'ROD FAILURE (PART)', 'POLISHED ROD FAILURE']
    FAILURE_DICT = {}
    for pfail in PRIME_FAIL_MODES:
        FAILURE_DICT[pfail] = event_summ_df[event_summ_df['PRIMARY_FAILURE'] == pfail]['SECONDARY_FAILURE'].tolist()

    last_date = pd.to_datetime(parameters_seg['end_date'])
    start_date = pd.to_datetime(parameters_seg['start_date'])

    well_segments = segment_well_history(api_list_fails, event_data_filtered, FAILURE_DICT,
                                                              start_date, last_date)

    return api_list_fails, well_segments, FAILURE_DICT, event_summ_df



def segmented_consolidate_data(api_list_fails, well_segments, con_opts, card_opts):
    # Consolidate Failure Data [Analog+Card]
    RMT = con_opts['RMT_LIST'][0]

    fname_dict = {key: RMT + '_' + con_opts['fname_dict_base'][key] for key in con_opts['fname_dict_base']}

    well_data = consolidate_failure_data(api_list_fails, fname_dict,
                                         con_opts['info_col_filter'], con_opts['data_dir'], con_opts['min_date'])

    # Add Card Features
    card_types = card_opts['card_types']

    well_data_cons = add_card_features_prll(well_data, card_opts['card_features_list'], card_opts['card_types'])

    # Segmented Consolidated data
    segmented_failure_data, segments_info = segment_consolidated_data(api_list_fails, well_data_cons, well_segments)

    return segmented_failure_data, segments_info, well_data_cons