
import pandas as pd
import numpy as np
import pickle
import gc
import os
from multiprocessing import cpu_count, Pool
from source.failure_consolidate_data_prll  import consolidate_failure_dataset_prll

cwd = os.getcwd()

def filter_wells( wellInfo, col_filter ):
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


def pivot_analog_table(filename, API_LIST):

    with open(filename, "rb") as f:
        Well_Analog_All = pickle.load(f)

    columns_keep = ['Fluid Lvl (meas)', 'Gas Rate', 'Tbg Pressure',
                    'Well Test Oil', 'Pump Size', 'SPM', 'Runtime',
                    'Est Gross Prod', 'Water Rate', 'Well Test Gross',
                    'Idle Time', 'Csg Pressure', 'Firmware Build',
                    'Cycles', 'Stroke Length', 'Comm Pct', 'MinLoad',
                    'PeakLoad', 'Current Motor RPM', 'Differential Pressure',
                    'Gas Flow Rate', 'Orifice Diameter',
                    'Pump Fillage', 'Current Inferred Production', 'Current Percent Run']

    Well_Analog_Dict = {}

    for api in API_LIST:
        if len(Well_Analog_All[api]) > 0:
            wa_df = pd.pivot_table(Well_Analog_All[api][0], index='Date',
                                   columns='AddressName', values='Value').filter(items=columns_keep)

            Well_Analog_Dict.update({api: wa_df})
        else:
            Well_Analog_Dict.update({api: []})

    del Well_Analog_All
    gc.collect()
    return Well_Analog_Dict

def load_PKL_data(data_dir,fname_dict, col_filter):
    info_file = fname_dict['info']
    analog_file = fname_dict['analog']
    card_file = fname_dict['card']
    event_file = fname_dict['event']

    filename = data_dir + info_file + '.pkl'
    with open(filename, "rb") as f:
        Well_Info_Dict = pickle.load(f)
    API_LIST = list(Well_Info_Dict.keys())

    df_init=0
    for api in API_LIST:
        if len( Well_Info_Dict[api] )>0:
            if df_init==0:
                wellInfo = pd.DataFrame(Well_Info_Dict[api][0],columns=Well_Info_Dict[api][0].columns)
                df_init=1
                continue
            wellInfo = wellInfo.append(Well_Info_Dict[api][0])

    # Filter Wells
    cond = filter_wells(wellInfo,col_filter)

    wellInfo_filtered = wellInfo[cond].filter(items=['API14', 'WELL_AUTO_NAME','COMP_TYPE',
                                                     'WELL_STATUS','STATUS_DATE','MOP','RMT_TEAM'])

    wellInfo_filtered.set_index('API14', inplace=True)

    API_LIST = list(wellInfo_filtered.index)

    #
    filename = data_dir + event_file + '.pkl'
    with open(filename, "rb") as f:
        Well_Event_All = pickle.load(f)
    Well_Event_Dict = {}
    for api in API_LIST:
        Well_Event_Dict.update({api:Well_Event_All[api][0]})

    del Well_Event_All
    gc.collect()

    # ANALOG DATA
    filename = data_dir + analog_file +'.pkl'
    Well_Analog_Dict = pivot_analog_table(filename, API_LIST)


    # CARD DATA
    filename = data_dir + card_file + '.pkl'
    with open(filename, "rb") as f:
        Well_Card_All= pickle.load(f)
    Well_Card_Dict = {}
    an_apis = list( Well_Card_All.keys() )
    for api in API_LIST:
        if api in an_apis:
            if len(Well_Card_All[api])>0:
                Well_Card_Dict.update({api:Well_Card_All[api][0].filter(
                    items=['plot_date', 'card_type', 'SurfaceCardB','DownholeCardB','POCDownholeCardB'])})
            else:
                Well_Card_Dict.update({api:[]})


    del Well_Card_All
    gc.collect()

    return wellInfo_filtered, Well_Event_Dict, Well_Analog_Dict, Well_Card_Dict

def RMT_PKLs_toDict(RMT_LIST, data_dir, fname_dict_base, col_filter, save_name_root, run_prll):
    # Loads pkl data for RMT's
    # Filters for specific wells
    #
    # Writes them back in a dictionary format


    for RMT in RMT_LIST:
        fname_dict = {key: RMT + '_' + fname_dict_base[key] for key in fname_dict_base}

        #     wellInfo_filtered = load_PKL_data(data_dir,fname_dict,col_filter)

        [wellInfo, Well_Event, Well_Analog, Well_Card] = load_PKL_data(data_dir, fname_dict,col_filter)

        API_LIST = list(wellInfo.index)

        # CONSOLIDATE FAILURE DATA
        (well_data_cons, annual_failure_stat, failure_events_stat, primary_failure_keywords,
         secondary_failure_keywords) = consolidate_failure_dataset_prll(API_LIST, Well_Card, Well_Analog, Well_Event, run_prll)

        # SAVE CONSOLIDATE DATA
        failure_keywords = {'primary': primary_failure_keywords, 'secondary': secondary_failure_keywords}
        with open(cwd + '\\' + RMT + save_name_root , 'wb') as f:
            pickle.dump([well_data_cons, failure_keywords, annual_failure_stat, failure_events_stat], f)

        del (wellInfo, Well_Event, Well_Analog, Well_Card, well_data_cons, annual_failure_stat, failure_events_stat,
             primary_failure_keywords, secondary_failure_keywords)
        gc.collect()


def filter_for_failures(parameters_failures):
    RMT_LIST            = parameters_failures['RMT_LIST']
    fname_dict_base     = parameters_failures['fname_dict_base']
    year_min_to_filter  = parameters_failures['year_min_to_filter']
    save_name_root      = parameters_failures['save_name_root'][0]
    PRIMARY_FAIL        = parameters_failures['PRIMARY_FAIL']
    SECONDARY_FAIL      = parameters_failures['SECONDARY_FAIL']

    cons_data_failure   = {}
    failure_events_stat = {}
    annual_failure_stat = {}
    k=0
    for RMT in RMT_LIST:
        # LOAD CONSOLIDATE DATA
        fname_dict={key:RMT + '_' + fname_dict_base[key] for key in fname_dict_base}

        with open(cwd+'\\' + RMT + save_name_root,'rb') as f:
            well_data_cons, failure_keywords, annual_failure_stat[RMT], failure_events_stat[RMT] = pickle.load(f)

        annual_failure_stat[RMT].set_index('Year', inplace=True)

        # annual_failure_stat[RMT] = AnFaSt
        # failure_events_stat[RMT] = FaEvSt

        k = k + 1
        if k==1:
            annual_failure_stat['ALL'] = annual_failure_stat[RMT].copy()
            failure_events_stat['ALL'] = failure_events_stat[RMT].copy()
        else:
            annual_failure_stat['ALL'] = annual_failure_stat['ALL'].add(annual_failure_stat[RMT], fill_value=0)
            for key1 in failure_events_stat[RMT]:
                for key2 in failure_events_stat[RMT][key1]:
                    if key2 in failure_events_stat['ALL'][key1]:
                        failure_events_stat['ALL'][key1][key2] = failure_events_stat['ALL'][key1][key2].append(
                            failure_events_stat[RMT][key1][key2], ignore_index=True).sort_values(by=['num_failure'],ascending =False)
                    else:
                        failure_events_stat['ALL'][key1][key2] = failure_events_stat[RMT][key1][key2].copy().sort_values(by=['num_failure'], ascending=False)


        fail_data = failure_events_stat[RMT][PRIMARY_FAIL][SECONDARY_FAIL]

        firstEventYear = [min(fail_data['event_date'].iloc[ind]).year for ind in range(fail_data.shape[0]) ]
        lastEventYear  = [max(fail_data['event_date'].iloc[ind]).year for ind in range(fail_data.shape[0]) ]
        firstEventYear = np.asarray(firstEventYear)
        lastEventYear = np.asarray(lastEventYear)

        iloc_target = np.where(lastEventYear>year_min_to_filter)[0]
        api_list = failure_events_stat[RMT][PRIMARY_FAIL][SECONDARY_FAIL]['API'].iloc[iloc_target].tolist()
        fail_dates = failure_events_stat[RMT][PRIMARY_FAIL][SECONDARY_FAIL]['event_date'].iloc[iloc_target].tolist()

        # ISOLATE FAILURE WELLS
        fail_dates_dict = dict(zip(api_list,fail_dates))
        for api in api_list:
            fail_dates_dict[api]= sorted(fail_dates_dict[api])
            cons_data_failure.update({api:well_data_cons[api]})

        del well_data_cons
        gc.collect()


    return cons_data_failure, failure_keywords, annual_failure_stat, failure_events_stat


