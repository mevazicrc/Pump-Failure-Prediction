import os
cwd = os.getcwd()
import pickle
import numpy as np
import pandas as pd
from source.failure_load_functions import RMT_PKLs_toDict, filter_for_failures
from source.cardData_Model import enumerate_cards_prll

from multiprocessing import cpu_count, Pool
import  cProfile

def failSummary_filtered_wells(RMT_LIST, api_list, data_dir ):
    well_fail_info = pd.DataFrame(columns=['RMT', 'WNAME'], index=api_list)
    for RMT in RMT_LIST:
        with open(data_dir + "\\" + RMT + "_Well_List_Info.pkl", "rb") as f:
            well_info = pickle.load(f)
        for api in api_list:
            if api in list(well_info.keys()):
                well_fail_info.loc[api]['RMT'] = RMT
                well_fail_info.loc[api]['WNAME'] = well_info[api][0]['WELL_AUTO_NAME'][0]

    well_fail_events = {}
    for RMT in RMT_LIST:
        with open(data_dir + "\\" + RMT + "_Well_Event_Hist.pkl", "rb") as f:
            well_event = pickle.load(f)
        for api in api_list:
            if api in list(well_event.keys()):
                well_fail_events[api] = well_event[api][0]

    return well_fail_info, well_fail_events


def add_card_features_seq(job_args):

    cons_data_failure, card_features_list, card_types, card_prll_run = job_args
    # ADD CARD FEATURES TO THE WELL CONSOLIDATED DATA
    api_list = list(cons_data_failure.keys())

    for api in api_list:
        cards = cons_data_failure[api].filter(items=['card_type', 'DownholeCardB', 'SurfaceCardB'])

        if cards.shape[0]==0:
            continue

        card_enum = enumerate_cards_prll(cards[cards['card_type'].isin(card_types)], card_prll_run)

        iloc_valid = np.where(cards['card_type'].isin(card_types))[0]
        if len(iloc_valid) > 1:
            # cols_ftr = np.full((cards.shape[0], len(card_features_list)), np.nan)
            for ftr in card_features_list:
                col_ftr = np.full((cards.shape[0], 1), np.nan)
                np.put(col_ftr, iloc_valid, card_enum[ftr])
                cons_data_failure[api][ftr] = col_ftr

    return cons_data_failure


def add_card_features_prll(cons_data_failure, card_features_list, card_types):
    # ADD CARD FEATURES TO THE WELL CONSOLIDATED DATA
    api_list = list(cons_data_failure.keys())

    card_prll_run = False

    # parallelize
    cores = cpu_count()  # Number of CPU cores on your system
    partitions = cores  # Define as many partitions as you want
    API_split = np.array_split(api_list, partitions)

    well_data = []
    flist = []
    ctypes = []
    prun=[]
    for i in range(partitions):
        flist.append(card_features_list)
        prun.append(card_prll_run)
        ctypes.append(card_types)

        well_data.append({})
        for api in API_split[i]:
            well_data[i].update({api: cons_data_failure[api]})

    pool = Pool(cores)
    output = []
    job_args = [(well_data[i], flist[i], ctypes[i], prun[i]) for i in range(partitions)]
    output.append(pool.map(add_card_features_seq, job_args))
    pool.close()
    pool.join()

    cons_data_failure = {}
    for i in range(partitions):
        cons_data_failure.update(output[0][i])

    return cons_data_failure

def load_prepare_fail_data(inputs):

    load_pkl, data_dir, fname_dict_base, col_filter, run_filter_failure, parameters_failures, card_types, card_features_list, run_prll = inputs

    RMT_LIST = parameters_failures['RMT_LIST']
    # pr = cProfile.Profile()
    # pr.enable()
    if load_pkl==True:
        # Loads RMT PKL data, filters them, consolidates data and writes well dictionaries to a new PKL
        RMT_PKLs_toDict(RMT_LIST, data_dir, fname_dict_base, col_filter, parameters_failures['save_name_root'][0], run_prll)

    # pr.disable()
    # pr.print_stats(sort='time')

    # FILTER AND CONSOLIDATE FAILURES
    data_save_name = parameters_failures['proj_name'] + '_' + parameters_failures['SECONDARY_FAIL'] + parameters_failures['save_name_root'][1]

    if run_filter_failure == True:
        cons_data_failure, failure_keywords, annual_failure_stat, failure_events_stat = filter_for_failures( parameters_failures)

        with open(cwd + '\\' + data_save_name, 'wb') as f:
            pickle.dump([cons_data_failure, failure_keywords, annual_failure_stat, failure_events_stat], f)
    else:
        with open(cwd + '\\' + data_save_name, 'rb') as f:
            cons_data_failure, failure_keywords, annual_failure_stat, failure_events_stat = pickle.load(f)

    api_list = list(cons_data_failure.keys())


    # ADD CARD FEATURES TO CONSOLIDATED DATA
    data_failure_with_card = add_card_features_prll(cons_data_failure, card_features_list, card_types)


    return data_failure_with_card, failure_keywords, annual_failure_stat, failure_events_stat

