import numpy as np
import pandas as pd
import pickle
import matplotlib
import matplotlib.pyplot as plt
import os
from sys import getsizeof
import gc

font = {'family': 'normal',
        'weight': 'bold',
        'size': 22}

matplotlib.rc('font', **font)

import datetime

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from failure_load_control import filterFailures_and_getWellsegments, segmented_consolidate_data
from failure_featureEngineering import featureEng_fail_data


def prepare_train_data(RMT, data_dir, start_date, end_date,  failureDataSetName):
    ############################################# DATA PREPARATION/LOAD #######################################
    run_prep=True
    if run_prep:
        RMT_LIST = [RMT]

        ### Load Event Data, filter failures by time and type and get Well segments ###
        event_file_dir = {'well_event': data_dir + RMT + '_Well_Event_Hist.pkl',
                          'well_info':  data_dir + RMT + '_Well_LIST_Info.pkl' }

        # Filter for time-of-failure
        year_min_failure = '2018'

        # Filter wells with Specific Failure Types
        # failure_filter_dict = {'PRIMARY_FAILURE': ['ROD PUMP FAILURE'],
        #                        'SECONDARY_FAILURE': ['SANDED']}
        #
        # failure_word_in_col = {'FOREIGN_MATERIAL_TYPE': 'SAND'}

        failure_filter_dict = {}
        failure_word_in_col = {}

        parameters_seg = {'start_date': start_date,
                          'end_date': end_date}

        api_list_fails, well_segments, FAILURE_DICT, event_summ_df = filterFailures_and_getWellsegments(event_file_dir,
                                                                                                        year_min_failure,
                                                                                                        failure_filter_dict,
                                                                                                        failure_word_in_col,
                                                                                                        parameters_seg)

        ##### Consolidate Failure Data [Analog+Card] and Add Card Features #####
        consolidation_opts = {
            'info_col_filter': {'cols': ['COMP_TYPE', 'MOP'],
                                'vals': ['PROD_OIL', ['BEAM', 'BEAM_POC']]},

            'fname_dict_base': {'info': 'Well_List_Info',
                                'analog': 'Well_Analog_Hist_CleanName',
                                'card': 'Well_Card_Hist_Decoded'},

            'RMT_LIST': RMT_LIST,
            'min_date': parameters_seg['start_date'],
            'data_dir': data_dir}

        # Add Card Features
        card_features_opts = {'card_types': ['P', 'N'],
                              'card_features_list': ['area', 'area_above', 'area_below', 'perimeter',
                                                     'load_norm_center', 'position_norm_center', 'areaToPerimeter',
                                                     'load_max', 'load_sum', 'position_max', 'maxpos_ratio',
                                                     'maxPosition_at_minLoad', 'minPosition_at_maxLoad',
                                                     'load_atMaxPos', 'position_max',
                                                     'st_delvel_mean', 'st_delvel_var',
                                                     'st_vel_mean', 'st_vel_var', 'st_acc_mean', 'st_acc_var',
                                                     'ups_vel_mean', 'ups_vel_var', 'ups_acc_mean', 'ups_acc_var',
                                                     'dns_vel_mean', 'dns_vel_var', 'dns_acc_mean', 'dns_acc_var',
                                                     'ups_n_inflc', 'dns_n_inflc']}

        segmented_failure_data, segments_info, well_data_cons = segmented_consolidate_data(api_list_fails,
                                                                                           well_segments,
                                                                                           consolidation_opts,
                                                                                           card_features_opts)

        ##### save Consolidated failure data #####
        data_save_dir = data_dir + RMT + '_consFailureData.pkl'

        with open(data_save_dir, 'wb') as f:
            pickle.dump([well_data_cons, segmented_failure_data, segments_info, well_segments, FAILURE_DICT, event_summ_df], f)
    else:
        ##### load Consolidated failure data #####
        data_load_dir = data_dir + RMT + '_consFailureData.pkl'
        with open(data_load_dir, 'rb') as f:
            well_data_cons, segmented_failure_data, segments_info, well_segments, FAILURE_DICT, event_summ_df = pickle.load(f)

    ############################################# Feature Engineering #######################################

    # Sliding Window Feature Generation
    cols_analog = ['Runtime_Daily', 'SPM', 'CommPct_Daily', 'PumpFillage', 'Cycles_Daily', 'Load_Min_Daily',
                   'Load_Max_Daily', 'IdleTime_Daily', 'AGAGas']
    cols_cards = ['position_max', 'ups_vel_var', 'maxpos_ratio', 'maxPosition_at_minLoad',
                  'load_norm_center', 'dns_vel_var', 'st_vel_var']

    main_features = cols_analog + cols_cards

    slideWin_pars = {'global_days': 50,
                     'longTerm_days': 20,
                     'shortTerm_days': 5,
                     'main_features': main_features,
                     'window_stat_mtd': ['mean', 'var'],
                     'YLABEL': [],
                     'timeDerivative': True,
                     'unchanged': True,
                     'unchThreshold': 0.1,
                     'min_daysOfData': 10,
                     'parallel_run': True}

    time_res_pars = {'freq': '1D',
                     'interp_tol': '3day'}

    # Change data resolution
    timeRes_pars = {'data_dir': data_dir,
                    'RMT': RMT,
                    'start_date': start_date,
                    'end_date': end_date,
                    'freq': '1D',
                    'interp_tol': '3day'}

    model_pars = {'slideWin_pars': slideWin_pars,
                  'timeRes_pars': timeRes_pars}


    data_failure_slidewin, data_failure_ts, incomplete_wells = featureEng_fail_data( segmented_failure_data,
                                                                  timeRes_pars, slideWin_pars)

    ml_data_dir = data_dir + failureDataSetName
    with open(ml_data_dir, 'wb') as f:
        pickle.dump([data_failure_slidewin, data_failure_ts, segments_info, model_pars], f)

    return incomplete_wells