import numpy as np
import pandas as pd
import matplotlib
import os
import matplotlib.pyplot as plt
import datetime as dt

import matplotlib.dates as mdates

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
matplotlib.rc('font', **font)
mingw_path = 'C:\\Program Files\\mingw-w64\\mingw64\\bin'

os.environ['PATH'] =  os.environ['PATH'] + ';' + mingw_path
# import xgboost as xgb

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

# from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (recall_score, accuracy_score, precision_score,
                             roc_auc_score, average_precision_score, precision_recall_curve)
# from sklearn import cross_validation

# from sklearn.grid_search import GridSearchCV   #Perforing grid search

from matplotlib.dates import (MONTHLY, DateFormatter,
                              rrulewrapper, RRuleLocator, drange)

from multiprocessing import cpu_count, Pool

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.constraints import maxnorm


import matplotlib
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import matplotlib.dates as mdates

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def training_evaluation(lcurve, plot_loss, loss_vecs, model_ml, X_train, y_train):

    if plot_loss == True:
        tr_loss, val_loss = loss_vecs
        fig = plt.figure(figsize=(20,10))
        plt.plot(tr_loss,'g.-')
        plt.plot(val_loss,'k.-')
        plt.title('model training vs validation loss')
        plt.ylabel('loss')
        plt.xlabel('n_estimator')
        plt.legend(['training', 'validation'], loc='upper right')
        plt.grid()
        plt.show()

    if lcurve==True:
        train_sizes = [0.1, 0.3, 0.5, 0.8, 1]
        train_size, train_scores, valid_scores = learning_curve(model_ml, X_train, y_train, train_sizes=train_sizes)
        plt.figure(figsize=(15, 10))
        plt.plot(train_sizes, train_scores, label='Training')
        plt.plot(train_sizes, valid_scores, 'k', label='validation')
        plt.legend()
        plt.grid()

def filterSegs_for_TS(segs_TS_dict, segments_info, seg_filter, pre_fail_days):

    segments_info.loc[ segments_info['SECONDARY_FAILURE'].isna() , ['SECONDARY_FAILURE'] ] = 'None'

    segs_TS_dict_model, seg_df_model = extract_segments_for_modeling( segs_TS_dict , segments_info, seg_filter)

    segs_TS_dict_model = get_target_vector(segs_TS_dict_model, seg_df_model, pre_fail_days)

    return segs_TS_dict_model, seg_df_model

def filterSegs_and_getTrainTest(segs_ML_dict, segments_info, parameters_split, pre_fail_days):
    
    seg_filter = {'ROD PUMP FAILURE': ['SANDED', 'LOW PRODUCTION'],
                   'CENSORED': ['CENSORED']}

    segments_info.loc[ segments_info['SECONDARY_FAILURE'].isna() , ['SECONDARY_FAILURE'] ] = 'None'


    segs_ML_dict_model, seg_df_model = extract_segments_for_modeling( segs_ML_dict , segments_info, seg_filter)


    segs_ML_dict_model = get_target_vector(segs_ML_dict_model, seg_df_model, pre_fail_days)

    train_set, test_set, columns_all= split_and_extractXY(parameters_split, segs_ML_dict_model )

    return segs_ML_dict_model, seg_df_model, columns_all, train_set, test_set

def extract_segments_for_modeling(segs_ML_dict, seg_df_all, seg_filter):
    key_list_model=[]
    for key_p in seg_filter:
        if len(key_p)==0:
            if  len(seg_filter[key_p])==0:
                inds = seg_df_all
            else:
                inds = seg_df_all['SECONDARY_FAILURE'].isin(seg_filter[key_p])
        else:
            if  len(seg_filter[key_p])==0:
                inds = seg_df_all['PRIMARY_FAILURE'].isin([key_p])
            else:
                inds = seg_df_all['PRIMARY_FAILURE'].isin([key_p]) & seg_df_all['SECONDARY_FAILURE'].isin(seg_filter[key_p])


        key_list_model.extend(seg_df_all[inds].index.tolist())

    key_list_model = list(set(key_list_model))

    segs_ML_dict_model = {}
    for key in key_list_model:
        if key in segs_ML_dict.keys():
            if len(segs_ML_dict[key]) > 0:
                segs_ML_dict_model.update({key: segs_ML_dict[key]})

    key_list_model = sorted(list(segs_ML_dict_model.keys()))
    seg_df_model = seg_df_all.loc[key_list_model]


    return segs_ML_dict_model, seg_df_model


def get_target_vector(segs_ML_dict_model, seg_df_model, pre_fail_days):

    key_list_model = sorted(list(segs_ML_dict_model.keys()))

    for key in key_list_model:
        if segs_ML_dict_model[key].shape[0]==0:
            continue

        segs_ML_dict_model[key]['Target'] = 0
        segs_ML_dict_model[key]['daysFromStart'] = np.arange(1,segs_ML_dict_model[key].shape[0]+1)/365
        if seg_df_model.loc[key]['PRIMARY_FAILURE'] != 'CENSORED':
            fail_date = min([seg_df_model.loc[key]['SEGMENT_END'], segs_ML_dict_model[key].index[-1]])
            inds1 = (segs_ML_dict_model[key].index <= fail_date) & (
                        segs_ML_dict_model[key].index >= (fail_date - dt.timedelta(days=pre_fail_days)))
            segs_ML_dict_model[key].loc[inds1, 'Target'] = 1

    return segs_ML_dict_model

def blind_test_well(ALG, ALG_name, well_data, YLABEL, holdout_time_start, xlim, ptitle):

    # Test Period
    X_test, y_test, date_vec = get_XandY(well_data[well_data.index >= holdout_time_start], YLABEL)

    if X_test.shape[0]>0:

        y_pred = ALG.predict(X_test)
        print('Recall score: {0:0.2f}'.format(recall_score(y_test, y_pred)))
        print('Precision score: {0:0.2f}'.format(precision_score(y_test, y_pred)))

        # HISTORY
        X_api, y_api, date_vec = get_XandY(well_data, YLABEL)
        y_api_pred = ALG.predict(X_api)

        # print( date_vec.to_series().diff().dt.days)

        normal_after = np.ones((y_api_pred.shape))
        normal_after[ y_api_pred > 0 ] = 0
        for i in range(1, len(normal_after)):
            if normal_after[i]>0:
                normal_after[i] = normal_after[i] + normal_after[i - 1]

        cum_faildays = np.cumsum(y_api_pred)

        fail_start = np.where( np.diff(y_api_pred)==1)[0]
        for rs in fail_start:
            if (normal_after[rs]>5):
                cum_faildays[rs+1:] = cum_faildays[rs+1:] - cum_faildays[rs+1]


        miss_dates = np.where( date_vec.to_series().diff().dt.days > 5)[0]
        for md in miss_dates:
            cum_faildays[md:] = cum_faildays[md:] - cum_faildays[md]

        faildays = cum_faildays - normal_after
        faildays[faildays<0]=0

        y_pred_prob = faildays / 14
        y_pred_prob[y_pred_prob > 1] = 1
        y_pred_prob[y_pred_prob < 0] = 0
        # tfail = np.where(np.diff(y_api)==-1)[0]
        # for tf in tfail:
        #     y_pred_prob[tf+1:tf+14]=0

        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)
        plt.plot_date(pd.to_datetime([holdout_time_start, holdout_time_start]), [-5, 5], 'k-', linewidth=5)
        plt.plot_date(date_vec,y_api_pred,'ro',label='Predicted',markersize=10)
        plt.plot_date(date_vec, y_pred_prob, 'go-', label='Failure probability', markersize=7)
        plt.plot_date(date_vec,y_api,'bo',label='Actual',markersize=5)
        plt.legend(loc=6)
        plt.grid(color='k')
        plt.title(ptitle)
        plt.xlim(xlim)
        plt.ylim([-0.01, 1.01])
        # plt.setp(ax.get_yticklabels(), visible=False)

        ax.fmt_xdata = mdates.DateFormatter('%m-%d-%y')
        mloc = mdates.MonthLocator(range(1, 13), bymonthday=1, interval=1)
        monthsFmt = mdates.DateFormatter("%b '%y")
        ax.xaxis.set_major_locator(mloc)
        ax.xaxis.set_major_formatter(monthsFmt)
        ax.grid(axis='y')
        fig.autofmt_xdate()
        plt.show()


def print_classification_evals(y_test ,y_pred, test_score, roc_auc):

    print("\nModel Report")
    print("Test Accuracy : %.4g" % accuracy_score(y_test, y_pred))
    print('Test Recall score: {0:0.2f}'.format(recall_score(y_test, y_pred)))
    print('Test Precision score: {0:0.2f}'.format(precision_score(y_test, y_pred)))
    print("Test ROC-AUC Score : %f" % roc_auc)
    print('Test score = {0:0.2f}'.format(test_score))
    print('Test average precision-recall score: {0:0.2f}'.format(average_precision_score(y_test, y_pred)))






def modelFit(ALG, X_train, y_train, X_test, y_test, predictors, nCV, featureImp, n_estimators, val_loss, staged_loss, nn_pars):

    # Perform cross-validation:
    cv_score = []
    feat_imp = []
    if nCV>0:
        cv_score = cross_validation.cross_val_score(ALG, X_train, y_train, cv=nCV, scoring='roc_auc')

        print("CV roc_auc Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" %
              (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))

    if len(nn_pars)>0:        # input layer

        ALG.add(Dropout(nn_pars['dropout'], input_shape=(X_train.shape[1],)))

        # 1st Hidden layer
        ALG.add(Dense(nn_pars['layer_nodes'][0], input_dim=X_train.shape[1], activation=nn_pars['layer_acfun'][0]))
        for lay in range(1, len(nn_pars['layer_nodes']) ):
            ALG.add(Dense(nn_pars['layer_nodes'][lay], activation=nn_pars['layer_acfun'][lay]))

        # sgd = SGD(lr=0.001, momentum=0.8, decay=1e-6, nesterov=True)
        optim_opt = ['adam', 'rmsprop', 'sgd']
        loss_opt = ['mse','binary_crossentropy']

        ALG.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

        fit_hist = ALG.fit(X_train, y_train, nb_epoch=nn_pars['epoch'],
                           batch_size=round(0.2*X_train.shape[0]), verbose=1,
                                  validation_data=(X_test, y_test))

        loss = (fit_hist.history['loss'], fit_hist.history['val_loss'])

        y_predprob = ALG.predict_proba(X_test)

        y_predprob = np.concatenate((1-y_predprob,y_predprob),axis=1)


    else:
        ALG.fit(X_train, y_train)

        # y_score = ALG.decision_function(X_test)
        y_pred = ALG.predict(X_test)
        y_predprob = ALG.predict_proba(X_test)

        # Print model report:
        test_score = ALG.score(X_test, y_test)
        roc_auc = roc_auc_score(y_test, y_predprob[:,1])

        print_classification_evals(y_test, y_pred, test_score, roc_auc)

        # Print Feature Importance:
        if featureImp>0:
            feat_imp = pd.Series(ALG.feature_importances_, predictors).sort_values(ascending=False)
            plt.figure(figsize=(20,10))
            feat_imp[:featureImp].plot(kind='bar', title='Feature Importances')
            plt.ylabel('Feature Importance Score')
            plt.show()


        val_score=[]
        loss=[]
        if val_loss==True:
            if staged_loss==True:
                ALG.staged_predict(X_test)
                val_score = np.zeros((n_estimators,), dtype=np.float64)
                for i, y_pred in enumerate(ALG.staged_predict(X_test)):
                    val_score[i] = ALG.loss_(y_test, y_pred.reshape(-1, 1))

            loss = (ALG.train_score_,val_score)

    return  y_predprob, cv_score, feat_imp, loss


def Training_flat_matrix( data_flat_dict, val_ratio, holdout_ratio):
    key_list_all = list(data_flat_dict.keys())
    n_ho = max([1, int(holdout_ratio * len(key_list_all))])
    key_list = key_list_all[0:-n_ho]


    # GET VALID COLUMNS
    flat_mat_all = []
    for key in key_list_all:
        if len(data_flat_dict[key]) == 0:
            continue

        if len(flat_mat_all) == 0:
            flat_mat_all = data_flat_dict[key].copy(deep=True)
        else:
            flat_mat_all = flat_mat_all.append( data_flat_dict[key])

    # Drop columns
    flat_mat_all.dropna(thresh=int(0.7 * flat_mat_all.shape[0]), axis='columns', inplace=True)
    valid_columns = list(flat_mat_all.columns)

    kfold = 1
    flatMatrix_tr={}
    flatMatrix_ts={}
    for k in range(kfold):

        np.random.shuffle(key_list)

        tr_fm, ts_fm = get_test_train_parallel(key_list,val_ratio, data_flat_dict)

        # Drop columns
        tr_fm= tr_fm.filter(items=valid_columns)
        ts_fm = ts_fm.filter(items=valid_columns)

        # Drop rows
        tr_fm.dropna(axis='rows', inplace=True)
        ts_fm.dropna(axis='rows', inplace=True)

        flatMatrix_tr.update({k:tr_fm})
        flatMatrix_ts.update({k:ts_fm})

    return flatMatrix_tr, flatMatrix_ts

def get_test_train_parallel(key_list, val_ratio, data_flat_dict):

    key_list_tr = key_list[int(val_ratio * len(key_list)):]
    key_list_ts = key_list[0:int(val_ratio * len(key_list))]

    cores = cpu_count()-1  # Number of CPU cores on your system
    partitions = cores  # Define as many partitions as you want

    pool = Pool(cores)

    data_list = []
    for _ in range(cores):
        data_list.append(data_flat_dict)

    # Train set
    kl_tr = np.array_split(key_list_tr, partitions)
    data_in = zip(data_list,kl_tr)
    data_out = []
    data_out.append(pool.map(get_data_flatmat, data_in))
    pool.close()
    pool.join()
    flatMatrix_tr = data_out[0][0].copy(deep=True)
    for i in range(1, len(data_out[0])):
            flatMatrix_tr = flatMatrix_tr.append(data_out[0][i])

    pool = Pool(cores)
    # Test set
    kl_ts = np.array_split(key_list_ts, partitions)
    data_in = zip(data_list, kl_ts)
    data_out = []
    data_out.append(pool.map(get_data_flatmat, data_in))
    pool.close()
    pool.join()
    flatMatrix_ts = data_out[0][0].copy(deep=True)
    for i in range(1, len(data_out[0])):
            flatMatrix_ts = flatMatrix_ts.append(data_out[0][i])

    return flatMatrix_tr, flatMatrix_ts

def get_data_flatmat(data_in):
    data_dict, key_list = data_in
    flatmat = []
    for key in key_list:
        if len(data_dict[key]) == 0:
            continue
        # data_flat_dict[api]['api'] = api
        if len(flatmat) == 0:
            flatmat = data_dict[key].copy(deep=True)
        else:
            flatmat = flatmat.append(data_dict[key])

    return flatmat

def get_test_train(key_list, val_ratio, data_flat_dict):
    np.random.shuffle(key_list)

    key_list_tr = key_list[int(val_ratio * len(key_list)):]
    key_list_ts = key_list[0:int(val_ratio * len(key_list))]

    # TRAINING FLAT MATRIX
    flatMatrix_tr = []
    for key in key_list_tr:
        if len(data_flat_dict[key]) == 0:
            continue
        # data_flat_dict[api]['api'] = api
        if len(flatMatrix_tr) == 0:
            flatMatrix_tr = data_flat_dict[key].copy(deep=True)
        else:
            flatMatrix_tr = flatMatrix_tr.append(data_flat_dict[key])

    # TEST FLAT MATRIX
    flatMatrix_ts = []
    for key in key_list_ts:
        if len(data_flat_dict[key]) == 0:
            continue
        if len(flatMatrix_ts) == 0:
            flatMatrix_ts = data_flat_dict[key].copy(deep=True)
        else:
            flatMatrix_ts = flatMatrix_ts.append(data_flat_dict[key])

def get_XandY(ml_df, parameters):

    inds_norm = np.where(ml_df[parameters['YLABEL']] == 0)[0]
    inds_fail = np.where(ml_df[parameters['YLABEL']] == 1)[0]

    n_fail = int(min([parameters['fail_ratio'] * parameters['n_samples'], len(inds_fail)]))
    n_norm = int(min([n_fail * (1 - parameters['fail_ratio']) / parameters['fail_ratio'], len(inds_norm)]))

    inds_sample = np.append(np.random.choice(inds_norm, n_norm, replace=False),
                            np.random.choice(inds_fail, n_fail, replace=False))

    print('df-len=%s'%inds_sample.shape[0])


    ml_df = ml_df.iloc[inds_sample]

    ml_df.dropna(axis='rows', inplace=True)

    print('new df-len=%s' % ml_df.shape[0])

    X = ml_df.drop( columns=parameters['drop_cols_x'] ).values.astype(float)
    y = ml_df[parameters['YLABEL']].values.astype(int)

    # date_vec = ml_df.index
    #     api_vec = ml_df['api'].values

    return X, y


def sample_data_instances(flatMatrix_tr, flatMatrix_ts, parameters):
    np.random.seed(0)

    train_set = []
    test_set  = []
    for k in range(len(flatMatrix_tr)):

        x,y = get_XandY(flatMatrix_tr[k], parameters)
        train_set.append( {'X':x, 'y':y} )

        x, y = get_XandY(flatMatrix_ts[k], parameters)
        test_set.append({'X': x, 'y': y})

    return train_set, test_set

def split_and_extractXY(parameters, seg_ML_dict):
    # Split wells into Train/holdout
    holdout_ratio = parameters['holdout']

    flatMatrix_tr, flatMatrix_ts = Training_flat_matrix( seg_ML_dict, parameters['val_size'], holdout_ratio)

    # Sample rows from flatMatrix_tr
    train_set, test_set = sample_data_instances(flatMatrix_tr, flatMatrix_ts, parameters)

    # shuffle and split training and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parameters['val_size'], random_state=10)
    columns_all = list(flatMatrix_tr[0].columns)


    return train_set, test_set, columns_all
	


def plot_binary_metrics(y_true,y_predprob,plot_opts):
    fpr, tpr, thresholds = roc_curve(y_true,y_predprob)
    roc_auc = auc(fpr, tpr)

    if plot_opts['roc_curve']:
        plt.figure(figsize=(15,10))
        plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid()
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()

    precision, recall, thresholds = precision_recall_curve(y_true, y_predprob)

    if plot_opts['PR_curve']:
        plt.figure(figsize=(15,10))
        plt.plot(recall, precision, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid()
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('2-class Precision_Recall curve: AP={0:0.2f}'.format(average_precision_score(y_true, y_predprob)))
        plt.legend(loc="lower right")
        plt.show()

    if plot_opts['threshold_curve']:
        plt.figure(figsize=(15,10))
        plt.plot(thresholds, recall[:-1], color='darkorange', lw=1, label='Recall')
        plt.plot(thresholds, precision[:-1], color='navy', lw=1, label='precision')
        plt.grid()
        plt.legend(loc="lower right")
        plt.xlabel('Decision threshold')
        plt.ylabel('Precision and Recall')
        plt.show()



def Failure_Decision(date_vec, y_pred, fail_decision_opts):
    decision_threshold = fail_decision_opts['decision_threshold']
    days_to_failure = fail_decision_opts['days_to_failure']
    max_delta_days = fail_decision_opts['max_delta_days']
    alarm_confidence = fail_decision_opts['alarm_confidence']
    min_diffConfidence = fail_decision_opts['min_diffConfidence']
    days_to_reset_cumFailDays = fail_decision_opts['days_to_reset_cumFailDays']

    days_int = date_vec.to_series().diff().dt.days
    days_int[0] = 1
    days = np.cumsum(days_int)


    y_pred_cum = np.cumsum(y_pred)
    y_pred_cum_s = y_pred_cum[days_to_reset_cumFailDays:]
    reset_cums = np.where((y_pred_cum_s - y_pred_cum[:-days_to_reset_cumFailDays]) == 0)[0]

    conf_scale = np.full((y_pred.shape[0],),y_pred.shape[0])

    for t_reset in reset_cums:
        y_pred_cum[t_reset + 1:] = y_pred_cum[t_reset + 1:] - y_pred_cum[t_reset + 1]
        conf_scale[t_reset + 1:] = y_pred.shape[0] - t_reset

    conf_scale[conf_scale<days_to_failure] = days_to_failure

    # y_pred_pn = np.copy(y_pred)
    # inds_n_to_f = np.where(np.diff(y_pred)==-1)[0]
    #
    # for i in range(inds_n_to_f.shape[0]):
    #
    #     if i==0:
    #         inds_n_to_f = np.append(inds_n_to_f,[len(y_pred)])
    #
    #     vc = np.full(y_pred.shape, False, dtype=bool)
    #     vc[inds_n_to_f[i]:inds_n_to_f[i+1]]=True
    #
    #     y_pred_pn[np.logical_and(y_pred==0,vc)] = -0.5
    #
    #     y_pred_pn_cum = np.cumsum(y_pred_pn)
    #     y_pred_pn[np.logical_and(y_pred_pn_cum < 0, y_pred == 0)] = 0

    # y_pred_cum = np.cumsum(y_pred)

    #


    cumFailDays_ratio = y_pred_cum / conf_scale

    cumFailDays_ratio[cumFailDays_ratio>1] = 1

    fail_inds = np.where(cumFailDays_ratio > alarm_confidence)[0]
    fail_candidates = {}

    for i in fail_inds:
        ind_delta = np.where(np.logical_and((cumFailDays_ratio[i] - cumFailDays_ratio[:i]) > min_diffConfidence,
                                            (days[i] - days[:i]) <= max_delta_days))[0]
        if len(ind_delta) > 0:
            fail_candidates[i] = date_vec[i]

    if len(fail_candidates) > 1:
        inds = list(fail_candidates.keys())
        diff_days = np.where(np.diff(inds) < days_to_failure)[0]
        for i in diff_days:
            del fail_candidates[inds[i + 1]]

    return cumFailDays_ratio, fail_candidates


def predict_failure_model(ml_type, ML_MODEL, data_model, ML_dict_model,columns_all, seg_df_all, holdout_ratio,
						  failure_decision_pars, date_lim, future_fails):

    key_list_model = sorted(list(ML_dict_model.keys()))
    key_vec = key_list_model[-int(holdout_ratio * len(key_list_model)):]

    if len(future_fails) > 0:
        key_vec = seg_df_all[seg_df_all['api'].isin(future_fails['api'])].index.tolist()

    df_start = 1
    failure_results = {}
    show_point_preds = True

    for k, key in enumerate(key_vec):

        if ML_dict_model[key].shape[0] == 0:
            continue

        api = seg_df_all.loc[key].api

        # well_x_test = ML_dict_model[key].drop(columns=['Target']).values.astype(float)
        well_x_test = ML_dict_model[key].filter(columns_all).values.astype(float)
        well_y_test = ML_dict_model[key]['Target'].values
        date_vec = ML_dict_model[key].index

        n_nan = [len(np.where(np.isnan(well_x_test[i]))[0]) for i in range(well_x_test.shape[0])]
        t_valid = np.where(np.array(n_nan) == 0)[0]

        if len(t_valid) > 0:
            #         well_x_test  = mmscaler.transform(well_x_test[t_valid])

            try:
                y_pred = ML_MODEL.predict_proba(well_x_test[t_valid])[:, 1]
            except:
                continue

            y_pred[y_pred >= failure_decision_pars['decision_threshold']] = 1
            y_pred[y_pred < failure_decision_pars['decision_threshold']] = 0


            # Failure Decision
            cumFailDays_ratio, fail_candidates = Failure_Decision(date_vec[t_valid],
                                                                                     y_pred, failure_decision_pars)
            #         yactual = np.zeros((well_y_test[k][t_valid].shape[0],1))
            #         ind_fail = np.where(well_y_test[k][t_valid]==1)[0][-1]
            #         yactual[ind_fail]=1


            yactual = well_y_test

            if yactual.max()==0:
                failure_results[key] = {'yactual': yactual,
                                        'y_pred': y_pred,
                                        'cumFailDays_ratio': cumFailDays_ratio,
                                        'fail_candidates': fail_candidates,
                                        'date_vec': date_vec,
                                        't_valid': t_valid}
                if df_start == 1:
                    failure_stat = pd.DataFrame({'key': [key],
                                                 'last_date': [date_vec[t_valid[-1]]],
                                                 'failure_confidence': [cumFailDays_ratio[-1]]})
                    df_start = 0
                else:
                    failure_stat = failure_stat.append({'key': key,
                                                        'last_date': date_vec[t_valid[-1]],
                                                        'failure_confidence': cumFailDays_ratio[-1]}, ignore_index=True)


            if len(future_fails)>0:
                ptitle = ' [%s] %s' % (key, api)  # seg_df_all[seg_df_all.index==key]['api'].values[0]
                ptitle = ptitle + ' [Primary = ' + future_fails[future_fails['api']==api]['PRIMARY_FAILURE'].values[0]
                ptitle = ptitle + ', Secondary=' + future_fails[future_fails['api']==api]['SECONDARY_FAILURE'].values[0]
                ptitle = ptitle + ', Date=' + future_fails[future_fails['api']==api]['date'].values[0] + ' ]'
                ylabel = ''
                plot_well_forecast(yactual, y_pred, cumFailDays_ratio, fail_candidates, date_vec,
                                   t_valid, date_lim, ylabel, ptitle, show_point_preds)

            if yactual.max()>0:
                ptitle = ' [%s] %s' % (key, api)  # seg_df_all[seg_df_all.index==key]['api'].values[0]
                ptitle = ptitle + ' [Primary = ' + seg_df_all[seg_df_all.index == key]['PRIMARY_FAILURE'].values[0]
                ptitle = ptitle + ', Secondary=' + seg_df_all[seg_df_all.index == key]['SECONDARY_FAILURE'].values[
                    0] + ' ]'
                ylabel = ''
                plot_well_forecast(yactual, y_pred, cumFailDays_ratio, fail_candidates, date_vec,
                           t_valid, date_lim, ylabel, ptitle, show_point_preds)

    if df_start==0:
        failure_stat.set_index('key', inplace=True)
        failure_stat.sort_values(by=['failure_confidence', 'last_date'], ascending=False, inplace=True)
    else:
        failure_stat=[]
        failure_results=[]

    return failure_stat, failure_results


def test_failure_model(ml_type, ML_MODEL, data_model, columns_all, ML_dict_model, seg_df_all, holdout_ratio, failure_decision_pars, configs, xlim):

    key_list_model = list(ML_dict_model.keys())
    n_ho = max([1, int(holdout_ratio * len(key_list_model))])
    key_vec = key_list_model[-n_ho:]

    show_point_preds = True

    ylabel = 'Prob( t_fail <= ' + str(failure_decision_pars['days_to_failure']) + ' days )'


    for k,key in enumerate(key_vec):

        well_x_test = ML_dict_model[key].filter(columns_all).values.astype(float)
        well_y_test = ML_dict_model[key]['Target'].values
        date_vec = ML_dict_model[key].index

        api = seg_df_all.loc[key].api

        ptitle = ' [%s] %s' % (key, api)  # seg_df_all[seg_df_all.index==key]['api'].values[0]
        ptitle = ptitle + ' [Primary = ' + seg_df_all[seg_df_all.index == key]['PRIMARY_FAILURE'].values[0]
        ptitle = ptitle + ', Secondary=' + seg_df_all[seg_df_all.index == key]['SECONDARY_FAILURE'].values[0] + ' ]'
        print(ptitle)

        n_nan = [len(np.where(np.isnan(well_x_test[i]))[0]) for i in range(well_x_test.shape[0])]

        t_valid = np.where(np.array(n_nan) == 0)[0]

        print(len(t_valid))

        if len(t_valid) > 0:
            #         well_x_test  = mmscaler.transform(well_x_test[t_valid])

            y_pred = ML_MODEL.predict_proba( well_x_test[t_valid] )[:,1]


            y_pred[y_pred >= failure_decision_pars['decision_threshold']] = 1
            y_pred[y_pred < failure_decision_pars['decision_threshold']] = 0

            print('date_vec: %s'%len(date_vec))
            print('t_valid %s' % len(t_valid))
            print('y_pred %s' % len(y_pred))

            # Failure Decision
            cumFailDays_ratio, fail_candidates = Failure_Decision(date_vec[t_valid], y_pred, failure_decision_pars)
            #         yactual = np.zeros((well_y_test[k][t_valid].shape[0],1))
            #         ind_fail = np.where(well_y_test[k][t_valid]==1)[0][-1]
            #         yactual[ind_fail]=1
            yactual = well_y_test

            plot_well_forecast(yactual, y_pred, cumFailDays_ratio, fail_candidates, date_vec,
                                                  t_valid, xlim, ylabel, ptitle, show_point_preds)


def plot_well_forecast( y_test, y_pred, cumFailDays_ratio, fail_candidates, date_vec, t_valid, xlim, ylabel, ptitle, show_point_preds):

    if y_pred.shape[0] > 0:
        fig = plt.figure( figsize=(20, 10) )
        ax = fig.add_subplot(111)

        if show_point_preds:
            plt.plot_date(date_vec[t_valid], y_pred, 'ro', label='Predicted', markersize=10)


        plt.plot_date(date_vec, y_test, 'b-', label='Actual', linewidth=5)
        plt.legend(loc=6)
        plt.grid(color='k')
        plt.title(ptitle)
        plt.xlim(xlim)
        plt.ylim([-0.01, 1.01])
        # plt.setp(ax.get_yticklabels(), visible=False)

        ax.fmt_xdata = mdates.DateFormatter('%m-%d-%y')
        mloc = mdates.MonthLocator(range(1, 13), bymonthday=1, interval=1)
        monthsFmt = mdates.DateFormatter("%b '%y")
        ax.xaxis.set_major_locator(mloc)
        ax.xaxis.set_major_formatter(monthsFmt)
        ax.grid(axis='y')
        fig.autofmt_xdate()


        plt.plot_date(date_vec[t_valid], cumFailDays_ratio, 'y-', label='Failure confidence', markersize=8)
        for key in fail_candidates:
            dd = [fail_candidates[key], fail_candidates[key]]
            plt.plot_date(dd, [0, 1], 'r--', linewidth=3, label='Predicted failure time')
        plt.legend(loc=6)
        plt.grid(color='k')
        plt.ylabel(ylabel)
        plt.title(ptitle)
        plt.xlim(xlim)
        plt.show()
