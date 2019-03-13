

from source.failure_machine_learning import filterSegs_and_getTrainTest, modelFit, training_evaluation

import copy, pickle

import matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from source.failure_machine_learning import  (get_target_vector, extract_segments_for_modeling,
									   plot_binary_metrics, predict_failure_model,
									   test_failure_model)

from keras.models import Sequential

def deploy_failure_model(trained_model, model_pars, deploy_opts, date_lim, future_fails):

	if deploy_opts['run_model'] == True:
		data_model = model_pars['data_model']
		ml_model_dir = deploy_opts['data_dir'] + deploy_opts['input_data_name']
		with open(ml_model_dir, 'rb') as f:
			data_failure_slidewin, data_failure_ts, segments_info, _ = pickle.load(f)

		segs_ML_dict = copy.deepcopy(data_failure_slidewin)
		seg_filter = {'CENSORED': ['CENSORED'],
					  'ROD PUMP FAILURE': ['SANDED', 'LOW PRODUCTION']}

		segments_info.loc[segments_info['SECONDARY_FAILURE'].isna(), ['SECONDARY_FAILURE']] = 'None'

		segs_ML_dict_model, seg_df_model = extract_segments_for_modeling(segs_ML_dict, segments_info, seg_filter)

		segs_ML_dict_model = get_target_vector(segs_ML_dict_model, seg_df_model, model_pars['pre_fail_days'])


		holdout_ratio = 0
		failure_stat_ml, failure_results_ml = predict_failure_model('ML', trained_model, model_pars['data_model'],segs_ML_dict_model,model_pars['columns_all'],
																	seg_df_model, holdout_ratio,model_pars['failure_decision_pars'], date_lim, future_fails)

		if len(future_fails)==0:
			deploy_results_dir = deploy_opts['data_dir'] + 'ML_Deploy_Results_' +  deploy_opts['results_root_name']
			with open(deploy_results_dir, 'wb') as f:
				pickle.dump([failure_stat_ml, failure_results_ml, segs_ML_dict_model,seg_df_model],f)

        
def test_trained_failure_model(trained_model, model_pars, failure_decision_pars, train_opts, date_lim):
	configs = model_pars['configs']
	if train_opts['plot_test'] == True:
		data_model = model_pars['data_model']

		ml_data_dir = train_opts['data_dir'] + 'ML_' + train_opts['input_data_name']
		with open(ml_data_dir, 'rb') as f:
			segs_ML_dict_model, seg_df_model,_, _, _ = pickle.load(f)

		test_failure_model('ML', trained_model, data_model, model_pars['columns_all'], segs_ML_dict_model,
						   seg_df_model, configs['holdout'], failure_decision_pars, [], date_lim)

		# SAVE VERIFIED MODEL
	if train_opts['save_trained_model'] == True:
		model_pars.update({'failure_decision_pars': failure_decision_pars})

		model_save_name = train_opts['data_dir'] + train_opts['trained_model_name']
		with open(model_save_name, 'wb') as f:
			pickle.dump([trained_model,  model_pars], f)


def train_failure_model(RMT, data_dir, failureDataSetName):

	configs   =  {'YLABEL': 'Target',
				  'drop_cols_x': ['Target'],
				  'val_size': 0.2,
				  'holdout': 0.02,
				  'fail_ratio': 0.1,
				  'n_samples': 100000}

	train_opts =  {'model_type': 'ML',
					'data_dir': data_dir,
					'input_data_name': failureDataSetName,
					'model_save_name': RMT + failureDataSetName[:-4] + 'RF_trainedModel.pkl',
					'run_split': True,
					'pre_fail_days': 30,
					'lurning_curve': False,
					'plot_loss': True}


	# LOAD PREPPED DATA

	ml_data_dir = train_opts['data_dir'] + RMT +  train_opts['input_data_name']
	with open(ml_data_dir, 'rb') as f:
		df_sw, df_ts, seginf, model_pars= pickle.load(f)

	# CORRECT INDICES
	nex_index = dict((indx, RMT + '_' + str(indx)) for indx in seginf.index.tolist())
	seginf.rename(index=nex_index, inplace=True)

	key_list = list(df_sw.keys())
	for key in key_list:
		new_key = RMT + '_' + str(key)
		df_sw[new_key] = df_sw.pop(key)


	key_list = list(df_ts.keys())
	for key in key_list:
		new_key = RMT + '_' + str(key)
		df_ts[new_key] = df_ts.pop(key)

	segments_info = seginf
	data_failure_slidewin = df_sw
	data_failure_ts = df_ts



	# Split wells into Train/holdout and Generate Global Training Matrix
	if train_opts['run_split']:
		segs_ML_dict = copy.deepcopy(data_failure_slidewin)

		segs_ML_dict_model, seg_df_model, columns_all, train_set, test_set = filterSegs_and_getTrainTest( segs_ML_dict,
																										  segments_info,
																										  configs,
																										  train_opts['pre_fail_days'] )

		# mmscaler = MinMaxScaler()
		# mmscaler.fit(np.append(X_train,X_test,axis=0))
		# X_train = mmscaler.transform(X_train)
		# X_test  = mmscaler.transform(X_test)
		ml_data_dir = train_opts['data_dir'] + 'ML_' + train_opts['input_data_name']
		with open(ml_data_dir, 'wb') as f:
			pickle.dump([segs_ML_dict_model, seg_df_model, columns_all, train_set, test_set], f)
	else:
		ml_data_dir = train_opts['data_dir'] + 'ML_' + train_opts['input_data_name']
		with open(ml_data_dir, 'rb') as f:
			segs_ML_dict_model, seg_df_model, columns_all, train_set, test_set = pickle.load(f)

	ml_models = {}
	prec_rec_intersect=[]
	for v in range(len(train_set)):

		X_train = train_set[v]['X']
		y_train = train_set[v]['y']
		X_test = test_set[v]['X']
		y_test = test_set[v]['y']

		print( '\n *** RUNNING RANDOM SAMPLE # %s *** \n' % str(v+1) )

		print('# columns: %s' % X_train.shape[1] )
		print('# examples: training=%s  test=%s' % (X_train.shape[0],X_test.shape[0]) )
		print('# training positives   = %s' % y_train.mean())
		print('# test positives   = %s' % y_test.mean())

		# SCALER
		input_scaler = MinMaxScaler()
		input_scaler.fit(X_train)
		scale_inputs=False
		if scale_inputs:
			X_train = input_scaler.transform(X_train)
			X_test = input_scaler.transform(X_test)

		data_model = []
		model_pars.update( {'pre_fail_days': train_opts['pre_fail_days'],
							'train_opts':train_opts,
							'configs':configs,
							'columns_all':columns_all,
							'input_scaler':input_scaler,
							'data_model':data_model} )

		if 'Target' in columns_all:
			columns_all.remove('Target')

		# RUN ML MODEL
		model_ml , pr_int = train_ml_model(train_opts, X_train, y_train, X_test, y_test, columns_all)

		ml_models.update({v:model_ml})
		prec_rec_intersect.append(pr_int)

	# print('Precision & Recall Intersection = %s'% prec_rec_intersect)

	if len(train_set)>2:
		print('Precision & Recall [ Mean = %s , Std = %s ]' % (np.mean(prec_rec_intersect),np.std(prec_rec_intersect)))


	model_save_name = train_opts['data_dir'] + train_opts['model_save_name']
	with open(model_save_name, 'wb') as f:
		pickle.dump([ml_models, model_pars], f)


	return ml_models, model_pars


def train_ml_model(train_opts, X_train, y_train, X_test, y_test,predictors):
	# features_retained = select_features(X_train)

	# Model Training
	### Sample rows from flatMatrix
	# key_list = list(segs_ML_dict_model.keys())
	# predictors = list(segs_ML_dict_model[key_list[0]].columns)
	# for rm in ['Target', 'api']:
	# 	predictors.remove(rm)

	n_estimators = 0
	featureIm = 20
	val_loss = False
	staged_loss = False

	model_opt = 2
	n_estimators_list = [500]
	error_rate = []

	nn_pars = []
	for n_estimators in n_estimators_list:
		if model_opt == 1:
			print('Applying XGB algorithm ... ')
			n_estimators = 50
			featureIm = 0
			val_loss = True
			staged_loss = True
			model_ml = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=0.01, subsample=0.9,
												  max_depth=None, min_weight_fraction_leaf=0.001,
												  max_features=None, random_state=0)
		elif model_opt == 2:
			print('Applying RF algorithm ... ')
			# n_estimators = 500
			featureIm = 10
			val_loss = False
			staged_loss = False
			model_ml = RandomForestClassifier(n_estimators=n_estimators, warm_start=True,
											  oob_score=True, random_state=10)
		elif model_opt == 3:
			print('Applying LR algorithm ... ')
			model_ml = LogisticRegression()
		elif model_opt == 4:
			print('Applying MLP algorithm ... ')
			val_loss = True
			model_ml = MLPClassifier(hidden_layer_sizes=[1000, 1000, 500, 100], random_state=10, verbose=True,
									 early_stopping=False,
									 alpha=0.0001, activation='relu', learning_rate='adaptive', learning_rate_init=0.01,
									 solver='adam')
		elif model_opt == 5:
			print('Applying Nueral network ... ')
			val_loss = True

			model_ml = Sequential()

			# actfun = ['linear', 'relu', 'sigmoid', 'tanh']

			nn_pars = {'layer_nodes': [500, 100, 50, 1],
					   'layer_acfun': ['relu', 'relu', 'relu', 'sigmoid'],
					   'dropout': 0.3,
					   'epoch': 50}



		elif model_opt == 6:
			print('Applying SVC algorithm ... ')
			model_ml = SVC(probability=True, kernel='rbf')

		elif model_opt == 7:
			model_ml = AdaBoostClassifier(n_estimators=10)

		# X_summ = pd.DataFrame(data=np.amax(X_train,axis=0),index=predictors)
		# print(X_summ)

		y_predprob, cv_score, featImp, loss_vecs = modelFit(model_ml, X_train, y_train, X_test,
															y_test, predictors, nCV=0, featureImp=featureIm,
															n_estimators=n_estimators, val_loss=val_loss,
															staged_loss=staged_loss, nn_pars=nn_pars)
		if len(n_estimators_list) > 1:
			error_rate.append(1 - model_ml.oob_score_)

	if len(n_estimators_list) > 1:
		plt.plot(n_estimators_list, error_rate)
		plt.show()

	if model_opt not in [1, 4, 5]:
		train_opts['plot_loss'] = False

	training_evaluation(train_opts['lurning_curve'], train_opts['plot_loss'], loss_vecs, model_ml, X_train, y_train)

	# Model Selection
	plot_opts = {'roc_curve': False,
				 'PR_curve': False,
				 'threshold_curve': True}

	prec_rec_intersect = plot_binary_metrics(y_test, y_predprob[:, 1], plot_opts)

	return model_ml, prec_rec_intersect


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

		output = precision[np.argmin(np.abs(precision-recall))]
		return output