from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation, svm
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.grid_search import GridSearchCV

from scipy import sparse
import pandas as pd
import numpy as np
import gc

import scipy.sparse as sparse
import scipy.io
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from sklearn.utils import shuffle

print "Reading Data"
train_feats = pd.read_csv('data/feats_train.csv')
test_feats = pd.read_csv('data/feats_test.csv')
train = pd.read_csv('data/train_convai_wordchars.csv')
test = pd.read_csv('data/test_convai_wordchars.csv')

test_ids = test['id']
train_ids = train['id']

print "concatenating"
train_feats.drop(['comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate',], axis = 1, inplace = True)
test_feats.drop(['comment_text'], axis = 1, inplace = True)

train = pd.concat([train.set_index('id'), train_feats.set_index('id')], axis=1, join='inner').reset_index()
test = pd.concat([test.set_index('id'), test_feats.set_index('id')], axis=1, join='inner').reset_index()

train.drop(['comment_text','id'], axis = 1, inplace = True)
test.drop(['comment_text','id'], axis = 1, inplace = True)


def runXGB(xgtrain):
	param = {}
	param['objective'] = 'binary:logistic'
	param['eta'] = 0.12
	param['max_depth'] = 5
	param['silent'] = 1
	param['eval_metric'] = 'logloss'
	param['min_child_weight'] = 1
	param['subsample'] = 0.5
	param['colsample_bytree'] = 0.7
	param['seed'] = 2017
	num_rounds = 400

	plst = list(param.items())
	model = xgb.train(plst, xgtrain, num_rounds)
	return model

labels = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']
y = train[labels]

train.drop(labels, axis = 1, inplace = True)

train_preds = np.zeros((train.shape[0],y.shape[1])) # create an empty predictions data frame
test_preds = np.zeros((test.shape[0],y.shape[1])) # create an empty predictions data frame

for i, label in enumerate(labels):
	print "Model Fitting for ", label

	dtrain = xgb.DMatrix(train, label=y[label])
	dtest = xgb.DMatrix(test)

	tune_params = {
		'eta': 0.1, # eta
		# 'n_estimators' : 1000, # 
		'scale_pos_weight': 0.8,
		'subsample': 0.4,
		'gamma' : 1.0,
		'colsample_bytree': 0.7,
		'min_child_weight': 1,
		'max_depth': 6,

		'num_parallel_tree': 1,
		'silent': True,
		'seed' : 2013,
		'nthread' : 4,
		'objective' : 'reg:logistic',
		'eval_metric': 'logloss',
	}

	# Tuning 
	# xgb_model = XGBClassifier(**tune_params)
	# param_test = {
	# 	'learning_rate':[0.1, 0.5]
	# }
	# gsearch = GridSearchCV(estimator = xgb_model, param_grid = param_test,  n_jobs = 16, iid = False, cv = 5)
	# gsearch.fit(train, y[label])
	# print gsearch.best_params_
	# print gsearch.best_score_

	## Use Cross Validation
	xgb_cv_res = xgb.cv(tune_params, dtrain, num_boost_round = 1500, nfold = 5, stratified = False, early_stopping_rounds = 50, 
										seed = 2013, verbose_eval = 5, show_stdv = False)
	best_nrounds = xgb_cv_res.shape[0] - 1
	model = xgb.train(tune_params, dtrain, best_nrounds) 
	train_output = model.predict(dtrain)
	print "LogLoss (Train): %f" % metrics.log_loss(y[label], train_output)	
	# feat_imp = pd.Series(model.get_fscore()).sort_values(ascending=False)
	# print feat_imp

	# model = runXGB(dtrain)
	train_preds[:,i] = model.predict(dtrain)
	test_preds[:,i] = model.predict(dtest)
	gc.collect()
	# exit(0)

print "Writing"

labels_rf = [g+"_rf" for g in labels]

test_res = pd.DataFrame(test_preds, columns=labels_rf)
test_pred = pd.concat([test_ids, test_res], axis=1)
test_pred.to_csv('sub/xgb_feats_test.csv', index=False)

train_res = pd.DataFrame(train_preds, columns=labels_rf)
train_pred = pd.concat([train_ids, train_res], axis=1)
train_pred.to_csv('sub/xgb_feats_train.csv', index=False)