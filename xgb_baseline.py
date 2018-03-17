from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation, svm
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import pandas as pd
import numpy as np
import gc

import scipy.sparse as sparse
import scipy.io
import xgboost as xgb

from sklearn.utils import shuffle

print "Reading Data"
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train.fillna(' ',inplace=True) # replace NA Values from train data
test.fillna(' ',inplace=True) # replace NA Values from train data
gc.collect()

# print "Fitting Word Vectors"
# vect_word = TfidfVectorizer(max_features=50000, lowercase=True, analyzer='word',
# 						stop_words= 'english',ngram_range=(1,3), dtype=np.float32) # Generate TF-IDF for Ngram Terms
# vect_word.fit(list(train['comment_text']) + list(test['comment_text'])) # Fit the complete data in TF-IDFs

# print "Fitting Character Vectors"
# vect_char = TfidfVectorizer(max_features=20000, lowercase=True, analyzer='char',
# 						stop_words= 'english',ngram_range=(1,3), dtype=np.float32) # Generate TF-IDF for Ngram Characters
# vect_char.fit(list(train['comment_text']) + list(test['comment_text'])) # Fit the complete data in TF-IDFs

# vect_char_bg = TfidfVectorizer(max_features=20000, lowercase=True, analyzer='char',
# 						stop_words= 'english',ngram_range=(4,5), dtype=np.float32) # Generate TF-IDF for Ngram Characters
# vect_char_bg.fit(list(train['comment_text']) + list(test['comment_text'])) # Fit the complete data in TF-IDFs

# vect_char_only_bg = TfidfVectorizer(max_features=20000, lowercase=True, analyzer='char',
# 						stop_words= 'english',ngram_range=(3,5), dtype=np.float32) # Generate TF-IDF for Ngram Characters
# vect_char_only_bg.fit(list(train['comment_text']) + list(test['comment_text'])) # Fit the complete data in TF-IDFs

# print "Transforming Words"
# tr_vect = vect_word.transform(train['comment_text']) # Transform the data into vectors
# sparse.save_npz('data/cln_tr_vect.npz', tr_vect)

# ts_vect = vect_word.transform(test['comment_text']) # Transform the data into vectors
# sparse.save_npz('data/cln_ts_vect.npz', ts_vect)

# print "Transforming Characters"
# tr_vect_char = vect_char.transform(train['comment_text']) # Transform the data into vectors
# sparse.save_npz('data/cln_tr_vect_char.npz', tr_vect_char)

# ts_vect_char = vect_char.transform(test['comment_text']) # Transform the data into vectors
# sparse.save_npz('data/cln_ts_vect_char.npz', ts_vect_char)

# tr_vect_char_bg = vect_char_bg.transform(train['comment_text']) # Transform the data into vectors
# ts_vect_char_bg = vect_char_bg.transform(test['comment_text']) # Transform the data into vectors

# tr_vect_char_only_bg = vect_char_only_bg.transform(train['comment_text']) # Transform the data into vectors
# ts_vect_char_only_bg = vect_char_only_bg.transform(test['comment_text']) # Transform the data into vectors
# gc.collect()

print "Saving"
# sparse.save_npz('data/tr_vect_char_bg.npz', tr_vect_char_bg)
# sparse.save_npz('data/ts_vect_char_bg.npz', ts_vect_char_bg)
# exit(0)

print "Loading Vectors"
tr_vect = sparse.load_npz("data/tr_vect.npz")
ts_vect = sparse.load_npz("data/ts_vect.npz")
# tr_vect_char = sparse.load_npz("data/tr_vect_char.npz")
# ts_vect_char = sparse.load_npz("data/ts_vect_char.npz")

# tr_vect_char_bg = sparse.load_npz("data/tr_vect_char_bg.npz") # No Improvement
# ts_vect_char_bg = sparse.load_npz("data/ts_vect_char_bg.npz") # No Improvement


### MayTryThis
# numFts = nm
# if numFts < tr_vect.shape[1]:
#     ch2 = SelectKBest(chi2, k=numFts)
#     tr_vect = ch2.fit_transform(tr_vect, y)
#     ts_vect = ch2.transform(ts_vect)

# save_sparse_matrix('data/tr_vect',tr_vect)
# tr_vect = load_sparse_matrix('data/tr_vect.npz').tolil()

# exit(0)


print "Combining Chars and Vects"

try_flag = 0
if try_flag == 0:
	# tr_vect = tr_vect[:100]
	train_combined = [tr_vect]
	test_combined = [ts_vect]
elif try_flag == 1:
	train_combined = [tr_vect, tr_vect_char]
	test_combined = [ts_vect, ts_vect_char]
elif try_flag == 2:
	train_combined = [tr_vect, tr_vect_char, tr_vect_char_bg]
	test_combined = [ts_vect, ts_vect_char, ts_vect_char_bg]
elif try_flag == 3:
	train_combined = [tr_vect, tr_vect_char_bg]
	test_combined = [ts_vect, ts_vect_char_bg]
## TRY TO REDUCE THE DIMENSION OF X, X_TEXT
X = sparse.hstack(train_combined) 
x_test = sparse.hstack(test_combined) 

del tr_vect, ts_vect#, tr_vect_char, ts_vect_char
gc.collect()


def runXGB(train_X, train_y, feature_names=None, seed_val=2017, num_rounds=400):
	param = {}
	param['objective'] = 'binary:logistic'
	param['eta'] = 0.12
	param['max_depth'] = 5
	param['silent'] = 1
	param['eval_metric'] = 'logloss'
	param['min_child_weight'] = 1
	param['subsample'] = 0.5
	param['colsample_bytree'] = 0.7
	param['seed'] = seed_val
	num_rounds = num_rounds

	plst = list(param.items())
	print "Convert"
	xgtrain = xgb.DMatrix(train_X, label=train_y)
	print "Modelling"
	model = xgb.train(plst, xgtrain, num_rounds)
	return model


labels = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']
y = train[labels]

predictions = np.zeros((x_test.shape[0],y.shape[1])) # create an empty predictions data frame
for i, label in enumerate(labels):

	print "Model Fitting for ", label
	model = runXGB(X, y[label])
	predictions[:,i] = model.predict(xgb.DMatrix(x_test))
	gc.collect()

	# model = LogisticRegression(C = 4, random_state = i)
	# model.fit(X,y[label]) # Train the LR model on each target column
	# predictions[:,i] = model.predict_proba(x_test)[:,1] # predict the probability

	predicted = cross_validation.cross_val_predict(model, X, y[label], cv=5)
	print metrics.accuracy_score(y[label], predicted)


results = pd.DataFrame(predictions, columns = labels) # write the results to dataframe
submit = pd.concat([test['id'], results], axis=1)
submit.to_csv('sub/submission_cln_xgb.csv', index=False)