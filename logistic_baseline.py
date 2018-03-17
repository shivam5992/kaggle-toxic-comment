from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation, svm
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import TruncatedSVD

from sklearn.utils import shuffle
from scipy import sparse
import pandas as pd
import numpy as np
import gc

import scipy.sparse as sparse
import scipy.io

print "Reading Data"

############### try this average thing #######

train_cln = pd.read_csv('data/train_convai_wordchars_cln.csv')
test_cln = pd.read_csv('data/test_convai_wordchars_cln.csv')
train_cln.drop(['comment_text'], axis = 1, inplace = True)
test_cln.drop(['comment_text'], axis = 1, inplace = True)

train = pd.read_csv('data/train_convai_wordchars4.csv')
test = pd.read_csv('data/test_convai_wordchars4.csv')

train['toxic_level'] = train['toxic_level']*0.50 + train_cln['toxic_level_cln']*0.50
train['attack'] = train['attack']*0.50 + train_cln['attack_cln']*0.50
train['aggression'] = train['aggression']*0.50 + train_cln['aggression_cln']*0.50
test['toxic_level'] = test['toxic_level']*0.50 + test_cln['toxic_level_cln']*0.50
test['attack'] = test['attack']*0.50 + test_cln['attack_cln']*0.50
test['aggression'] = test['aggression']*0.50 + test_cln['aggression_cln']*0.50

train_feats = pd.read_csv('data/feats_train.csv')
train_feats.drop(['comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate',], axis = 1, inplace = True)
test_feats = pd.read_csv('data/feats_test.csv')
test_feats.drop(['comment_text'], axis = 1, inplace = True)
train = pd.concat([train.set_index('id'), train_feats.set_index('id')], axis=1, join='inner').reset_index()
test = pd.concat([test.set_index('id'), test_feats.set_index('id')], axis=1, join='inner').reset_index()



############## DO Text Work ########################################

train.fillna(' ',inplace=True) # replace NA Values from train data
test.fillna(' ',inplace=True) # replace NA Values from train data
gc.collect()

# print "Fitting Word Vectors"
# vect_word = TfidfVectorizer(max_features=50000, lowercase=True, analyzer='word',
						# stop_words= 'english',ngram_range=(1,3), dtype=np.float32) # Generate TF-IDF for Ngram Terms
# vect_word.fit(list(train['comment_text']) + list(test['comment_text'])) # Fit the complete data in TF-IDFs

# print "Transforming Words"
# tr_vect = vect_word.transform(train['comment_text']) # Transform the data into vectors
# sparse.save_npz('data/tr_vect.npz', tr_vect)
# ts_vect = vect_word.transform(test['comment_text']) # Transform the data into vectors
# sparse.save_npz('data/ts_vect.npz', ts_vect)

# print "Fitting Character Vectors"
# vect_char = TfidfVectorizer(max_features=5000, lowercase=True, analyzer='char',
# 						stop_words= 'english',ngram_range=(5,5), dtype=np.float32) # Generate TF-IDF for Ngram Characters
# vect_char.fit(list(train['comment_text']) + list(test['comment_text'])) # Fit the complete data in TF-IDFs

# print "Transforming Characters"
# tr_vect_char = vect_char.transform(train['comment_text']) # Transform the data into vectors
# sparse.save_npz('data/tr_vect_char4.npz', tr_vect_char)
# ts_vect_char = vect_char.transform(test['comment_text']) # Transform the data into vectors
# sparse.save_npz('data/ts_vect_char4.npz', ts_vect_char)

# tr_vect_char = vect_char.transform(train['comment_text']) # Transform the data into vectors
# sparse.save_npz('data/tr_vect_char5.npz', tr_vect_char)
# ts_vect_char = vect_char.transform(test['comment_text']) # Transform the data into vectors
# sparse.save_npz('data/ts_vect_char5.npz', ts_vect_char)

# exit(0)

print "Loading Vectors"
tr_vect = sparse.load_npz("data/tr_vect.npz")
ts_vect = sparse.load_npz("data/ts_vect.npz")
tr_vect_char = sparse.load_npz("data/tr_vect_char.npz")
ts_vect_char = sparse.load_npz("data/ts_vect_char.npz")
tr_vect_char4 = sparse.load_npz("data/tr_vect_char4.npz")
ts_vect_char4 = sparse.load_npz("data/ts_vect_char4.npz")
# tr_vect_char5 = sparse.load_npz("data/tr_vect_char5.npz")
# ts_vect_char5 = sparse.load_npz("data/ts_vect_char5.npz")

print "Combining Chars and Vects"
train_combined = [tr_vect, tr_vect_char, tr_vect_char4]
test_combined = [ts_vect, ts_vect_char, ts_vect_char4]

print "Feature Engineering"
adds = ["bad_word_count","negative_word_count","you_phrases"] # No Improvement
# SELECTED_COLS = ['toxic_rf', 'severe_toxic_rf', 'obscene_rf', 'threat_rf','insult_rf', 'identity_hate_rf']
SELECTED_COLS = ['toxic_level', 'attack', 'aggression']
SELECTED_COLS.extend(adds)

X = sparse.hstack((tr_vect,tr_vect_char,train[SELECTED_COLS])).tocsr()
x_test = sparse.hstack((ts_vect,ts_vect_char,test[SELECTED_COLS])).tocsr()

########### other way ##################################
# X = sparse.hstack(train_combined) 
# x_test = sparse.hstack(test_combined) 

################################### MayTryThis #########

# numFts = 25000
# ch2 = SelectKBest(chi2, k=numFts)
# X = ch2.fit_transform(X, y[label])
# x_test = ch2.transform(x_test)

##### PCA ######################################################

del tr_vect, ts_vect, tr_vect_char, ts_vect_char
gc.collect()

labels = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']
y = train[labels]

predictions = np.zeros((x_test.shape[0],y.shape[1])) # create an empty predictions data frame

cc = {'toxic' : 1, 'severe_toxic': 1, 'obscene' : 2, 'threat' : 4, 'insult' : 0.5, 'identity_hate' : 2}

for i, label in enumerate(labels):

	print "Model Fitting for ", label
	model = LogisticRegression(C = cc[label], random_state=i)

	# param_grid = {'C' : [2, 3, 4]}
	# gsearch = GridSearchCV(estimator = model, param_grid = param_grid, scoring='log_loss', n_jobs = 16, cv = 5)
	# gsearch.fit(X, y[label])
	# print gsearch.best_params_
	# print gsearch.best_score_
 
	model.fit(X,y[label]) # Train the LR model on each target column
	predictions[:,i] = model.predict_proba(x_test)[:,1] # predict the probability
	
	# predicted = cross_validation.cross_val_predict(model, X, y[label], cv=5)
	# print metrics.accuracy_score(y[label], predicted)

results = pd.DataFrame(predictions, columns = labels) # write the results to dataframe
submit = pd.concat([test['id'], results], axis=1)

# submit.to_csv('sub/submission_convai_feats1.csv', index=False) # word convai + raw feats
# submit.to_csv('sub/submission_convai_feats2.csv', index=False) # char, word convai + rawfeats
# submit.to_csv('sub/submission_convai_feats3.csv', index=False) # clean convai + raw feats
# submit.to_csv('sub/submission_convai_feats5.csv', index=False) # clean convai + raw feats + tuned + nonclean <best
submit.to_csv('sub/seed_41.csv', index=False) # clean convai + raw feats + tuned + nonclean

deep = pd.read_csv("sub/kg.csv")
p_res = deep.copy()
p_res[labels] = deep[labels]*0.40 + submit[labels]*0.60
p_res.to_csv('sub/ens_submission.csv', index=False)