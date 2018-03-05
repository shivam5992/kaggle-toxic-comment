from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation
from sklearn.grid_search import GridSearchCV
import pandas as pd
import numpy as np
import gc

import scipy.sparse as sparse

print "Reading Data"
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

save = False
load = True

if save:
	train.fillna(' ',inplace=True) # replace NA Values from train data
	test.fillna(' ',inplace=True) # replace NA Values from train data
	gc.collect()

	print "Fitting Word Vectors"
	vect_word = TfidfVectorizer(max_features=50000, lowercase=True, analyzer='word', stop_words= 'english',ngram_range=(1,3), dtype=np.float32) # Generate TF-IDF for Ngram Terms
	vect_word.fit(list(train['comment_text']) + list(test['comment_text'])) # Fit the complete data in TF-IDFs

	print "Transforming Words"
	tr_vect = vect_word.transform(train['comment_text']) # Transform the data into vectors
	sparse.save_npz('models/vectors/tr_vect_cln.npz', tr_vect)

	ts_vect = vect_word.transform(test['comment_text']) # Transform the data into vectors
	sparse.save_npz('models/vectors/ts_vect_cln.npz', ts_vect)

	print "Fitting Character Vectors"
	vect_char = TfidfVectorizer(max_features=25000, lowercase=True, analyzer='char', stop_words= 'english',ngram_range=(2,4), dtype=np.float32) # Generate TF-IDF for Ngram Characters
	vect_char.fit(list(train['comment_text']) + list(test['comment_text'])) # Fit the complete data in TF-IDFs

	print "Transforming Characters"
	tr_vect_char = vect_char.transform(train['comment_text']) # Transform the data into vectors
	sparse.save_npz('models/vectors/tr_vect_char_cln.npz', tr_vect_char)

	ts_vect_char = vect_char.transform(test['comment_text']) # Transform the data into vectors
	sparse.save_npz('models/vectors/ts_vect_char_cln.npz', ts_vect_char)

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

if load:
	print "Loading Vectors"
	tr_vect = sparse.load_npz("models/vectors/tr_vect.npz")
	ts_vect = sparse.load_npz("models/vectors/ts_vect.npz")
	tr_vect_char = sparse.load_npz("models/vectors/tr_vect_char.npz")
	ts_vect_char = sparse.load_npz("models/vectors/ts_vect_char.npz")

	print "Combining Chars and Vects"
	# train_combined = [tr_vect, tr_vect_char]
	# test_combined = [ts_vect, ts_vect_char]

	print "Feature Engineering" # adding features from other files
	SELECTED_COLS = ['toxic_level', 'attack', 'aggression']
	X = sparse.hstack((tr_vect,tr_vect_char, train[SELECTED_COLS])).tocsr()
	x_test = sparse.hstack((ts_vect,ts_vect_char, test[SELECTED_COLS])).tocsr()
	
	X = sparse.hstack((tr_vect,tr_vect_char)).tocsr()
	x_test = sparse.hstack((ts_vect,ts_vect_char)).tocsr()
	del tr_vect, ts_vect, tr_vect_char, ts_vect_char
	gc.collect()

	labels = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']
	y = train[labels]

	predictions = np.zeros((x_test.shape[0],y.shape[1])) # create an empty predictions data frame
	cc = {'toxic' : 1, 'severe_toxic': 1, 'obscene' : 2, 'threat' : 4, 'insult' : 0.5, 'identity_hate' : 2}

	for i, label in enumerate(labels):
		if i != 0:
			continue

		print "Model Fitting for ", label
		model = LogisticRegression(C = cc[label], random_state=i)

		## Parameter Tuning
		param_grid = {'C' : [0.5, 1, 2]}
		gsearch = GridSearchCV(estimator = model, param_grid = param_grid, scoring='neg_log_loss', n_jobs = 16, cv = 5)
		gsearch.fit(X, y[label])
		print gsearch.best_params_
		print gsearch.best_score_
		exit(0)
	 
		# model.fit(X,y[label]) # Train the LR model on each target column
		# predictions[:,i] = model.predict_proba(x_test)[:,1] # predict the probability
		
		## Cross Validation
		# predicted = cross_validation.cross_val_predict(model, X, y[label], cv=5)
		# print metrics.log_loss(y[label], predicted)

	# results = pd.DataFrame(predictions, columns = labels) # write the results to dataframe
	# submit = pd.concat([test['id'], results], axis=1)
	# submit.to_csv('sub/lr_conv.csv', index=False)