from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation
from sklearn.utils import shuffle
import scipy.sparse as sparse
import pandas as pd
import numpy as np
import gc

train_cln = pd.read_csv('data/train_convai_wordchars_cln.csv')
train_cln.drop(['comment_text'], axis = 1, inplace = True)
train = pd.read_csv('data/train_convai_wordchars.csv')
train['toxic_level'] = train['toxic_level']*0.50 + train_cln['toxic_level_cln']*0.50
train['attack'] = train['attack']*0.50 + train_cln['attack_cln']*0.50
train['aggression'] = train['aggression']*0.50 + train_cln['aggression_cln']*0.50
train_feats = pd.read_csv('data/feats_train.csv')
train_feats.drop(['comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate',], axis = 1, inplace = True)
train = pd.concat([train.set_index('id'), train_feats.set_index('id')], axis=1, join='inner').reset_index()
train.fillna(' ',inplace=True)

test_cln = pd.read_csv('data/test_convai_wordchars_cln.csv')
test_cln.drop(['comment_text'], axis = 1, inplace = True)
test = pd.read_csv('data/test_convai_wordchars.csv')
test['toxic_level'] = test['toxic_level']*0.50 + test_cln['toxic_level_cln']*0.50
test['attack'] = test['attack']*0.50 + test_cln['attack_cln']*0.50
test['aggression'] = test['aggression']*0.50 + test_cln['aggression_cln']*0.50
test_feats = pd.read_csv('data/feats_test.csv')
test_feats.drop(['comment_text'], axis = 1, inplace = True)
test = pd.concat([test.set_index('id'), test_feats.set_index('id')], axis=1, join='inner').reset_index()
test.fillna(' ',inplace=True)

gc.collect()

def fit_word_vectors(maxf):
	vect_word = TfidfVectorizer(max_features=maxf, lowercase=True, analyzer='word', stop_words= 'english',ngram_range=(1,3), dtype=np.float32)
	vect_word.fit(list(train['comment_text']) + list(test['comment_text'])) 
	tr_vect = vect_word.transform(train['comment_text']) 
	# ts_vect = vect_word.transform(test['comment_text']) 
	return tr_vect

def fit_char_vectors(maxf):
	vect_char = TfidfVectorizer(max_features=maxf, lowercase=True, analyzer='char', stop_words= 'english',ngram_range=(1,3), dtype=np.float32)
	vect_char.fit(list(train['comment_text']) + list(test['comment_text'])) 
	tr_vect_char = vect_char.transform(train['comment_text']) 
	# ts_vect_char = vect_char.transform(test['comment_text']) 
	return tr_vect_char

def get_char_vectors():
	tr_vect_char = sparse.load_npz("data/tr_vect_char.npz")
	# ts_vect_char = sparse.load_npz("data/ts_vect_char.npz")
	return tr_vect_char

def get_word_vectors():
	tr_vect = sparse.load_npz("data/tr_vect.npz")
	# ts_vect_char = sparse.load_npz("data/ts_vect_char.npz")
	return tr_vect

maxf_list = [10000, 20000, 30000, 40000, 50000]
labels_c = {'toxic' : 1, 'severe_toxic': 1, 'obscene' : 2, 'threat' : 4, 'insult' : 0.5, 'identity_hate' : 2}
adds = ["bad_word_count","negative_word_count","you_phrases"]
SELECTED_COLS = ['toxic_level', 'attack', 'aggression']
SELECTED_COLS.extend(adds)
y = train[labels_c.keys()]

for maxf in maxf_list:
	print
	print maxf

	tr_vect = get_word_vectors()
	tr_vect_char = fit_char_vectors(maxf)

	train_combined = [tr_vect, tr_vect_char]
	X = sparse.hstack((tr_vect,tr_vect_char,train[SELECTED_COLS])).tocsr()
	del tr_vect, tr_vect_char
	gc.collect()

	label = 'toxic'
	model = LogisticRegression(C = labels_c[label], random_state=0)
	predicted = cross_validation.cross_val_predict(model, X, y[label], cv=5)
	print metrics.accuracy_score(y[label], predicted)