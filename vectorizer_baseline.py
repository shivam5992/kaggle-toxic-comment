from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation, svm
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import pandas as pd
import gc
from collections import defaultdict
from gensim import models
import gensim
import numpy

import scipy.sparse as sparse
import scipy.io

from sklearn.utils import shuffle

print "Reading Data"
train = pd.read_csv('data/cleaned_train.csv')
test = pd.read_csv('data/cleaned_test.csv')

train.fillna(' ',inplace=True) # replace NA Values from train data
test.fillna(' ',inplace=True) # replace NA Values from train data
gc.collect()

w2v = models.Word2Vec.load("data/vectors.pkl")

def get_sent_vec(txt):
	if txt.strip():
		vectors = [w2v.wv[str(word)] for word in str(txt).split()]
		sentence_vector = list(numpy.mean( vectors, axis=0 ))
	else:
		sentence_vector = []
	return sentence_vector

# train['sent_vec'] = train['comment_text'].apply(lambda x: get_sent_vec(x)) 
# test['sent_vec'] = test['comment_text'].apply(lambda x: get_sent_vec(x)) 

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = 100

    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
    	A = numpy.array([
                numpy.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [numpy.zeros(self.dim)], axis=0)
                for words in X
            ])
        sA = sparse.csr_matrix(A)
        return sA

vect_tf = TfidfEmbeddingVectorizer(w2v)
vect_tf.fit(list(train['comment_text']) + list(test['comment_text'])) 

# tr_vect_w2v = vect_tf.transform(train['comment_text']) 
# sparse.save_npz('data/tr_vect_w2v.npz', tr_vect_w2v)
ts_vect_w2v = vect_tf.transform(test['comment_text']) 
sparse.save_npz('data/ts_vect_w2v.npz', ts_vect_w2v)
exit(0)
