from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import pandas as pd
import numpy as np
import gc

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train.fillna(' ',inplace=True) # replace NA Values from train data
test.fillna(' ',inplace=True) # replace NA Values from train data
gc.collect()

vect_word = TfidfVectorizer(max_features=50000, lowercase=True, analyzer='word',
                        stop_words= 'english',ngram_range=(1,3), dtype=np.float32) # Generate TF-IDF for Ngram Terms
vect_char = TfidfVectorizer(max_features=20000, lowercase=True, analyzer='char',
                        stop_words= 'english',ngram_range=(1,3), dtype=np.float32) # Generate TF-IDF for Ngram Characters

vect_word.fit(list(train['comment_text']) + list(test['comment_text'])) # Fit the complete data in TF-IDFs
vect_char.fit(list(train['comment_text']) + list(test['comment_text'])) # Fit the complete data in TF-IDFs

tr_vect = vect_word.transform(train['comment_text']) # Transform the data into vectors
ts_vect = vect_word.transform(test['comment_text']) # Transform the data into vectors
tr_vect_char = vect_char.transform(train['comment_text']) # Transform the data into vectors
ts_vect_char = vect_char.transform(test['comment_text']) # Transform the data into vectors
gc.collect()

X = sparse.hstack([tr_vect, tr_vect_char]) # Combine the char and terms vectors
x_test = sparse.hstack([ts_vect, ts_vect_char]) # Combine the char and terms vectors

del tr_vect, ts_vect, tr_vect_char, ts_vect_char
gc.collect()

labels = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']
y = train[labels]

predictions = np.zeros((x_test.shape[0],y.shape[1])) # create an empty predictions data frame
for i, label in enumerate(labels):
    model = LogisticRegression(C = 4, random_state = i)
    model.fit(X,y[label]) # Train the LR model on each target column
    predictions[:,i] = model.predict_proba(x_test)[:,1] # predict the probability

results = pd.DataFrame(predictions, columns = labels) # write the results to dataframe
submit = pd.concat([test['id'], results], axis=1)
submit.to_csv('submission_lr.csv', index=False)