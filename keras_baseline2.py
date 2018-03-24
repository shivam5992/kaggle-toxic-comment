from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.embeddings import Embedding
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from keras.layers import GRU
import pandas as pd 

def predict_test(model, xtest, classes):
    sample_submission = pd.read_csv("input/sample_submission.csv", encoding='latin-1')
    sample_submission[classes] =  model.predict(xtest)
    sample_submission.to_csv("sub/keras_toxic.csv", index=False)

def _get_sequences(train_texts, test_texts, top_words, padded_length):
    tokenizer = Tokenizer(num_words=top_words)
    tokenizer.fit_on_texts(train_texts)

    train_sequences = tokenizer.texts_to_sequences(train_texts)
    train_sequences = pad_sequences(train_sequences, maxlen=padded_length)

    test_sequences = tokenizer.texts_to_sequences(test_texts)
    test_sequences = pad_sequences(test_sequences, maxlen=padded_length)
    
    return train_sequences, test_sequences

def train_model(xtrain, ytrain, config):
    model= Sequential()
    
    # Sequence Embedding Layer
    model.add(Embedding(config['max_words'], config['vector_len'], input_length = config['padding_len']))
    
    # Convolution Layer, MaxPolling, Dropout
    model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.3))

    # Conv Layer - 2, MaxPolling, Dropout
    model.add(Conv1D(64,kernel_size=3,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.35))

    # Conv Layer - 3, MaxPooling, Dropout
    model.add(Conv1D(128,kernel_size=3,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.4))

    # GRU Layer
    model.add(LSTM(100))
    model.add(GRU(50,return_sequences=True))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.45))
    model.add(Dense(6, activation='sigmoid'))

    # Compile, Fit and Save 
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(xtrain, ytrain, batch_size=config['batch_size'], epochs=config['epochs'])
    model.save("models/toxic.h5")    
    return model

def _process(config):
    # Read Data
    trainDF = pd.read_csv("input/train.csv", encoding='latin-1')
    testDF = pd.read_csv("input/test.csv", encoding='latin-1')

    # Handle NA
    trainDF['comment_text'].fillna(' ', inplace = True)
    testDF['comment_text'].fillna(' ', inplace = True)

    # Get Y Labels
    classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    ytrain = trainDF[classes].values

    # Get Padded Sequences
    train_text  = trainDF['comment_text']
    test_text = testDF['comment_text']
    xtrain, xtest = _get_sequences(train_text, test_text, config['max_words'], config['padding_len'])

    model = train_model(xtrain, ytrain, config)
    predict_test(model, xtest, classes)


config = {
    'max_words': 50000,
    'padding_len': 100,
    'vector_len': 50,
    'batch_size': 1000,
    'epochs': 10
}
_process(config)