import pandas as pd
import numpy as np
import pickle
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
from tensorflow.data import Dataset, AUTOTUNE
from tensorflow.keras import Model, layers, optimizers, losses, saving
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

import pandas as pd
from pandas.api.types import CategoricalDtype, is_list_like
from scipy import sparse

from sklearn.metrics import confusion_matrix as cm
from sklearn.svm import SVC

from src.metrics import accuracy_score, plot_history


MODEL_PATH = 'models'
TFIDF_PATH = os.path.join(MODEL_PATH, 'tfdif.pkl')
TFIDF_SVC_PATH = os.path.join(MODEL_PATH, 'tfidf_svc.pkl')
ENCODER_PATH = os.path.join(MODEL_PATH, 'encoder.pkl')
RNN_PATH = os.path.join(MODEL_PATH, 'RNN.pkl')

def load_tfidf():
    with open(TFIDF_PATH, 'rb') as pf:
        tfidf = pickle.load(pf)
    return tfidf

def load_tfidf_svc():
    with open(TFIDF_SVC_PATH, 'rb') as pf:
        tfidf_svc = pickle.load(pf)
    return tfidf_svc

def load_encoder():
    with open(ENCODER_PATH, 'rb') as pf:
        encoder = pickle.load(pf)
    return encoder

def load_rnn():
    with open(RNN_PATH, 'rb') as pf:
        rnn = pickle.load(pf)
    return rnn

class TFIDF:

    def __init__(self):
        
        self.__idf = None
        # self.__all_tokens
        # self.__token_cat

        return
    
    def fit(self, data):

        # establish number of documents
        docs_in_corpus = len(data)

        token_df = self.__make_token_df(data)
        
        idf = (token_df
               .copy()
               .drop('doc_len', axis=1)
               .drop_duplicates())

        # get document frequency
        idf['df'] = [0] * len(idf)
        idf = (idf
               .groupby('token', as_index=False)
               .count()
               .drop(['index'], axis=1)
               )
        
        # add dummy token for unknown characters
        idf = pd.concat([idf, pd.DataFrame([['#',0]], columns=idf.columns)], ignore_index=True)
        self.__all_tokens = list(idf['token'].unique())

        # calculate idf
        idf['idf'] = np.log((1 + docs_in_corpus) / (1 + idf['df'])) + 1

        self.__idf = idf.copy()

        # establish vocabulary
        token = sorted(idf['token'].unique())
        self.__token_cat = CategoricalDtype(categories=token, ordered=True)

        return
    
    def transform(self, data):
        
        if not self.is_fit():
            raise ValueError("IDF has not been established.  Use the 'fit' or 'fit_transform' functions first.")

        token_df = self.__make_token_df(data)
        
        # replace unknown tokens with dummy character
        token_df['token'] = [x if x in self.__all_tokens else '#' for x in token_df['token']]
        
        # get term frequencies
        tf = token_df.copy()
        tf['tf'] = [0] * len(tf)
        tf = (tf
              .groupby(['index','token','doc_len'], as_index=False)
              .count()
              )
        tf['tf'] = [tf/dl for tf, dl in zip(tf['tf'], tf['doc_len'])]

        # calculate tfidf
        tfidf = tf.merge(self.__idf, how='left', on='token')
        tfidf['tfidf'] = tfidf['tf'] * tfidf['idf']
        
        # convert to csr (condensed sparse row) format (matching sci-kit learn's TfidfVectorizer)

        shape = (len(tfidf['index'].unique()), len(self.__all_tokens))
        
        token_index = tfidf['token'].astype(self.__token_cat).cat.codes
        coo = sparse.coo_matrix((tfidf['tfidf'], (tfidf['index'], token_index)), 
                                shape=shape)
        csr = coo.tocsr()
        return csr

    def fit_transform(self, data):
        
        # establish number of documents
        docs_in_corpus = len(data)

        token_df = self.__make_token_df(data)

        # get term frequencies
        tf = token_df.copy()
        tf['tf'] = [0] * len(tf)
        tf = (tf
              .groupby(['index','token','doc_len'], as_index=False)
              .count()
              )
        tf['tf'] = [tf/dl for tf, dl in zip(tf['tf'], tf['doc_len'])]

        idf = (token_df
               .copy()
               .drop('doc_len', axis=1)
               .drop_duplicates())

        # get document frequency
        idf['df'] = [0] * len(idf)
        idf = (idf
               .groupby('token', as_index=False)
               .count()
               .drop(['index'], axis=1)
               )
        
        # add dummy token for unknown characters
        idf = pd.concat([idf, pd.DataFrame([['#',0]], columns=idf.columns)], ignore_index=True)

        self.__all_tokens = list(idf['token'].unique())

        # calculate idf
        idf['idf'] = np.log((1 + docs_in_corpus) / (1 + idf['df'])) + 1

        # Save idf for validation
        self.__idf = idf.copy()

        # get tfidf
        tfidf = tf.merge(idf, how='left', on='token')
        tfidf['tfidf'] = tfidf['tf'] * tfidf['idf']

        # establish vocabulary
        token = sorted(idf['token'].unique())
        self.__token_cat = CategoricalDtype(categories=token, ordered=True)

        shape = (len(tfidf['index'].unique()), len(self.__all_tokens))
        
        token_index = tfidf['token'].astype(self.__token_cat).cat.codes
        coo = sparse.coo_matrix((tfidf['tfidf'], (tfidf['index'], token_index)), 
                                shape=shape)
        csr = coo.tocsr()
        return csr
    
    def __make_token_df(self, data):
        
        # Make dataframe for easier calculations
        df = pd.DataFrame({'x':data})

        # establish number of documents
        docs_in_corpus = len(df)

        # get token lists
        df['token'] = [x.split(' ') for x in df['x']]
        df['doc_len'] = [len(x) for x in df['token']]

        # Make token dataframe
        # - reset index to get doc ids
        # - drop duplicates so documents are only counted once
        token_df = (df
                    .explode('token')
                    .reset_index()
                    .drop('x', axis=1)
                    )

        return token_df
    
    def save_model(self):
        with open(TFIDF_PATH, 'wb+') as pf:
            pickle.dump(self, pf, pickle.HIGHEST_PROTOCOL)
        return
    
    def is_fit(self) -> bool:
        return (self.__idf is not None)

class TFIDF_SVC:

    def __init__(self, svc_args: dict = None, load_tfidf: bool = False):

        if load_tfidf:
            self.__tfidf = load_tfidf()
        else:
            self.__tfidf = TFIDF()

        if svc_args is not None:
            self.__svc = SVC(**svc_args)
        else:
            self.__svc = SVC()
        return
    
    def fit(self, x, y):
        if self.__tfidf.is_fit():
            x_tfidf = self.__tfidf.transform(x)
        else:
            x_tfidf = self.__tfidf.fit_transform(x)
        self.__svc.fit(x_tfidf, y)
        return
    
    def predict(self, x):
        if (not self.__tfidf.is_fit()) or self.__svc.fit_status_ :
            raise ValueError("Models have not been established.  Use the 'fit' or 'fit_transform' functions first.")
        
        print('Predicting TF-IDF ...')
        x_tfidf = self.__tfidf.transform(x)
        return self.__svc.predict(x_tfidf)
    
    def predict_and_score(self, x, y):
        yhat = self.predict(x)
        return accuracy_score(y, yhat)
    
    def save_model(self):
        with open(TFIDF_SVC_PATH, 'wb+') as pf:
            pickle.dump(self, pf, pickle.HIGHEST_PROTOCOL)
        return
    
    def is_fit(self) -> bool:
        return self.__svc.fit_status_ == 0

class Encoder:
    def __init__(self, 
                 categories = None):
        if categories is not None:
            self.categories_ = {z:i for i, z in enumerate(categories)}
        else:
            self.categories_ = None

    def fit(self, y):
        categories = sorted(set(y))
        self.categories_ = {z:i for i, z in enumerate(categories)}
    
    def transform(self, y):
        if not self.is_fit():
            raise ValueError("Encoder has not been established.  Use the 'fit' or 'fit_transform' functions first.")
        
        output = np.zeros(shape=(len(y), len(self.categories_)), dtype=float)

        for i, n in enumerate(y):
            if n in self.categories_:
                output[i][self.categories_[n]] = 1
            else:
                print(f"Warning: {n} is not in the trained categories.")

        return output
    
    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)
    
    def inverse_transform(self, 
                          yhat: np.ndarray
                          ) -> list:
        if yhat.shape[1] != len(self.categories_):
            raise IndexError(f"Input shape {yhat.shape} does not match number of categories ({len(self.categories_)})")
        # np.argmax
        inv_map = {v:k for k, v in self.categories_.items()}
        
        output = np.zeros(len(yhat), dtype=object)
        for i, n in enumerate(yhat):
            output[i] = inv_map[np.argmax(n)]

        return list(output)

    def save_model(self):
        with open(ENCODER_PATH, 'wb+') as pf:
            pickle.dump(self, pf, pickle.HIGHEST_PROTOCOL)
        return

    def is_fit(self) -> bool:
        return self.categories_ is not None

class RNN:
    def __init__(self, load_encoder_: bool = True, load_keras: bool = True):
        self.__KERAS_PATH = os.path.join(MODEL_PATH, 'RNN.keras')
        
        if load_encoder_:
            self.__enc = load_encoder()
        else:
            self.__enc = Encoder()

        if load_keras:
            self.load_keras()
        else:
            self.rnn = None
    
    def fit(self, 
            xt, xv, yt, yv, 
            learning_rate = 0.0001, 
            optimizer = 'adam', 
            optimizer_args = None,
            loss = 'binary_crossentropy', 
            loss_args: dict = None,
            epochs = 50,
            monitor = 'val_accuracy'):
        raise(NotImplementedError)
        if optimizer == 'adam':
            if optimizer_args is not None:
                optimizer_ = optimizer.Adam(**optimizer_args)
            else:
                optimizer_ = optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer_ = optimizer
        
        if loss == 'binary_focal_crossentropy':
            if loss_args is not None:
                loss_ = losses.BinaryFocalCrossentropy(**loss_args)
            else:
                loss_ = losses.BinaryFocalCrossentropy(apply_class_balancing=True)
        else:
            loss_ = loss

        # OneHotEncode y_labels
        if not self.__enc.is_fit():
            self.__enc.fit(yt)
        
        yt_enc = self.__enc.transform(yt)
        yv_enc = self.__enc.transform(yv)
        num_classes = len(self.__enc.categories_)

        BATCH_SIZE = 64

        train_dataset = Dataset.from_tensor_slices((xt.values, yt_enc))
        val_dataset = Dataset.from_tensor_slices((xv.values, yv_enc))

        train_batches = train_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
        val_batches = val_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

        text_vectorizer = layers.TextVectorization(standardize=None, split='whitespace')
        text_vectorizer.adapt(xt.values)
        vocab = text_vectorizer.get_vocabulary()

        # Architecture
        inputs = layers.Input(shape=(1,), dtype=tf.string)
        x = text_vectorizer(inputs)
        x = layers.Embedding(input_dim=len(vocab),
                                        output_dim=128,
                                        mask_zero=True,
                                        name='token_embedding')(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Bidirectional(layers.SimpleRNN(128, return_sequences=True))(x)
        x = layers.Dropout(0.4)(x)
        x = layers.BatchNormalization()(x)

        x = layers.SimpleRNN(64, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.BatchNormalization()(x)

        x = layers.LSTM(16, return_sequences=False)(x)
        x = layers.Dropout(0.2)(x)

        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = Model(inputs, outputs)
        model.compile(loss=loss_, optimizer=optimizer_, metrics=['accuracy'])

        filepath = "models/temp.keras"
        checkpoint = ModelCheckpoint(filepath, 
                             monitor=monitor,
                             verbose=1,
                             save_best_only=True,
                             mode='auto')

        # Train the model with the ModelCheckpoint callback
        history = model.fit(train_batches,
                            validation_data=val_batches,
                            epochs=epochs,
                            shuffle=True,
                            callbacks=[checkpoint])
        
        plot_history(history)

        self.load_temp()

        return self.rnn

    def predict(self, 
                x, 
                get_probs: bool = True):
        print('Predicting RNN ...')
        if type(x) != pd.Series:
            if is_list_like(x):
                x = pd.Series(x)
            else:
                x = pd.Series([x])
        
        x_rnn_probs = self.rnn.predict(x)
        x_rnn = self.__enc.inverse_transform(x_rnn_probs)

        if get_probs:
            return x_rnn, x_rnn_probs
        else:
            return x_rnn

    def load_keras(self):
        
        self.rnn = saving.load_model(self.__KERAS_PATH)
        return

    def load_temp(self):
        self.rnn = load_model(os.path.join(MODEL_PATH, 'temp.keras'))
        return
    
    def save_model(self):
        with open(RNN_PATH, 'wb+') as pf:
            pickle.dump(self, pf, pickle.HIGHEST_PROTOCOL)
        return