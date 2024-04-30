import pandas as pd
import numpy as np
import pickle
import os

from pandas.api.types import CategoricalDtype
from scipy import sparse

from sklearn.metrics import confusion_matrix as cm
from sklearn.svm import SVC

from src.metrics import accuracy_score

MODEL_PATH = 'models'

# TODO: add TFIDF_SVC class
# TODO: - train, predict, score
# TODO: - save svc model?

class TFIDF:

    DATA_PATH = os.path.join(MODEL_PATH, 'tfdif.pkl')

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
        
        if self.__idf is None:
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
        with open(self.DATA_PATH, 'wb+') as pf:
            pickle.dump(self, pf, pickle.HIGHEST_PROTOCOL)
        return
    
    def load_model(self):
        with open(self.DATA_PATH, 'rb') as pf:
            tfidf = pickle.load(pf)
        return tfidf
    
    def is_fit(self) -> bool:
        return (self.__idf is not None)

class TFIDF_SVC:

    DATA_PATH = os.path.join(MODEL_PATH, 'tfidf_svc.pkl')
    TFIDF_PATH = os.path.join(MODEL_PATH, 'tfidf.pkl')

    def __init__(self, svc_args: dict = None, load_tfidf: bool = False):

        if load_tfidf:
            self.load_tfidf()
        else:
            self.tfidf = TFIDF()

        if svc_args is not None:
            self.svc = SVC(**svc_args)
        else:
            self.svc = SVC()
        return
    
    def fit(self, x, y):
        if self.tfidf.is_fit():
            x_tfidf = self.tfidf.transform(x)
        else:
            x_tfidf = self.tfidf.fit_transform(x)
        self.svc.fit(x_tfidf, y)
        return
    
    def predict(self, x):
        if (not self.tfidf.is_fit()) or self.svc.fit_status_ :
            raise ValueError("Models have not been established.  Use the 'fit' or 'fit_transform' functions first.")
        
        x_tfidf = self.tfidf.transform(x)
        return self.svc.predict(x_tfidf)
    
    def predict_and_score(self, x, y):
        yhat = self.predict(x)
        return accuracy_score(y, yhat)
    
    def save_model(self):
        with open(self.DATA_PATH, 'wb+') as pf:
            pickle.dump(self, pf, pickle.HIGHEST_PROTOCOL)
        return
    
    def load_model(self):
        with open(self.DATA_PATH, 'rb') as pf:
            tfidf_svc = pickle.load(pf)
        return tfidf_svc
    
    def load_tfidf(self):
        self.tfidf = self.tfidf.load_model()
        return
    
    def is_fit(self) -> bool:
        return self.svc.fit_status_ == 0

