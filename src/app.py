from typing import Union
import streamlit as st
import numpy as np
import os

import src.preprocessing as pp
from src.models import *
from src.metrics import accuracy_score, confusion_matrix

class App:

    demoF_path = os.path.join('data','demo_F1.txt')
    demoM_path = os.path.join('data','demo_M1.txt')
    conversion = {'F':'Woman', 'M':'Man'}
    model_selection = ['Term Frequency-Inverse Document Frequency (TF-IDF)', 'Recurrent Neural Network (RNN)']

    def __init__(self):

        if 'svc' not in st.session_state:
            st.session_state.svc = load_tfidf_svc()

        if 'rnn' not in st.session_state:
            st.session_state.rnn = load_rnn()

        if 'demoF' not in st.session_state:
            with open(self.demoF_path, 'r') as f:
                demoF_text = f.read()
            st.session_state.demoF = demoF_text.split('\n\n')[1]

        if 'demoM' not in st.session_state:
            with open(self.demoM_path, 'r') as f:
                demoM_text = f.read()
            st.session_state.demoM = demoM_text.split('\n\n')[1]

        if 'input_text' not in st.session_state:
            st.session_state.input_text = ''

        return
    
    def __get_results(self, text, model):
        print(text, model)
        text_processed, _ = pp.reformat_poem(text)
        if model == self.model_selection[0]:
            yhat = st.session_state.svc.predict([text_processed])

        elif model == self.model_selection[1]:
            yhat = st.session_state.rnn.predict([text_processed], get_probs=False)

        print('Results returned')
        return self.conversion[yhat[0]]
    
    def window(self):

        st.title('ChARM - Chinese Authorship Recognition Model')

        col1, col2 = st.columns(2)

        col1.subheader('Inputs')

        demo_opt = col1.radio('Demo Data', options=['None','Demo Data 1','Demo Data 2'], horizontal=True)
        if demo_opt == 'Demo Data 1':
            input_value = st.session_state.demoF
        elif demo_opt == 'Demo Data 2':
            input_value = st.session_state.demoM
        else:
            input_value = st.session_state.input_text

        input_text = col1.text_area(label='Type a poem:',
                                    value=input_value,
                                    key='input_text',
                                    placeholder='Type a poem:',
                                    disabled=(demo_opt != 'None'))

        
        input_model = col1.selectbox('Choose a Model:',
                                     options = self.model_selection,
                                     key='input_model')

        submit = col1.button('Run!', key='submit_button')

        if submit and input_text != '':
            result = self.__get_results(input_text, input_model)
        else: 
            result = ''
        
        col2.subheader('Output')

        if submit and result != '':
            col2.write(f"This poem was likely written by a:")
            col2.header(result)

        return