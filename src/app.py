import streamlit as st
import numpy as np
import os

import src.preprocessing as pp
from src.models import *
from src.metrics import accuracy_score, confusion_matrix

class App:

    conversion = {'F':'Woman', 'M':'Man'}
    demoF_path = os.path.join('data','demo_F1.txt')
    demoM_path = os.path.join('data','demo_M1.txt')

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

        return
    
    def get_results(self, text, model):
        text_processed, _ = pp.reformat_poem(text)
        if model == 'TF-IDF and Support Vector Machine':
            
            yhat = st.session_state.svc.predict([text_processed])
        else:
            yhat = st.session_state.rnn.predict([text_processed], get_probs=False)

        return self.conversion[yhat[0]]
    
    def window(self):

        st.title('ChARM')
        st.subheader('Chinese Authorship Recognition Model')

        col1, col2 = st.columns(2)

        input_form = col1.form('Input')
        demoF = input_form.toggle('Use Demo Data 1')
        demoM = input_form.toggle('Use Demo Data 2')
        if demoF:
            input_value = st.session_state.demoF
        elif demoM:
            input_value = st.session_state.demoM
        else:
            input_value = ''

        input_text = input_form.text_area(label='Type a poem:', value=input_value, key='input_text')

        model_selection = ['TF-IDF and Support Vector Machine', 'RNN']
        input_model = input_form.selectbox('Choose a model:', 
                                              options=model_selection,
                                              key='input_model')
        
        input_submit = input_form.form_submit_button('Submit!')
        
        if input_text != '':
            result = self.get_results(input_text, input_model)
        else:
            result = ''

        col2.text(f"This poem was likely written by a\n{result}")
        # col2.text(result)
        # col2.text(st.session_state)
        return