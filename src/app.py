import streamlit as st
import numpy as np

import src.preprocessing as pp
from src.models import *
from src.metrics import accuracy_score, confusion_matrix

class App:

    conversion = {'F':'Woman', 'M':'Man'}

    def __init__(self):

        if 'svc' not in st.session_state:
            st.session_state.svc = load_tfidf_svc()

        if 'rnn' not in st.session_state:
            st.session_state.rnn = load_rnn()

        return
    
    def get_results(self, text, model):
        text_processed, _ = pp.reformat_poem(text)
        if model == 'TF-IDF and Support Vector Machine':
            
            yhat = st.session_state.svc.predict([text_processed])
        else:
            yhat = st.session_state.rnn.predict([text_processed], get_prob=False)

        return self.conversion[yhat[0]]
    
    def window(self):

        st.title('ChARM')
        st.subheader('Chinese Authorship Recognition Model')

        col1, col2 = st.columns(2)

        input_form = col1.form('Input')
        demo1 = input_form.toggle('Use Demo Data 1')
        if demo1:
            value = """帝裏重陽節，
香園萬乘來。
卻邪萸入佩，
獻壽菊傳杯。
塔類承天湧，
門疑待佛開。
睿詞懸日月，
長得仰昭回。"""
        else:
            value = ''

        input_text = input_form.text_area(label='Type a poem:', value=value, key='input_text')

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