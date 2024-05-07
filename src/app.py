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
    model_selection = ['TF-IDF', 'RNN']

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
        st.write('This app is designed to predict if a piece of traditional Tang poetry was written by a man or a woman. There are two possible models, one a term frequency-inverse document frequency model (TF-IDF), and the other a recurrent neural network (RNN). More information on the models and training data can be found below.')

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
        
        st.divider()
        st.subheader('Model Information')

        st.write('#### TF-IDF:')
        st.write('The term frequency-inverse document frequency model uses a support vector machine (SVM) to classify the vectors created using TF-IDF weights. These weights are calculated by dividing the count of each word in each document by the logarithm of the ratio of documents containing the word to the total number of documents.')
        st.write('**Validation Results**')
        st.write('- Accuracy Score: 82.43%\n- Precision: 71.43%\n- Recall: 31.35%\n- F-Score: 43.48%')

        st.write('#### RNN:')
        st.write('The recurrent neural network model is built with tensorflow layers. There are three recurrent layers: a bidirectional simple RNN, a second simple RNN (not bidirectional), and finally a bidirectional long short-term memory (LSTM) layer. There are approximately 430,000 trainable parameters. The model is saved as a keras file in the source code for further inspection.')
        st.write('**Validation Results**')
        st.write('- Accuracy Score: 86.49%\n- Precision: 75.00%\n- Recall: 56.25%\n- F-Score: 64.29%')

        st.subheader('Training Information')

        st.write('#### The Data')
        st.write('The complete dataset included 317 poems by men and 85 poems by women. Punctuation is stripped from the poems, but an underscore is used to indicate a new line.  When splitting the dataset into training and validation sets, the data was stratified, such that there was an approximately even ratio of poems by women to poems by men.  The training set contained 328 poems (259 by men, 69 by women), and the validation set contained 74 poems (58 by men, 16 by women). The demo poems were not included in the training or validation sets.')
        st.write('**Sources**')
        st.write('The poems were sourced from around the web.  The men\'s poems were scraped from the book *300 Tang Poems*, alongside some additional poems collected by the Chinese Text Initiative.  The women\'s poems were copy and pasted from journals and articles from across the internet. The full citations can be found in the repository\'s README.')

        return