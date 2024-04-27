import streamlit as st
import numpy as np

class App:

    def __init__(self):
        st.session_state['hi'] = 'Test'
        return
    
    def window(self):
        input_form = st.form('Input')

        input_text = input_form.text_area(label='Type a poem:', key='input_text')

        # TODO: add model dropdown
        
        input_submit = input_form.form_submit_button('Submit!')
        st.session_state['help'] = input_text
        
        rand = np.random.random()
        if rand > 0.5:
            result = 'Woman'
        else:
            result = 'Man'
        st.text(result)
        # st.text(st.session_state)
        return