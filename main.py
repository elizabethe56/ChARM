import tensorflow as tf
from src.app import App
import streamlit as st

if __name__ == '__main__':
    st.set_page_config(layout='wide')
    app = App()
    app.window()