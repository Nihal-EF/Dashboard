import pandas as pd
import streamlit as st
import requests
from datetime import date
from datetime import datetime
from streamlit_echarts import st_echarts
import js2py
from PIL import Image



def main():
    MLFLOW_URI = 'http://127.0.0.1:5000/invocations'

    #st.set_page_config(layout="wide")
    st.title('Scoring crédit')

    col1, col2 = st.columns(2)

    with col1:
       client_nouveau = st.checkbox('Client nouveau')
       client_existant = st.checkbox('Client existant')
    with col2:
        image = Image.open('credit.png')
        st.image(image, caption='', width = 250)
        
    st.session_state['client_nouveau'] = client_nouveau
    st.session_state['client_existant'] = client_existant
    







if __name__ == '__main__':
    main()
