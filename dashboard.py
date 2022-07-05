import pandas as pd
import streamlit as st
import requests
from datetime import date
from datetime import datetime
from streamlit_echarts import st_echarts
import js2py


def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'columns':["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3","DAYS_EMPLOYED", "DAYS_BIRTH", "DAYS_ID_PUBLISH", "PAYMENT_RATE", "ANNUITY_INCOME_PERC", "DPD_BOOL"], 'data': data}
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


def main():
    MLFLOW_URI = 'http://127.0.0.1:5000/invocations'

    st.set_page_config(layout="wide")   
    st.title('Scoring crédit')

    col1, col2, col3 = st.columns(3)
    with col1:
    
        EXT_SOURCE_1 = st.number_input('Données source extérieur 1',
                                     min_value=0., value=.5, step=.1)
                                     
        EXT_SOURCE_2 = st.number_input('Données source extérieur 2',
                                     min_value=0., value=.5, step=.1)
        
        EXT_SOURCE_3 = st.number_input('Données source extérieur 3',
                                 min_value=0., value=.5, step=.1)
    with col2:
        st.write('Retard sur des crédits précédents')
        DPD_TRUE= st.checkbox('oui')
        PAYMENT_RATE =st.slider('Taux de remboursement',
                             min_value=1, max_value = 20, value=5, step=1)


        ANNUITY_INCOME_PERC = st.slider('Taux d''endettement',
                         min_value=1, max_value = 35, value=5, step=1)
    with col3:
                                 
        DATE_EMPLOYED = st.date_input('Date de début d''emploi')
                                     
        
        DATE_BIRTH = st.date_input('Date de naissance', min_value=datetime.strptime("01/01/1950", "%d/%m/%Y"), max_value=date.today())
                                     
        DATE_ID_PUBLISH = st.date_input('Date d''édition de la pièce d''identité')
        
        
    

                                 
    today = date.today()
    DAYS_EMPLOYED = (today - DATE_EMPLOYED).days
    DAYS_BIRTH = (today - DATE_BIRTH).days
    DAYS_ID_PUBLISH = (today - DATE_ID_PUBLISH).days
    DPD_BOOL = DPD_TRUE
    
    predict_btn = st.button('Prédire')
    if predict_btn:
        data = [[EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3,DAYS_EMPLOYED, DAYS_BIRTH, DAYS_ID_PUBLISH, PAYMENT_RATE, ANNUITY_INCOME_PERC, DPD_BOOL]]
        pred = None
        pred = request_prediction(MLFLOW_URI, data)[0]

            
            
        option = {
        "tooltip": {
            "formatter": '{a} <br/>{b} : {c}%'
        },
        "series": [{
            "name": 'Risque de non-solvabilité du client',
            "type": 'gauge',
            "startAngle": 180,
            "endAngle": 0,
            "progress": {
                "show": "true"
            },
            "radius":'100%',
            "axisLine": {
                "lineStyle": {
                    "width": 6,
                    "color": [
                        [0.25,  "#64C88A"],
                        [0.5, "#FDDD60"],
                        [0.75, "#ffa500" ],
                        [1, "#FF403F"],
                    ],
                }
            },

            "itemStyle": {
                "color": "auto",
                #"shadowColor": 'rgba(0,138,255,0.45)',
                "shadowBlur": 10,
                "shadowOffsetX": 2,
                "shadowOffsetY": 2,
                "radius": '55%',
            },
            "axisTick": {"length": 10, "lineStyle": {"color": "auto", "width": 2}},
            "splitLine": {"length": 15, "lineStyle": {"color": "auto", "width": 5}},
            "progress": {
                "show": "true",
                "roundCap": "true",
                "width": 15
            },
            "pointer": {
                "length": '60%',
                "width": 8,
                "offsetCenter": [0, '5%']
            },
            "detail": {
                "valueAnimation": "true",
                "formatter": '{value}%',
                "width": '60%',
                "lineHeight": 20,
                "fontSize": 30,
                "height": 20,
                "borderRadius": 188,
                "offsetCenter": [0, '40%'],
            },
            "data": [{
                "value": pred,
                "name": 'Risque de non-solvabilité du client'
            }]
        }]
        };




        st_echarts(options=option, key="0")
        










if __name__ == '__main__':
    main()
