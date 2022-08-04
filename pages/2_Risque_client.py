import pandas as pd
import streamlit as st
import requests
from datetime import date
from datetime import datetime, timedelta
from streamlit_echarts import st_echarts
import js2py
from PIL import Image
import joblib

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'columns':["EXT_SOURCE_1","EXT_SOURCE_2","DAYS_EMPLOYED", "DAYS_BIRTH", "DAYS_ID_PUBLISH", "PAYMENT_RATE", "ANNUITY_INCOME_PERC", "DPD_BOOL"], 'data': data}
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
    
    return response.json()

def predict_joblib(data):
    joblib_model = joblib.load("pipeline_credit.joblib")
    return int(joblib_model.predict_proba(data)[0][1]*100)

if 'client_nouveau' not in st.session_state:
    st.session_state['client_nouveau'] = True
if 'client_existant' not in st.session_state:
    st.session_state['client_existant'] = False
    
def main():
    MLFLOW_URI = 'http://127.0.0.1:5000/invocations'
    st.set_page_config(layout="wide")
    test_data = pd.read_csv("test_df_dropna.csv")
    train_data = pd.read_csv("train_df_dropna.csv")
    if (st.session_state['client_nouveau']):
        

        
        st.title('Scoring crédit')
        col1, col2, col3 = st.columns(3)
        with col1:
            
            EXT_SOURCE_1 = st.number_input('Score externe 1',
                                     min_value=0., value=.5, step=.1)
            EXT_SOURCE_2 = st.number_input('Score externe 2',
                                     min_value=0., value=.5, step=.1)
            st.write('Retard sur des crédits précédents')
            DPD_TRUE= st.checkbox('oui')
                                         
        with col2:

            DATE_EMPLOYED = st.date_input('Date de début d\'emploi')
            DATE_BIRTH = st.date_input('Date de naissance', min_value=datetime.strptime("01/01/1950", "%d/%m/%Y"), max_value=date.today())
            DATE_ID_PUBLISH = st.date_input('Date d\'édition de la pièce d\’identité')
        with col3 :
            PAYMENT_RATE =st.slider('Taux de remboursement',
                                 min_value=1, max_value = 20, value=5, step=1)/100


            ANNUITY_INCOME_PERC = st.slider('Taux d\'endettement',
                             min_value=1, max_value = 35, value=5, step=1)/100
        today = date.today()
        DAYS_EMPLOYED = (today - DATE_EMPLOYED).days
        DAYS_BIRTH = (today - DATE_BIRTH).days
        DAYS_ID_PUBLISH = (today - DATE_ID_PUBLISH).days
        DPD_BOOL = DPD_TRUE
        data = [[EXT_SOURCE_1,EXT_SOURCE_2,DAYS_EMPLOYED, DAYS_BIRTH, DAYS_ID_PUBLISH, PAYMENT_RATE, ANNUITY_INCOME_PERC, DPD_BOOL]]
    else:
        SK_ID_CURR = st.selectbox(
            'Identifiant du client :', test_data["SK_ID_CURR"])
        data = test_data[test_data["SK_ID_CURR"]==SK_ID_CURR][["EXT_SOURCE_1","EXT_SOURCE_2","DAYS_EMPLOYED", "DAYS_BIRTH", "DAYS_ID_PUBLISH", "PAYMENT_RATE", "ANNUITY_INCOME_PERC", "DPD_BOOL"]].values
        DATE_EMPLOYED = (date.today()+timedelta(int(data[0][2])))
        DATE_BIRTH = (date.today()+timedelta(int(data[0][3])))
        DATE_ID_PUBLISH = (date.today()+timedelta(int(data[0][4])))
        col1, col2, col3 = st.columns(3)
        with col1:
            
            EXT_SOURCE_1= st.number_input('Score externe 1',
                                     min_value=0., value=data[0][0], step=.1)
            EXT_SOURCE_2 = st.number_input('Score externe 2',
                                     min_value=0., value=data[0][1], step=.1)
            st.write('Retard sur des crédits précédents')
            DPD_TRUE= st.checkbox('oui', value = data[0][7])
                                         
        with col2:
            DATE_EMPLOYED = st.date_input('Date de début d\'emploi', value = DATE_EMPLOYED)
            DATE_BIRTH = st.date_input('Date de naissance', min_value=datetime.strptime("01/01/1950", "%d/%m/%Y"), max_value=date.today(), value = DATE_BIRTH)
            DATE_ID_PUBLISH = st.date_input('Date d''édition de la pièce d\’identité', value = DATE_ID_PUBLISH)
        with col3 :
            PAYMENT_RATE =st.slider('Taux de remboursement',
                                 min_value=1, max_value = 20, value=int(data[0][5]*100), step=1)


            ANNUITY_INCOME_PERC = st.slider('Taux d\'endettement',
                             min_value=1, max_value = 35, value=int(data[0][6]*100), step=1)
        today = date.today()
        DAYS_EMPLOYED = (today - DATE_EMPLOYED).days
        DAYS_BIRTH = (today - DATE_BIRTH).days
        DAYS_ID_PUBLISH = (today - DATE_ID_PUBLISH).days
        DPD_BOOL = DPD_TRUE
        data = [[EXT_SOURCE_1,EXT_SOURCE_2,DAYS_EMPLOYED, DAYS_BIRTH, DAYS_ID_PUBLISH, PAYMENT_RATE, ANNUITY_INCOME_PERC, DPD_BOOL]]
        
    predict_btn = st.button('Prédire')
    st.session_state['data'] = data
    
    if predict_btn:
        
        pred = None
        #pred = request_prediction(MLFLOW_URI, data)[0]
        pred = predict_joblib(data)

            
            
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


        st.session_state['pred'] = pred
        col1, col2 = st.columns(2)
        with col1:
            st_echarts(options=option, key="0")
        with col2:
            if pred > 50:
                image = Image.open('credit_non.png')
                st.image(image, caption='', width = 150)
            else:
                image = Image.open('credit_oui.png')
                st.image(image, caption='', width = 150)

            
            
        










if __name__ == '__main__':
    main()
