import pandas as pd
import numpy as np
import streamlit as st
import requests
from datetime import date
from datetime import datetime, timedelta
from streamlit_echarts import st_echarts
import js2py
from PIL import Image
import seaborn as sns
from matplotlib import pyplot as plt
import lime.lime_tabular
import altair as alt
import joblib
import streamlit.components.v1 as components


test_data = pd.read_csv("test_df_dropna.csv")
train_data = pd.read_csv("train_df_dropna.csv")
if 'data' not in st.session_state:
    st.session_state['data'] = [train_data.iloc[0][["EXT_SOURCE_1","EXT_SOURCE_2","DAYS_EMPLOYED", "DAYS_BIRTH", "DAYS_ID_PUBLISH", "PAYMENT_RATE", "ANNUITY_INCOME_PERC", "DPD_BOOL"]].values]
if 'pred' not in st.session_state:
    st.session_state['pred'] = train_data.iloc[0]["TARGET"]
    
def main():
    joblib_model = joblib.load("pipeline_credit.joblib")
    MLFLOW_URI = 'http://127.0.0.1:5000/invocations'
    st.set_page_config(layout="wide")
    feature_names= ["Score externe 1", "Score externe 2", "jours travaillés",  "Âge", "Date ID", "Taux de remboursement","Taux d'endettement","Retard de paiements"]
    feats = ["EXT_SOURCE_1","EXT_SOURCE_2","DAYS_EMPLOYED", "DAYS_BIRTH", "DAYS_ID_PUBLISH", "PAYMENT_RATE", "ANNUITY_INCOME_PERC", "DPD_BOOL"]
    data = st.session_state['data']
    pred = st.session_state['pred']
    data_series = pd.DataFrame(data = np.array(data).reshape(1,-1), columns = feats).iloc[0]
    if pred > 50 :
        TARGET_CLIENT = "Non solvable"
    else :
        TARGET_CLIENT = "Solvable"
    COLOR_CLIENT = ["#64C88A" if pred < 25 else "#FDDD60" if pred < 50 else "#ffa500" if pred < 75 else "#FF403F" ][0]

    ANALYSE_OPTION = st.selectbox('Choisissez une option d\'analyse :', ["Importance des variables", "Explication des résultats", "Graphiques"])
    if (ANALYSE_OPTION == "Importance des variables") :
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = joblib_model.named_steps['regressor'].feature_importances_
        display_importances(fold_importance_df)
    elif (ANALYSE_OPTION == "Explication des résultats") :
        target_names = ['credit worthy', 'credit unworthy']
        classifier_lime = lime.lime_tabular.LimeTabularExplainer(train_data[feats].values,
                                             mode='classification',
                                             class_names = target_names,
                                             categorical_features = [6],
                                             feature_names= feature_names, discretize_continuous=True, feature_selection = 'auto')
        MLFLOW_URI = 'http://127.0.0.1:5000/invocations'
        lime_results = classifier_lime.explain_instance(data_series, joblib_model.predict_proba, num_features=len(feats))
        lime_resuts_list = lime_results.as_list()
        lime_dataframe = lime_result_dataframe(lime_resuts_list)
        feature_names= feature_names
        fig = plt.figure(figsize=(10, 4))
        plt.rcParams.update({'font.size': 10})
        sns.barplot(data = lime_dataframe, x = "Feature", y = "Value")
        #ax.set_xticklabels(labels = feature_names, rotation=80)
        plt.xticks(rotation = 60)
        plt.xlabel('Variable', fontsize=10)
        plt.ylabel('Contribution', fontsize=10)
        plt.axhline(y=0, color='r', linestyle='-', linewidth=1.5)
        st.pyplot(fig)
            #st.write(lime_dataframe)
    else :
        subset_0 = train_data[train_data.TARGET == 0][:3000]
        subset_1 = train_data[train_data.TARGET == 1][:1000]
        df_subset = pd.concat([subset_0, subset_1])
        df_subset["DAYS_EMPLOYED"] = abs(df_subset["DAYS_EMPLOYED"])
        df_subset["TARGET"] = ["Non solvable" if x else "Solvable" for x in df_subset["TARGET"]]
        DAYS_EMPLOYED_CLIENT = data[0][2]
        EXT_SOURCE_1_CLIENT = data[0][0]
        col1, col2 = st.columns(2)
        with col1:
            PARAMETRE = st.selectbox('Paramètres à analyser :', ["Score externe 1", "Nombres de jours travaillés"])
            if PARAMETRE == "Score externe 1" :
                FEATURE = "EXT_SOURCE_1"
                FEATURE_CLIENT = EXT_SOURCE_1_CLIENT
            else :
                FEATURE = "DAYS_EMPLOYED"
                FEATURE_CLIENT = DAYS_EMPLOYED_CLIENT
            fig = plt.figure(figsize=(4, 4))
            plt.rcParams.update({'font.size': 10})
            # plt.boxplot(df_subset[["TARGET", FEATURE]])
            sns.boxplot(data = df_subset, x= "TARGET", y = FEATURE, palette = ["#9ACD32", "#FF0000"])
            TARGET_CLIENT_NUM = [1 if TARGET_CLIENT == "Non solvable" else 0][0]
            plt.scatter(TARGET_CLIENT_NUM, FEATURE_CLIENT, marker='X', s=100, c = COLOR_CLIENT)
            plt.ylabel(PARAMETRE)
            plt.xlabel("")
            #fig_html = mpld3.fig_to_html(fig)
            #components.html(fig_html, height=600)
            st.pyplot(fig)
        with col2 :
            st.markdown("***")
            st.markdown("***")
            palette = ["#9ACD32", "#FF0000"]
            # domain = ["Score externe 1", "Nbre de jours travaillés"]
            alt_scatter = alt.Chart(df_subset).mark_circle(size=60).encode(
                x='EXT_SOURCE_1',
                y='DAYS_EMPLOYED',
                color=alt.Color('TARGET', scale=alt.Scale(range=palette)),
                tooltip=['EXT_SOURCE_1', 'DAYS_EMPLOYED', 'TARGET']
            ).properties(width=500, height = 300).interactive()
            st.altair_chart(alt_scatter, use_container_width=True)

            
            
            
            
def request_prediction_analyse(data):
    MLFLOW_URI = 'http://127.0.0.1:5000/invocations'
    headers = {"Content-Type": "application/json"}

    data_json = {'columns':["EXT_SOURCE_1","EXT_SOURCE_2","DAYS_EMPLOYED", "DAYS_BIRTH", "DAYS_ID_PUBLISH", "PAYMENT_RATE", "ANNUITY_INCOME_PERC", "DPD_BOOL"], 'data': data}
    response = requests.request(
        method='POST', headers=headers, url=MLFLOW_URI, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
            
    pred_array = np.array([response.json()[0]/100])
    pred_proba_array = np.array(list(zip(1-pred_array, pred_array))).tolist()

    return pred_proba_array

def predict_joblib_lime(data):
    joblib_model = joblib.load("pipeline_credit.joblib")
    return int(joblib_model.predict_proba(data)[0][1]*100)
    
def predict_joblib(data):
    joblib_model = joblib.load("pipeline_credit.joblib")
    return int(joblib_model.predict_proba(data)[0][1]*100)

def lime_result_dataframe(lime_resuts_list):
    lime_dataframe = pd.DataFrame()
    feature_column = [lime_resuts_list[i][0] for i in range(len(lime_resuts_list))]
    value_column = [-lime_resuts_list[i][1] for i in range(len(lime_resuts_list))]
    feature_names= ["Score externe 1", "Score externe 2", "jours travaillés",  "Âge", "Date ID", "Taux de remboursement","Taux d'endettement","Retard de paiements"]
    feature_column_cor = []
    for feat in feature_column:
        for feat_name in feature_names:
            if feat_name in feat:
                feature_column_cor.append(feat_name)
    lime_dataframe.index = feature_column_cor
    lime_dataframe["Value"] = value_column
    lime_dataframe = lime_dataframe.reindex(feature_names, copy = False)
    lime_dataframe["Feature"] = feature_column_cor
    return lime_dataframe

def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False).index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    fig = plt.figure(figsize=(4, 2.5))
    plt.rcParams.update({'font.size': 5})
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('Variables selon leur degré d\'impact sur les prédictions de façon globale', fontsize = 8)
    plt.tight_layout()
    plt.xlabel('Importance', fontsize=8)
    plt.ylabel('Variable', fontsize=8)
    st.pyplot(fig)

if __name__ == '__main__':
    main()
