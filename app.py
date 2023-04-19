import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn
import plotly.express as px
from statistics import mean
import plotly.graph_objects as go
import requests

st.set_option('deprecation.showPyplotGlobalUse', False)
#############################################
## Importations
#############################################
## Model ------------------------------------
model  =  pickle.load(open('data/model.pkl','rb'))
## DATA ------------------------------------
data = pd.read_csv('data/application_test.csv')

#############################################
## Functions
#############################################
## Data preparation ------------------------------------
def processData(features):
    import pickle
    input1 = open("data/scaler.pickle", "rb")
    scaler = pickle.load(input1)
    input1.close()
    
    # Remove the ids and target
    features = features.drop(columns = ['SK_ID_CURR'])  
      
    # OHE
    features = pd.get_dummies(features)   
   
   # Age information into a separate dataframe
    age_data = features[['DAYS_BIRTH']]
    age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

    # Bin the age data
    age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
    age_data.head(10)
    # Bin the age data
    features['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
    
    features['CREDIT_INCOME_PERCENT'] = features['AMT_CREDIT'] / features['AMT_INCOME_TOTAL']
    features['ANNUITY_INCOME_PERCENT'] = features['AMT_ANNUITY'] / features['AMT_INCOME_TOTAL']
    features['CREDIT_TERM'] = features['AMT_ANNUITY'] / features['AMT_CREDIT']
    features['DAYS_EMPLOYED_PERCENT'] = features['DAYS_EMPLOYED'] / features['DAYS_BIRTH']
    
    features.dropna(axis='columns', inplace=True)
    
    from sklearn.preprocessing import MinMaxScaler
    # Transform both training and testing data
    #features = imputer.transform(features)    
    # Repeat with the scaler
    #features = scaler.transform(features)  
  
    # Convert to np arrays
    #features = np.array(features)
    
    return features

## Model Visualization ------------------------------------
import shap
def modelVisualization(m, X_test, select = range(20)):
    explainer = shap.TreeExplainer(m.estimator)   

    #P1
    import warnings
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value
    print(f"Explainer expected value: {expected_value}")
    
    features = X_test.iloc[select]
    features = np.array(features).reshape(1, -1)
    features_display = X_test.columns
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(features)[1]
        shap_interaction_values = explainer.shap_interaction_values(features)
    if isinstance(shap_interaction_values, list):
        shap_interaction_values = shap_interaction_values[1]
    shap.decision_plot(expected_value, shap_values, features_display)

## Model Visualization shap js in streamlite------------------------------------
def st_shap(plot, height=None):
    import streamlit.components.v1 as components
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)    

#############################################
## APP
#############################################
## IDS ------------------------------------
clientsIDS = data.SK_ID_CURR
id = st.selectbox("Client ID", clientsIDS)
index = np.where(data.SK_ID_CURR == id)
backup = data
## Process data for ML ------------------------------------

r = requests.get("https://mlflowbank.drsosa.repl.co/get/" + str(index[0][0])).json()
data = processData(data)
model.predict_proba(data[clientsIDS == id])

idLOC = np.where(clientsIDS == id)
st.markdown("# Predicted probabilities:")
st.write(r["p1"], r["p2"])
#############################################
## PLots
#############################################
# Plot1 ------------------------------------
st.markdown("# Features importance")
explainer = shap.TreeExplainer(model.estimator)
shap_values = explainer.shap_values(data.iloc[idLOC[0],:])
shap.initjs()
st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0][0:], feature_names=data.columns))

# Plot2 ------------------------------------
st.pyplot(modelVisualization(model, data, select = idLOC))

# Plot3 ------------------------------------
st.markdown("# Comparison of client features with the database")
var = data.columns
varSelected = st.selectbox("Client variable to compare", var)

fig = go.Figure()
fig.add_trace(go.Box(x = np.zeros(data.shape[0]), y=data[varSelected].tolist(), showlegend=False))

fig.add_trace(go.Scatter(
    x=[0.5], y=data[varSelected][idLOC[0]],
    mode="markers",
    name="Client position compare to other clients",
    text=["Client"],
    textposition="bottom center",
    textfont=dict(
        family="sans serif",
        size=12,
        color="LightSeaGreen"
    ),
    marker=dict(
            color='LightSkyBlue',
            size=1,
            line=dict(
                color='MediumPurple',
                width=12
            ),
             symbol="diamond"
        ),
    showlegend=True
))
fig.update_layout(legend = dict(orientation="h", x = 0, y = -0.10))
fig.update_xaxes(visible=False, showticklabels=False)
fig.update_yaxes(title = varSelected)
fig

#############################################
## sidebar
#############################################
st.sidebar.header("Modify client informations")
var2 = data.columns
var2Selected = st.sidebar.selectbox("Client variable to modify", var2)

