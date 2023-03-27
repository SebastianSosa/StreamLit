import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn
st.set_option('deprecation.showPyplotGlobalUse', False)
#############################################
## Importations
#############################################
## Model ------------------------------------
with open('MLFLOW/model.pkl', 'rb') as file:
    model = pickle.load(file)
    
## DATA ------------------------------------
data = pd.read_csv('./data/application_test.csv')

#############################################
## Functions
#############################################
## Data preparation ------------------------------------
def processData(features):
    import pickle
    input1 = open("scaler.pickle", "rb")
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
backup = data
## Process data for ML ------------------------------------
data = processData(data)
model.predict_proba(data[clientsIDS == id])
idLOC = np.where(clientsIDS == id)
st.markdown("# Predicted probabilities:")
st.write(model.predict_proba(np.array(data.iloc[idLOC[0],:])))

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
import plotly.express as px
from statistics import mean
fig = px.histogram(data.CNT_FAM_MEMBERS)
fig.add_vline(x = mean(data.CNT_FAM_MEMBERS[idLOC[0]]), line_dash = 'dash', line_color = 'firebrick')
st.plotly_chart(fig)


## RUN ML ML ------------------------------------
def show_predict_page():
    st.title("Software client loane risk prediction")
    st.write("### Import data")