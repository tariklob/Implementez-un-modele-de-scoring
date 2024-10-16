import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import lime
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import pickle
import numpy as np

# URL of the API Flask
BASE_URL = "http://127.0.0.1:5000"

# Load the test data
df_test = pd.read_csv('data_test.csv')
X_sample = df_test.drop(columns=['TARGET', 'SK_ID_CURR'])

# Load the model (pipeline)
with open('best_xgb_model.pkl', 'rb') as f:
    bestmodel = pickle.load(f)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Extract the final model from the pipeline
model = bestmodel.named_steps['xgb']

# Function to get customer data
def get_customer_data(SK_ID_CURR):
    response = requests.get(f'{BASE_URL}/app/data_cust_test/', params={'SK_ID_CURR': SK_ID_CURR})
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API error: {response.status_code}")
        return None

# Streamlit UI
st.title("Credit Risk Prediction Dashboard")

# Select client
client_ids = [''] + df_test['SK_ID_CURR'].astype(str).tolist()  # Ajouter un élément vide au début
SK_ID_CURR = st.selectbox("Select Client ID (SK_ID_CURR)", client_ids)

# Get customer data
if st.button("Get Customer Data"):
    if SK_ID_CURR:
        customer_data = get_customer_data(SK_ID_CURR)
        if customer_data:
            # Convertir les données en DataFrame et traiter les valeurs booléennes
            data = {k: (v if not isinstance(v, dict) else list(v.values())[0]) for k, v in customer_data['data'].items()}
            st.table(pd.DataFrame([data]))  # Afficher les données sous forme de tableau
    else:
        st.warning("Please select a Client ID.")

# Get scoring
if st.button("Get Credit Score"):
    if SK_ID_CURR:
        response = requests.get(f'{BASE_URL}/app/scoring_cust_test/', params={'SK_ID_CURR': SK_ID_CURR})
        if response.status_code == 200:
            score_data = response.json()
            score = score_data['score']
            decision = score_data['decision']
            st.markdown(f"<h3>Score: {score * 100:.2f}% - Decision: {decision}</h3>", unsafe_allow_html=True)

            # Plot Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score * 100,
                title={'text': "Credit Score"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "blue"},
                       'steps': [
                           {'range': [0, 40], 'color': "green"},  # Accepted
                           {'range': [40, 100], 'color': "red"}  # Rejected
                       ],
                       'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 40}}
            ))
            st.plotly_chart(fig)
    else:
        st.warning("Please select a Client ID.")

# For Global Explanation using model's feature importances
if st.button("Get Global Explanation"):
    # Obtenir l'importance des caractéristiques du modèle
    feature_importances = model.feature_importances_
    features = X_sample.columns

    # Créer un DataFrame pour les importances des caractéristiques
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(15)  # Limiter à 15 variables les plus importantes

    # Créer un graphique à barres avec plotly
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="Global Feature Importance (Model's Feature Importances)")
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=600)  # Ajuster la hauteur du graphique

    # Afficher les résultats globaux
    st.plotly_chart(fig)

# For LIME explanations
if st.button("Get Local Explanation"):
    if SK_ID_CURR:
        # Obtenir les données du client pour l'explication LIME
        X_cust = X_sample.loc[df_test['SK_ID_CURR'] == int(SK_ID_CURR)]
        X_cust_scaled = scaler.transform(X_cust)

        # Créer un explainer LIME pour les données
        explainer = LimeTabularExplainer(X_sample.values, feature_names=X_sample.columns,
                                         class_names=['Rejected', 'Accepted'], mode='classification')

        # Générer l'explication pour le client avec LIME
        exp = explainer.explain_instance(X_cust_scaled[0], model.predict_proba, num_features=10)

        # Afficher le graphique LIME dans Streamlit
        fig = exp.as_pyplot_figure()
        plt.title('Importance Locale des Variables (LIME)')
        st.pyplot(fig)
    else:
        st.warning("Please select a Client ID.")
