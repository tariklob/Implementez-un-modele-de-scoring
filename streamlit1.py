import pandas as pd
import pickle
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import matplotlib.pyplot as plt


# Définir l'URL de base pour les appels à l'API Flask
BASE_URL = "http://127.0.0.1:8080"

# Charger les données de test localement
df_test = pd.read_csv('data_test.csv')
X_sample = df_test.drop(columns=['TARGET', 'SK_ID_CURR'])

# Charger le modèle (pipeline)
with open('best_xgb_model.pkl', 'rb') as f:
    bestmodel = pickle.load(f)

# Extraire le modèle final du pipeline (sans scaler)
model = bestmodel.named_steps['xgb']


# Interface utilisateur Streamlit
st.title("Credit Risk Prediction Dashboard")

# Sélectionner le client
client_ids = [''] + df_test['SK_ID_CURR'].astype(str).tolist()  # Ajouter un élément vide au début
SK_ID_CURR = st.selectbox("Select Client ID (SK_ID_CURR)", client_ids)

# Fonction pour obtenir les données du client via l'API Flask
def get_customer_data(SK_ID_CURR):
    response = requests.get(f'{BASE_URL}/app/data_cust_test/', params={'SK_ID_CURR': SK_ID_CURR})
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API error: {response.status_code}")
        return None

# Fonction pour obtenir le score de crédit via l'API Flask
def get_credit_score(SK_ID_CURR):
    response = requests.get(f'{BASE_URL}/app/scoring_cust_test/', params={'SK_ID_CURR': SK_ID_CURR})
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API error: {response.status_code}")
        return None

# Fonction pour obtenir les explications locales via LIME
def get_local_explanation(X_cust):
    explainer = LimeTabularExplainer(X_sample.values,
                                     feature_names=X_sample.columns,
                                     class_names=['Rejected', 'Accepted'],
                                     mode='classification')

    # Générer l'explication pour le client avec LIME
    exp = explainer.explain_instance(X_cust.values[0], model.predict_proba, num_features=10)
    return exp

# Fonction pour obtenir les explications globales via LIME
def get_global_explanation():
    explainer = LimeTabularExplainer(X_sample.values,
                                     feature_names=X_sample.columns,
                                     class_names=['Rejected', 'Accepted'],
                                     mode='classification')

    # Calculer l'importance moyenne des caractéristiques pour chaque instance
    feature_importances = np.zeros(X_sample.shape[1])
    for i in range(X_sample.shape[0]):
        exp = explainer.explain_instance(X_sample.values[i], model.predict_proba, num_features=X_sample.shape[1])
        for feature, importance in exp.as_list():
            clean_feature = feature.split(' <= ')[0].split(' > ')[0].strip()
            if clean_feature in X_sample.columns:
                feature_index = X_sample.columns.get_loc(clean_feature)
                feature_importances[feature_index] += importance

    feature_importances /= X_sample.shape[0]
    return pd.DataFrame({'Feature': X_sample.columns, 'Importance': feature_importances})

# Obtenir les données du client
if st.button("Get Customer Data"):
    if SK_ID_CURR:
        customer_data = get_customer_data(SK_ID_CURR)
        if customer_data:
            # Convertir les données en DataFrame et afficher sous forme de tableau
            data = {k: (v if not isinstance(v, dict) else list(v.values())[0]) for k, v in customer_data['data'].items()}
            st.table(pd.DataFrame([data]))
    else:
        st.warning("Veuillez sélectionner un ID client.")

# Obtenir le score de crédit
if st.button("Get Credit Score"):
    if SK_ID_CURR:
        score_data = get_credit_score(SK_ID_CURR)
        if score_data:
            score = score_data['score']
            decision = score_data['decision']
            st.markdown(f"<h3>Score: {score * 100:.2f}% - Decision: {decision}</h3>", unsafe_allow_html=True)

            # Graphique de jauge pour le score
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score * 100,
                title={'text': "Credit Score"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "blue"},
                       'steps': [
                           {'range': [0, 50], 'color': "red"},
                           {'range': [50, 100], 'color': "green"}
                       ],
                       'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 50}}
            ))
            st.plotly_chart(fig)
    else:
        st.warning("Veuillez sélectionner un ID client.")

# Obtenir les explications globales
if st.button("Get Global Explanation"):
    importance_df = get_global_explanation()

    # Afficher les 10 caractéristiques les plus importantes
    high_importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)
    fig_high = px.bar(high_importance_df, x='Importance', y='Feature', orientation='h',
                      title="Global Feature Importance (High Values)", color_discrete_sequence=['blue'])

    # Afficher les 10 caractéristiques les moins importantes
    low_importance_df = importance_df.sort_values(by='Importance', ascending=True).head(10)
    fig_low = px.bar(low_importance_df, x='Importance', y='Feature', orientation='h',
                     title="Global Feature Importance (Low Values)", color_discrete_sequence=['red'])

    st.plotly_chart(fig_high)
    st.plotly_chart(fig_low)

# Obtenir les explications locales
if st.button("Get Local Explanation"):
    if SK_ID_CURR:
        SK_ID_CURR_int = int(SK_ID_CURR)
        X_cust = X_sample.loc[df_test['SK_ID_CURR'] == SK_ID_CURR_int]
        exp = get_local_explanation(X_cust)

        # Afficher le graphique LIME dans Streamlit
        fig = exp.as_pyplot_figure()
        plt.title('Importance Locale des Variables (LIME)')
        st.pyplot(fig)
    else:
        st.warning("Veuillez sélectionner un ID client.")
