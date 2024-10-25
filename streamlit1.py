import pandas as pd
import pickle
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import matplotlib.pyplot as plt

BASE_URL = "https://appli-42e4dc055e71.herokuapp.com/"
#BASE_URL = "http://127.0.0.1:8080"

# Charger les données de test
df_test = pd.read_csv('data_sample.csv')
X_sample = df_test.drop(columns=['TARGET', 'SK_ID_CURR'])

# Charger le modèle (pipeline)
with open('best_xgb_model.pkl', 'rb') as f:
    bestmodel = pickle.load(f)

# Extraire le modèle final du pipeline (sans scaler)
model = bestmodel.named_steps['xgb']

# Fonction pour obtenir les données du client
def get_customer_data(SK_ID_CURR):
    response = requests.get(f'{BASE_URL}/app/data_cust_test/', params={'SK_ID_CURR': SK_ID_CURR})
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API error: {response.status_code}")
        return None

# Interface utilisateur Streamlit
st.title("Credit Risk Prediction Dashboard")

# Sélectionner le client
client_ids = [''] + df_test['SK_ID_CURR'].astype(str).tolist()  # Ajouter un élément vide au début
SK_ID_CURR = st.selectbox("Select Client ID (SK_ID_CURR)", client_ids)

# Obtenir les données du client
if st.button("Get Customer Data"):
    if SK_ID_CURR:
        customer_data = get_customer_data(SK_ID_CURR)
        if customer_data:
            # Convertir les données en DataFrame et traiter les valeurs booléennes
            data = {k: (v if not isinstance(v, dict) else list(v.values())[0]) for k, v in
                    customer_data['data'].items()}
            st.table(pd.DataFrame([data]))  # Afficher les données sous forme de tableau
    else:
        st.warning("Veuillez sélectionner un ID client.")

# Obtenir le score de crédit
if st.button("Get Credit Score"):
    if SK_ID_CURR:
        response = requests.get(f'{BASE_URL}/app/scoring_cust_test/', params={'SK_ID_CURR': SK_ID_CURR})
        if response.status_code == 200:
            score_data = response.json()
            score = score_data['score']
            decision = score_data['decision']
            st.markdown(f"<h3>Score: {score * 100:.2f}% - Decision: {decision}</h3>", unsafe_allow_html=True)

            # Graphique de jauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score * 100,
                title={'text': "Credit Score"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "blue"},
                       'steps': [
                           {'range': [0, 50], 'color': "red"},  # Rejected
                           {'range': [50, 100], 'color': "green"}  # Accepted
                       ],
                       'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 50}}
            ))
            fig.update_layout(height=400)  # Ajuster la hauteur
            st.plotly_chart(fig)
    else:
        st.warning("Veuillez sélectionner un ID client.")

# Pour les explications globales en utilisant LIME (sans scaler)
if st.button("Get Global Explanation"):
    # Créer un explainer LIME pour les données
    explainer = LimeTabularExplainer(X_sample.values, feature_names=X_sample.columns,
                                     class_names=['Rejected', 'Accepted'], mode='classification')

    # Sélectionner un échantillon de données pour les explications locales
    sample_data = X_sample.copy()

    # Générer des explications locales pour chaque instance de l'échantillon
    feature_importances = np.zeros(X_sample.shape[1])
    for i in range(sample_data.shape[0]):
        exp = explainer.explain_instance(sample_data.values[i], model.predict_proba, num_features=X_sample.shape[1])
        for feature, importance in exp.as_list():
            clean_feature = feature.split(' <= ')[0].split(' > ')[0].strip()
            if clean_feature in X_sample.columns:
                feature_index = X_sample.columns.get_loc(clean_feature)
                feature_importances[feature_index] += importance

    # Calculer l'importance moyenne des caractéristiques
    feature_importances /= sample_data.shape[0]

    # Créer un DataFrame pour les importances des caractéristiques
    importance_df = pd.DataFrame({'Feature': X_sample.columns, 'Importance': feature_importances})

    # Graphique pour les valeurs fortes (les 10 caractéristiques les plus importantes)
    high_importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)
    fig_high = px.bar(high_importance_df, x='Importance', y='Feature', orientation='h',
                      title="Global Feature Importance (High Values)", color_discrete_sequence=['blue'])

    # Graphique pour les valeurs faibles (les 10 caractéristiques les moins importantes)
    low_importance_df = importance_df.sort_values(by='Importance', ascending=True).head(10)
    fig_low = px.bar(low_importance_df, x='Importance', y='Feature', orientation='h',
                     title="Global Feature Importance (Low Values)", color_discrete_sequence=['red'])

    # Afficher les graphiques dans Streamlit
    st.plotly_chart(fig_high)
    st.plotly_chart(fig_low)

# Pour les explications locales avec LIME (sans scaler)
if st.button("Get Local Explanation"):
    if SK_ID_CURR != '':
        SK_ID_CURR_int = int(SK_ID_CURR)

        # Obtenir les données du client pour l'explication LIME
        X_cust = X_sample.loc[df_test['SK_ID_CURR'] == SK_ID_CURR_int]

        # Créer un explainer LIME pour les données
        explainer = LimeTabularExplainer(X_sample.values,
                                         feature_names=X_sample.columns,
                                         class_names=['Rejected', 'Accepted'],
                                         mode='classification')

        # Générer l'explication pour le client avec LIME
        exp = explainer.explain_instance(X_cust.values[0], model.predict_proba, num_features=10)

        # Afficher le graphique LIME dans Streamlit
        fig = exp.as_pyplot_figure()
        plt.title('Importance Locale des Variables (LIME)')
        st.pyplot(fig)
    else:
        st.warning("Veuillez sélectionner un ID client.")
