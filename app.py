#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
from flask import Flask, request, jsonify
import joblib  
import numpy as np

# Initialiser Flask
app = Flask(__name__)

# Charger le modèle XGBoost sauvegardé
model = joblib.load("best_xgb_model.pkl")

# Définir la route pour faire des prédictions
@app.route('/predict', methods=['POST'])
def predict():
    # Obtenir les données du POST request
    data = request.json  # Les données devraient être envoyées au format JSON
    try:
        # Extraire les features (ici supposons que les données d'entrée sont un tableau de features)
        features = np.array(data['features']).reshape(1, -1)  # Adaptation selon tes données
        
        # Faire la prédiction avec le modèle
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)

        # Retourner la prédiction et la probabilité sous forme JSON
        result = {
            'prediction': int(prediction[0]),
            'prediction_proba': prediction_proba.tolist()  # Probabilité sous forme de liste
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

# Point de départ de l'API
if __name__ == '__main__':
    app.run(debug=True)

