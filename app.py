from flask import Flask, jsonify, request
import pandas as pd
import pickle
import shap
import os

app = Flask(__name__)

# Load the data and the model
df_test = pd.read_csv('data_test.csv')
X_sample = df_test.drop(columns=['TARGET', 'SK_ID_CURR'])
y_sample = df_test['TARGET']

with open('best_xgb_model.pkl', 'rb') as f:
    bestmodel = pickle.load(f)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def homepage():
    return "APP loaded, model and data loaded"

@app.route('/app/data_cust_test/', methods=['GET'])
def get_customer_data():
    """Return customer data excluding 'TARGET' and 'SK_ID_CURR'."""
    SK_ID_CURR = request.args.get('SK_ID_CURR')
    try:
        client_index = X_sample.index[df_test['SK_ID_CURR'] == int(SK_ID_CURR)].tolist()
        if len(client_index) == 0:
            return jsonify({'error': 'Client not found'}), 404

        data_cust = X_sample.loc[client_index]
        y_cust = y_sample.loc[client_index].values[0]
        return jsonify({'data': data_cust.to_dict(), 'y_cust': y_cust})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/app/scoring_cust_test/', methods=['GET'])
def scoring_cust_test():
    """Return the scoring for a customer."""
    SK_ID_CURR = request.args.get('SK_ID_CURR')
    try:
        client_index = X_sample.index[df_test['SK_ID_CURR'] == int(SK_ID_CURR)].tolist()
        if len(client_index) == 0:
            return jsonify({'error': 'Client not found'}), 404

        X_cust = X_sample.loc[client_index]
        X_cust_scaled = scaler.transform(X_cust)
        score_cust = float(bestmodel.predict_proba(X_cust_scaled)[:, 1][0])
        decision = 'accepted' if score_cust >= 0.40 else 'rejected'
        return jsonify({'score': score_cust, 'decision': decision})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/app/shap_explanation/', methods=['GET'])
def shap_explanation():
    """Return SHAP values for the specified customer."""
    SK_ID_CURR = request.args.get('SK_ID_CURR')
    client_index = X_sample.index[df_test['SK_ID_CURR'] == int(SK_ID_CURR)].tolist()
    if len(client_index) == 0:
        return jsonify({'error': 'Client not found'}), 404

    # Extract the model from the pipeline
    model = bestmodel.named_steps['xgb']

    explainer = shap.TreeExplainer(model)
    X_cust = X_sample.loc[client_index]
    X_cust_scaled = scaler.transform(X_cust)
    shap_values = explainer.shap_values(X_cust_scaled)
    return jsonify({'shap_values': shap_values.tolist()})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))  # Utilise le port fourni ou 8080 par défaut
    app.run(debug=True, port=port)