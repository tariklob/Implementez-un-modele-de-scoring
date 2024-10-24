BASE_URL = "https://appli-42e4dc055e71.herokuapp.com/"

# Appel à l'API Flask pour obtenir le score de crédit
response = requests.get(f'{BASE_URL}/api/scoring_cust_test', params={'SK_ID_CURR': SK_ID_CURR})
score_data = response.json()

# Appel à l'API Flask pour les explications locales avec LIME
response = requests.get(f'{BASE_URL}/api/local_explanation', params={'SK_ID_CURR': SK_ID_CURR})
local_explanation = response.json()

# Appel à l'API Flask pour les explications globales
response = requests.get(f'{BASE_URL}/api/global_explanation')
global_explanation = response.json()

# Appel à l'API Flask pour les explications SHAP
response = requests.get(f'{BASE_URL}/api/shap_explanation', params={'SK_ID_CURR': SK_ID_CURR})
shap_values = response.json()
