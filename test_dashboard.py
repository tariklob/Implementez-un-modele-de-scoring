import pytest
import pandas as pd
import requests
import pickle
import streamlit as st

# Test de l'importation des modules principaux
def test_imports():
    try:
        import pandas as pd
        import requests
        import streamlit as st
        import plotly.graph_objects as go
        import lime
        import numpy as np
        import matplotlib.pyplot as plt
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

# Test de l'accès à l'API
def test_api_access():
    BASE_URL = "https://appli-42e4dc055e71.herokuapp.com/"
    response = requests.get(f'{BASE_URL}/app/data_cust_test/', params={'SK_ID_CURR': 100001})
    assert response.status_code == 200, f"API returned status code {response.status_code}"

# Test de chargement des données de test
def test_data_loading():
    try:
        df_test = pd.read_csv('data_test.csv')
        assert not df_test.empty, "DataFrame is empty"
    except Exception as e:
        pytest.fail(f"Data loading failed: {e}")
