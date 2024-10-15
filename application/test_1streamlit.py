import pytest
import requests
from streamlit import session_state
from app_streamlit import get_customer_data  # Importez votre fonction de l'application Streamlit


def test_get_customer_data(mocker):
    # Mock l'appel à l'API
    mock_response = {
        'data': {'feature_1': 1, 'feature_2': 2},
        'y_cust': 0
    }
    mocker.patch('requests.get', return_value=mock_response)

    # Appeler la fonction avec un ID client
    result = get_customer_data(100001)

    # Vérifiez que le résultat est correct
    assert result['data']['feature_1'] == 1
    assert result['y_cust'] == 0
