import unittest
import json
from app import app  # Assurez-vous que le nom de votre fichier principal est `app.py`


class FlaskApiTestCase(unittest.TestCase):
    def setUp(self):
        # Créez un client de test
        self.app = app.test_client()
        self.app.testing = True

    def test_homepage(self):
        """Test de la page d'accueil."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode(), "APP loaded, model and data loaded")

    def test_get_customer_data(self):
        """Test de la récupération des données d'un client."""
        response = self.app.get('/app/data_cust_test/?SK_ID_CURR=100001')
        self.assertEqual(response.status_code, 200)

        # Vérifiez que la réponse contient les données
        data = json.loads(response.data)
        self.assertIn('data', data)
        self.assertIn('y_cust', data)

    def test_get_customer_data_not_found(self):
        """Test de la récupération des données d'un client non trouvé."""
        response = self.app.get('/app/data_cust_test/?SK_ID_CURR=999999')  # ID qui n'existe pas
        self.assertEqual(response.status_code, 404)
        self.assertIn('Client not found', response.get_data(as_text=True))

    def test_scoring_customer(self):
        """Test de la notation d'un client."""
        response = self.app.get('/app/scoring_cust_test/?SK_ID_CURR=100001')
        self.assertEqual(response.status_code, 200)

        # Vérifiez que la réponse contient le score
        data = json.loads(response.data)
        self.assertIn('score', data)
        self.assertIn('decision', data)


if __name__ == '__main__':
    unittest.main()