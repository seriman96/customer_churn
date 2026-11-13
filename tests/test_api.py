
import pytest
from fastapi.testclient import TestClient

from src.app.main import app


client = TestClient(app)

'''
@pytest.fixture
def valid_customer():
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
    }
'''

def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model"] == "loaded"


def test_predict_endpoint(valid_customer):
    response = client.post("/predict", json=valid_customer)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in ["Likely to churn", "Unlikely to churn"]
    assert isinstance(result["churn_probability"], float)
    assert 0 <= result["churn_probability"] <= 1
    assert result["threshold"] == 0.35


def test_predict_rejects_invalid_gender(valid_customer):
    valid_customer["gender"] = "Unknown"

    response = client.post("/predict", json=valid_customer)

    assert response.status_code == 422


def test_predict_rejects_negative_tenure(valid_customer):
    valid_customer["tenure"] = -1

    response = client.post("/predict", json=valid_customer)

    assert response.status_code == 422


def test_predict_rejects_missing_required_field(valid_customer):
    del valid_customer["SeniorCitizen"]

    response = client.post("/predict", json=valid_customer)

    assert response.status_code == 422
