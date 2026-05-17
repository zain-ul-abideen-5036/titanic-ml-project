"""tests/test_api.py"""
import sys, os, pytest
from unittest.mock import patch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with patch("os.path.exists", return_value=True):
    from fastapi.testclient import TestClient
    from api.main import app

client = TestClient(app)

VALID = {"pclass":1,"sex":"female","age":29.0,"sibsp":0,"parch":0,"fare":100.0,"embarked":"S"}
MOCK  = {"survived":True,"probability":0.91,"confidence":"high"}

def test_health():
    with patch("api.main.os.path.exists", return_value=True):
        r = client.get("/health")
    assert r.status_code == 200

def test_health_missing_model():
    with patch("api.main.os.path.exists", return_value=False):
        r = client.get("/health")
    assert r.status_code == 503

def test_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "total_predictions" in r.json()

def test_predict_valid():
    with patch("api.main.ml_predict", return_value=MOCK):
        r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    d = r.json()
    assert d["survived"] is True
    assert 0 <= d["probability"] <= 1

def test_predict_bad_sex():
    r = client.post("/predict", json={**VALID,"sex":"alien"})
    assert r.status_code == 422

def test_predict_bad_pclass():
    r = client.post("/predict", json={**VALID,"pclass":9})
    assert r.status_code == 422

def test_dashboard():
    r = client.get("/")
    assert r.status_code in (200, 404)   # 404 only if template missing in CI
