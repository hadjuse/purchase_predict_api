import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch


# Patch env vars and model loading before importing app
@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    monkeypatch.setenv("ENV", "staging")
    monkeypatch.setenv("MLFLOW_SERVER", " http://10.128.0.8/")
    monkeypatch.setenv("MLFLOW_REGISTRY_NAME", "purchase_predict")


@pytest.fixture()
def mock_model():
    model = MagicMock()
    model.predict.return_value = np.array([0, 1])
    return model


@pytest.fixture()
def client(mock_model):
    with patch("src.model.Model.load_model", return_value=None):
        from app import app, _model
        import app as app_module

        app_module._model = mock_model
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client, mock_model

        # Reset singleton after each test
        app_module._model = None


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

class TestHome:
    def test_status_ok(self, client):
        c, _ = client
        resp = c.get("/")
        assert resp.status_code == 200
        assert resp.get_json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_single_object(self, client):
        c, mock = client
        mock.predict.return_value = np.array([1])
        payload = {"feature1": 0.5, "feature2": 3}
        resp = c.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "predictions" in data
        assert data["predictions"] == [1]

    def test_predict_list_of_objects(self, client):
        c, mock = client
        mock.predict.return_value = np.array([0, 1])
        payload = [{"feature1": 0.1}, {"feature1": 0.9}]
        resp = c.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["predictions"] == [0, 1]

    def test_predict_missing_body(self, client):
        c, _ = client
        resp = c.post("/predict", data="not json", content_type="text/plain")
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_predict_invalid_json_type(self, client):
        c, _ = client
        # Send a raw JSON scalar (not object or list)
        resp = c.post(
            "/predict",
            data=json.dumps(42),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_predict_model_exception(self, client):
        c, mock = client
        mock.predict.side_effect = ValueError("bad input")
        resp = c.post("/predict", json={"feature1": 1})
        assert resp.status_code == 500
        data = resp.get_json()
        assert "error" in data
        assert "bad input" in data["error"]

    def test_predict_returns_integers(self, client):
        c, mock = client
        mock.predict.return_value = np.array([0.0, 1.0])
        resp = c.post("/predict", json=[{"f": 1}, {"f": 2}])
        assert resp.status_code == 200
        predictions = resp.get_json()["predictions"]
        assert all(isinstance(p, int) for p in predictions)
