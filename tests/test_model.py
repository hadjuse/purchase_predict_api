import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, call


@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    monkeypatch.setenv("ENV", "staging")
    monkeypatch.setenv("MLFLOW_SERVER", " http://10.128.0.8/")
    monkeypatch.setenv("MLFLOW_REGISTRY_NAME", "purchase_predict")


@pytest.fixture()
def model_instance():
    """Return a Model with load_model stubbed out."""
    with patch("src.model.Model.load_model", return_value=None):
        from src.model import Model
        m = Model()
        # Attach mock sklearn model and empty pipeline
        m.model = MagicMock()
        m.transform_pipeline = None
        return m


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

class TestModelPredict:
    def test_predict_no_pipeline(self, model_instance):
        model_instance.model.predict.return_value = np.array([1])
        df = pd.DataFrame([{"feature1": 0.5}])
        result = model_instance.predict(df)
        model_instance.model.predict.assert_called_once()
        np.testing.assert_array_equal(result, np.array([1]))

    def test_predict_with_dict_pipeline(self, model_instance):
        """Pipeline stored as dict of {column: encoder}."""
        encoder = MagicMock()
        encoder.transform.return_value = pd.Series([0])
        model_instance.transform_pipeline = {"category": encoder}
        model_instance.model.predict.return_value = np.array([0])

        df = pd.DataFrame([{"category": "A", "feature1": 1.0}])
        model_instance.predict(df)

        encoder.transform.assert_called_once()

    def test_predict_with_list_pipeline(self, model_instance):
        """Pipeline stored as list of (column, encoder) tuples."""
        encoder = MagicMock()
        encoder.transform.return_value = pd.Series([1])
        model_instance.transform_pipeline = [("category", encoder)]
        model_instance.model.predict.return_value = np.array([1])

        df = pd.DataFrame([{"category": None, "feature1": 0.2}])
        model_instance.predict(df)

        encoder.transform.assert_called_once()

    def test_predict_drops_reserved_columns(self, model_instance):
        model_instance.model.predict.return_value = np.array([0])
        df = pd.DataFrame([{
            "user_id": "u1",
            "user_session": "s1",
            "purchased": 1,
            "feature1": 0.5,
        }])
        model_instance.predict(df)

        called_df = model_instance.model.predict.call_args[0][0]
        assert "user_id" not in called_df.columns
        assert "user_session" not in called_df.columns
        assert "purchased" not in called_df.columns

    def test_predict_returns_none_when_no_model(self, model_instance):
        model_instance.model = None
        df = pd.DataFrame([{"feature1": 1.0}])
        result = model_instance.predict(df)
        assert result is None

    def test_predict_fillna_unknown_for_pipeline(self, model_instance):
        """NaN values in categorical columns should be filled with 'unknown'."""
        encoder = MagicMock()
        encoder.transform.return_value = pd.Series([0])
        model_instance.transform_pipeline = {"cat": encoder}
        model_instance.model.predict.return_value = np.array([0])

        df = pd.DataFrame([{"cat": None}])
        model_instance.predict(df)

        # The value passed to encoder.transform should not contain NaN
        transformed_series = encoder.transform.call_args[0][0]
        assert "unknown" in transformed_series.values