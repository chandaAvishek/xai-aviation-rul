import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from xai_aviation_rul.models import (
    train_linear_regression,
    train_random_forest,
    train_xgboost,
    evaluate_model,
    save_model,
    load_model,
)


@pytest.fixture
def sample_data():
    """Create sample training and test data."""
    np.random.seed(42)
    n_train = 100
    n_test = 20
    n_features = 14
    
    # Create training data
    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f"sensor_{i+1}" for i in range(n_features)]
    )
    y_train = pd.Series(np.random.uniform(0, 125, n_train), name="RUL_capped")
    
    # Create test data
    X_test = pd.DataFrame(
        np.random.randn(n_test, n_features),
        columns=[f"sensor_{i+1}" for i in range(n_features)]
    )
    y_test = pd.Series(np.random.uniform(0, 125, n_test), name="RUL_capped")
    
    return X_train, y_train, X_test, y_test


def test_train_linear_regression(sample_data):
    """Test Linear Regression training."""
    X_train, y_train, _, _ = sample_data
    
    model = train_linear_regression(X_train, y_train)
    
    # Check model type and attributes
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "coef_")
    assert len(model.coef_) == X_train.shape[1]
    
    # Check predictions are reasonable
    predictions = model.predict(X_train)
    assert predictions.shape[0] == X_train.shape[0]
    assert np.all(np.isfinite(predictions))


def test_train_random_forest(sample_data):
    """Test Random Forest training with specified hyperparameters."""
    X_train, y_train, _, _ = sample_data
    
    model = train_random_forest(X_train, y_train, n_estimators=100, random_state=42)
    
    # Check model type and attributes
    assert model is not None
    assert hasattr(model, "predict")
    assert model.get_params()["n_estimators"] == 100
    assert model.get_params()["random_state"] == 42
    
    # Check predictions are reasonable
    predictions = model.predict(X_train)
    assert predictions.shape[0] == X_train.shape[0]
    assert np.all(np.isfinite(predictions))


def test_train_xgboost(sample_data):
    """Test XGBoost training with specified hyperparameters."""
    X_train, y_train, _, _ = sample_data
    
    model = train_xgboost(
        X_train, y_train,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    )
    
    # Check model type and attributes
    assert model is not None
    assert hasattr(model, "predict")
    assert model.get_params()["n_estimators"] == 200
    assert model.get_params()["max_depth"] == 6
    assert model.get_params()["learning_rate"] == 0.05
    assert model.get_params()["random_state"] == 42
    
    # Check predictions are reasonable
    predictions = model.predict(X_train)
    assert predictions.shape[0] == X_train.shape[0]
    assert np.all(np.isfinite(predictions))


def test_evaluate_model(sample_data):
    """Test model evaluation returns correct metrics."""
    X_train, y_train, X_test, y_test = sample_data
    
    model = train_linear_regression(X_train, y_train)
    scores = evaluate_model(model, X_test, y_test)
    
    # Check return type and structure
    assert isinstance(scores, dict)
    assert set(scores.keys()) == {"RMSE", "MAE", "R²"}
    
    # Check metric values are reasonable
    assert all(np.isfinite(v) for v in scores.values())
    assert scores["RMSE"] >= 0
    assert scores["MAE"] >= 0
    assert -1 <= scores["R²"] <= 1


def test_evaluate_model_perfect_prediction():
    """Test evaluation with perfect predictions (y_test = y_pred)."""
    y_test = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Create a mock model that returns perfect predictions
    class PerfectModel:
        def predict(self, X):
            return np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    model = PerfectModel()
    X_test = pd.DataFrame(np.zeros((5, 10)))
    scores = evaluate_model(model, X_test, y_test)
    
    # Perfect predictions should have RMSE ≈ 0, MAE ≈ 0, R² ≈ 1
    assert scores["RMSE"] < 1e-10
    assert scores["MAE"] < 1e-10
    assert scores["R²"] > 0.9999


def test_save_and_load_model(sample_data):
    """Test saving and loading models with joblib."""
    X_train, y_train, X_test, y_test = sample_data
    
    # Train a model
    model = train_linear_regression(X_train, y_train)
    original_pred = model.predict(X_test)
    
    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pkl"
        save_model(model, model_path)
        
        # Check file was created
        assert model_path.exists()
        
        # Load model
        loaded_model = load_model(model_path)
        loaded_pred = loaded_model.predict(X_test)
        
        # Check predictions are identical
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)


def test_save_model_creates_directory(sample_data):
    """Test that save_model creates parent directories if they don't exist."""
    X_train, y_train, _, _ = sample_data
    
    model = train_linear_regression(X_train, y_train)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_path = Path(tmpdir) / "nested" / "dir" / "model.pkl"
        
        # Directory should not exist yet
        assert not nested_path.parent.exists()
        
        # Save should create it
        save_model(model, nested_path)
        
        # Check it was created
        assert nested_path.exists()
        assert nested_path.parent.exists()


def test_model_comparison(sample_data):
    """Test that all three models can be trained and evaluated."""
    X_train, y_train, X_test, y_test = sample_data
    
    # Train all three models
    lr_model = train_linear_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    
    # Evaluate all three
    lr_scores = evaluate_model(lr_model, X_test, y_test)
    rf_scores = evaluate_model(rf_model, X_test, y_test)
    xgb_scores = evaluate_model(xgb_model, X_test, y_test)
    
    # All should be valid
    for scores in [lr_scores, rf_scores, xgb_scores]:
        assert isinstance(scores, dict)
        assert all(np.isfinite(v) for v in scores.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
