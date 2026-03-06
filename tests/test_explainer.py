import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import shap

from xai_aviation_rul import explainer


@pytest.fixture
def sample_data():
    """Create sample training and test data."""
    np.random.seed(42)
    n_train, n_test, n_features = 100, 20, 10
    
    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f"sensor_{i}" for i in range(n_features)]
    )
    y_train = np.random.randn(n_train)
    
    X_test = pd.DataFrame(
        np.random.randn(n_test, n_features),
        columns=[f"sensor_{i}" for i in range(n_features)]
    )
    
    return X_train, y_train, X_test


@pytest.fixture
def trained_model(sample_data):
    """Train a simple RandomForest model for testing."""
    X_train, y_train, _ = sample_data
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return model


class TestComputeShapValues:
    """Tests for compute_shap_values function."""
    
    def test_compute_shap_values_returns_tuple(self, trained_model, sample_data):
        """Test that compute_shap_values returns a tuple of (shap_values, explainer)."""
        _, _, X_test = sample_data
        result = explainer.compute_shap_values(trained_model, X_test)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_compute_shap_values_returns_correct_types(self, trained_model, sample_data):
        """Test that returned values have correct types."""
        _, _, X_test = sample_data
        shap_values, exp = explainer.compute_shap_values(trained_model, X_test)
        
        assert isinstance(shap_values, np.ndarray)
        assert isinstance(exp, shap.TreeExplainer)
    
    def test_compute_shap_values_shape(self, trained_model, sample_data):
        """Test that SHAP values have correct shape (n_samples, n_features)."""
        _, _, X_test = sample_data
        shap_values, _ = explainer.compute_shap_values(trained_model, X_test)
        
        assert shap_values.shape == X_test.shape
    
    def test_compute_shap_values_with_numpy_array(self, trained_model, sample_data):
        """Test compute_shap_values works with numpy array input."""
        _, _, X_test = sample_data
        X_test_numpy = X_test.values
        
        shap_values, exp = explainer.compute_shap_values(trained_model, X_test_numpy)
        
        assert isinstance(shap_values, np.ndarray)
        assert isinstance(exp, shap.TreeExplainer)
    
    def test_explainer_has_expected_value(self, trained_model, sample_data):
        """Test that explainer object has expected_value attribute."""
        _, _, X_test = sample_data
        _, exp = explainer.compute_shap_values(trained_model, X_test)
        
        assert hasattr(exp, 'expected_value')
        assert isinstance(exp.expected_value, (float, np.floating, np.ndarray))


class TestPlotShapSummary:
    """Tests for plot_shap_summary function."""
    
    def test_plot_shap_summary_creates_file(self, trained_model, sample_data, tmp_path):
        """Test that plot_shap_summary saves figure to file."""
        _, _, X_test = sample_data
        shap_values, _ = explainer.compute_shap_values(trained_model, X_test)
        
        output_path = tmp_path / "shap_summary.png"
        explainer.plot_shap_summary(shap_values, X_test, save_path=output_path)
        
        assert output_path.exists()
    
    def test_plot_shap_summary_without_save_path(self, trained_model, sample_data):
        """Test that plot_shap_summary works without save_path (just shows)."""
        _, _, X_test = sample_data
        shap_values, _ = explainer.compute_shap_values(trained_model, X_test)
        
        # Should not raise an exception
        explainer.plot_shap_summary(shap_values, X_test, save_path=None)
    
    def test_plot_shap_summary_with_string_path(self, trained_model, sample_data, tmp_path):
        """Test that save_path works with string input."""
        _, _, X_test = sample_data
        shap_values, _ = explainer.compute_shap_values(trained_model, X_test)
        
        output_path = str(tmp_path / "shap_summary.png")
        explainer.plot_shap_summary(shap_values, X_test, save_path=output_path)
        
        assert Path(output_path).exists()
    
    def test_plot_shap_summary_file_format(self, trained_model, sample_data, tmp_path):
        """Test that saved file is a valid PNG."""
        _, _, X_test = sample_data
        shap_values, _ = explainer.compute_shap_values(trained_model, X_test)
        
        output_path = tmp_path / "shap_summary.png"
        explainer.plot_shap_summary(shap_values, X_test, save_path=output_path)
        
        # PNG files
        with open(output_path, 'rb') as f:
            header = f.read(4)
            assert header == b'\x89PNG'


class TestPlotShapWaterfall:
    """Tests for plot_shap_waterfall function."""
    
    def test_plot_shap_waterfall_creates_file(self, trained_model, sample_data, tmp_path):
        """Test that plot_shap_waterfall saves figure to file."""
        _, _, X_test = sample_data
        shap_values, exp = explainer.compute_shap_values(trained_model, X_test)
        
        output_path = tmp_path / "shap_waterfall.png"
        explainer.plot_shap_waterfall(exp, shap_values, X_test, engine_idx=0, save_path=output_path)
        
        assert output_path.exists()
    
    def test_plot_shap_waterfall_without_save_path(self, trained_model, sample_data):
        """Test that plot_shap_waterfall works without save_path."""
        _, _, X_test = sample_data
        shap_values, exp = explainer.compute_shap_values(trained_model, X_test)
        
        # Should not raise an exception
        explainer.plot_shap_waterfall(exp, shap_values, X_test, engine_idx=0, save_path=None)
    
    def test_plot_shap_waterfall_different_indices(self, trained_model, sample_data, tmp_path):
        """Test waterfall plot for different engine indices."""
        _, _, X_test = sample_data
        shap_values, exp = explainer.compute_shap_values(trained_model, X_test)
        
        for idx in [0, 5, 10]:
            output_path = tmp_path / f"shap_waterfall_{idx}.png"
            explainer.plot_shap_waterfall(exp, shap_values, X_test, engine_idx=idx, save_path=output_path)
            assert output_path.exists()
    
    def test_plot_shap_waterfall_invalid_index(self, trained_model, sample_data):
        """Test that invalid engine_idx raises appropriate error."""
        _, _, X_test = sample_data
        shap_values, exp = explainer.compute_shap_values(trained_model, X_test)
        n_samples = X_test.shape[0]
        
        with pytest.raises(IndexError):
            explainer.plot_shap_waterfall(exp, shap_values, X_test, engine_idx=n_samples + 10, save_path=None)
    
    def test_plot_shap_waterfall_with_string_path(self, trained_model, sample_data, tmp_path):
        """Test that save_path works with string input."""
        _, _, X_test = sample_data
        shap_values, exp = explainer.compute_shap_values(trained_model, X_test)
        
        output_path = str(tmp_path / "shap_waterfall.png")
        explainer.plot_shap_waterfall(exp, shap_values, X_test, engine_idx=0, save_path=output_path)
        
        assert Path(output_path).exists()
    
    def test_plot_shap_waterfall_file_format(self, trained_model, sample_data, tmp_path):
        """Test that saved file is a valid PNG."""
        _, _, X_test = sample_data
        shap_values, exp = explainer.compute_shap_values(trained_model, X_test)
        
        output_path = tmp_path / "shap_waterfall.png"
        explainer.plot_shap_waterfall(exp, shap_values, X_test, engine_idx=0, save_path=output_path)
        
        # PNG files start with magic bytes: 89 50 4E 47
        with open(output_path, 'rb') as f:
            header = f.read(4)
            assert header == b'\x89PNG'


class TestExplainerIntegration:
    """Integration tests combining multiple functions."""
    
    def test_full_explainer_workflow(self, trained_model, sample_data, tmp_path):
        """Test a complete SHAP explanation workflow."""
        _, _, X_test = sample_data
        
        # Step 1: Compute SHAP values
        shap_values, exp = explainer.compute_shap_values(trained_model, X_test)
        assert shap_values.shape == X_test.shape
        
        # Step 2: Create summary plot
        summary_path = tmp_path / "summary.png"
        explainer.plot_shap_summary(shap_values, X_test, save_path=summary_path)
        assert summary_path.exists()
        
        # Step 3: Create waterfall plot
        waterfall_path = tmp_path / "waterfall.png"
        explainer.plot_shap_waterfall(exp, shap_values, X_test, engine_idx=0, save_path=waterfall_path)
        assert waterfall_path.exists()
    
    def test_multiple_waterfall_plots(self, trained_model, sample_data, tmp_path):
        """Test creating waterfall plots for multiple engines."""
        _, _, X_test = sample_data
        shap_values, exp = explainer.compute_shap_values(trained_model, X_test)
        
        indices = [0, 5, 10, 15, 19]
        for idx in indices:
            output_path = tmp_path / f"waterfall_{idx}.png"
            explainer.plot_shap_waterfall(exp, shap_values, X_test, engine_idx=idx, save_path=output_path)
            assert output_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
