import pytest
import numpy as np
from PIL import Image
from io import BytesIO
from app.models.LIMExplainer import LIMExplainer
from app.core.exceptions import ModelProcessingError

@pytest.fixture
def sample_image():
    # Создаем тестовое изображение
    img = Image.new('RGB', (224, 224), color='white')
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

@pytest.fixture
def mock_model():
    class MockModel:
        def predict(self, x):
            # Возвращаем shape (n_samples, 4)
            return np.tile([0.25, 0.25, 0.25, 0.25], (len(x), 1))
    return MockModel()

class TestLIMExplainer:
    def test_init(self, mock_model):
        explainer = LIMExplainer(mock_model)
        assert explainer.model == mock_model
        assert explainer.num_samples == 1000  # default value
        assert explainer.explainer is not None
        
    def test_init_with_custom_samples(self, mock_model):
        explainer = LIMExplainer(mock_model, num_samples=500)
        assert explainer.num_samples == 500
        
    def test_explain(self, sample_image, mock_model):
        explainer = LIMExplainer(mock_model)
        img_array = np.array(Image.open(sample_image))
        img_array = np.expand_dims(img_array, axis=0)
        
        explanation = explainer.explain(img_array[0])
        assert explanation is not None
        assert hasattr(explanation, 'local_exp')
        assert hasattr(explanation, 'score')
        assert hasattr(explanation, 'top_labels')
        assert len(explanation.top_labels) > 0
        
    def test_explain_with_normalized_input(self, sample_image, mock_model):
        explainer = LIMExplainer(mock_model)
        img_array = np.array(Image.open(sample_image))
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)
        
        explanation = explainer.explain(img_array[0])
        assert explanation is not None
        assert hasattr(explanation, 'local_exp')
        
    def test_explain_with_invalid_input(self, mock_model):
        explainer = LIMExplainer(mock_model)
        with pytest.raises(ValueError):
            explainer.explain(None)
            
    def test_get_visualization(self, sample_image, mock_model):
        explainer = LIMExplainer(mock_model)
        img_array = np.array(Image.open(sample_image))
        img_array = np.expand_dims(img_array, axis=0)
        
        explanation = explainer.explain(img_array[0])
        vis = explainer.get_visualization(explanation)
        assert vis is not None
        assert isinstance(vis, np.ndarray)
        assert vis.shape == (224, 224, 3)
        assert vis.dtype == np.uint8
        assert np.all(vis >= 0) and np.all(vis <= 255)
        
    def test_visualization_parameters(self, sample_image, mock_model):
        explainer = LIMExplainer(mock_model)
        img_array = np.array(Image.open(sample_image))
        img_array = np.expand_dims(img_array, axis=0)
        
        explanation = explainer.explain(img_array[0])
        
        # Test different alpha values
        for alpha in [0.1, 0.5, 0.9]:
            vis = explainer.get_visualization(explanation, alpha=alpha)
            assert vis is not None
            assert isinstance(vis, np.ndarray)
            assert vis.shape == (224, 224, 3)
            assert vis.dtype == np.uint8
        
        # Test different num_features values
        for num_features in [3, 5, 7]:
            vis = explainer.get_visualization(explanation, num_features=num_features)
            assert vis is not None
            assert isinstance(vis, np.ndarray)
            assert vis.shape == (224, 224, 3)
            assert vis.dtype == np.uint8
            
    def test_visualization_with_invalid_explanation(self, mock_model):
        explainer = LIMExplainer(mock_model)
        with pytest.raises(ValueError):
            explainer.get_visualization(None)
            
    def test_visualization_with_invalid_parameters(self, sample_image, mock_model):
        explainer = LIMExplainer(mock_model)
        img_array = np.array(Image.open(sample_image))
        img_array = np.expand_dims(img_array, axis=0)
        explanation = explainer.explain(img_array[0])
        
        with pytest.raises(ValueError):
            explainer.get_visualization(explanation, alpha=-0.1)
        
        with pytest.raises(ValueError):
            explainer.get_visualization(explanation, alpha=1.1)
        
        with pytest.raises(ValueError):
            explainer.get_visualization(explanation, num_features=0)
            
    def test_model_prediction_error(self, sample_image):
        class ErrorModel:
            def predict(self, x):
                raise Exception("Model prediction error")
                
        explainer = LIMExplainer(ErrorModel())
        img_array = np.array(Image.open(sample_image))
        img_array = np.expand_dims(img_array, axis=0)
        
        with pytest.raises(ModelProcessingError):
            explainer.explain(img_array[0]) 