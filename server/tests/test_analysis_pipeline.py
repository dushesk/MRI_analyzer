import pytest
import numpy as np
from fastapi import UploadFile
from io import BytesIO
from PIL import Image
from unittest.mock import patch, MagicMock
from app.services.analysis_pipeline import AnalysisPipeline
from app.core.exceptions import InvalidImageError, ModelProcessingError

@pytest.fixture
def sample_image():
    # Создаем тестовое изображение 224x224
    img = Image.new('RGB', (224, 224), color='white')
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return UploadFile(
        file=img_byte_arr,
        filename="test.png"
    )

@pytest.fixture
def invalid_image():
    # Создаем неверный файл (не изображение)
    return UploadFile(
        file=BytesIO(b"not an image"),
        filename="test.txt"
    )

@pytest.fixture
def wrong_size_image():
    # Создаем изображение неправильного размера
    img = Image.new('RGB', (100, 100), color='white')
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return UploadFile(
        file=img_byte_arr,
        filename="test.png"
    )

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
    return model

@pytest.mark.asyncio
class TestAnalysisPipeline:
    @patch('app.models.GradCAM.GradCAM.prepare_heatmap_image', return_value=Image.new('RGB', (224, 224)))
    @patch('app.models.GradCAM.GradCAM.generate_heatmap', return_value=np.zeros((224, 224)))
    @patch('app.services.analysis_pipeline.get_model')
    async def test_process_image_valid(self, mock_get_model, mock_heatmap, mock_heatmap_img, sample_image, mock_model):
        mock_get_model.return_value = mock_model
        result = await AnalysisPipeline.process_image(sample_image)
        
        # Структуру ответа
        assert result is not None
        assert 'classification' in result
        assert 'interpretation' in result
        assert 'processing_time' in result
        assert 'model_version' in result
        
        # Поля классификации
        classification = result['classification']
        assert 'class_name' in classification
        assert 'confidence' in classification
        assert 'class_id' in classification
        assert 'probabilities' in classification
        assert len(classification['probabilities']) == 4
        
        # Поля интерпретации
        interpretation = result['interpretation']
        assert 'findings' in interpretation
        assert 'recommendations' in interpretation
        assert 'severity' in interpretation
        assert 'additional_info' in interpretation
        assert 'heatmap_img' in interpretation['additional_info']
        assert 'lime_explanation' in interpretation['additional_info']

    @patch('app.models.GradCAM.GradCAM.prepare_heatmap_image', return_value=Image.new('RGB', (224, 224)))
    @patch('app.models.GradCAM.GradCAM.generate_heatmap', return_value=np.zeros((224, 224)))
    @patch('app.services.analysis_pipeline.get_model')
    async def test_process_image_invalid(self, mock_get_model, mock_heatmap, mock_heatmap_img, invalid_image):
        with pytest.raises(InvalidImageError):
            await AnalysisPipeline.process_image(invalid_image)

    @patch('app.models.GradCAM.GradCAM.prepare_heatmap_image', return_value=Image.new('RGB', (224, 224)))
    @patch('app.models.GradCAM.GradCAM.generate_heatmap', return_value=np.zeros((224, 224)))
    @patch('app.services.analysis_pipeline.get_model')
    async def test_process_image_wrong_size(self, mock_get_model, mock_heatmap, mock_heatmap_img, wrong_size_image, mock_model):
        mock_get_model.return_value = mock_model
        result = await AnalysisPipeline.process_image(wrong_size_image)
        assert result is not None
        assert 'classification' in result
        assert 'interpretation' in result
        assert 'processing_time' in result
        assert 'model_version' in result

    @patch('app.models.GradCAM.GradCAM.prepare_heatmap_image', return_value=Image.new('RGB', (224, 224)))
    @patch('app.models.GradCAM.GradCAM.generate_heatmap', return_value=np.zeros((224, 224)))
    @patch('app.services.analysis_pipeline.get_model')
    async def test_process_image_model_error(self, mock_get_model, mock_heatmap, mock_heatmap_img, sample_image):
        mock_get_model.side_effect = Exception("Model loading error")
        with pytest.raises(ModelProcessingError):
            await AnalysisPipeline.process_image(sample_image)

    @patch('app.models.GradCAM.GradCAM.prepare_heatmap_image', return_value=Image.new('RGB', (224, 224)))
    @patch('app.models.GradCAM.GradCAM.generate_heatmap', return_value=np.zeros((224, 224)))
    @patch('app.services.analysis_pipeline.get_model')
    async def test_classify_image_valid(self, mock_get_model, mock_heatmap, mock_heatmap_img, sample_image, mock_model):
        mock_get_model.return_value = mock_model
        result = await AnalysisPipeline.classify_image(sample_image)
        
        assert result is not None
        assert 'class_name' in result
        assert 'confidence' in result
        assert 'class_id' in result
        assert 'probabilities' in result
        assert len(result['probabilities']) == 4
        assert all(key in result['probabilities'] for key in [
            'MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented'
        ])

    @patch('app.models.GradCAM.GradCAM.prepare_heatmap_image', return_value=Image.new('RGB', (224, 224)))
    @patch('app.models.GradCAM.GradCAM.generate_heatmap', return_value=np.zeros((224, 224)))
    @patch('app.services.analysis_pipeline.get_model')
    async def test_interpret_image_valid(self, mock_get_model, mock_heatmap, mock_heatmap_img, sample_image, mock_model):
        mock_get_model.return_value = mock_model
        result = await AnalysisPipeline.interpret_image(sample_image)
        
        assert result is not None
        assert 'findings' in result
        assert 'recommendations' in result
        assert 'severity' in result
        assert 'additional_info' in result
        assert 'heatmap_img' in result['additional_info']
        assert 'lime_explanation' in result['additional_info']
        assert 'top_features' in result['additional_info']['lime_explanation']

    @patch('app.models.GradCAM.GradCAM.prepare_heatmap_image', return_value=Image.new('RGB', (224, 224)))
    @patch('app.models.GradCAM.GradCAM.generate_heatmap', return_value=np.zeros((224, 224)))
    @patch('app.services.analysis_pipeline.get_model')
    async def test_interpret_image_gradcam_error(self, mock_get_model, mock_heatmap, mock_heatmap_img, sample_image, mock_model):
        mock_heatmap.side_effect = Exception("GradCAM error")
        mock_get_model.return_value = mock_model
        with pytest.raises(ModelProcessingError):
            await AnalysisPipeline.interpret_image(sample_image)

    @patch('app.models.GradCAM.GradCAM.prepare_heatmap_image', return_value=Image.new('RGB', (224, 224)))
    @patch('app.models.GradCAM.GradCAM.generate_heatmap', return_value=np.zeros((224, 224)))
    @patch('app.services.analysis_pipeline.get_model')
    async def test_interpret_image_lime_error(self, mock_get_model, mock_heatmap, mock_heatmap_img, sample_image, mock_model):
        mock_get_model.return_value = mock_model
        with patch('app.models.LIMExplainer.LIMExplainer.explain', side_effect=Exception("LIME error")):
            with pytest.raises(ModelProcessingError):
                await AnalysisPipeline.interpret_image(sample_image) 