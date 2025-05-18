import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np
from app.models.ImageProcessor import ImageProcessor
from app.core.config import settings

@pytest.fixture
def sample_image_rgb():
    # Создаем тестовое RGB изображение
    return Image.new('RGB', (100, 100), color='red')

@pytest.fixture
def sample_image_grayscale():
    # Создаем тестовое grayscale изображение
    return Image.new('L', (100, 100), color=128)

@pytest.fixture
def mock_settings():
    # Мокаем настройки
    with patch('app.models.ImageProcessor.settings') as mock:
        mock.IMAGE_SIZE = (224, 224)  # Стандартный размер для моделей CNN
        yield mock

class TestImageProcessor:
    def test_preprocess_rgb_image(self, sample_image_rgb, mock_settings):
        # Тестируем обработку RGB изображения
        result = ImageProcessor.preprocess(sample_image_rgb)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 224, 224, 3)  # Проверяем форму массива
        assert result.min() >= 0 and result.max() <= 1  # Проверяем нормализацию

    def test_preprocess_grayscale_conversion(self, sample_image_grayscale, mock_settings):
        # Тестируем автоматическое преобразование grayscale в RGB
        result = ImageProcessor.preprocess(sample_image_grayscale)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 224, 224, 3)  # Должно быть 3 канала
        assert result.min() >= 0 and result.max() <= 1

    def test_preprocess_resizing(self, sample_image_rgb, mock_settings):
        # Проверяем изменение размера изображения
        with patch('PIL.Image.Image.resize') as mock_resize:
            mock_resize.return_value = Image.new('RGB', mock_settings.IMAGE_SIZE)
            ImageProcessor.preprocess(sample_image_rgb)
            mock_resize.assert_called_once_with(mock_settings.IMAGE_SIZE)

    def test_preprocess_normalization(self, sample_image_rgb, mock_settings):
        # Проверяем нормализацию значений пикселей
        result = ImageProcessor.preprocess(sample_image_rgb)
        assert np.all(result <= 1.0) and np.all(result >= 0.0)

    def test_preprocess_batch_dimension(self, sample_image_rgb, mock_settings):
        # Проверяем добавление batch-размерности
        result = ImageProcessor.preprocess(sample_image_rgb)
        assert result.shape[0] == 1  # Batch dimension

    @patch('PIL.Image.Image.convert')
    def test_grayscale_conversion_call(self, mock_convert, sample_image_grayscale, mock_settings):
        # Проверяем вызов convert для grayscale изображения
        mock_convert.return_value = Image.new('RGB', mock_settings.IMAGE_SIZE)
        ImageProcessor.preprocess(sample_image_grayscale)
        mock_convert.assert_called_once_with('RGB')

    def test_preprocess_invalid_input(self):
        # Тестируем обработку невалидного ввода
        with pytest.raises(AttributeError):
            ImageProcessor.preprocess(None)
        
        with pytest.raises(AttributeError):
            ImageProcessor.preprocess("not_an_image")