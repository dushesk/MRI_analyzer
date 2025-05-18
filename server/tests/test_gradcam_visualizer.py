import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.models.GradCAM import GradCAM
from PIL import Image
import tensorflow as tf
import cv2

@pytest.fixture
def mock_model():
    # Создаем более реалистичную mock-модель
    mock = MagicMock()
    
    # Создаем реальные тензоры для тестов
    input_tensor = tf.keras.Input(shape=(224, 224, 3))
    conv_output = tf.keras.layers.Conv2D(64, (3, 3), name='conv2d_5')(input_tensor)
    output = tf.keras.layers.Dense(2)(conv_output)
    
    # Настраиваем mock модель
    mock.inputs = [input_tensor]
    mock.outputs = [output]
    
    # Настраиваем get_layer для возврата реального слоя
    def get_layer_side_effect(name):
        if name == 'conv2d_5':
            return MagicMock(output=conv_output)
        return None
    
    mock.get_layer.side_effect = get_layer_side_effect
    
    return mock

@pytest.fixture
def sample_image_array():
    # Создаем реалистичный массив изображения правильной формы
    return np.random.rand(1, 224, 224, 3).astype(np.float32)

@pytest.fixture
def sample_heatmap():
    # Создаем реалистичную heatmap для тестов визуализации
    return np.random.rand(224, 224).astype(np.float32)

class TestGradCAM:
    def test_generate_heatmap(self, mock_model, sample_image_array):
        # Подготовка тестовых данных
        mock_conv_outputs = np.random.rand(1, 14, 14, 64).astype(np.float32)
        mock_prediction = np.array([[0.1, 0.2, 0.6, 0.1]], dtype=np.float32)
        
        # Настройка mock-объектов
        with patch('tensorflow.GradientTape') as mock_tape:
            mock_tape_instance = MagicMock()
            mock_tape.return_value.__enter__.return_value = mock_tape_instance
            
            # Настраиваем возвращаемые значения для модели и градиентов
            mock_model.predict.return_value = mock_prediction
            mock_tape_instance.gradient.return_value = [np.random.rand(1, 14, 14, 64).astype(np.float32)]
            
            # Вызов тестируемого метода
            heatmap = GradCAM.generate_heatmap(mock_model, sample_image_array)
            
            # Проверки
            assert heatmap is not None
            assert isinstance(heatmap, np.ndarray)
            assert heatmap.shape == (224, 224)
            assert np.all(heatmap >= 0)  # Проверка неотрицательных значений
            assert np.all(heatmap <= 1)   # Проверка нормализации
            
            # Проверка вызовов
            mock_model.predict.assert_called_once()
            mock_tape_instance.watch.assert_called_once()

    def test_prepare_heatmap_image(self, sample_heatmap):
        # Патчим cv2 функции, чтобы не зависеть от реальных
        with patch('cv2.resize', return_value=sample_heatmap), \
             patch('cv2.applyColorMap', return_value=np.zeros((224, 224, 3), dtype=np.uint8)):
            
            visualization = GradCAM.prepare_heatmap_image(sample_heatmap)
            
            assert visualization is not None
            assert isinstance(visualization, Image.Image)
            assert visualization.size == (224, 224)

    def test_generate_heatmap_with_invalid_input(self, mock_model):
        # Тест с None
        with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
            GradCAM.generate_heatmap(mock_model, None)
            
        # Тест с неправильным типом данных
        with pytest.raises((ValueError, TypeError)):
            GradCAM.generate_heatmap(mock_model, "invalid_input")
            
        # Тест с неправильной формой массива
        with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
            GradCAM.generate_heatmap(mock_model, np.random.rand(224, 224))  # Должно быть 4D

    @patch('cv2.resize')
    @patch('cv2.applyColorMap')
    @patch('os.makedirs')
    def test_save_heatmap(self, mock_makedirs, mock_apply, mock_resize, sample_heatmap, tmp_path):
        # Настраиваем моки
        mock_resize.return_value = sample_heatmap
        mock_apply.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        
        save_path = GradCAM.save_heatmap(sample_heatmap, save_dir=str(tmp_path))
        
        assert save_path is not None
        assert str(tmp_path) in save_path
        assert save_path.endswith('.png')
        mock_makedirs.assert_called_once()
        mock_resize.assert_called_once()
        mock_apply.assert_called_once()