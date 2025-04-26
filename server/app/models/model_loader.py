from tensorflow.keras.models import Model
from app.core.config import settings
from .model import build_cnn_model
import logging
import os

_model = None

def get_model():
    global _model
    if _model is None:
        try:
            # Строим архитектуру модели
            _model = build_cnn_model()
            
            # Проверяем существование файла весов
            if not os.path.exists(settings.MODEL_PATH):
                raise FileNotFoundError(f"Weights file not found at {settings.MODEL_PATH}")
            
            # Загрузка весов
            _model.load_weights(settings.MODEL_PATH)
            logging.info("Веса модели успешно загружены!")
            
        except Exception as e:
            logging.error(f"Ошибка загрузки весов модели: {str(e)}")
            raise
    return _model