import numpy as np
from tensorflow.keras.preprocessing import image
from app.core.config import settings
from .model_loader import get_model
from PIL import Image
import cv2

class ImageProcessor:
    """Обработка изображений для модели"""
    
    @staticmethod
    def preprocess(img):
        """Подготовка изображения для модели"""
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize(settings.IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return img_array
    
    @staticmethod
    def prepare_for_display(img):
        """Подготовка изображения для визуализации"""
        img = img.resize(settings.IMAGE_SIZE)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

class AlzheimerPredictor:
    """Классификатор болезни Альцгеймера"""
    
    @staticmethod
    def predict(img_array):
        """Выполнение предсказания"""
        model = get_model()
        return model.predict(img_array)[0].tolist()
    
    @staticmethod
    def get_class_name(predictions):
        """Получение имени предсказанного класса"""
        classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
        return classes[np.argmax(predictions)]