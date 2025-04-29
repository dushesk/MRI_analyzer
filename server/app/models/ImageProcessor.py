import numpy as np
from tensorflow.keras.preprocessing import image
from app.core.config import settings

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