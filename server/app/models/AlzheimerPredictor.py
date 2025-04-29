import numpy as np
from app.models.model_loader import get_model
  
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