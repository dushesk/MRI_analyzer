import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image
from app.core.exceptions import ModelProcessingError

class LIMExplainer:
    """Класс для работы с LIME объяснениями"""
    
    def __init__(self, model, num_samples=1000):
        self.explainer = lime_image.LimeImageExplainer()
        self.model = model
        self.num_samples = num_samples
    
    def explain(self, image_array):
        """Генерация объяснения для одного изображения"""
        if image_array is None:
            raise ValueError("Input image array cannot be None")
            
        # Проверка и нормализация входных данных
        if image_array.max() > 1.0:
            image_array = image_array / 255.0
            
        def predict_fn(images):
            try:
                # Подготовка входных данных для модели
                if images.ndim == 3:
                    images = np.expand_dims(images, axis=0)
                
                # Получаем предсказания от модели
                predictions = self.model.predict(images)
                
                # Нормализуем форму предсказаний
                if len(predictions.shape) == 1:
                    predictions = np.expand_dims(predictions, axis=0)
                
                # Если предсказания имеют форму (1, n_classes), повторяем их для каждого изображения
                if predictions.shape[0] == 1 and len(images) > 1:
                    predictions = np.tile(predictions, (len(images), 1))
                
                return predictions
            except Exception as e:
                raise ModelProcessingError(f"Error in model prediction: {str(e)}")
        
        return self.explainer.explain_instance(
            image_array.astype('double'),
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=self.num_samples,
            batch_size=10
        )
    
    @staticmethod
    def get_visualization(explanation, positive_only=True, num_features=5):
        """Визуализация объяснения"""
        if explanation is None:
            raise ValueError("Explanation cannot be None")
            
        if num_features <= 0:
            raise ValueError("Number of features must be positive")
            
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=positive_only,
            num_features=num_features,
            hide_rest=False
        )
        return (mark_boundaries(temp, mask) * 255).astype(np.uint8)
    
    @staticmethod
    def explanation_to_image(explanation_array):
        """Конвертация numpy array в PIL Image"""
        return Image.fromarray(explanation_array)