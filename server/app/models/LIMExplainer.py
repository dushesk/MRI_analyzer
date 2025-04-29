import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image

class LIMExplainer:
    """Класс для работы с LIME объяснениями"""
    
    def __init__(self, model, num_samples=1000):
        self.explainer = lime_image.LimeImageExplainer()
        self.model = model
        self.num_samples = num_samples
    
    def explain(self, image_array):
        """Генерация объяснения для одного изображения"""
        # Проверка и нормализация входных данных
        if image_array.max() > 1.0:
            image_array = image_array / 255.0
            
        def predict_fn(images):
            # Подготовка входных данных для модели
            if images.ndim == 3:
                images = np.expand_dims(images, axis=0)
            return self.model.predict(images)
        
        return self.explainer.explain_instance(
            image_array.astype('double'),
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=self.num_samples
        )
    
    @staticmethod
    def get_visualization(explanation, positive_only=True, num_features=5):
        """Визуализация объяснения"""
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=positive_only,
            num_features=num_features,
            hide_rest=False
        )
        return mark_boundaries(temp, mask)
    
    @staticmethod
    def explanation_to_image(explanation_array):
        """Конвертация numpy array в PIL Image"""
        return Image.fromarray((explanation_array * 255).astype(np.uint8))