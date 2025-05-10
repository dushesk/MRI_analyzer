import io
import base64
import numpy as np
import time
from typing import Dict, Any
from fastapi import UploadFile
from PIL import Image
from app.models.AlzheimerPredictor import AlzheimerPredictor
from app.models.ImageProcessor import ImageProcessor
from app.models.GradCAM import GradCAM
from app.models.LIMExplainer import LIMExplainer
from app.models.model_loader import get_model


class AnalysisPipeline:
    @staticmethod
    async def process_image(file: UploadFile) -> Dict[str, Any]:
        """Основной метод обработки изображения
        
        Args:
            file: UploadFile - загруженный файл изображения
            
        Returns:
            Dict[str, Any]: Результаты анализа с предсказаниями и визуализациями
        """
        start_time = time.time()
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        # Предобработка
        img_array = ImageProcessor.preprocess(img)
        model = get_model()
        
        # Предсказание
        predictions = model.predict(img_array)
        confidence = float(np.max(predictions))
        predicted_class = AlzheimerPredictor.get_class_name(predictions)
        class_id = np.argmax(predictions)
        
        # Grad-CAM
        heatmap = GradCAM.generate_heatmap(model, img_array)
        heatmap_img = GradCAM.prepare_heatmap_image(heatmap)
        
        # LIME
        lime_explainer = LIMExplainer(model)
        lime_explanation = lime_explainer.explain(img_array[0])
        
        # Формирование ответа в новом формате
        response = {
            "classification": {
                "class_name": predicted_class,
                "confidence": confidence,
                "class_id": int(class_id),
                "probabilities": {
                    "MildDemented": float(predictions[0][0]),
                    "ModerateDemented": float(predictions[0][1]),
                    "NonDemented": float(predictions[0][2]),
                    "VeryMildDemented": float(predictions[0][3])
                }
            },
            "interpretation": {
                "findings": [
                    f"Обнаружена {predicted_class} степень деменции",
                    f"Уверенность модели: {confidence:.2%}"
                ],
                "recommendations": [
                    "Рекомендуется консультация невролога",
                    "Провести дополнительные исследования"
                ],
                "severity": "moderate" if confidence > 0.8 else "low",
                "additional_info": {
                    "heatmap_img": AnalysisPipeline._image_to_base64(heatmap_img),
                    "lime_explanation": {
                        "top_features": [
                            {"feature": int(f[0]), "weight": float(f[1])} 
                            for f in lime_explanation.local_exp[lime_explanation.top_labels[0]][:5]
                        ]
                    }
                }
            },
            "processing_time": time.time() - start_time,
            "model_version": "1.0.0"  # TODO: Добавить версионирование модели
        }
        
        if hasattr(lime_explainer, 'get_visualization'):
            lime_img = lime_explainer.explanation_to_image(
                lime_explainer.get_visualization(lime_explanation)
            )
            response["interpretation"]["additional_info"]["lime_img"] = AnalysisPipeline._image_to_base64(lime_img)
            
        return response

    @staticmethod
    async def classify_image(file: UploadFile) -> Dict[str, Any]:
        """Только классификация изображения
        
        Args:
            file: UploadFile - загруженный файл изображения
            
        Returns:
            Dict[str, Any]: Результаты классификации
        """
        start_time = time.time()
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        # Предобработка
        img_array = ImageProcessor.preprocess(img)
        model = get_model()
        
        # Предсказание
        predictions = model.predict(img_array)
        confidence = float(np.max(predictions))
        predicted_class = AlzheimerPredictor.get_class_name(predictions)
        class_id = np.argmax(predictions)
        
        return {
            "class_name": predicted_class,
            "confidence": confidence,
            "class_id": int(class_id),
            "probabilities": {
                "MildDemented": float(predictions[0][0]),
                "ModerateDemented": float(predictions[0][1]),
                "NonDemented": float(predictions[0][2]),
                "VeryMildDemented": float(predictions[0][3])
            }
        }

    @staticmethod
    async def interpret_image(file: UploadFile) -> Dict[str, Any]:
        """Только интерпретация изображения
        
        Args:
            file: UploadFile - загруженный файл изображения
            
        Returns:
            Dict[str, Any]: Результаты интерпретации
        """
        start_time = time.time()
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        # Предобработка
        img_array = ImageProcessor.preprocess(img)
        model = get_model()
        
        # Предсказание
        predictions = model.predict(img_array)
        confidence = float(np.max(predictions))
        predicted_class = AlzheimerPredictor.get_class_name(predictions)
        
        # Grad-CAM
        heatmap = GradCAM.generate_heatmap(model, img_array)
        heatmap_img = GradCAM.prepare_heatmap_image(heatmap)
        
        # LIME
        lime_explainer = LIMExplainer(model)
        lime_explanation = lime_explainer.explain(img_array[0])
        
        response = {
            "findings": [
                f"Обнаружена {predicted_class} степень деменции",
                f"Уверенность модели: {confidence:.2%}"
            ],
            "recommendations": [
                "Рекомендуется консультация невролога",
                "Провести дополнительные исследования"
            ],
            "severity": "moderate" if confidence > 0.8 else "low",
            "additional_info": {
                "heatmap_img": AnalysisPipeline._image_to_base64(heatmap_img),
                "lime_explanation": {
                    "top_features": [
                        {"feature": int(f[0]), "weight": float(f[1])} 
                        for f in lime_explanation.local_exp[lime_explanation.top_labels[0]][:5]
                    ]
                }
            }
        }
        
        if hasattr(lime_explainer, 'get_visualization'):
            lime_img = lime_explainer.explanation_to_image(
                lime_explainer.get_visualization(lime_explanation)
            )
            response["additional_info"]["lime_img"] = AnalysisPipeline._image_to_base64(lime_img)
            
        return response

    @staticmethod
    def _image_to_base64(img: Image.Image) -> str:
        """Конвертирует PIL Image в base64 строку
        
        Args:
            img: Image.Image - изображение для конвертации
            
        Returns:
            str: base64-encoded строка
        """
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')