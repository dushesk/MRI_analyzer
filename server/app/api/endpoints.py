from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io
import numpy as np
import base64
from app.models.AlzheimerPredictor import AlzheimerPredictor
from app.models.ImageProcessor import ImageProcessor
from app.models.GradCAM import GradCAM
from app.models.LIMExplainer import LIMExplainer
from app.schemas.predictions import PredictionResult
from app.models.model_loader import get_model
from typing import Dict, Any
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/analyze", response_model=PredictionResult)
async def analyze_mri(file: UploadFile = File(...)):
    """Эндпоинт для классификации с Grad-CAM и LIME"""
    try:
        return await AnalysisPipeline.process_image(file)        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
class AnalysisPipeline:
    """Централизованный обработчик анализа изображений"""
    @staticmethod
    async def process_image(file: UploadFile) -> Dict[str, Any]:
        # Загрузка и проверка изображения
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        # Предобработка изображения
        img_array = ImageProcessor.preprocess(img)
        logger.debug(f"Image shape after preprocessing: {img_array.shape}")
        
        # Получение модели
        model = get_model()
        
        # Основное предсказание
        predictions = model.predict(img_array)
        confidence = float(np.max(predictions))
        predicted_class = AlzheimerPredictor.get_class_name(predictions)
        
        # Grad-CAM
        heatmap = GradCAM.generate_heatmap(model, img_array)
        heatmap_img = GradCAM.prepare_heatmap_image(heatmap)
        
        # LIME объяснение
        lime_explainer = LIMExplainer(model)
        lime_explanation = lime_explainer.explain(img_array[0])  # Берем первое изображение из батча
        
        # Подготовка результатов
        def image_to_base64(img):
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Конвертируем все NumPy типы в Python-native типы
        predictions_list = [float(x) for x in predictions[0]]
        
        # Формируем LIME объяснения
        lime_features = []
        for feature in lime_explanation.local_exp[lime_explanation.top_labels[0]][:5]:
            lime_features.append({
                "feature": int(feature[0]),  # Конвертируем в int
                "weight": float(feature[1])  # Конвертируем в float
            })
        
        # Формируем ответ
        response = {
            "filename": file.filename,
            "predictions": predictions_list,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "heatmap_img": image_to_base64(heatmap_img),
            "explanations": {
                "lime": {
                    "top_features": lime_features
                }
            }
        }
        
        # Для LIME изображения 
        if hasattr(lime_explainer, 'get_visualization'):
            lime_img = lime_explainer.get_visualization(lime_explanation)
            lime_img_pil = lime_explainer.explanation_to_image(lime_img)
            response["lime_img"] = image_to_base64(lime_img_pil)
        
        return response