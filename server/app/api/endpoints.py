from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi_cache.decorator import cache
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from PIL import Image
import io
import numpy as np
import base64
import hashlib
import logging
from typing import Dict, Any
from app.models.AlzheimerPredictor import AlzheimerPredictor
from app.models.ImageProcessor import ImageProcessor
from app.models.GradCAM import GradCAM
from app.models.LIMExplainer import LIMExplainer
from app.schemas.predictions import PredictionResult
from app.models.model_loader import get_model
import json

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/analyze", response_model=PredictionResult)
@cache(
    expire=3600,
    key_builder=lambda *args, **kwargs: 
        "mri:" + hashlib.md5(kwargs["file"].file.read()).hexdigest()
)
async def analyze_mri(file: UploadFile = File(...)):
    """Эндпоинт для классификации с Grad-CAM и LIME"""
    try:
        file_content = await file.read()
        cache_key = FastAPICache.get_prefix() + "mri:" + hashlib.md5(file_content).hexdigest()
        print(f"Ключ кэша: {cache_key}")  # Логирование в консоль

        test_data = json.dumps({"test": "value"})
        await FastAPICache.get_backend().set(cache_key, test_data, expire=10)
        print(f"Тестовые данные сохранены. Ключ: {cache_key}")

        # Сброс позиции файла после генерации ключа
        file.file.seek(0)
        result = await AnalysisPipeline.process_image(file)
        
        # Логирование для отладки
        cache_key = FastAPICache.get_prefix() + "mri:" + hashlib.md5(await file.read()).hexdigest()
        logger.info(f"Кэшируем данные с ключом: {cache_key}")
        file.file.seek(0)
        
        return result
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class AnalysisPipeline:
    @staticmethod
    async def process_image(file: UploadFile) -> Dict[str, Any]:
        """Обработка изображения и генерация результатов"""
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
        
        # Формирование ответа
        def image_to_base64(img):
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        response = {
            "filename": file.filename,
            "predictions": [float(x) for x in predictions[0]],
            "predicted_class": predicted_class,
            "confidence": confidence,
            "heatmap_img": image_to_base64(heatmap_img),
            "explanations": {
                "lime": {
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
            response["lime_img"] = image_to_base64(lime_img)
            
        return response