from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io
import numpy as np
import base64
from app.models.predictor import ImageProcessor, AlzheimerPredictor
from app.models.gradcam import GradCAMVisualizer
from app.schemas.predictions import PredictionResult
from app.models.model_loader import get_model
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/predict", response_model=PredictionResult)
async def predict_mri(file: UploadFile = File(...)):
    """Эндпоинт для классификации без Grad-CAM"""
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        img_array = ImageProcessor.preprocess(img)
        predictions = AlzheimerPredictor.predict(img_array)
        
        return {
            "filename": file.filename,
            "predictions": predictions,
            "predicted_class": AlzheimerPredictor.get_class_name(predictions),
            "confidence": float(np.max(predictions))
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze", response_model=PredictionResult)
async def predict_mri_with_grad(file: UploadFile = File(...)):
    """Эндпоинт для классификации с Grad-CAM"""
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        # Подготовка данных
        img_array = ImageProcessor.preprocess(img)
        
        # Получение модели и предсказаний
        model = get_model()
        predictions = AlzheimerPredictor.predict(img_array)
        
        # Генерация и сохранение Grad-CAM heatmap
        heatmap = GradCAMVisualizer.generate_heatmap(model, img_array)
        heatmap_img = GradCAMVisualizer.prepare_heatmap_image(heatmap)
        GradCAMVisualizer.save_heatmap(heatmap)
        
        # Конвертация heatmap в base64
        buffered = io.BytesIO()
        heatmap_img.save(buffered, format="PNG")
        heatmap_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {
            "filename": file.filename,
            "predictions": predictions,
            "predicted_class": AlzheimerPredictor.get_class_name(predictions),
            "confidence": float(np.max(predictions)),
            "heatmap_img": heatmap_base64  # Возвращаем heatmap как base64 строку
        }
        
    except Exception as e:
        logger.error(f"GradCAM prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))