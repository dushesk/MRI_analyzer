from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi_cache.decorator import cache
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
import hashlib
import logging
import json
from app.services.analysis_pipeline import AnalysisPipeline
from app.schemas.predictions import PredictionResult, ClassificationResult, InterpretationResult
from app.schemas.dicom import DicomExportData
from app.core.exceptions import (
    MRIAnalysisError,
    InvalidImageError,
    ModelProcessingError,
    ImageSizeError,
    CacheError
)
from fastapi.responses import FileResponse
from typing import Optional
import os
import tempfile
from app.services.dicom_handler import DicomHandler
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
dicom_handler = DicomHandler()

def get_file_hash(file: UploadFile) -> str:
    """Получение хэша файла для кэширования"""
    try:
        # Читаем первые 1024 байта для хэширования
        chunk = file.file.read(1024)
        file.file.seek(0)  # Возвращаем указатель в начало
        cache_key = f"mri:{hashlib.md5(chunk).hexdigest()}"
        logger.info(f"Generated cache key: {cache_key}")
        return cache_key
    except Exception as e:
        logger.error(f"Error generating file hash: {str(e)}")
        raise CacheError(f"Failed to generate cache key: {str(e)}")

@router.post("/analyze", response_model=PredictionResult)
async def analyze_mri(file: UploadFile = File(...)):
    """Полный анализ МРТ (классификация + интерпретация)"""
    try:
        if not file.content_type.startswith('image/'):
            raise InvalidImageError("Загруженный файл не является изображением")

        try:
            # Проверяем кэш перед обработкой
            cache_key = get_file_hash(file)
            logger.info(f"Checking cache for key: {cache_key}")
            
            # Пытаемся получить результат из кэша
            backend = FastAPICache.get_backend()
            cached_result = await backend.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for key: {cache_key}")
                cached_data = json.loads(cached_result)
                # Если в кэше только классификация, делаем полный анализ
                if "classification" not in cached_data:
                    logger.info("Cache contains only classification, performing full analysis...")
                    result = await AnalysisPipeline.process_image(file)
                    result_dict = dict(result)
                    await backend.set(cache_key, json.dumps(result_dict), expire=3600)
                    return result_dict
                return cached_data
            
            logger.info(f"Cache miss for key: {cache_key}, processing image...")
            result = await AnalysisPipeline.process_image(file)
            
            # Преобразуем результат в словарь для корректной сериализации
            result_dict = dict(result)
            
            # Сохраняем в кэш
            await backend.set(cache_key, json.dumps(result_dict), expire=3600)
            logger.info(f"Result cached for key: {cache_key}")
            
            return result_dict
            
        except ValueError as ve:
            if "size" in str(ve).lower():
                raise ImageSizeError(str(ve))
            raise InvalidImageError(str(ve))
        except Exception as e:
            logger.error(f"Ошибка при обработке изображения: {str(e)}", exc_info=True)
            raise ModelProcessingError(f"Ошибка при обработке изображения: {str(e)}")

    except CacheError as ce:
        logger.error(f"Ошибка кэширования: {str(ce)}", exc_info=True)
        raise
    except MRIAnalysisError:
        raise
    except Exception as e:
        logger.error(f"Непредвиденная ошибка: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Произошла непредвиденная ошибка: {str(e)}"
        )

@router.post("/classify", response_model=ClassificationResult)
async def classify_mri(file: UploadFile = File(...)):
    """Только классификация МРТ без интерпретации"""
    try:
        if not file.content_type.startswith('image/'):
            raise InvalidImageError("Загруженный файл не является изображением")

        try:
            # Проверяем кэш перед обработкой
            cache_key = get_file_hash(file)
            logger.info(f"Checking cache for key: {cache_key}")
            
            # Пытаемся получить результат из кэша
            backend = FastAPICache.get_backend()
            cached_result = await backend.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for key: {cache_key}")
                cached_data = json.loads(cached_result)
                # Если в кэше полный анализ, берем только часть с классификацией
                if "classification" in cached_data:
                    return cached_data["classification"]
                return cached_data
            
            logger.info(f"Cache miss for key: {cache_key}, processing image...")
            result = await AnalysisPipeline.classify_image(file)
            
            # Преобразуем результат в словарь для корректной сериализации
            result_dict = {
                "class_name": result["class_name"],
                "confidence": result["confidence"],
                "class_id": result["class_id"],
                "probabilities": result["probabilities"]
            }
            
            # Сохраняем в кэш
            await backend.set(cache_key, json.dumps(result_dict), expire=3600)
            logger.info(f"Result cached for key: {cache_key}")
            
            return result_dict
            
        except ValueError as ve:
            if "size" in str(ve).lower():
                raise ImageSizeError(str(ve))
            raise InvalidImageError(str(ve))
        except Exception as e:
            logger.error(f"Ошибка при классификации: {str(e)}", exc_info=True)
            raise ModelProcessingError(f"Ошибка при классификации: {str(e)}")

    except CacheError as ce:
        logger.error(f"Ошибка кэширования: {str(ce)}", exc_info=True)
        raise
    except MRIAnalysisError:
        raise
    except Exception as e:
        logger.error(f"Непредвиденная ошибка: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Произошла непредвиденная ошибка: {str(e)}"
        )

@router.post("/export/dicom")
async def export_to_dicom(
    file: UploadFile = File(...),
    export_data: DicomExportData = None
):
    """
    Экспортирует изображение в DICOM формат с добавлением метаданных пациента.
    
    Args:
        file: Загруженный файл (JPG, PNG)
        export_data: Данные пациента и исследования
    """
    try:
        if not file.content_type.startswith('image/'):
            raise InvalidImageError("Загруженный файл не является изображением")

        # Создаем временный файл для загруженного изображения
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Создаем временный файл для DICOM
            output_dicom = tempfile.mktemp(suffix='.dcm')
            
            # Подготавливаем метаданные
            metadata = {
                'PatientName': export_data.patient_name,
                'PatientID': export_data.patient_id or '',
                'PatientBirthDate': export_data.patient_birth_date or '',
                'PatientSex': export_data.patient_sex or '',
                'StudyDate': export_data.study_date or datetime.now().strftime('%Y%m%d'),
                'StudyDescription': export_data.study_description or 'MRI Analysis',
                'ReferringPhysicianName': export_data.referring_physician_name or '',
            }
            
            # Добавляем дополнительные метаданные, если они есть
            if export_data.additional_metadata:
                metadata.update(export_data.additional_metadata)
            
            # Конвертируем в DICOM
            dicom_handler.convert_to_dicom(temp_file.name, output_dicom, metadata)
            
            # Возвращаем DICOM файл
            return FileResponse(
                output_dicom,
                media_type='application/dicom',
                filename=f'mri_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.dcm'
            )
            
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        # Удаляем временные файлы
        if 'temp_file' in locals():
            os.unlink(temp_file.name)
        if 'output_dicom' in locals() and os.path.exists(output_dicom):
            os.unlink(output_dicom)

@router.post("/import/dicom")
async def import_from_dicom(
    file: UploadFile = File(...),
    output_format: str = "json"
):
    """
    Импортирует DICOM файл и возвращает изображение и метаданные.
    
    Args:
        file: Загруженный DICOM файл
        output_format: Формат вывода (json или image)
    """
    try:
        if not file.filename.lower().endswith('.dcm'):
            raise InvalidImageError("Загруженный файл не является DICOM файлом")

        # Создаем временный файл для загруженного DICOM
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Получаем метаданные
            metadata = dicom_handler.get_dicom_metadata(temp_file.name)
            
            # Если запрошен формат изображения, конвертируем DICOM в изображение
            if output_format.lower() == "image":
                output_image = tempfile.mktemp(suffix='.png')
                dicom_handler.convert_from_dicom(temp_file.name, output_image)
                
                return FileResponse(
                    output_image,
                    media_type='image/png',
                    filename='imported_image.png'
                )
            
            # Возвращаем метаданные в формате JSON
            return {
                "metadata": metadata,
                "message": "DICOM file successfully imported"
            }
            
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        # Удаляем временные файлы
        if 'temp_file' in locals():
            os.unlink(temp_file.name)
        if 'output_image' in locals() and os.path.exists(output_image):
            os.unlink(output_image)