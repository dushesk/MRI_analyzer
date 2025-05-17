from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from fastapi_cache.decorator import cache
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
import hashlib
import logging
import json
from app.services.analysis_pipeline import AnalysisPipeline
from app.schemas.predictions import PredictionResult, ClassificationResult
from app.schemas.dicom import DicomExportData
from app.core.exceptions import (
    MRIAnalysisError,
    InvalidImageError,
    ModelProcessingError,
    ImageSizeError,
    CacheError
)
from fastapi.responses import FileResponse
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
        if not file.content_type == 'image/jpeg':
            raise InvalidImageError("Загруженный файл должен быть в формате JPG")

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
            logger.info(f"File info - name: {file.filename}, content_type: {file.content_type}")
            
            # Сохраняем временную копию для отладки
            temp_path = f"/tmp/debug_analyze_{file.filename}"
            contents = await file.read()
            with open(temp_path, 'wb') as f:
                f.write(contents)
            logger.info(f"Saved debug copy to {temp_path}")
            
            # Возвращаем указатель в начало файла
            await file.seek(0)
            
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
        if not file.content_type == 'image/jpeg':
            raise InvalidImageError("Загруженный файл должен быть в формате JPG")

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
    export_data: str = Form(...)
):
    """
    Экспортирует изображение в DICOM формат с добавлением метаданных пациента.
    
    Args:
        file: Загруженный файл (JPG)
        export_data: JSON строка с данными пациента и исследования
    """
    temp_file = None
    output_dicom = None
    
    try:
        if not file.content_type == 'image/jpeg':
            raise InvalidImageError("Загруженный файл должен быть в формате JPG")

        # Парсим JSON данные
        export_data_dict = json.loads(export_data)
        export_data_model = DicomExportData(**export_data_dict)

        # Создаем временный файл для загруженного изображения
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        logger.info(f"Created temporary input file at: {temp_file.name}")
        content = await file.read()
        temp_file.write(content)
        temp_file.flush()
        temp_file.close()
        
        # Создаем временный файл для DICOM
        output_dicom = tempfile.mktemp(suffix='.dcm')
        logger.info(f"Will save DICOM file to: {output_dicom}")
        
        # Подготавливаем метаданные
        metadata = {
            'PatientName': export_data_model.patient_name,
            'PatientID': export_data_model.patient_id or '',
            'PatientBirthDate': export_data_model.patient_birth_date or '',
            'PatientSex': export_data_model.patient_sex or '',
            'StudyDate': export_data_model.study_date or datetime.now().strftime('%Y%m%d'),
            'StudyDescription': export_data_model.study_description or 'MRI Analysis',
            'ReferringPhysicianName': export_data_model.referring_physician_name or '',
        }
        
        # Добавляем дополнительные метаданные, если они есть
        if export_data_model.additional_metadata:
            metadata.update(export_data_model.additional_metadata)
        
        # Конвертируем в DICOM
        logger.info("Starting DICOM conversion...")
        dicom_handler.convert_to_dicom(temp_file.name, output_dicom, metadata)
        
        # Проверяем, что файл создался
        if not os.path.exists(output_dicom):
            raise Exception(f"DICOM file was not created at {output_dicom}")
        
        logger.info(f"DICOM file exists at: {output_dicom}")
        logger.info(f"File size: {os.path.getsize(output_dicom)} bytes")
        
        # Возвращаем DICOM файл
        return FileResponse(
            output_dicom,
            media_type='application/dicom',
            filename=f'mri_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.dcm',
            headers={
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )
        
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="Invalid JSON data")
    except Exception as e:
        logger.error(f"Error in export_to_dicom: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

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
    temp_files = []  # Список для отслеживания временных файлов
    try:
        if not file.filename.lower().endswith('.dcm'):
            raise InvalidImageError("Загруженный файл не является DICOM файлом")

        # Создаем временный файл для загруженного DICOM
        temp_dicom = tempfile.NamedTemporaryFile(delete=False, suffix='.dcm')
        temp_files.append(temp_dicom.name)
        content = await file.read()
        temp_dicom.write(content)
        temp_dicom.flush()
        temp_dicom.close()
        
        # Получаем метаданные
        metadata = dicom_handler.get_dicom_metadata(temp_dicom.name)
        
        # Если запрошен формат изображения, конвертируем DICOM в изображение
        if output_format.lower() == "image":
            output_image = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_files.append(output_image.name)
            output_image.close()
            
            dicom_handler.convert_from_dicom(temp_dicom.name, output_image.name)
            
            # Создаем функцию для очистки временных файлов после отправки
            async def cleanup():
                for temp_file in temp_files:
                    try:
                        os.unlink(temp_file)
                    except Exception as e:
                        logger.error(f"Error cleaning up temp file {temp_file}: {str(e)}")
            
            # Возвращаем файл с функцией очистки
            return FileResponse(
                output_image.name,
                media_type='image/jpeg',
                filename='imported_image.jpg',
                headers={
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type'
                },
                background=cleanup
            )
        
        # Возвращаем метаданные в формате JSON
        return {
            "metadata": metadata,
            "message": "DICOM file successfully imported"
        }
        
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in import_from_dicom: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        # Удаляем временные файлы, если они не были переданы в FileResponse
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.error(f"Error cleaning up temp file {temp_file}: {str(e)}")