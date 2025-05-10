from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi_cache.decorator import cache
import hashlib
import logging
from app.services.analysis_pipeline import AnalysisPipeline
from app.schemas.predictions import PredictionResult, ClassificationResult, InterpretationResult
from app.core.exceptions import (
    MRIAnalysisError,
    InvalidImageError,
    ModelProcessingError,
    ImageSizeError,
    CacheError
)

router = APIRouter()
logger = logging.getLogger(__name__)

def get_file_hash(file: UploadFile) -> str:
    """Получение хэша файла для кэширования"""
    file_content = file.file.read()
    file.file.seek(0)
    return "mri:" + hashlib.md5(file_content).hexdigest()

@router.post("/analyze", response_model=PredictionResult)
@cache(expire=3600, key_builder=lambda *args, **kwargs: get_file_hash(kwargs["file"]))
async def analyze_mri(file: UploadFile = File(...)):
    """Полный анализ МРТ (классификация + интерпретация)"""
    try:
        if not file.content_type.startswith('image/'):
            raise InvalidImageError("Загруженный файл не является изображением")

        try:
            result = await AnalysisPipeline.process_image(file)
            return result
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
@cache(expire=3600, key_builder=lambda *args, **kwargs: get_file_hash(kwargs["file"]))
async def classify_mri(file: UploadFile = File(...)):
    """Только классификация МРТ без интерпретации"""
    try:
        if not file.content_type.startswith('image/'):
            raise InvalidImageError("Загруженный файл не является изображением")

        try:
            result = await AnalysisPipeline.classify_image(file)
            return result
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

@router.post("/interpret", response_model=InterpretationResult)
@cache(expire=3600, key_builder=lambda *args, **kwargs: get_file_hash(kwargs["file"]))
async def interpret_mri(file: UploadFile = File(...)):
    """Только интерпретация МРТ без классификации"""
    try:
        if not file.content_type.startswith('image/'):
            raise InvalidImageError("Загруженный файл не является изображением")

        try:
            result = await AnalysisPipeline.interpret_image(file)
            return result
        except ValueError as ve:
            if "size" in str(ve).lower():
                raise ImageSizeError(str(ve))
            raise InvalidImageError(str(ve))
        except Exception as e:
            logger.error(f"Ошибка при интерпретации: {str(e)}", exc_info=True)
            raise ModelProcessingError(f"Ошибка при интерпретации: {str(e)}")

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