from fastapi import HTTPException
from typing import Optional

class MRIAnalysisError(HTTPException):
    """Базовый класс для ошибок анализа МРТ"""
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: Optional[str] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code

class InvalidImageError(MRIAnalysisError):
    """Ошибка при обработке изображения"""
    def __init__(self, detail: str = "Некорректный формат или повреждение изображения"):
        super().__init__(
            status_code=400,
            detail=detail,
            error_code="INVALID_IMAGE"
        )

class ModelProcessingError(MRIAnalysisError):
    """Ошибка при обработке модели машинного обучения"""
    def __init__(self, detail: str = "Ошибка при обработке модели"):
        super().__init__(
            status_code=500,
            detail=detail,
            error_code="MODEL_ERROR"
        )

class ImageSizeError(MRIAnalysisError):
    """Ошибка при несоответствии размера изображения"""
    def __init__(self, detail: str = "Некорректный размер изображения"):
        super().__init__(
            status_code=400,
            detail=detail,
            error_code="INVALID_SIZE"
        )

class CacheError(MRIAnalysisError):
    """Ошибка при работе с кэшем"""
    def __init__(self, detail: str = "Ошибка при работе с кэшем"):
        super().__init__(
            status_code=500,
            detail=detail,
            error_code="CACHE_ERROR"
        ) 