from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ClassificationResult(BaseModel):
    """Результат классификации МРТ"""
    class_name: str
    confidence: float
    class_id: int
    probabilities: Dict[str, float]

class InterpretationResult(BaseModel):
    """Результат интерпретации МРТ"""
    findings: List[str]
    recommendations: List[str]
    severity: str
    additional_info: Optional[Dict[str, Any]] = None

class PredictionResult(BaseModel):
    """Полный результат анализа МРТ"""
    classification: ClassificationResult
    interpretation: InterpretationResult
    processing_time: float
    model_version: str