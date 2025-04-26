from pydantic import BaseModel
from typing import List, Optional

class PredictionResult(BaseModel):
    """Схема ответа API"""
    filename: str
    predictions: List[float]
    predicted_class: str
    confidence: float
    heatmap_img: Optional[str] = None