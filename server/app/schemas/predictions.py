from pydantic import BaseModel
from typing import List, Optional

class PredictionResult(BaseModel):
    filename: str
    predictions: List[float]
    predicted_class: str
    confidence: float
    heatmap_img: str
    lime_img: str
    explanations: dict