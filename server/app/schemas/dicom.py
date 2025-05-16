from pydantic import BaseModel
from typing import Optional

class DicomExportData(BaseModel):
    """Схема данных для экспорта DICOM файла"""
    patient_name: str
    patient_id: Optional[str] = None
    patient_birth_date: Optional[str] = None
    patient_sex: Optional[str] = None
    study_date: Optional[str] = None
    study_description: Optional[str] = None
    referring_physician_name: Optional[str] = None
    additional_metadata: Optional[dict] = None 