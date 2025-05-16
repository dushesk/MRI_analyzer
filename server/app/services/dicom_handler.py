import os
from typing import Union, Dict, Any, Optional
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid
from datetime import datetime
import numpy as np
from PIL import Image
import io

class DicomHandler:
    """Класс для работы с DICOM форматом (импорт/экспорт)."""
    
    def __init__(self):
        self.default_metadata = {
            'Modality': 'MR',
            'Manufacturer': 'MRI Classifier',
            'ManufacturerModelName': 'MRI Classifier v1.0',
            'SoftwareVersions': '1.0',
            'StudyDescription': 'MRI Analysis',
            'SeriesDescription': 'MRI Classification',
            'PatientPosition': 'HFS', 
            'StudyInstanceUID': generate_uid(),
            'SeriesInstanceUID': generate_uid(),
            'SOPInstanceUID': generate_uid(),
        }

    def _create_dicom_dataset(self, image_data: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> FileDataset:
        """
        Создает базовый DICOM dataset с изображением и метаданными.
        
        Args:
            image_data: numpy array с изображением
            metadata: дополнительные метаданные для DICOM файла
            
        Returns:
            FileDataset: DICOM dataset
        """
        # Базовый dataset
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
        file_meta.MediaStorageSOPInstanceUID = self.default_metadata['SOPInstanceUID']
        file_meta.ImplementationClassUID = '1.2.3.4.5.6.7.8.9.0'
        
        # Создаем dataset
        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
        
        # Добавляем базовые метаданные
        for key, value in self.default_metadata.items():
            setattr(ds, key, value)
        
        # Добавляем пользовательские метаданные
        if metadata:
            for key, value in metadata.items():
                setattr(ds, key, value)
        
        # Добавляем данные изображения
        ds.Rows, ds.Columns = image_data.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelData = image_data.tobytes()
        
        # Добавляем временные метки
        dt = datetime.now()
        ds.ContentDate = dt.strftime('%Y%m%d')
        ds.ContentTime = dt.strftime('%H%M%S.%f')
        
        return ds

    def convert_to_dicom(self, 
                        image_path: str, 
                        output_path: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Конвертирует изображение в DICOM формат.
        
        Args:
            image_path: путь к исходному изображению
            output_path: путь для сохранения DICOM файла
            metadata: дополнительные метаданные для DICOM файла
            
        Returns:
            str: путь к созданному DICOM файлу
        """
        try:
            # Загружаем изображение
            image = Image.open(image_path)
            image_array = np.array(image)
            
            # Создаем DICOM dataset
            ds = self._create_dicom_dataset(image_array, metadata)
            
            # Сохраняем DICOM файл
            ds.save_as(output_path)
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Ошибка при конвертации в DICOM: {str(e)}")

    def convert_from_dicom(self, 
                          dicom_path: str, 
                          output_path: str, 
                          format: str = 'PNG') -> str:
        """
        Конвертирует DICOM файл в обычное изображение.
        
        Args:
            dicom_path: путь к DICOM файлу
            output_path: путь для сохранения изображения
            format: формат выходного изображения (PNG, JPEG и т.д.)
            
        Returns:
            str: путь к созданному изображению
        """
        try:
            # Загружаем DICOM файл
            ds = pydicom.dcmread(dicom_path)
            
            # Получаем данные изображения
            image_array = ds.pixel_array
            
            # Нормализуем значения пикселей
            image_array = ((image_array - image_array.min()) * 
                          (255.0 / (image_array.max() - image_array.min())))
            image_array = image_array.astype(np.uint8)
            
            # Создаем изображение
            image = Image.fromarray(image_array)
            
            # Сохраняем изображение
            image.save(output_path, format=format)
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Ошибка при конвертации из DICOM: {str(e)}")

    def get_dicom_metadata(self, dicom_path: str) -> Dict[str, Any]:
        """
        Получает метаданные из DICOM файла.
        
        Args:
            dicom_path: путь к DICOM файлу
            
        Returns:
            Dict[str, Any]: словарь с метаданными
        """
        try:
            ds = pydicom.dcmread(dicom_path)
            metadata = {}
            
            # Собираем все доступные метаданные
            for elem in ds:
                if elem.keyword and elem.value:
                    metadata[elem.keyword] = str(elem.value)
            
            return metadata
            
        except Exception as e:
            raise Exception(f"Ошибка при чтении метаданных DICOM: {str(e)}")

    def update_dicom_metadata(self, 
                            dicom_path: str, 
                            metadata: Dict[str, Any]) -> str:
        """
        Обновляет метаданные в DICOM файле.
        
        Args:
            dicom_path: путь к DICOM файлу
            metadata: новые метаданные
            
        Returns:
            str: путь к обновленному DICOM файлу
        """
        try:
            ds = pydicom.dcmread(dicom_path)
            
            # Обновляем метаданные
            for key, value in metadata.items():
                if hasattr(ds, key):
                    setattr(ds, key, value)
            
            # Сохраняем обновленный файл
            ds.save_as(dicom_path)
            
            return dicom_path
            
        except Exception as e:
            raise Exception(f"Ошибка при обновлении метаданных DICOM: {str(e)}") 