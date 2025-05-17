import os
from typing import Dict, Any, Optional
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid
from datetime import datetime
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

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
            'PatientName': 'Anonymous',
            'PatientID': '000000',
            'PatientSex': 'O',
            'PatientBirthDate': '',
            'StudyInstanceUID': generate_uid(),
            'SeriesInstanceUID': generate_uid(),
            'SOPInstanceUID': generate_uid(),
            'PixelSpacing': [1.0, 1.0],
            'SliceThickness': 1.0,
        }

    def _create_dicom_dataset(self, image_data: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> FileDataset:
        """
        Создает корректный DICOM dataset с изображением и метаданными.
        
        Args:
            image_data: numpy array с изображением (должен быть uint16)
            metadata: дополнительные метаданные для DICOM файла
            
        Returns:
            FileDataset: Валидный DICOM dataset
            
        Raises:
            ValueError: Если входные данные некорректны
            RuntimeError: При ошибках создания DICOM
        """
        try:
            logger.info("Создание DICOM dataset...")
            
            # Проверка входных данных
            if not isinstance(image_data, np.ndarray):
                raise ValueError("Изображение должно быть массивом numpy")
                
            if len(image_data.shape) != 2:
                raise ValueError("Изображение должно быть 2D массивом")

            # Конвертация в uint16 если нужно
            if image_data.dtype != np.uint16:
                logger.warning("Конвертация изображения в 16-битный формат...")
                image_data = ((image_data - image_data.min()) * 
                            (65535.0 / (image_data.max() - image_data.min()))).astype(np.uint16)
            
            # Создание метаинформации
            file_meta = Dataset()
            file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'  # MR Image Storage
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
            
            # Основной dataset
            ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
            
            # Обязательные теги DICOM
            required_tags = {
                'SOPClassUID': file_meta.MediaStorageSOPClassUID,
                'SOPInstanceUID': file_meta.MediaStorageSOPInstanceUID,
                'StudyDate': datetime.now().strftime('%Y%m%d'),
                'SeriesDate': datetime.now().strftime('%Y%m%d'),
                'ContentDate': datetime.now().strftime('%Y%m%d'),
                'StudyTime': datetime.now().strftime('%H%M%S.%f'),
                'SeriesTime': datetime.now().strftime('%H%M%S.%f'),
                'ContentTime': datetime.now().strftime('%H%M%S.%f'),
                'AccessionNumber': '',
                'StudyID': '1',
                'SeriesNumber': 1,
                'InstanceNumber': 1,
                'ImagePositionPatient': ['0', '0', '0'],
                'ImageOrientationPatient': ['1', '0', '0', '0', '1', '0']
            }
            
            for tag, value in required_tags.items():
                setattr(ds, tag, value)
            
            # Добавление базовых метаданных
            for key, value in self.default_metadata.items():
                if not hasattr(ds, key):
                    setattr(ds, key, value)
            
            # Пользовательские метаданные
            if metadata:
                for key, value in metadata.items():
                    if value is not None:
                        setattr(ds, key, value)
            
            # Параметры изображения
            ds.Rows, ds.Columns = image_data.shape
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.PixelData = image_data.tobytes()
            
            logger.info("DICOM dataset успешно создан")
            return ds
            
        except ValueError as ve:
            logger.error(f"Ошибка входных данных: {str(ve)}")
            raise ValueError(f"Некорректные входные данные: {str(ve)}")
        except Exception as e:
            logger.error(f"Ошибка создания DICOM: {str(e)}", exc_info=True)
            raise RuntimeError(f"Не удалось создать DICOM dataset: {str(e)}")

    def convert_to_dicom(self, 
                       image_path: str, 
                       output_path: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Конвертирует изображение в DICOM формат.
        
        Args:
            image_path: путь к исходному изображению
            output_path: путь для сохранения DICOM файла
            metadata: дополнительные метаданные
            
        Returns:
            str: Путь к созданному DICOM файлу
            
        Raises:
            FileNotFoundError: Если файл не найден
            RuntimeError: При ошибках конвертации
        """
        try:
            logger.info(f"Конвертация {image_path} в DICOM...")
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Файл не найден: {image_path}")
            
            # Загрузка изображения
            try:
                img = Image.open(image_path).convert('L')  # В grayscale
                img_array = np.array(img)
            except Exception as e:
                raise ValueError(f"Невозможно загрузить изображение: {str(e)}")
            
            # Создание DICOM
            ds = self._create_dicom_dataset(img_array, metadata)
            
            # Сохранение
            try:
                ds.save_as(output_path, write_like_original=False)
            except Exception as e:
                raise IOError(f"Ошибка сохранения DICOM: {str(e)}")
            
            if not os.path.exists(output_path):
                raise RuntimeError("DICOM файл не был создан")
                
            logger.info(f"DICOM успешно сохранен в {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Ошибка конвертации: {str(e)}", exc_info=True)
            raise RuntimeError(f"Не удалось конвертировать в DICOM: {str(e)}")

    def convert_from_dicom(self, dicom_path: str, output_path: str, format: str = 'JPEG') -> str:
        """
        Конвертирует DICOM в обычное изображение.
        
        Args:
            dicom_path: путь к DICOM файлу
            output_path: путь для сохранения изображения
            format: формат выходного изображения
            
        Returns:
            str: Путь к созданному изображению
            
        Raises:
            ValueError: Если файл не является валидным DICOM
            RuntimeError: При ошибках конвертации
        """
        try:
            # Проверяем, что файл является DICOM
            if not pydicom.misc.is_dicom(dicom_path):
                raise ValueError("Файл не является валидным DICOM файлом")

            # Читаем DICOM файл
            ds = pydicom.dcmread(dicom_path)
            
            # Проверяем наличие пиксельных данных
            if not hasattr(ds, 'pixel_array'):
                raise ValueError("DICOM файл не содержит пиксельных данных")

            # Получаем пиксельные данные
            pixel_array = ds.pixel_array
            
            # Проверяем тип данных и диапазон значений
            if pixel_array.dtype != np.uint8:
                # Нормализуем значения в диапазон 0-255
                if pixel_array.max() != pixel_array.min():
                    pixel_array = ((pixel_array - pixel_array.min()) * 
                                (255.0 / (pixel_array.max() - pixel_array.min()))).astype(np.uint8)
                else:
                    pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
            
            # Создаем изображение
            image = Image.fromarray(pixel_array)
            
            # Конвертируем в RGB если нужно
            if format.upper() == 'JPEG':
                if image.mode == 'L':
                    image = image.convert('RGB')
                elif image.mode not in ['RGB', 'RGBA']:
                    image = image.convert('RGB')
            
            # Сохраняем с максимальным качеством
            image.save(output_path, format=format, quality=95)
            
            logger.info(f"Изображение успешно сохранено в {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Ошибка конвертации DICOM: {str(e)}", exc_info=True)
            raise RuntimeError(f"Не удалось конвертировать DICOM: {str(e)}")

    def validate_dicom(self, filepath: str) -> bool:
        """Проверяет валидность DICOM файла."""
        try:
            pydicom.dcmread(filepath)
            return True
        except Exception as e:
            logger.warning(f"Невалидный DICOM файл: {str(e)}")
            return False

    def get_dicom_metadata(self, dicom_path: str) -> Dict[str, Any]:
        """Возвращает метаданные DICOM файла в сериализуемом формате."""
        try:
            ds = pydicom.dcmread(dicom_path)
            metadata = {}
            
            for elem in ds:
                if elem.keyword:
                    # Преобразуем значения в сериализуемые типы
                    if elem.VR == "SQ":  # Для последовательностей
                        metadata[elem.keyword] = [
                            {item_elem.keyword: str(item_elem.value) 
                            for item_elem in item if item_elem.keyword}
                            for item in elem
                        ]
                    else:
                        # Для обычных значений
                        if elem.value is None:
                            metadata[elem.keyword] = None
                        elif isinstance(elem.value, (str, int, float, bool)):
                            metadata[elem.keyword] = elem.value
                        elif isinstance(elem.value, bytes):
                            continue  # Пропускаем бинарные данные
                        else:
                            metadata[elem.keyword] = str(elem.value)
            
            # Добавляем основные поля, если их нет
            basic_fields = {
                'PatientName': str(ds.get('PatientName', '')),
                'PatientID': str(ds.get('PatientID', '')),
                'StudyDate': str(ds.get('StudyDate', '')),
                'Modality': str(ds.get('Modality', ''))
            }
            
            metadata.update({k: v for k, v in basic_fields.items() if k not in metadata})
            
            return metadata

        except Exception as e:
            logger.error(f"Ошибка чтения метаданных: {str(e)}", exc_info=True)
            raise RuntimeError(f"Не удалось прочитать метаданные: {str(e)}")

    def update_dicom_metadata(self, 
                            dicom_path: str, 
                            metadata: Dict[str, Any]) -> str:
        """Обновляет метаданные DICOM файла."""
        try:
            ds = pydicom.dcmread(dicom_path)
            
            for key, value in metadata.items():
                if hasattr(ds, key):
                    setattr(ds, key, value)
                else:
                    logger.warning(f"Тег не найден: {key}")
            
            ds.save_as(dicom_path)
            return dicom_path
            
        except Exception as e:
            logger.error(f"Ошибка обновления метаданных: {str(e)}")
            raise RuntimeError(f"Не удалось обновить метаданные: {str(e)}")