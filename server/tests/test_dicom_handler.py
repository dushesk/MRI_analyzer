import pytest
from unittest.mock import patch, MagicMock
import pydicom
import numpy as np
from PIL import Image
import os
import tempfile
from app.services.dicom_handler import DicomHandler  

@pytest.fixture
def dicom_handler():
    return DicomHandler()

@pytest.fixture
def sample_image_array():
    return np.random.randint(0, 256, (256, 256), dtype=np.uint16)

@pytest.fixture
def sample_dicom_file(tmp_path):
    handler = DicomHandler()
    img_array = np.random.randint(0, 256, (256, 256), dtype=np.uint16)
    ds = handler._create_dicom_dataset(img_array)
    file_path = os.path.join(tmp_path, "test.dcm")
    ds.save_as(file_path)
    return file_path

class TestDicomHandler:
    def test_create_dicom_dataset_valid(self, dicom_handler, sample_image_array):
        ds = dicom_handler._create_dicom_dataset(sample_image_array)
        
        assert isinstance(ds, pydicom.dataset.FileDataset)
        assert ds.Rows == 256
        assert ds.Columns == 256
        assert ds.SamplesPerPixel == 1
        assert ds.PhotometricInterpretation == "MONOCHROME2"
        assert ds.BitsAllocated == 16

    def test_create_dicom_dataset_invalid_input(self, dicom_handler):
        with pytest.raises(ValueError):
            dicom_handler._create_dicom_dataset("not_an_array")
            
        with pytest.raises(ValueError):
            dicom_handler._create_dicom_dataset(np.zeros((256, 256, 3)))

    def test_create_dicom_dataset_conversion(self, dicom_handler):
        float_array = np.random.rand(256, 256).astype(np.float32)
        ds = dicom_handler._create_dicom_dataset(float_array)
        assert ds.pixel_array.dtype == np.uint16

    def test_convert_from_dicom_success(self, dicom_handler, sample_dicom_file, tmp_path):
        output_path = os.path.join(tmp_path, "output.jpg")
        result = dicom_handler.convert_from_dicom(sample_dicom_file, output_path)
        
        assert os.path.exists(result)
        assert Image.open(result)  # Проверяем, что это валидное изображение

    def test_validate_dicom_valid(self, dicom_handler, sample_dicom_file):
        assert dicom_handler.validate_dicom(sample_dicom_file)

    def test_validate_dicom_invalid(self, dicom_handler, tmp_path):
        invalid_path = os.path.join(tmp_path, "invalid.dcm")
        with open(invalid_path, 'w') as f:
            f.write("Not a DICOM file")
            
        assert not dicom_handler.validate_dicom(invalid_path)

    def test_get_dicom_metadata(self, dicom_handler, sample_dicom_file):
        metadata = dicom_handler.get_dicom_metadata(sample_dicom_file)
        
        assert isinstance(metadata, dict)
        assert 'PatientName' in metadata
        assert 'Modality' in metadata
        assert metadata['Modality'] == 'MR'

    def test_update_dicom_metadata(self, dicom_handler, sample_dicom_file):
        new_metadata = {
            'PatientName': 'NewName',
            'StudyDescription': 'NewDescription'
        }
        
        result = dicom_handler.update_dicom_metadata(sample_dicom_file, new_metadata)
        
        assert os.path.exists(result)
        updated_metadata = dicom_handler.get_dicom_metadata(result)
        assert updated_metadata['PatientName'] == 'NewName'
        assert updated_metadata['StudyDescription'] == 'NewDescription'

    @patch('pydicom.dcmread')
    def test_get_metadata_error_handling(self, mock_dcmread, dicom_handler):
        mock_dcmread.side_effect = Exception("Test error")
        with pytest.raises(RuntimeError):
            dicom_handler.get_dicom_metadata("any_path.dcm")

    @patch('pydicom.dcmread')
    def test_update_metadata_error_handling(self, mock_dcmread, dicom_handler):
        mock_dcmread.side_effect = Exception("Test error")
        with pytest.raises(RuntimeError):
            dicom_handler.update_dicom_metadata("any_path.dcm", {})