import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import AsyncClient
from io import BytesIO
from PIL import Image
from unittest.mock import patch, AsyncMock, MagicMock
from app.main import app
from app.api.endpoints import router
from app.schemas.predictions import ClassificationResult, InterpretationResult, PredictionResult
from app.core.exceptions import InvalidImageError, ImageSizeError, ModelProcessingError, CacheError
import numpy as np
import io

app = FastAPI()

@pytest_asyncio.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
def sample_image():
    img = Image.new('RGB', (224, 224), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return {'file': ('test.png', img_byte_arr, 'image/png')}

@pytest.fixture
def invalid_image():
    return {'file': ('test.txt', b'not an image', 'text/plain')}

@pytest.fixture
def mock_cache():
    with patch('fastapi_cache.FastAPICache.get_backend') as mock:
        backend = MagicMock()
        mock.return_value = backend
        yield backend

class TestEndpoints:
    @pytest.mark.asyncio
    async def test_analyze_endpoint_valid(self, client, sample_image):
        with patch('app.services.analysis_pipeline.AnalysisPipeline.process_image') as mock_process:
            mock_process.return_value = {
                'classification': {'class': 'normal', 'confidence': 0.95},
                'interpretation': {'findings': 'No abnormalities detected'},
                'recommendations': 'No further action required',
                'severity': 'low'
            }
            response = await client.post("/api/analyze", files=sample_image)
            assert response.status_code == 200
            data = response.json()
            assert 'classification' in data
            assert 'interpretation' in data
            assert 'recommendations' in data
            assert 'severity' in data

    @pytest.mark.asyncio
    async def test_analyze_endpoint_invalid(self, client, invalid_image):
        response = await client.post("/api/analyze", files=invalid_image)
        assert response.status_code == 400
        data = response.json()
        assert 'detail' in data
        assert 'Invalid image format' in data['detail']

    @pytest.mark.asyncio
    async def test_classify_endpoint_valid(self, client, sample_image):
        with patch('app.services.analysis_pipeline.AnalysisPipeline.classify_image') as mock_classify:
            mock_classify.return_value = {'class': 'normal', 'confidence': 0.95}
            response = await client.post("/api/classify", files=sample_image)
            assert response.status_code == 200
            data = response.json()
            assert 'class' in data
            assert 'confidence' in data

    @pytest.mark.asyncio
    async def test_interpret_endpoint_valid(self, client, sample_image):
        with patch('app.services.analysis_pipeline.AnalysisPipeline.interpret_image') as mock_interpret:
            mock_interpret.return_value = {
                'findings': 'No abnormalities detected',
                'recommendations': 'No further action required',
                'severity': 'low'
            }
            response = await client.post("/api/interpret", files=sample_image)
            assert response.status_code == 200
            data = response.json()
            assert 'findings' in data
            assert 'recommendations' in data
            assert 'severity' in data

    @pytest.mark.asyncio
    async def test_cache_behavior(self, client, sample_image):
        with patch('app.services.analysis_pipeline.AnalysisPipeline.process_image') as mock_process:
            mock_process.return_value = {
                'classification': {'class': 'normal', 'confidence': 0.95},
                'interpretation': {'findings': 'No abnormalities detected'},
                'recommendations': 'No further action required',
                'severity': 'low'
            }
            # First request
            response1 = await client.post("/api/analyze", files=sample_image)
            assert response1.status_code == 200
            # Second request (should use cache)
            response2 = await client.post("/api/analyze", files=sample_image)
            assert response2.status_code == 200
            assert mock_process.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_error_handling(self, client, sample_image):
        with patch('app.services.analysis_pipeline.AnalysisPipeline.process_image') as mock_process:
            mock_process.side_effect = CacheError("Cache error occurred")
            response = await client.post("/api/analyze", files=sample_image)
            assert response.status_code == 500
            data = response.json()
            assert 'detail' in data
            assert 'Cache error occurred' in data['detail']

    @pytest.mark.asyncio
    async def test_model_error_handling(self, client, sample_image):
        with patch('app.services.analysis_pipeline.AnalysisPipeline.process_image') as mock_process:
            mock_process.side_effect = ModelProcessingError("Model processing error occurred")
            response = await client.post("/api/analyze", files=sample_image)
            assert response.status_code == 500
            data = response.json()
            assert 'detail' in data
            assert 'Model processing error occurred' in data['detail']

    @pytest.mark.asyncio
    async def test_image_size_error_handling(self, client, sample_image):
        with patch('app.services.analysis_pipeline.AnalysisPipeline.process_image') as mock_process:
            mock_process.side_effect = ImageSizeError("Image size error occurred")
            response = await client.post("/api/analyze", files=sample_image)
            assert response.status_code == 400
            data = response.json()
            assert 'detail' in data
            assert 'Image size error occurred' in data['detail']

    @pytest.mark.asyncio
    async def test_invalid_image_error_handling(self, client, invalid_image):
        response = await client.post("/api/analyze", files=invalid_image)
        assert response.status_code == 400
        data = response.json()
        assert 'detail' in data
        assert 'Invalid image format' in data['detail'] 