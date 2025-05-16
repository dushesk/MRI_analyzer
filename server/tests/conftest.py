import pytest
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

# Добавляем корневую директорию проекта в PYTHONPATH
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

# Настройка тестового окружения
@pytest.fixture(autouse=True)
def setup_test_environment():
    # Устанавливаем переменные окружения для тестов
    os.environ["TESTING"] = "1"
    os.environ["MODEL_PATH"] = str(Path(project_root) / "models" / "test_model.h5")
    os.environ["REDIS_URL"] = "redis://localhost:6479/0"
    
    # Создаем временную директорию для тестовых файлов
    test_files_dir = Path(project_root) / "tests" / "test_files"
    test_files_dir.mkdir(exist_ok=True)
    
    # Мокаем Redis для тестов
    with patch('fastapi_cache.FastAPICache.init') as mock_cache:
        mock_cache.return_value = None
        yield
    
    # Очистка после тестов
    for file in test_files_dir.glob("*"):
        file.unlink()
    test_files_dir.rmdir()

# Общие фикстуры для всех тестов
@pytest.fixture
def test_files_dir():
    return Path(project_root) / "tests" / "test_files"

@pytest.fixture
def mock_redis():
    with patch('fastapi_cache.backends.redis.RedisBackend') as mock:
        mock.get.return_value = None
        mock.set.return_value = True
        yield mock

@pytest.fixture
def mock_model():
    model = AsyncMock()
    model.predict.return_value = [[0.25, 0.25, 0.25, 0.25]]
    return model

@pytest.fixture
def mock_get_model(mock_model):
    with patch('app.services.analysis_pipeline.get_model', return_value=mock_model) as mock:
        yield mock 