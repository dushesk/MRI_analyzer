from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as api_router
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Alzheimer MRI Classifier",
    description="API для классификации МРТ-снимков"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем все источники в режиме разработки
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

@app.get("/")
def health_check():
    return {"status": "OK"}

@app.on_event("startup")
async def startup():
    try:
        logger.info("Initializing Redis connection...")
        redis = aioredis.from_url(
            "redis://redis:6379",
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=5,
            retry_on_timeout=True
        )
        
        # Проверяем соединение
        await redis.ping()
        logger.info("Redis connection successful")
        
        # Инициализируем кэш
        FastAPICache.init(
            RedisBackend(redis),
            prefix="alzheimer-cache",
            expire=3600,  # Время жизни кэша по умолчанию
        )
        logger.info("FastAPI cache initialized successfully")
        
        # Проверяем работу кэша
        await redis.set("test_key", "test_value", ex=10)
        test_value = await redis.get("test_key")
        logger.info(f"Cache test successful: {test_value}")
        
    except Exception as e:
        logger.error(f"Failed to initialize Redis cache: {str(e)}", exc_info=True)
        raise

@app.get("/redis_test")
async def test():
    try:
        redis = FastAPICache.get_backend().redis
        logger.info("Testing Redis connection...")
        
        # Проверяем соединение
        await redis.ping()
        logger.info("Redis ping successful")
        
        # Проверяем запись
        await redis.set("test", "works", ex=10)
        logger.info("Test value written to Redis")
        
        # Проверяем чтение
        val = await redis.get("test")
        logger.info(f"Test value read from Redis: {val}")
        
        # Проверяем все ключи
        keys = await redis.keys("*")
        logger.info(f"All Redis keys: {keys}")
        
        return {"status": bool(val), "keys": keys}
    except Exception as e:
        logger.error(f"Redis test failed: {str(e)}", exc_info=True)
        return {"error": str(e)}