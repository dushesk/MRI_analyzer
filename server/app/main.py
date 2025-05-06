from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as api_router
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis

app = FastAPI(
    title="Alzheimer MRI Classifier",
    description="API для классификации МРТ-снимков"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    redis = aioredis.from_url("redis://redis:6379")
    FastAPICache.init(RedisBackend(redis), prefix="alzheimer-cache")

@app.get("/redis_test")
async def test():
    try:
        await FastAPICache.get_backend().redis.set("test", "works", ex=10)
        val = await FastAPICache.get_backend().redis.get("test")
        return {"status": bool(val)}
    except Exception as e:
        return {"error": str(e)}