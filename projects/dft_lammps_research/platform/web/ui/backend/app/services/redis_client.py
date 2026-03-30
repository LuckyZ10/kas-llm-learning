"""
Redis Client - Pub/Sub and caching
"""
import json
from typing import Optional, Any
import aioredis
import structlog

from app.core.config import settings

logger = structlog.get_logger()

redis_client: Optional[aioredis.Redis] = None


async def init_redis():
    """Initialize Redis connection"""
    global redis_client
    
    try:
        redis_client = aioredis.from_url(
            settings.REDIS_URL,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,
        )
        await redis_client.ping()
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Redis", error=str(e))
        redis_client = None


async def close_redis():
    """Close Redis connection"""
    global redis_client
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")


def get_redis() -> Optional[aioredis.Redis]:
    """Get Redis client instance"""
    return redis_client


async def cache_set(key: str, value: Any, expire: int = 300) -> bool:
    """Set cache value"""
    if not redis_client:
        return False
    
    try:
        serialized = json.dumps(value) if not isinstance(value, (str, bytes)) else value
        await redis_client.setex(key, expire, serialized)
        return True
    except Exception as e:
        logger.error(f"Cache set failed", key=key, error=str(e))
        return False


async def cache_get(key: str) -> Optional[Any]:
    """Get cache value"""
    if not redis_client:
        return None
    
    try:
        value = await redis_client.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return None
    except Exception as e:
        logger.error(f"Cache get failed", key=key, error=str(e))
        return None


async def cache_delete(key: str) -> bool:
    """Delete cache key"""
    if not redis_client:
        return False
    
    try:
        await redis_client.delete(key)
        return True
    except Exception as e:
        logger.error(f"Cache delete failed", key=key, error=str(e))
        return False


async def publish_message(channel: str, message: dict) -> bool:
    """Publish message to Redis channel"""
    if not redis_client:
        return False
    
    try:
        await redis_client.publish(channel, json.dumps(message))
        return True
    except Exception as e:
        logger.error(f"Publish failed", channel=channel, error=str(e))
        return False
