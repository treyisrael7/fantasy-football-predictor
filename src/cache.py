"""
Redis cache for players, teams, stats, and predictions. Reduces DB load.
"""
import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_redis_client = None
_CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "300"))  # 5 min default


def _get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _redis_client = redis.from_url(url, decode_responses=True)
        _redis_client.ping()
        return _redis_client
    except Exception as e:
        logger.debug("Redis not available: %s", e)
        return None


def get_cached(key: str) -> Optional[Any]:
    """Get JSON value from Redis. Returns None if miss or Redis down."""
    r = _get_redis()
    if not r:
        return None
    try:
        raw = r.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as e:
        logger.debug("Cache get failed for %s: %s", key, e)
        return None


def set_cached(key: str, value: Any, ttl_seconds: int = _CACHE_TTL) -> bool:
    """Store JSON value in Redis. Returns True on success."""
    r = _get_redis()
    if not r:
        return False
    try:
        r.setex(key, ttl_seconds, json.dumps(value, default=str))
        return True
    except Exception as e:
        logger.debug("Cache set failed for %s: %s", key, e)
        return False


def cache_key_players() -> str:
    return "ff:players"
def cache_key_teams() -> str:
    return "ff:teams"
def cache_key_stats(season: int) -> str:
    return f"ff:stats:{season}"
