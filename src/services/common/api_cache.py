import os
import json
import time
import hashlib
import functools
import threading
from typing import Dict, Any, Optional, Callable

from src.core.logger import get_logger

logger = get_logger(__name__)

class TokenBucketRateLimiter:
    """简单的令牌桶限流器"""
    def __init__(self, capacity: int, fill_rate: float):
        """
        :param capacity: 桶的总体积（最大爆发量）
        :param fill_rate: 每秒补充的令牌数量
        """
        self.capacity = capacity
        self.fill_rate = fill_rate
        self.tokens = capacity
        self.last_fill_time = time.time()
        self.lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        """尝试消费指定数量的令牌，如果足够返回 True，否则返回 False"""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_fill_time
            # 补充令牌
            self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
            self.last_fill_time = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
            
    def wait_and_consume(self, tokens: int = 1):
        """阻塞直到可以消费到指定数量的令牌"""
        while True:
            if self.consume(tokens):
                return
            time.sleep(0.1)


class APICacheManager:
    """
    进程安全的简易磁盘 API 缓存管理器。
    """
    def __init__(self, cache_dir: str = "data/cache/api", default_ttl_seconds: int = 7 * 24 * 3600):
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl_seconds
        os.makedirs(self.cache_dir, exist_ok=True)
        # 单独设置一个 Materials Project 的速率限制：例如 100 req / minute -> rate=1.6
        self.rate_limiters = {
            "materials_project": TokenBucketRateLimiter(capacity=100, fill_rate=1.5),
            "semantic_scholar": TokenBucketRateLimiter(capacity=10, fill_rate=1.0) # S2 API typically 1 req/s
        }

    def _get_cache_path(self, key_string: str) -> str:
        key_hash = hashlib.md5(key_string.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.json")

    def get(self, key: str) -> Optional[Any]:
        path = self._get_cache_path(key)
        if not os.path.exists(path):
            return None
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Check TTL
            timestamp = data.get("_timestamp", 0)
            ttl = data.get("_ttl", self.default_ttl)
            
            if time.time() - timestamp > ttl:
                # Expired
                try:
                    os.remove(path)
                except Exception:
                    pass
                return None
                
            return data.get("payload")
        except Exception as e:
            logger.warning(f"Cache read error for key {key}: {e}")
            return None

    def set(self, key: str, payload: Any, ttl_seconds: Optional[int] = None):
        path = self._get_cache_path(key)
        data = {
            "_timestamp": time.time(),
            "_ttl": ttl_seconds if ttl_seconds is not None else self.default_ttl,
            "payload": payload
        }
        try:
            # Atomic-like write
            tmp_path = path + ".tmp"
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            os.replace(tmp_path, path)
        except Exception as e:
            logger.warning(f"Cache write error for key {key}: {e}")

# 单例模式
_api_cache_manager = APICacheManager()

def get_api_cache() -> APICacheManager:
    return _api_cache_manager

def with_cache(namespace: str, ttl_seconds: Optional[int] = None, limiter_key: Optional[str] = None):
    """
    装饰器：自动对同步 API 客户端方法进行缓存和限流。
    要求方法的输入和输出均能够 JSON 序列化。
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_mgr = get_api_cache()
            
            # Serialize arguments to generate a unique cache key
            # Skip 'self' if it's a method
            args_for_key = args[1:] if args and hasattr(args[0], func.__name__) else args
            key_dict = {
                "namespace": namespace,
                "func": func.__name__,
                "args": args_for_key,
                "kwargs": kwargs
            }
            try:
                key_str = json.dumps(key_dict, sort_keys=True)
            except TypeError:
                # Fallback if arguments are not fully serializable
                key_str = f"{namespace}:{func.__name__}:{args_for_key}:{kwargs}"
                
            cached_result = cache_mgr.get(key_str)
            if cached_result is not None:
                logger.debug(f"[Cache HITT] {namespace} / {func.__name__}")
                return cached_result
                
            # If not cached, enforce rate limit if specified
            if limiter_key and limiter_key in cache_mgr.rate_limiters:
                logger.debug(f"Applying rate limit constraint: {limiter_key}")
                cache_mgr.rate_limiters[limiter_key].wait_and_consume(1)
                
            # Execute actual function
            logger.debug(f"[Cache MISS] {namespace} / {func.__name__}")
            result = func(*args, **kwargs)
            
            # Cache the result
            if result is not None:
                cache_mgr.set(key_str, result, ttl_seconds)
                
            return result
        return wrapper
    return decorator
