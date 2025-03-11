"""
装饰器模块
"""
import time
from functools import wraps
from loguru import logger

def log_function_call(func):
    """函数调用日志装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        if elapsed_time > 0.1:
            logger.info(f"Completed {func.__name__} in {elapsed_time:.2f}s")
        
        return result
    return wrapper

def cache_data(func):
    """数据缓存装饰器"""
    cache = {}
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper 