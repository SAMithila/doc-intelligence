"""Structured logging configuration for production observability."""
import logging
import sys
import time
from functools import wraps
from typing import Any, Callable


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure structured logging."""
    logger = logging.getLogger("docint")
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '{"time": "%(asctime)s", "level": "%(levelname)s", '
            '"module": "%(module)s", "message": "%(message)s"}'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def log_latency(operation: str):
    """Decorator to log operation latency."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = logging.getLogger("docint")
            start = time.time()
            try:
                result = func(*args, **kwargs)
                latency_ms = (time.time() - start) * 1000
                logger.info(f'{operation} completed latency_ms={latency_ms:.0f}')
                return result
            except Exception as e:
                latency_ms = (time.time() - start) * 1000
                logger.error(f'{operation} failed latency_ms={latency_ms:.0f} error={str(e)}')
                raise
        return wrapper
    return decorator


# Global logger instance
logger = setup_logging()
