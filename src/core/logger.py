import logging
import os
import sys
from logging.handlers import RotatingFileHandler
import functools

# Ensure log directory exists
os.makedirs("logs", exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.
    Writes to logs/system.log and console.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File Handler (5 MB per file, max 3 backups)
        file_handler = RotatingFileHandler(
            "logs/system.log", maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def log_exception(logger):
    """Decorator to log exceptions automaticallly with traceback."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Exception in {func.__name__}: {str(e)}", exc_info=True)
                raise e
        return wrapper
    return decorator
