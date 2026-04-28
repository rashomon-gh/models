import sys

from loguru import logger


def setup_logging():
    logger.remove()
    logger.add(
        sys.stdout, 
        colorize=False, 
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    
    return logger
