import logging
import sys
from src.cores import config


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        ))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
    return logger
