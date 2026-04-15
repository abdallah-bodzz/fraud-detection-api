"""
utils.py
--------
Logging configuration using loguru.
One import everywhere: from src.utils import logger
"""

import sys
from loguru import logger
from src.config import LOG_LEVEL

# Remove default handler, add structured one
logger.remove()
logger.add(
    sys.stdout,
    level=LOG_LEVEL,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    ),
    colorize=True,
)

# Also write to file for monitoring (rotates at 10 MB, keeps 7 days)
logger.add(
    "logs/api.log",
    level="INFO",
    rotation="10 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} — {message}",
)
