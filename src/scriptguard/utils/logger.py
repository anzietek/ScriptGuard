"""
Loguru configuration for ScriptGuard.
Centralized logging setup with file rotation, colorized output, and structured logging.
"""

import sys
from pathlib import Path
from loguru import logger

# Remove default handler
logger.remove()

# Console handler with colorized output
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# File handler with rotation
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger.add(
    log_dir / "scriptguard_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="100 MB",
    retention="30 days",
    compression="zip",
    enqueue=True,  # Thread-safe
)

# Error log file
logger.add(
    log_dir / "errors_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
    level="ERROR",
    rotation="50 MB",
    retention="90 days",
    compression="zip",
    backtrace=True,
    diagnose=True,
    enqueue=True,
)

# JSON log for structured logging (optional, for production)
logger.add(
    log_dir / "scriptguard_{time:YYYY-MM-DD}.json",
    format="{message}",
    level="INFO",
    rotation="100 MB",
    retention="30 days",
    compression="zip",
    serialize=True,  # JSON format
    enqueue=True,
)


def get_logger(name: str = None):
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured loguru logger
    """
    if name:
        return logger.bind(name=name)
    return logger


__all__ = ["logger", "get_logger"]
