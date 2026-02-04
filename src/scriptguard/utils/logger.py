"""
Loguru configuration for ScriptGuard.
Centralized logging setup with file rotation, colorized output, and structured logging.
Includes secret redaction (P2.2 fix).
"""

import sys
import re
from pathlib import Path
from loguru import logger


# Secret Redaction Filter (P2.2 fix)
class SecretRedactionFilter:
    """Filter to redact secrets from log messages."""

    # Patterns for common secrets
    SECRET_PATTERNS = [
        # API tokens
        (r'(hf_[a-zA-Z0-9]{20,})', r'hf_***REDACTED***'),
        (r'(github_pat_[a-zA-Z0-9_]{22,})', r'github_pat_***REDACTED***'),
        (r'(ghp_[a-zA-Z0-9]{36,})', r'ghp_***REDACTED***'),

        # Environment variable values that look like secrets
        (r'(["\']?)([A-Z_]*TOKEN|[A-Z_]*API_KEY|[A-Z_]*SECRET|[A-Z_]*PASSWORD)\1\s*[:=]\s*["\']?([^"\'\s]{20,})["\']?',
         r'\1\2\1: ***REDACTED***'),

        # Generic long alphanumeric strings that might be keys (conservative)
        (r'(api[_-]?key|access[_-]?token|secret[_-]?key|password)\s*[:=]\s*["\']?([a-zA-Z0-9+/]{32,})["\']?',
         r'\1: ***REDACTED***', re.IGNORECASE),

        # Bearer tokens
        (r'(Bearer\s+)([a-zA-Z0-9_\-\.]{20,})', r'\1***REDACTED***', re.IGNORECASE),
    ]

    @staticmethod
    def redact(message: str) -> str:
        """Redact secrets from log message."""
        redacted = message
        for pattern_info in SecretRedactionFilter.SECRET_PATTERNS:
            if len(pattern_info) == 2:
                pattern, replacement = pattern_info
                flags = 0
            else:
                pattern, replacement, flags = pattern_info

            redacted = re.sub(pattern, replacement, redacted, flags=flags)

        return redacted

    def __call__(self, record):
        """Called by loguru for each log record."""
        record["message"] = self.redact(record["message"])
        return True


# Create filter instance
secret_filter = SecretRedactionFilter()

# Remove default handler
logger.remove()

# Console handler with colorized output
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
    filter=secret_filter  # P2.2: Redact secrets
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
    filter=secret_filter  # P2.2: Redact secrets
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
    filter=secret_filter  # P2.2: Redact secrets
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
    filter=secret_filter  # P2.2: Redact secrets
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
