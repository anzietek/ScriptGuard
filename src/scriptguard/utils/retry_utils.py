"""
Retry utilities with exponential backoff for robust error handling.

This module provides reusable retry decorators and statistics tracking
for handling transient failures in external service calls (Qdrant, APIs).
"""

import time
import functools
from typing import Callable, TypeVar, Tuple, Type, Optional
from scriptguard.utils.logger import logger

T = TypeVar('T')


def retry_with_backoff(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Retry decorator with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        backoff_factor: Multiplier for delay between retries (default: 2.0)
        initial_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        exceptions: Tuple of exception types to catch (default: all)
        on_retry: Callback function(exception, attempt) called on each retry

    Returns:
        Decorated function with retry logic

    Example:
        @retry_with_backoff(max_retries=3, backoff_factor=2)
        def upload_to_qdrant(data):
            client.upsert(data)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt >= max_retries:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} retries: {e}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(
                        initial_delay * (backoff_factor ** attempt),
                        max_delay
                    )

                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    if on_retry:
                        on_retry(e, attempt + 1)

                    time.sleep(delay)

            raise last_exception

        return wrapper
    return decorator


class RetryStats:
    """Track retry statistics for monitoring."""

    def __init__(self):
        self.total_attempts = 0
        self.total_retries = 0
        self.total_failures = 0
        self.operation_stats = {}

    def record_attempt(self, operation: str, success: bool, retry_count: int):
        """Record an operation attempt."""
        self.total_attempts += 1
        self.total_retries += retry_count

        if not success:
            self.total_failures += 1

        if operation not in self.operation_stats:
            self.operation_stats[operation] = {
                "attempts": 0,
                "retries": 0,
                "failures": 0
            }

        stats = self.operation_stats[operation]
        stats["attempts"] += 1
        stats["retries"] += retry_count
        if not success:
            stats["failures"] += 1

    def get_summary(self) -> dict:
        """Get summary statistics."""
        success_rate = (
            (self.total_attempts - self.total_failures) / self.total_attempts * 100
            if self.total_attempts > 0 else 0
        )

        return {
            "total_attempts": self.total_attempts,
            "total_retries": self.total_retries,
            "total_failures": self.total_failures,
            "success_rate": f"{success_rate:.2f}%",
            "by_operation": self.operation_stats
        }

    def reset(self):
        """Reset all statistics."""
        self.total_attempts = 0
        self.total_retries = 0
        self.total_failures = 0
        self.operation_stats = {}
