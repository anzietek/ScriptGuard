"""
Dynamic step caching utility.
Allows controlling ZenML step cache through configuration.
"""

from zenml import step
from functools import wraps
from typing import Callable, Any


def cacheable_step(step_name: str = None):
    """
    Decorator that wraps ZenML @step with configurable caching.

    Usage:
        @cacheable_step("advanced_data_ingestion")
        def advanced_data_ingestion(config: dict):
            # Check cache setting from config
            enable_cache = get_cache_setting(config, "advanced_data_ingestion")
            # ... rest of function

    Args:
        step_name: Name of the step for cache lookup in config

    Returns:
        Decorated function with configurable cache
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract config from args/kwargs
            config = None
            if args and isinstance(args[0], dict):
                config = args[0]
            elif 'config' in kwargs:
                config = kwargs['config']

            # Determine cache setting
            enable_cache = get_cache_setting(config, step_name or func.__name__)

            # Apply ZenML step decorator with cache setting
            zen_step = step(enable_cache=enable_cache)
            decorated_func = zen_step(func)

            return decorated_func(*args, **kwargs)

        return wrapper
    return decorator


def get_cache_setting(config: dict, step_name: str) -> bool:
    """
    Get cache setting for a specific step from config.

    Priority:
    1. config['pipeline']['cache_steps'][step_name] (specific step)
    2. config['pipeline']['enable_cache'] (global setting)
    3. True (default ZenML behavior)

    Args:
        config: Configuration dictionary
        step_name: Name of the step

    Returns:
        Whether to enable cache for this step
    """
    if not config:
        return True

    pipeline_config = config.get('pipeline', {})

    # Check step-specific cache setting
    cache_steps = pipeline_config.get('cache_steps', {})
    if step_name in cache_steps:
        return cache_steps[step_name]

    # Check global cache setting
    if 'enable_cache' in pipeline_config:
        return pipeline_config['enable_cache']

    # Default: enabled
    return True


def apply_cache_to_step(func: Callable, config: dict, step_name: str = None):
    """
    Apply cache setting to a ZenML step function.

    Args:
        func: The step function to wrap
        config: Configuration dictionary
        step_name: Name of the step (defaults to function name)

    Returns:
        ZenML step with applied cache setting
    """
    if step_name is None:
        step_name = func.__name__

    enable_cache = get_cache_setting(config, step_name)
    return step(enable_cache=enable_cache)(func)


__all__ = ["cacheable_step", "get_cache_setting", "apply_cache_to_step"]
