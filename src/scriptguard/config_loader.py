"""
Configuration loader for ScriptGuard.
Handles YAML loading, environment variable substitution, and schema validation.
"""

import os
import yaml
from typing import Dict, Any, Optional
from scriptguard.schemas.config_schema import validate_config, ScriptGuardConfig
from scriptguard.utils.logger import logger

def load_raw_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file and substitute environment variables.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration with environment variables substituted
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    def convert_type(value: str):
        """Convert string value to appropriate type (int, float, bool, or str)."""
        if not isinstance(value, str):
            return value

        # Try to convert to boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # Try to convert to int
        try:
            return int(value)
        except ValueError:
            pass

        # Try to convert to float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def substitute_env_vars(obj):
        """Recursively substitute environment variables in config."""
        if isinstance(obj, dict):
            return {k: substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            # Parse ${ENV_VAR:-default}
            env_expr = obj[2:-1]
            if ":-" in env_expr:
                env_var, default = env_expr.split(":-", 1)
                result = os.getenv(env_var, default)
            else:
                env_var = env_expr
                result = os.getenv(env_var, "")
            return convert_type(result)
        else:
            return obj

    return substitute_env_vars(config)

def load_config(config_path: str = "config.yaml") -> ScriptGuardConfig:
    """
    Load, substitute, and validate configuration.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Validated ScriptGuardConfig object
    """
    try:
        raw_config = load_raw_config(config_path)
        validated_config = validate_config(raw_config)
        return validated_config
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise
