from pydantic import BaseModel, Field, field_validator
from pydantic import ConfigDict
from typing import Dict, List, Literal, Optional


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enable_cache: bool = False
    cache_steps: Dict[str, bool] = Field(default_factory=dict)
    cache_ttl_hours: int = Field(24, gt=0)
    cache_key_includes_version: bool = True
    cache_invalidation_on_config_change: bool = True


class APIKeysConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    github_token: Optional[str] = None
    vx_github_token: Optional[str] = None
    thezoo_github_token: Optional[str] = None
    nvd_api_key: Optional[str] = None
    malwarebazaar_api_key: Optional[str] = None
    huggingface_token: Optional[str] = None
    scriptguard_api_key: Optional[str] = None
    max_retries: int = Field(3, ge=0)
    retry_backoff_factor: float = Field(2.0, gt=0)
    timeout_seconds: int = Field(30, gt=0)
    connection_pool_size: int = Field(10, gt=0)


class PostgreSQLConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    host: str = "localhost"
    port: int = Field(5432, gt=0, lt=65536)
    database: str = "scriptguard"
    user: str = "scriptguard"
    password: str = "scriptguard"
    min_connections: int = Field(1, ge=1)
    max_connections: int = Field(10, ge=1)
    connection_timeout: int = Field(30, gt=0)
    command_timeout: int = Field(60, gt=0)


class DatabaseConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["postgresql", "sqlite"] = "postgresql"
    postgresql: PostgreSQLConfig = Field(default_factory=PostgreSQLConfig)
    enable_versioning: bool = True
    auto_backup: bool = False


class CodeBERTConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model_name: str = "microsoft/codebert-base"
    max_tokens: int = Field(512, gt=0)
    chunk_overlap: int = Field(50, ge=0)
    batch_size: int = Field(16, gt=0)
    learning_rate: float = Field(2.0e-5, gt=0.0)
    warmup_steps: int = Field(200, ge=0)
    weight_decay: float = Field(0.01, ge=0.0)
    num_epochs: int = Field(5, gt=0)
    eval_threshold: float = Field(0.92, ge=0.0, le=1.0)
    output_dir: str = "/workspace/models/codebert"

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_window(cls, v: int, info: object) -> int:
        return v


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    test_size: float = Field(0.15, gt=0.0, lt=1.0)
    val_size: float = Field(0.15, gt=0.0, lt=1.0)
    seed: int = 42
    early_stopping_patience: int = Field(3, gt=0)


class ValidationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    validate_syntax: bool = True
    skip_syntax_errors: bool = True
    min_length: int = Field(50, ge=0)
    max_length: int = Field(100000, gt=0)
    min_code_lines: int = Field(5, ge=0)
    max_comment_ratio: float = Field(0.5, ge=0.0, le=1.0)
    deduplicate: bool = True
    dedup_threshold: float = Field(0.92, ge=0.0, le=1.0)
    dedup_method: str = "auto"
    dedup_exact_first: bool = True
    dedup_minhash_num_perm: int = Field(128, gt=0)
    filter_test_leakage: bool = True
    allow_python2: bool = True


class AugmentationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    variants_per_sample: int = Field(2, ge=1)
    techniques: List[str] = Field(default_factory=lambda: ["base64", "hex", "rename_vars", "split_strings"])
    balance_dataset: bool = False
    augment_after_split: bool = True


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    level: str = "INFO"
    file: str = "/workspace/logs/scriptguard.log"


class ScriptGuardConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", use_enum_values=True)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    api_keys: APIKeysConfig = Field(default_factory=APIKeysConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    codebert: CodeBERTConfig = Field(default_factory=CodeBERTConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def validate_config(config_dict: dict) -> ScriptGuardConfig:
    return ScriptGuardConfig(**config_dict)
