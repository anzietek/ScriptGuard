"""
Pydantic schema for config.yaml validation.
Prevents typos and validates configuration structure.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Literal
from enum import Enum


class PaddingStrategy(str, Enum):
    """Tokenizer padding strategies"""
    MAX_LENGTH = "max_length"
    DYNAMIC = "dynamic"
    LONGEST = "longest"


class BalanceMethod(str, Enum):
    """Dataset balancing methods"""
    UNDERSAMPLE = "undersample"
    OVERSAMPLE = "oversample"
    HYBRID = "hybrid"


class EvaluationStrategy(str, Enum):
    """Training evaluation strategies"""
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class PipelineConfig(BaseModel):
    """Pipeline caching configuration"""
    enable_cache: bool = False
    cache_steps: Dict[str, bool] = Field(default_factory=dict)
    cache_ttl_hours: int = Field(24, gt=0)
    cache_key_includes_version: bool = True
    cache_invalidation_on_config_change: bool = True


class APIKeysConfig(BaseModel):
    """API keys configuration"""
    github_token: Optional[str] = None
    nvd_api_key: Optional[str] = None
    malwarebazaar_api_key: Optional[str] = None
    huggingface_token: Optional[str] = None
    scriptguard_api_key: Optional[str] = None
    
    # Network resilience
    max_retries: int = Field(3, ge=0)
    retry_backoff_factor: int = Field(2, ge=1)
    timeout_seconds: int = Field(30, gt=0)
    connection_pool_size: int = Field(10, gt=0)


class GitHubSourceConfig(BaseModel):
    """GitHub data source configuration"""
    enabled: bool = True
    fetch_malicious: bool = True
    fetch_benign: bool = True
    malicious_keywords: List[str] = Field(default_factory=list)
    benign_repos: List[str] = Field(default_factory=list)
    max_samples_per_keyword: int = Field(20, gt=0)
    max_files_per_repo: int = Field(50, gt=0)
    timeout: int = Field(30, gt=0)
    max_retries: int = Field(3, ge=0)


class MalwareBazaarConfig(BaseModel):
    """MalwareBazaar data source configuration"""
    enabled: bool = True
    tags: List[str] = Field(default_factory=list)
    max_samples: int = Field(100, gt=0)
    timeout: int = Field(60, gt=0)
    max_retries: int = Field(3, ge=0)


class HuggingFaceConfig(BaseModel):
    """HuggingFace datasets configuration"""
    enabled: bool = True
    datasets: List[str] = Field(default_factory=list)
    max_samples: int = Field(10000, gt=0)
    timeout: int = Field(120, gt=0)
    max_retries: int = Field(3, ge=0)


class CVEFeedsConfig(BaseModel):
    """CVE feeds configuration"""
    enabled: bool = True
    days_back: int = Field(30, gt=0)
    keywords: List[str] = Field(default_factory=list)
    timeout: int = Field(45, gt=0)
    max_retries: int = Field(3, ge=0)

class AdditionalHFConfig(BaseModel):
    """Additional HuggingFace datasets configuration"""
    enabled: bool = True
    timeout: int = Field(120, gt=0)
    max_retries: int = Field(3, ge=0)
    max_samples_per_dataset: int = Field(200, gt=0)
    malware_datasets: List[str] = Field(default_factory=list)
    classification_datasets: List[str] = Field(default_factory=list)
    url_datasets: List[str] = Field(default_factory=list)


class DataSourcesConfig(BaseModel):
    """Data sources configuration"""
    github: GitHubSourceConfig = Field(default_factory=GitHubSourceConfig)
    malwarebazaar: MalwareBazaarConfig = Field(default_factory=MalwareBazaarConfig)
    huggingface: HuggingFaceConfig = Field(default_factory=HuggingFaceConfig)
    cve_feeds: CVEFeedsConfig = Field(default_factory=CVEFeedsConfig)
    vxunderground: Optional[Dict[str, Any]] = None # Placeholder for now
    thezoo: Optional[Dict[str, Any]] = None # Placeholder for now
    additional_hf: AdditionalHFConfig = Field(default_factory=AdditionalHFConfig)


class PostgreSQLConfig(BaseModel):
    """PostgreSQL database configuration"""
    host: str = "localhost"
    port: int = Field(5432, gt=0, lt=65536)
    database: str = "scriptguard"
    user: str = "scriptguard"
    password: str = "scriptguard"
    min_connections: int = Field(1, ge=1)
    max_connections: int = Field(10, ge=1)
    connection_timeout: int = Field(30, gt=0)
    command_timeout: int = Field(60, gt=0)


class SQLiteConfig(BaseModel):
    """SQLite database configuration"""
    path: str = "./data/scriptguard.db"


class DatabaseConfig(BaseModel):
    """Database configuration"""
    type: Literal["postgresql", "sqlite"] = "postgresql"
    postgresql: PostgreSQLConfig = Field(default_factory=PostgreSQLConfig)
    sqlite: SQLiteConfig = Field(default_factory=SQLiteConfig)
    enable_versioning: bool = True
    auto_backup: bool = False


class QdrantConfig(BaseModel):
    """Qdrant vector database configuration"""
    host: str = "localhost"
    port: int = Field(6333, gt=0, lt=65536)
    collection_name: str = "malware_knowledge"
    embedding_model: str = "all-MiniLM-L6-v2"
    api_key: Optional[str] = None
    use_https: bool = False
    timeout: int = Field(30, gt=0)
    grpc_port: int = Field(6334, gt=0, lt=65536)
    prefer_grpc: bool = True
    bootstrap_on_startup: bool = False

class FewShotConfig(BaseModel):
    """Few-Shot RAG Configuration"""
    enabled: bool = True
    k: int = Field(3, ge=1)
    balance_labels: bool = True
    score_threshold_mode: str = "default"
    score_threshold: Optional[float] = None
    max_context_length: int = Field(300, gt=0)
    max_code_length: int = Field(500, gt=0)
    aggregate_chunks: bool = True
    enable_reranking: bool = True

class SanitizationConfig(BaseModel):
    """Code Sanitization Configuration"""
    enabled: bool = True
    min_entropy: float = 3.5
    max_line_length: int = 500
    min_valid_lines: int = 3
    max_empty_line_ratio: float = 0.5
    remove_license_headers: bool = True
    strict_mode: bool = False

class ContextInjectionConfig(BaseModel):
    """Context Injection Configuration"""
    enabled: bool = True
    injection_format: str = "structured"

class RerankingConfig(BaseModel):
    """Reranking Configuration"""
    enabled: bool = True
    strategy: str = "hybrid"
    heuristic: Dict[str, Any] = Field(default_factory=dict)
    cross_encoder: Dict[str, Any] = Field(default_factory=dict)

class CodeEmbeddingConfig(BaseModel):
    """Code Embedding Configuration"""
    model: str = "microsoft/unixcoder-base"
    cache_dir: str = "/workspace/cache"
    fewshot: FewShotConfig = Field(default_factory=FewShotConfig)
    pooling_strategy: str = "mean_pooling"
    normalize: bool = True
    enable_chunking: bool = True
    max_code_length: int = Field(512, gt=0)
    chunk_overlap: int = Field(64, ge=0)
    max_samples_to_vectorize: Optional[int] = None
    batch_size: int = Field(256, gt=0)
    sanitization: SanitizationConfig = Field(default_factory=SanitizationConfig)
    context_injection: ContextInjectionConfig = Field(default_factory=ContextInjectionConfig)
    aggregate_chunks: bool = True
    aggregation_strategy: str = "max_score"
    score_thresholds: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    graceful_fallback: Dict[str, Any] = Field(default_factory=dict)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)


class ValidationConfig(BaseModel):
    """Data validation configuration"""
    validate_syntax: bool = True
    skip_syntax_errors: bool = True
    min_length: int = Field(50, ge=0)
    max_length: int = Field(50000, gt=0)
    min_code_lines: int = Field(5, ge=0)
    max_comment_ratio: float = Field(0.5, ge=0.0, le=1.0)
    deduplicate: bool = True
    dedup_threshold: float = Field(0.85, ge=0.0, le=1.0)


class AugmentationConfig(BaseModel):
    """Data augmentation configuration"""
    enabled: bool = True
    variants_per_sample: int = Field(2, ge=1)
    techniques: List[str] = Field(default_factory=lambda: ["base64", "hex", "rename_vars", "split_strings"])
    balance_dataset: bool = True
    target_balance_ratio: float = Field(1.0, gt=0.0)
    balance_method: BalanceMethod = BalanceMethod.UNDERSAMPLE
    use_qdrant_patterns: bool = True
    qdrant_format_style: str = "detailed"
    augment_after_split: bool = True


class FeaturesConfig(BaseModel):
    """Feature extraction configuration"""
    extract_ast: bool = True
    extract_entropy: bool = True
    extract_api_patterns: bool = True
    extract_string_features: bool = True


class TrainingConfig(BaseModel):
    """Training configuration"""
    model_id: str = "bigcode/starcoder2-3b"
    output_dir: str = "./models/scriptguard-model"
    cache_dir: str = "/workspace/cache"
    logging_dir: str = "/workspace/logs/tensorboard"

    # GPU Configuration
    device: str = "cuda"
    max_memory_mb: int = 22000
    gradient_checkpointing: bool = True
    use_flash_attention_2: bool = True
    attn_implementation: str = "flash_attention_2"
    group_by_length: bool = True
    save_total_limit: int = 2
    cleanup_old_checkpoints: bool = True
    save_on_each_node: bool = True
    resume_from_checkpoint: str = "latest"
    load_best_model_at_end: bool = True
    auto_find_batch_size: bool = False
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    eval_default_on_unclear: str = "unknown"
    dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 4
    dataloader_persistent_workers: bool = True
    max_steps: int = -1
    lr_scheduler_type: str = "cosine_with_restarts"
    tf32: bool = True
    seed: int = 42
    report_to: List[str] = Field(default_factory=lambda: ["wandb"])
    run_name: str = "scriptguard-training"
    disable_tqdm: bool = False
    max_grad_norm: float = 1.0

    # QLoRA Configuration
    use_qlora: bool = True
    lora_r: int = Field(16, gt=0)
    lora_alpha: int = Field(32, gt=0)
    lora_dropout: float = Field(0.05, ge=0.0, le=1.0)
    target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])

    # Training Hyperparameters
    batch_size: int = Field(4, gt=0)
    gradient_accumulation_steps: int = Field(4, gt=0)
    num_epochs: int = Field(3, gt=0)
    learning_rate: float = Field(0.0002, gt=0.0)
    weight_decay: float = Field(0.01, ge=0.0)
    warmup_steps: int = Field(100, ge=0)
    max_seq_length: int = Field(2048, gt=0)
    label_smoothing_factor: float = Field(0.0, ge=0.0, le=1.0)

    # Tokenization
    tokenizer_max_length: int = Field(512, gt=0)
    tokenizer_padding: PaddingStrategy = PaddingStrategy.MAX_LENGTH
    tokenizer_truncation: bool = True

    # Optimization
    fp16: bool = False
    bf16: bool = True
    optim: str = "paged_adamw_8bit"

    # Evaluation
    evaluation_strategy: EvaluationStrategy = EvaluationStrategy.NO
    eval_steps: int = Field(100, gt=0)
    save_steps: int = Field(500, gt=0)
    logging_steps: int = Field(10, gt=0)
    test_split_size: float = Field(0.1, gt=0.0, lt=1.0)
    logging_first_step: bool = True

    # Early Stopping
    early_stopping: bool = True
    early_stopping_patience: int = Field(3, gt=0)
    early_stopping_threshold: float = 0.001
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Model Evaluation
    eval_max_new_tokens: int = Field(20, gt=0)
    eval_temperature: float = Field(0.1, gt=0.0, le=2.0)
    eval_batch_size: int = Field(1, gt=0)
    eval_max_code_length: int = Field(500, gt=0)

    @validator("lora_alpha")
    def validate_lora_alpha(cls, v, values):
        """LoRA alpha typically 2x lora_r"""
        lora_r = values.get("lora_r", 16)
        if v < lora_r:
            raise ValueError(f"lora_alpha ({v}) should be >= lora_r ({lora_r})")
        return v


class InferenceConfig(BaseModel):
    """Inference API configuration"""
    host: str = "0.0.0.0"
    port: int = Field(8000, gt=0, lt=65536)
    max_length: int = Field(2048, gt=0)
    temperature: float = Field(0.1, gt=0.0, le=2.0)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    device: Literal["cuda", "cpu"] = "cuda"
    graceful_shutdown_timeout: int = 30
    save_on_sigterm: bool = True
    batch_size: int = 8
    max_batch_wait_ms: int = 100


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "./logs/scriptguard.log"
    max_bytes: int = 104857600
    backup_count: int = 5
    console: bool = True
    handlers: List[str] = Field(default_factory=lambda: ["console", "file"])

class ExperimentalConfig(BaseModel):
    """Experimental features configuration"""
    use_torch_compile: bool = False
    use_4bit_inference: bool = False
    cpu_offload: bool = False

class RunPodConfig(BaseModel):
    """RunPod specific configuration"""
    pod_id: Optional[str] = None
    volume_mount: str = "/workspace"
    test_network_on_startup: bool = True
    auto_restart_on_oom: bool = False
    snapshot_before_training: bool = False

class ScriptGuardConfig(BaseModel):
    """Complete ScriptGuard configuration schema"""
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    api_keys: APIKeysConfig = Field(default_factory=APIKeysConfig)
    data_sources: DataSourcesConfig = Field(default_factory=DataSourcesConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    code_embedding: CodeEmbeddingConfig = Field(default_factory=CodeEmbeddingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    experimental: ExperimentalConfig = Field(default_factory=ExperimentalConfig)
    runpod: RunPodConfig = Field(default_factory=RunPodConfig)

    class Config:
        """Pydantic configuration"""
        use_enum_values = True  # Convert enums to their values
        validate_assignment = True  # Validate on assignment


def validate_config(config_dict: Dict[str, Any]) -> ScriptGuardConfig:
    """
    Validate configuration dictionary against schema.

    Args:
        config_dict: Configuration dictionary loaded from YAML

    Returns:
        Validated ScriptGuardConfig instance

    Raises:
        ValidationError: If configuration is invalid
    """
    return ScriptGuardConfig(**config_dict)
