from zenml import pipeline, step
from datasets import Dataset
from typing import List, Dict, Any, Tuple
from scriptguard.steps.data_ingestion import (
    github_data_ingestion,
    local_data_ingestion,
    generic_web_ingestion,
    synthetic_data_generation
)
from scriptguard.steps.advanced_ingestion import advanced_data_ingestion as _advanced_data_ingestion
from scriptguard.steps.data_validation import validate_samples as _validate_samples, filter_by_quality as _filter_by_quality
from scriptguard.steps.advanced_augmentation import (
    augment_malicious_samples as _augment_malicious_samples,
    balance_dataset
)
from scriptguard.steps.qdrant_augmentation import (
    augment_with_qdrant_patterns,
    validate_qdrant_augmentation
)
from scriptguard.steps.feature_extraction import extract_features, analyze_feature_importance
from scriptguard.steps.data_preprocessing import preprocess_data
from scriptguard.steps.model_training import train_model as _train_model
from scriptguard.steps.model_evaluation import evaluate_model
from scriptguard.steps.vectorize_samples import vectorize_samples
from scriptguard.materializers.dataset_materializer import HuggingFaceDatasetMaterializer

@step
def split_train_test(dataset: Dataset, test_size: float = 0.1) -> Tuple[Dataset, Dataset]:
    """
    Splits dataset into train and test sets.

    Args:
        dataset: Input dataset
        test_size: Fraction for test set (default 0.1 = 10%)

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    split_dict = dataset.train_test_split(test_size=test_size, seed=42)
    return split_dict['train'], split_dict['test']

@step(output_materializers={"output_2": HuggingFaceDatasetMaterializer})
def split_raw_data(
    data: List[Dict[str, Any]],
    test_size: float = 0.1
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dataset]:
    """
    Splits raw data (list of dicts) into train and test sets.

    Args:
        data: List of dictionaries with code samples
        test_size: Fraction for test set

    Returns:
        Tuple of (train_list, test_list, raw_test_dataset)
        - train_list: List for preprocessing
        - test_list: List for preprocessing
        - raw_test_dataset: HF Dataset for evaluation (not preprocessed)
    """
    from datasets import Dataset as HFDataset
    from scriptguard.utils.logger import logger

    logger.info(f"Splitting {len(data)} samples into train/test (test_size={test_size})")

    # Convert to HF Dataset for splitting
    temp_dataset = HFDataset.from_list(data)
    split_dict = temp_dataset.train_test_split(test_size=test_size, seed=42)

    # Convert to lists for preprocessing
    train_list = [dict(item) for item in split_dict['train']]
    test_list = [dict(item) for item in split_dict['test']]

    # Keep raw test as Dataset for evaluation
    raw_test_dataset = split_dict['test']

    logger.info(f"Split complete: {len(train_list)} train, {len(test_list)} test")

    return train_list, test_list, raw_test_dataset

@step
def merge_data_sources(data_sources: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Merges data from different sources into a single list."""
    combined = []
    for source in data_sources:
        combined.extend(source)
    return combined

@pipeline
def malware_detection_training_pipeline(
    gh_malicious_urls: List[str],
    gh_benign_urls: List[str],
    local_malicious_dir: str,
    local_benign_dir: str,
    web_urls: List[str],
    model_id: str
):
    """
    ZenML pipeline for training the malware detection model with explicit label separation.
    """
    gh_data = github_data_ingestion(
        malicious_urls=gh_malicious_urls,
        benign_urls=gh_benign_urls
    )
    local_data = local_data_ingestion(
        malicious_dir=local_malicious_dir,
        benign_dir=local_benign_dir
    )
    web_data = generic_web_ingestion(urls=web_urls)

    combined_data = merge_data_sources(data_sources=[gh_data, local_data, web_data])

    augmented_data = synthetic_data_generation(base_data=combined_data)
    processed_dataset = preprocess_data(data=augmented_data, config=None)

    adapter_path = _train_model(dataset=processed_dataset, model_id=model_id)
    metrics = evaluate_model(adapter_path=adapter_path)

    return metrics


@pipeline
def advanced_training_pipeline(
    config: Dict[str, Any],
    model_id: str
):
    """
    Advanced ZenML pipeline using new data sources and processing steps.
    Corrected to prevent Data Leakage by splitting BEFORE augmentation.

    Args:
        config: Configuration dictionary from config.yaml
        model_id: Model identifier for training
    """
    # Step 1: Advanced data ingestion from multiple sources
    raw_data = _advanced_data_ingestion(config=config)

    # Step 2: Validate samples (syntax, length, encoding, deduplication)
    validated_data = _validate_samples(
        data=raw_data,
        config=config,
        validate_syntax=config.get("validation", {}).get("validate_syntax", True),
        min_length=config.get("validation", {}).get("min_length", 50),
        max_length=config.get("validation", {}).get("max_length", 50000)
    )

    # Step 3: Filter by quality metrics
    quality_data = _filter_by_quality(
        data=validated_data,
        min_code_lines=config.get("validation", {}).get("min_code_lines", 5),
        max_comment_ratio=config.get("validation", {}).get("max_comment_ratio", 0.5)
    )

    # Step 4: Extract features (useful for analysis, but also for filtering)
    featured_data = extract_features(data=quality_data)

    # Step 5: Analyze feature importance (informational)
    feature_analysis = analyze_feature_importance(data=featured_data)

    # === CRITICAL: Check augmentation strategy ===
    augment_after_split = config.get("augmentation", {}).get("augment_after_split", True)
    test_size = config.get("training", {}).get("test_split_size", 0.1)

    if augment_after_split:
        # RECOMMENDED: Split BEFORE augmentation to prevent data leakage
        from scriptguard.utils.logger import logger
        logger.info("Using augment_after_split=True (prevents data leakage)")

        train_data_list, test_data_list, raw_test_data = split_raw_data(
            data=featured_data,
            test_size=test_size
        )
        data_to_augment = train_data_list
    else:
        # LEGACY: Augment before split (NOT RECOMMENDED - risk of data leakage)
        from scriptguard.utils.logger import logger
        logger.warning("Using augment_after_split=False - this may cause data leakage!")
        data_to_augment = featured_data

    # === AUGMENTATION ===

    # Step 6: Augment malicious samples
    if config.get("augmentation", {}).get("enabled", True):
        augmented_data = _augment_malicious_samples(
            data=data_to_augment,
            variants_per_sample=config.get("augmentation", {}).get("variants_per_sample", 2)
        )
    else:
        augmented_data = data_to_augment

    # === IMPORTANT: Vectorize BEFORE balancing ===
    # RAG needs access to ALL augmented data for maximum pattern diversity
    # Training will use balanced subset, but RAG retrieves from full dataset

    # Step 7: Vectorize ALL augmented data to Qdrant (BEFORE balancing)
    from scriptguard.utils.logger import logger
    logger.info("Vectorizing augmented samples to Qdrant (BEFORE balancing)...")

    vectorization_result = vectorize_samples(
        data=augmented_data,  # All augmented data, not balanced subset!
        config=config,
        clear_existing=True
    )
    logger.info("✓ Vectorized augmented data to Qdrant for RAG")

    # Step 7.5: Balance dataset (AFTER vectorization)
    # This ensures RAG has full dataset while training uses balanced subset
    # NOTE: Step always called - internal logic checks config.augmentation.balance_dataset
    logger.info(f"Calling balance_dataset step (enabled={config.get('augmentation', {}).get('balance_dataset', True)})...")
    balanced_data = balance_dataset(
        data=augmented_data,
        config=config
    )

    # Step 7.7: Augment with Qdrant CVE patterns
    # NOTE: Step always called - internal logic checks config.augmentation.use_qdrant_patterns
    logger.info(f"Calling augment_with_qdrant_patterns step (enabled={config.get('augmentation', {}).get('use_qdrant_patterns', False)})...")
    qdrant_augmented_data = augment_with_qdrant_patterns(
        data=balanced_data,
        config=config
    )

    # Step 7.8: Validate Qdrant augmentation
    logger.info("Calling validate_qdrant_augmentation step...")
    augmentation_stats = validate_qdrant_augmentation(
        data=qdrant_augmented_data,
        config=config
    )

    # === Handle split timing ===
    if not augment_after_split:
        # Split NOW (after augmentation) - legacy behavior
        train_data_list, test_data_list, raw_test_data = split_raw_data(
            data=qdrant_augmented_data,
            test_size=test_size
        )
        final_train_data = train_data_list
    else:
        # Already split before augmentation (recommended)
        final_train_data = qdrant_augmented_data

    # Step 9: Preprocess training and test data
    processed_train_dataset = preprocess_data(data=final_train_data, config=config)
    processed_test_dataset = preprocess_data(data=test_data_list, config=config)

    # Step 10: Train model with preprocessed evaluation dataset
    adapter_path = _train_model(
        dataset=processed_train_dataset,
        eval_dataset=processed_test_dataset,
        model_id=model_id,
        config=config
    )

    # Step 11: Evaluate model using RAW test data (not tokenized)
    # Enable Few-Shot RAG for improved evaluation
    metrics = evaluate_model(
        adapter_path=adapter_path,
        test_dataset=raw_test_data,
        base_model_id=model_id,
        config=config,
        use_fewshot_rag=True  # NEW: Enable Few-Shot RAG
    )

    return metrics
