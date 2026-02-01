from zenml import pipeline, step
from datasets import Dataset
from typing import List, Dict, Any, Tuple
from scriptguard.utils.step_cache import get_cache_setting
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
from scriptguard.steps.feature_extraction import extract_features, analyze_feature_importance
from scriptguard.steps.data_preprocessing import preprocess_data
from scriptguard.steps.model_training import train_model as _train_model
from scriptguard.steps.model_evaluation import evaluate_model

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
    processed_dataset = preprocess_data(data=augmented_data)

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

    Args:
        config: Configuration dictionary from config.yaml
        model_id: Model identifier for training
    """
    # Step 1: Advanced data ingestion from multiple sources
    raw_data = _advanced_data_ingestion(config=config)

    # Step 2: Validate samples (syntax, length, encoding)
    validated_data = _validate_samples(
        data=raw_data,
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

    # Step 4: Extract features
    featured_data = extract_features(data=quality_data)

    # Step 5: Analyze feature importance
    feature_analysis = analyze_feature_importance(data=featured_data)

    # Step 6: Augment malicious samples
    if config.get("augmentation", {}).get("enabled", True):
        augmented_data = _augment_malicious_samples(
            data=featured_data,
            variants_per_sample=config.get("augmentation", {}).get("variants_per_sample", 2)
        )
    else:
        augmented_data = featured_data

    # Step 7: Balance dataset
    if config.get("augmentation", {}).get("balance_dataset", True):
        balanced_data = balance_dataset(
            data=augmented_data,
            target_ratio=config.get("augmentation", {}).get("target_balance_ratio", 1.0),
            method=config.get("augmentation", {}).get("balance_method", "undersample")
        )
    else:
        balanced_data = augmented_data

    # Step 8: Preprocess for training
    processed_dataset = preprocess_data(data=balanced_data)

    # Step 9: Split into train/test
    test_size = config.get("training", {}).get("test_split_size", 0.1)
    train_dataset, test_dataset = split_train_test(
        dataset=processed_dataset,
        test_size=test_size
    )

    # Step 10: Train model
    adapter_path = _train_model(dataset=train_dataset, model_id=model_id, config=config)

    # Step 11: Evaluate model
    metrics = evaluate_model(
        adapter_path=adapter_path,
        test_dataset=test_dataset,
        base_model_id=model_id,
        config=config
    )

    return metrics
