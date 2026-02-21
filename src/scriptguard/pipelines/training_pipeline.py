from typing import Any, Dict
from zenml import pipeline
from scriptguard.steps.data_ingestion import ingest_data
from scriptguard.steps.data_preprocessing import split_data
from scriptguard.steps.tokenization import tokenize_data
from scriptguard.steps.extract_features import cache_features, extract_features
from scriptguard.steps.data_augmentation import augment_and_tokenize
from scriptguard.steps.model_training import train_codebert
from scriptguard.steps.evaluation import evaluate_codebert
from scriptguard.steps.model_registration import register_model


@pipeline
def codebert_training_pipeline(config: Dict[str, Any]) -> None:
    clean_data = ingest_data(config=config)
    cached_data = cache_features(all_data=clean_data)
    train_data, val_data, test_data = split_data(data=cached_data, config=config)
    train_tokens, val_tokens, test_tokens = tokenize_data(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        config=config,
    )
    train_tokens_f, val_tokens_f, test_tokens_f, scaler_path = extract_features(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        test_tokens=test_tokens,
        config=config,
    )
    augmented_train = augment_and_tokenize(
        train_tokens=train_tokens_f,
        train_data=train_data,
        config=config,
        scaler_path=scaler_path,
    )
    model_path = train_codebert(
        train_dataset=augmented_train,
        val_dataset=val_tokens_f,
        config=config,
        scaler_path=scaler_path,
    )
    metrics = evaluate_codebert(
        test_dataset=test_tokens_f,
        model_path=model_path,
        config=config,
        scaler_path=scaler_path,
    )
    register_model(
        metrics=metrics,
        model_path=model_path,
        config=config,
    )
