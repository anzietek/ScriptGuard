from typing import Any, Dict
from zenml import pipeline
from scriptguard.steps.data_ingestion import ingest_data
from scriptguard.steps.data_preprocessing import split_data
from scriptguard.steps.tokenization import tokenize_data
from scriptguard.steps.data_augmentation import augment_and_tokenize
from scriptguard.steps.model_training import train_codebert
from scriptguard.steps.evaluation import evaluate_codebert
from scriptguard.steps.model_registration import register_model


@pipeline
def codebert_training_pipeline(config: Dict[str, Any]) -> None:
    clean_data = ingest_data(config=config)
    train_data, val_data, test_data = split_data(data=clean_data, config=config)
    train_tokens, val_tokens, test_tokens = tokenize_data(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        config=config,
    )
    augmented_train = augment_and_tokenize(
        train_tokens=train_tokens,
        train_data=train_data,
        config=config,
    )
    model_path = train_codebert(
        train_dataset=augmented_train,
        val_dataset=val_tokens,
        config=config,
    )
    metrics = evaluate_codebert(
        test_dataset=test_tokens,
        model_path=model_path,
        config=config,
    )
    register_model(
        metrics=metrics,
        model_path=model_path,
        config=config,
    )
