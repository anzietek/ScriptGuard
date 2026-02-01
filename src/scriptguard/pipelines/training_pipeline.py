from zenml import pipeline, step
from scriptguard.steps.data_ingestion import (
    github_data_ingestion, 
    local_data_ingestion, 
    generic_web_ingestion,
    synthetic_data_generation
)
from scriptguard.steps.data_preprocessing import preprocess_data
from scriptguard.steps.model_training import train_model
from scriptguard.steps.model_evaluation import evaluate_model
from typing import List, Dict, Any

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
    
    adapter_path = train_model(dataset=processed_dataset, model_id=model_id)
    metrics = evaluate_model(adapter_path=adapter_path)
    
    return metrics
