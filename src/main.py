import os
from dotenv import load_dotenv
from scriptguard.pipelines.training_pipeline import malware_detection_training_pipeline
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Comet.ml configuration
    # os.environ["COMET_API_KEY"] = "your_api_key"
    # os.environ["COMET_PROJECT_NAME"] = "scriptguard"
    
    gh_malicious = [
        "https://github.com/example/malware/blob/main/shell.py"
    ]
    gh_benign = [
        "https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py"
    ]
    
    local_malicious = "./data/malicious_scripts"
    local_benign = "./data/benign_scripts"
    
    web_urls = ["https://raw.githubusercontent.com/python/cpython/main/Lib/os.py"]
    model_id = "bigcode/starcoder2-3b"
    
    logger.info("Starting ScriptGuard training pipeline with explicit label separation...")
    
    run = malware_detection_training_pipeline(
        gh_malicious_urls=gh_malicious,
        gh_benign_urls=gh_benign,
        local_malicious_dir=local_malicious,
        local_benign_dir=local_benign,
        web_urls=web_urls,
        model_id=model_id
    )
    
    logger.info("Pipeline run completed.")

if __name__ == "__main__":
    main()
