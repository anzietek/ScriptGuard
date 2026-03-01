import json
import os
from pathlib import Path
from typing import Optional
import re
from typing import Callable

import joblib
import numpy as np
import torch
from transformers import AutoTokenizer

from scriptguard.exceptions import InferenceError
from scriptguard.features.extractor import FeatureExtractor
from scriptguard.models.fused_classifier import load_fused_model
from scriptguard.utils.tokenization_utils import sliding_window_chunks
from scriptguard.utils.logger import logger


# =============================================================================
# TrustEngine helpers (module-level so lambdas can close over them)
# =============================================================================

def _has(code: str, *keywords: str) -> bool:
    """All keywords present (case-sensitive)."""
    return all(kw in code for kw in keywords)


def _has_i(code: str, *keywords: str) -> bool:
    """All keywords present (case-insensitive)."""
    cl = code.lower()
    return all(kw.lower() in cl for kw in keywords)


def _any(code: str, *keywords: str) -> bool:
    """Any keyword present (case-sensitive)."""
    return any(kw in code for kw in keywords)


def _none(code: str, *keywords: str) -> bool:
    """No keyword present (case-sensitive)."""
    return not any(kw in code for kw in keywords)


def _rx(pattern: str, code: str, flags: int = 0) -> bool:
    return bool(re.search(pattern, code, flags))


# ---------------------------------------------------------------------------
# Universal veto sets — any hit in these disqualifies a safe pattern match
# ---------------------------------------------------------------------------
_EXEC_FAMILY = ("exec(", "eval(", "compile(", "__import__", "marshal")
_NETWORK_SINK = ("requests.", "socket.", "urllib.", "httpx.", "aiohttp.",
                 "ftplib.", "smtplib.", "imaplib.", "poplib.")
_SHELL_SINK = ("subprocess.", "os.system", "os.popen", "Popen(",
               "check_output(", "os.execv", "os.execve", "os.spawn")
_INJECTION_API = ("VirtualAlloc", "WriteProcessMemory", "CreateRemoteThread",
                  "mmap.PROT_EXEC", "ctypes.cast", "CFUNCTYPE",
                  "NtCreateThreadEx", "RtlCreateUserThread")
_PERSISTENCE = ("crontab", "schtasks", "HKCU", "HKLM",
                "authorized_keys", "LD_PRELOAD", "rc.local", "launchctl")
_EXFIL = ("attacker.", "evil.", "c2.", "pastebin.com",
          "ngrok.io", "burpcollaborator", "webhook.site",
          "requestbin.", "pipedream.")
_DESER = ("pickle.loads", "marshal.loads", "yaml.load(",
          "shelve.open", "jsonpickle", "dill.loads")
_DANGEROUS_DUNDER = ("__reduce__", "__reduce_ex__", "__getattr__",
                     "__setattr__", "__missing__")


# =============================================================================
# TrustEngine
# =============================================================================

class TrustEngine:
    """
    Enterprise Behavioral Allowlist.

    Overrides ML false-positives for proven, strictly safe developer patterns.
    A sample is safe only if:
      1. At least one positive pattern matches (opt-in allowlist), AND
      2. None of the universal hard-veto disqualifiers fire.

    Positive patterns are (name, predicate) tuples for debuggability.
    Veto checks are (name, predicate) tuples applied after any positive match.

    Method explain() returns a full breakdown for debugging.
    """

    # -------------------------------------------------------------------------
    # Hard-veto disqualifiers — applied after any positive match
    # -------------------------------------------------------------------------
    _VETO_CHECKS: list[tuple[str, Callable[[str], bool]]] = [
        ("exec_family", lambda c: _any(c, *_EXEC_FAMILY)),
        ("injection_api", lambda c: _any(c, *_INJECTION_API)),
        ("persistence", lambda c: _any(c, *_PERSISTENCE)),
        ("exfil_domain", lambda c: _any(c, *_EXFIL)),
        #("shell_sink", lambda c: _any(c, *_SHELL_SINK)),
        ("deserialization", lambda c: _any(c, *_DESER)),
        ("dangerous_dunder", lambda c: _any(c, *_DANGEROUS_DUNDER)),
        ("attacker_ip", lambda c: _rx(
            r'(?:attacker|evil|c2)\s*\.\s*(?:example|local|com)', c, re.I
        )),
        ("base64_exec", lambda c: _rx(
            r'base64.*exec|exec.*base64', c, re.DOTALL
        )),
        ("open_self", lambda c: _rx(
            r'open\s*\(\s*__file__\s*\)', c
        )),
        ("mmap_exec", lambda c: _rx(
            r'mmap.*PROT_EXEC|PROT_WRITE.*PROT_EXEC', c
        )),
        ("shellcode_array", lambda c: _rx(
            r'(?:bytearray|bytes)\s*\(\s*\[[^\]]{200,}\]', c
        )),
        ("docker_escape", lambda c: _rx(
            r'docker\.sock|/:/host|Privileged.*True', c
        )),
        ("aws_metadata", lambda c: "169.254.169.254" in c),
        ("proc_mem", lambda c: _rx(
            r'/proc/\d+/mem|/proc/self/mem', c
        )),
        ("yaml_unsafe_load", lambda c: (
                "yaml" in c and "yaml.load(" in c
                and "Loader=yaml.SafeLoader" not in c
                and "yaml.safe_load" not in c
        )),
        ("token_pattern", lambda c: _rx(
            r'AKIA[0-9A-Z]{10}|ghp_[0-9a-zA-Z]{10}|xox[baprs]-', c
        )),
        ("reverse_shell", lambda c: _rx(
            r'bash\s+-i.*>&|/dev/tcp/|nc\s+-[elp]|ncat\s+--exec', c
        )),
        ("chr_exec", lambda c: (
                re.search(r'chr\s*\(', c) is not None
                and re.search(r'\bexec\b|\beval\b', c) is not None
                and re.search(r'(?:map|join)\s*\(\s*chr', c) is not None
        )),
        # ── NEW VETOS ──────────────────────────────────────────────────────
        ("gui_hijack", lambda c: _rx(r'BlockInput|ShowWindow.*SW_HIDE', c)),
        # C2 webhook / messaging exfil — these are never legitimate in code
        # being scanned for malware regardless of other positive patterns
        ("c2_webhook", lambda c: _rx(
            r'hooks\.slack\.com|api\.telegram\.org'
            r'|discord\.com/api/webhooks'
            r'|discord\.Client\s*\('
            r'|t\.me/|telegram\.me/', c, re.I
        )),
        # Port scan: socket.connect_ex across range of ports
        ("port_scan", lambda c: _rx(
            r'connect_ex\s*\(.*range\s*\('
            r'|range\s*\(.*connect_ex\s*\('
            r'|OPEN_PORTS|open_ports', c, re.DOTALL | re.I
        )),
        # sys.modules poisoning (replace stdlib module)
        ("modules_poison", lambda c: _rx(
            r'sys\.modules\s*\[\s*["\'](?:os|subprocess|socket|builtins)["\']'
            r'\s*\]\s*=', c
        )),
        # Network requests combined with env secret harvesting = exfil
        ("env_secret_exfil", lambda c: (
                _rx(r'os\.environ\.items\(\)', c)
                and _rx(r'(?:KEY|TOKEN|SECRET|PASSWORD)', c, re.I)
                and _any(c, "requests.", "socket.", "urllib.", "httpx.")
        )),
        # codecs.register() as exec smuggling vector
        ("codec_register", lambda c: (
                "codecs.register" in c
                and _rx(r'\bexec\b|\beval\b', c)
        )),
        # threading.Timer used as deferred exec
        ("timer_deferred_exec", lambda c: (
                _rx(r'threading\.Timer|Timer\s*\(', c)
                and _rx(r'\bexec\b|\beval\b', c)
        )),
        # __missing__ dunder used to exec on dict access
        ("missing_exec", lambda c: (
                _rx(r'def\s+__missing__', c)
                and _rx(r'\bexec\b|\beval\b', c)
        )),
        # contextlib.suppress wrapping exec (not just bare suppress)
        ("suppress_exec", lambda c: (
            _rx(r'suppress\s*\([^)]*\).*\bexec\b'
                r'|with\s+suppress\s*\([^)]*\)[^:]*:.*\bexec\b', c, re.DOTALL)
        )),
    ]

    def __init__(self) -> None:
        self._patterns: list[tuple[str, Callable[[str], bool]]] = [

            # ── GUI / TUI ─────────────────────────────────────────────────
            (
                "ctypes_messagebox",
                lambda c: (
                        _has(c, "ctypes", "MessageBoxW")
                        and _none(c, *_INJECTION_API, "subprocess", *_EXEC_FAMILY)
                ),
            ),
            (
                "rich_tui",
                lambda c: (
                        _any(c, "from rich", "import rich",
                             "from textual", "import textual")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_INJECTION_API)
                ),
            ),
            (
                "tkinter_gui",
                lambda c: (
                        _any(c, "import tkinter", "from tkinter", "import Tkinter")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK)
                ),
            ),
            (
                "curses_tui",
                lambda c: (
                        _any(c, "import curses", "from curses")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK)
                ),
            ),
            (
                "wxpython_gui",
                lambda c: (
                        _any(c, "import wx", "from wx")
                        and _any(c, "wx.App", "wx.Frame", "wx.Panel",
                                 "wx.Button", "EVT_")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK)
                ),
            ),
            (
                "pyqt_pyside_gui",
                lambda c: (
                        _any(c, "from PyQt", "from PySide", "import PyQt")
                        and _any(c, "QApplication", "QMainWindow", "QWidget",
                                 "QPushButton", "QLabel")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK)
                ),
            ),
            (
                "typer_cli",
                lambda c: (
                        _any(c, "import typer", "from typer")
                        and _any(c, "typer.run", "typer.Typer", "@app.command")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK)
                ),
            ),
            (
                "click_cli",
                lambda c: (
                        _any(c, "import click", "from click")
                        and _any(c, "@click.command", "@click.option",
                                 "@click.group", "click.echo")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK)
                ),
            ),
            (
                "argparse_cli",
                lambda c: (
                        "import argparse" in c
                        and _any(c, "ArgumentParser", "add_argument", "parse_args")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK, *_EXFIL)
                ),
            ),
            (
                "dotenv_config_runner",
                lambda c: (
                        _any(c, "from dotenv", "import dotenv")
                        and _any(c, "load_dotenv", "find_dotenv")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK, *_EXFIL)
                ),
            ),
            (
                "zenml_pipeline",
                lambda c: (
                        "zenml" in c.lower()
                        and "pipeline" in c.lower()
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK, *_EXFIL)
                ),
            ),
            # ── In-memory / local databases ───────────────────────────────
            (
                "sqlite_in_memory",
                lambda c: (
                        _has(c, "sqlite3", ":memory:")
                        and _none(c, *_NETWORK_SINK, *_EXEC_FAMILY, *_SHELL_SINK)
                ),
            ),
            (
                "sqlalchemy_orm",
                lambda c: (
                        _any(c, "from sqlalchemy", "import sqlalchemy")
                        and _has(c, "Base", "Column")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),
            (
                "peewee_orm",
                lambda c: (
                        _any(c, "from peewee import", "import peewee")
                        and _any(c, "Model", "CharField", "IntegerField",
                                 "database.connect")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),
            (
                "tortoise_orm",
                lambda c: (
                        _any(c, "from tortoise", "import tortoise")
                        and _any(c, "Model", "fields.", "Tortoise.init")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),
            (
                "redis_pubsub",
                lambda c: (
                        _any(c, "import redis", "from redis")
                        and _any(c, "StrictRedis", "pubsub()", "subscribe(",
                                 "publish(", "pipeline(")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),

            # ── Logging ───────────────────────────────────────────────────
            (
                "loguru_logging",
                lambda c: (
                        _any(c, "from loguru", "import loguru")
                        and _none(c, *_NETWORK_SINK, *_EXEC_FAMILY, *_SHELL_SINK)
                ),
            ),
            (
                "stdlib_logging",
                lambda c: (
                        "import logging" in c
                        and _any(c, "logging.getLogger", "basicConfig",
                                 "StreamHandler", "FileHandler",
                                 "RotatingFileHandler")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK)
                ),
            ),
            (
                "structlog_logging",
                lambda c: (
                        _any(c, "import structlog", "from structlog")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK)
                ),
            ),
            (
                "opentelemetry_tracing",
                lambda c: (
                        _any(c, "from opentelemetry", "import opentelemetry")
                        and _any(c, "tracer", "span", "trace.get_tracer",
                                 "MeterProvider")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),

            # ── Data science / ML ─────────────────────────────────────────
            (
                "numpy_math",
                lambda c: (
                        _any(c, "import numpy", "from numpy", "import np")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK,
                                  *_INJECTION_API, *_EXFIL)
                ),
            ),
            (
                "pandas_analysis",
                lambda c: (
                        _any(c, "import pandas", "from pandas")
                        and _any(c, "DataFrame", "read_csv", "describe()",
                                 "groupby", "merge(")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),
            (
                "matplotlib_plot",
                lambda c: (
                        _any(c, "import matplotlib", "from matplotlib")
                        and _any(c, "plt.show", "fig.savefig", "plt.savefig",
                                 "subplots(", "plt.plot")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK)
                ),
            ),
            (
                "scipy_signal",
                lambda c: (
                        _any(c, "from scipy", "import scipy")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK)
                ),
            ),
            (
                "sklearn_model",
                lambda c: (
                        _any(c, "from sklearn", "import sklearn")
                        and _any(c, "fit(", "predict(", "train_test_split",
                                 "Pipeline(", "GridSearchCV")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),
            (
                "pytorch_training",
                lambda c: (
                        _any(c, "import torch", "from torch")
                        and _any(c, "nn.Module", "optimizer.step()",
                                 "loss.backward()", "DataLoader")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),
            (
                "transformers_inference",
                lambda c: (
                        _any(c, "from transformers", "import transformers")
                        and _any(c, "AutoModel", "pipeline(", "tokenizer",
                                 "from_pretrained")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),
            (
                "plotly_visualization",
                lambda c: (
                        _any(c, "import plotly", "from plotly")
                        and _any(c, "fig.show", "go.Figure", "px.scatter",
                                 "update_layout")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK)
                ),
            ),

            # ── Testing frameworks ────────────────────────────────────────
            (
                "pytest_tests",
                lambda c: (
                        _any(c, "import pytest", "from pytest")
                        and _any(c, "def test_", "@pytest.fixture",
                                 "@pytest.mark", "@pytest.parametrize")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK,
                                  *_INJECTION_API)
                ),
            ),
            (
                "unittest_tests",
                lambda c: (
                        _any(c, "import unittest", "from unittest")
                        and _any(c, "TestCase", "setUp", "tearDown",
                                 "assertEqual", "assertRaises")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK)
                ),
            ),
            (
                "hypothesis_property",
                lambda c: (
                        _any(c, "from hypothesis", "import hypothesis")
                        and _any(c, "@given(", "strategies.", "settings(")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK)
                ),
            ),
            (
                "testcontainers",
                lambda c: (
                        _any(c, "from testcontainers", "import testcontainers")
                        and _any(c, "DockerContainer", "PostgreSqlContainer",
                                 "MySqlContainer", "with_command")
                        and _none(c, *_EXEC_FAMILY, *_EXFIL, "/:/host")
                ),
            ),

            # ── Architecture patterns ─────────────────────────────────────
            (
                "abc_registry",
                lambda c: (
                        _any(c, "from abc import", "import abc")
                        and _any(c, "ABC", "ABCMeta", "abstractmethod")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK)
                ),
            ),
            (
                "dataclass_model",
                lambda c: (
                        _any(c, "from dataclasses import", "@dataclass")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK,
                                  *_NETWORK_SINK, *_INJECTION_API)
                ),
            ),
            (
                "pydantic_model",
                lambda c: (
                        _any(c, "from pydantic import", "BaseModel")
                        and _any(c, "field_validator", "model_dump",
                                 "Field(", "model_validator", "validator")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_INJECTION_API)
                ),
            ),
            (
                "enum_statemachine",
                lambda c: (
                        _any(c, "from enum import", "import enum")
                        and _any(c, "Enum", "auto()", "IntEnum", "Flag",
                                 "StrEnum")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK)
                ),
            ),
            (
                "protocol_typing",
                lambda c: (
                        _any(c, "from typing import", "import typing")
                        and _any(c, "Protocol", "TypeVar", "Generic",
                                 "overload", "TypedDict")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK)
                ),
            ),
            (
                "attrs_model",
                lambda c: (
                        _any(c, "import attrs", "from attrs", "import attr")
                        and _any(c, "@attrs.define", "@attr.s", "attr.ib",
                                 "attrs.field")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK)
                ),
            ),

            # ── Web frameworks ────────────────────────────────────────────
            (
                "flask_endpoint",
                lambda c: (
                        _any(c, "from flask import", "import flask")
                        and _any(c, "@app.route", "jsonify(", "request.get_json",
                                 "Blueprint(", "Flask(__name__)")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK,
                                  *_INJECTION_API, *_EXFIL)
                ),
            ),
            (
                "fastapi_endpoint",
                lambda c: (
                        _any(c, "from fastapi import", "import fastapi")
                        and _any(c, "@app.get", "@app.post", "APIRouter",
                                 "Depends(", "HTTPException")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK,
                                  *_INJECTION_API, *_EXFIL)
                ),
            ),
            (
                "django_view",
                lambda c: (
                        _any(c, "from django", "import django")
                        and _any(c, "HttpResponse", "render(", "get_object_or_404",
                                 "models.Model", "urls.path")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK,
                                  *_INJECTION_API, *_EXFIL)
                ),
            ),
            (
                "starlette_app",
                lambda c: (
                        _any(c, "from starlette", "import starlette")
                        and _any(c, "Request", "Response", "Route(",
                                 "Middleware", "TestClient")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),
            (
                "aiohttp_server",
                lambda c: (
                        _any(c, "from aiohttp import", "import aiohttp")
                        and _any(c, "web.Application", "web.get", "web.post",
                                 "AppRunner", "web.Response")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK,
                                  *_INJECTION_API, *_EXFIL)
                ),
            ),

            # ── DevOps / infra tooling ────────────────────────────────────
            (
                "docker_sdk_read",
                lambda c: (
                        _any(c, "import docker", "from docker")
                        and _any(c, "containers.list", "images.list",
                                 "from_env()", "client.images", "client.containers")
                        and _none(c, *_EXEC_FAMILY, "/:/host", "Privileged",
                                  *_EXFIL)
                ),
            ),
            (
                "paramiko_devops",
                lambda c: (
                        _has(c, "paramiko")
                        and _any(c, "AutoAddPolicy", "exec_command", "connect(")
                        and not _rx(
                    r'for\s+\w+\s+in\s+\w*(wordlist|passwords|WORDLIST)',
                    c
                )
                        and _none(c, *_EXEC_FAMILY, *_EXFIL, *_PERSISTENCE)
                ),
            ),
            (
                "boto3_s3_sync",
                lambda c: (
                        _any(c, "import boto3", "from boto3")
                        and _any(c, "s3.upload_file", "s3.download_file",
                                 "s3_client", "boto3.client", "boto3.resource")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL,
                                  "169.254.169.254")
                ),
            ),
            (
                "terraform_sdk",
                lambda c: (
                        _any(c, "from python_terraform", "import python_terraform")
                        and _any(c, "Terraform(", "tf.apply", "tf.plan",
                                 "tf.destroy")
                        and _none(c, *_EXEC_FAMILY, *_EXFIL)
                ),
            ),
            (
                "kubernetes_sdk_read",
                lambda c: (
                        _any(c, "from kubernetes import", "import kubernetes")
                        and _any(c, "list_namespaced_pod", "read_namespaced",
                                 "config.load_kube_config",
                                 "config.load_incluster_config")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL,
                                  "/var/run/secrets/kubernetes")
                ),
            ),
            (
                "ansible_module",
                lambda c: (
                        _any(c, "from ansible", "import ansible")
                        and _any(c, "AnsibleModule", "module.run_command",
                                 "module.exit_json", "module.fail_json")
                        and _none(c, *_EXEC_FAMILY, *_EXFIL)
                ),
            ),
            (
                "fabric_deploy",
                lambda c: (
                        _any(c, "from fabric import", "import fabric")
                        and _any(c, "Connection(", "@task", "ctx.run",
                                 "c.put", "c.get")
                        and _none(c, *_EXFIL, *_INJECTION_API, *_PERSISTENCE)
                ),
            ),
            (
                "devops_automation_runner",
                lambda c: (
                    # Allow standard shell operations
                        _any(c, "os.system(", "subprocess.run(", "subprocess.Popen(", "subprocess.call(")
                        # Script must contain typical config/CLI utilities
                        and _any(c, "import yaml", "import argparse", "from pathlib import Path")
                        # LOCAL HARD VETO: No network communication, code injection, or exfiltration allowed
                        and _none(c, *_EXEC_FAMILY, *_NETWORK_SINK, *_INJECTION_API, *_EXFIL, *_DESER)
                        # Block explicit calls to dangerous/interactive shells
                        and not _rx(r'powershell|/bin/sh|/bin/bash|cmd\.exe|nc\s+-|ncat\s+-', c, re.I)
                ),
            ),

            # ── Config & serialization ────────────────────────────────────
            (
                "json_config",
                lambda c: (
                        "import json" in c
                        and _any(c, "json.loads", "json.dumps",
                                 "json.load", "json.dump")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK,
                                  *_EXFIL, *_INJECTION_API)
                        # Disqualify when json.dumps result is POSTed to external URL
                        # (data exfil pattern)
                        and not _rx(
                    r'requests\.post\s*\(.*json\s*=|json\.dumps.*requests',
                    c, re.DOTALL
                )
                        and not _rx(r'hooks\.slack\.com|api\.telegram\.org', c, re.I)
                ),
            ),
            (
                "yaml_safe_config",
                lambda c: (
                        _any(c, "import yaml", "from yaml")
                        and "yaml.safe_load" in c
                        and "yaml.load(" not in c
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),
            (
                "configparser_ini",
                lambda c: (
                        _any(c, "import configparser", "from configparser")
                        and _any(c, "ConfigParser(", "read_string(", "read(",
                                 "RawConfigParser")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK)
                ),
            ),
            (
                "toml_config",
                lambda c: (
                        _any(c, "import toml", "from toml", "import tomllib",
                             "import tomli")
                        and _any(c, "toml.load", "toml.loads", "tomllib.loads",
                                 "tomli.loads")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),
            (
                "msgpack_serialization",
                lambda c: (
                        _any(c, "import msgpack", "from msgpack")
                        and _any(c, "msgpack.packb", "msgpack.unpackb",
                                 "msgpack.pack", "msgpack.unpack")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),

            # ── Pure stdlib utilities ─────────────────────────────────────
            (
                "pathlib_ops",
                lambda c: (
                        _any(c, "from pathlib import", "import pathlib")
                        and _any(c, "Path(", ".rglob(", ".glob(",
                                 ".read_text", ".write_text", ".mkdir")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL,
                                  *_INJECTION_API, *_NETWORK_SINK)
                ),
            ),
            (
                "hashlib_hashing",
                lambda c: (
                        "import hashlib" in c
                        and _any(c, "hashlib.new", "hashlib.sha256",
                                 "hashlib.md5", "hashlib.sha512",
                                 "hashlib.sha3_256", ".hexdigest()")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),
            (
                "datetime_tz",
                lambda c: (
                        _any(c, "from datetime import", "import datetime")
                        and _any(c, "datetime.now", "timedelta", "timezone",
                                 "strftime", "ZoneInfo", "isoformat")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK,
                                  *_NETWORK_SINK, *_EXFIL)
                ),
            ),
            (
                "itertools_functional",
                lambda c: (
                        _any(c, "import itertools", "from itertools")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK)
                        # Disqualify reduce-based payload rebuilding
                        and not _rx(r'reduce.*chr|chr.*reduce', c)
                ),
            ),
            (
                "functools_cache",
                lambda c: (
                        _any(c, "import functools", "from functools")
                        and _any(c, "lru_cache", "cache", "partial(",
                                 "@functools.wraps", "total_ordering")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK)
                        and not _rx(r'reduce.*chr|chr.*reduce', c)
                ),
            ),
            (
                "contextlib_safe",
                lambda c: (
                        _any(c, "from contextlib import", "import contextlib")
                        and _any(c, "contextmanager", "suppress", "closing",
                                 "ExitStack", "asynccontextmanager")
                        # suppress is fine without exec
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK)
                ),
            ),
            (
                "decimal_precision",
                lambda c: (
                        _any(c, "from decimal import", "import decimal")
                        and _any(c, "Decimal(", "getcontext()",
                                 "ROUND_HALF_UP", "quantize")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK)
                ),
            ),
            (
                "heapq_priority",
                lambda c: (
                        _any(c, "import heapq", "from heapq")
                        and _any(c, "heappush", "heappop", "heapify",
                                 "nlargest", "nsmallest")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK)
                ),
            ),
            (
                "collections_named",
                lambda c: (
                        _any(c, "from collections import", "import collections")
                        and _any(c, "namedtuple", "Counter", "deque",
                                 "defaultdict", "OrderedDict")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK)
                        # OrderedDict.__missing__ can be abused
                        and not _rx(r'__missing__.*exec|exec.*__missing__', c)
                ),
            ),
            (
                "string_template",
                lambda c: (
                        "import string" in c
                        and _any(c, "string.Template", "string.ascii",
                                 "string.digits", "string.punctuation")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK)
                ),
            ),
            (
                "uuid_generation",
                lambda c: (
                        _any(c, "import uuid", "from uuid")
                        and _any(c, "uuid4()", "uuid1()", "uuid3(",
                                 "UUID(", "NAMESPACE_")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK)
                ),
            ),
            (
                "csv_processing",
                lambda c: (
                        "import csv" in c
                        and _any(c, "csv.reader", "csv.writer",
                                 "DictReader", "DictWriter")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),
            (
                "xml_safe_parse",
                lambda c: (
                        _any(c, "import xml", "from xml", "import lxml",
                             "from lxml")
                        and _any(c, "ElementTree", "fromstring", "parse(",
                                 "etree.parse", "defusedxml")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                        # Exclude XXE-prone patterns
                        and not _rx(
                    r'resolve_entities\s*=\s*True'
                    r'|XMLParser.*resolve_entities', c
                )
                ),
            ),
            (
                "markdown_processing",
                lambda c: (
                        _any(c, "import markdown", "from markdown",
                             "import mistune", "import commonmark")
                        and _any(c, "markdown(", "md.convert", "Markdown(",
                                 "mistune.create_markdown")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),
            (
                "jinja2_template",
                lambda c: (
                        _any(c, "from jinja2 import", "import jinja2")
                        and _any(c, "Environment(", "Template(", "render(",
                                 "FileSystemLoader", "PackageLoader")
                        # Disqualify sandbox-escape patterns
                        and not _rx(r'__subclasses__|__class__\s*\.\s*__init__',
                                    c)
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),
            (
                "httpx_client_safe",
                lambda c: (
                        _any(c, "import httpx", "from httpx")
                        and _any(c, "httpx.get", "httpx.post", "httpx.Client",
                                 "AsyncClient", "httpx.AsyncClient")
                        # Must be clearly benign endpoint (not exfil)
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK,
                                  *_INJECTION_API, *_EXFIL)
                        # Disqualify if URL looks like C2
                        and not _rx(
                    r'https?://(?:\d{1,3}\.){3}\d{1,3}'
                    r'(?::\d+)?/(?:beacon|c2|payload|stage)',
                    c, re.I
                )
                ),
            ),

            # ── Concurrency ───────────────────────────────────────────────
            (
                "threading_queue",
                lambda c: (
                        _any(c, "import threading", "from threading")
                        and _any(c, "Queue(", "Thread(", "Event(", "Lock(",
                                 "RLock(", "Semaphore(", "Barrier(")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK,
                                  *_INJECTION_API, *_EXFIL)
                        and "Timer(" not in c
                        # Exclude port scanners: Thread + socket + connect_ex over range
                        and not (
                        "socket" in c
                        and "connect_ex" in c
                )
                ),
            ),
            (
                "concurrent_futures",
                lambda c: (
                        _any(c, "from concurrent.futures", "import concurrent")
                        and _any(c, "ThreadPoolExecutor", "ProcessPoolExecutor",
                                 "as_completed", "submit(")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),
            (
                "asyncio_safe",
                lambda c: (
                        "import asyncio" in c
                        and _any(c, "asyncio.run(", "async def ", "await ",
                                 "gather(", "asyncio.sleep", "asyncio.Queue")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK,
                                  *_INJECTION_API, *_EXFIL)
                ),
            ),
            (
                "trio_async",
                lambda c: (
                        _any(c, "import trio", "from trio")
                        and _any(c, "trio.run", "async with trio",
                                 "trio.open_nursery", "nursery.start_soon")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),

            # ── gRPC / message queues / task queues ───────────────────────
            (
                "grpc_client",
                lambda c: (
                        _any(c, "import grpc", "from grpc")
                        and _any(c, "insecure_channel", "grpc.channel_ready_future",
                                 "stub =", "ServerReflection",
                                 "secure_channel", "add_servicer_to_server")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),
            (
                "celery_task",
                lambda c: (
                        _any(c, "from celery import", "import celery")
                        and _any(c, "@app.task", "Celery(", ".delay(",
                                 ".apply_async(", "@shared_task")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK,
                                  *_INJECTION_API, *_EXFIL)
                ),
            ),
            (
                "kafka_consumer",
                lambda c: (
                        _any(c, "from kafka import", "import kafka",
                             "from confluent_kafka", "import confluent_kafka")
                        and _any(c, "KafkaConsumer", "KafkaProducer",
                                 "Consumer(", "Producer(", "poll(")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),
            (
                "pika_rabbitmq",
                lambda c: (
                        _any(c, "import pika", "from pika")
                        and _any(c, "BlockingConnection", "channel.basic_publish",
                                 "channel.basic_consume", "ConnectionParameters")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),
            (
                "arq_worker",
                lambda c: (
                        _any(c, "from arq import", "import arq")
                        and _any(c, "Worker", "WorkerSettings", "create_pool",
                                 "@arq.cron")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),

            # ── Image / audio / video processing ─────────────────────────
            (
                "pillow_imaging",
                lambda c: (
                        _any(c, "from PIL import", "import PIL",
                             "from Pillow", "Image.open")
                        and _any(c, "Image.new", "img.save", "img.show",
                                 "ImageDraw", "ImageFilter")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK,
                                  *_NETWORK_SINK, *_EXFIL)
                ),
            ),
            (
                "opencv_imaging",
                lambda c: (
                        _any(c, "import cv2", "from cv2")
                        and _any(c, "cv2.imread", "cv2.imwrite", "cv2.resize",
                                 "cv2.VideoCapture", "cv2.cvtColor")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),
            (
                "pydub_audio",
                lambda c: (
                        _any(c, "from pydub import", "import pydub")
                        and _any(c, "AudioSegment", "from_file", "export(",
                                 "overlay(", "append(")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_NETWORK_SINK)
                ),
            ),

            # ── Email / calendar (benign reads) ───────────────────────────
            (
                "email_parse_only",
                lambda c: (
                        _any(c, "import email", "from email")
                        and _any(c, "email.parser", "message_from_string",
                                 "email.mime", "MIMEText")
                        # Must not contain sending code
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL,
                                  "smtplib", "sendmail")
                ),
            ),
            (
                "icalendar_parse",
                lambda c: (
                        _any(c, "from icalendar import", "import icalendar")
                        and _any(c, "Calendar.from_ical", "Component(",
                                 "vCalAddress", "vDatetime")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),

            # ── Static analysis / code quality tools ──────────────────────
            (
                "ast_analysis",
                lambda c: (
                        "import ast" in c
                        and _any(c, "ast.parse", "ast.walk", "ast.dump",
                                 "ast.NodeVisitor", "ast.unparse")
                        and _none(c, *_SHELL_SINK, *_INJECTION_API, *_EXFIL)
                        # ast.parse + exec is a known attack vector
                        and _none(c, "exec(", "eval(")
                ),
            ),
            (
                "pyflakes_lint",
                lambda c: (
                        _any(c, "import pyflakes", "from pyflakes",
                             "import pylint", "from pylint",
                             "import flake8", "from flake8")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),

            # ── Cryptography (safe: signing / hashing only) ───────────────
            (
                "cryptography_signing",
                lambda c: (
                        _any(c, "from cryptography import", "import cryptography")
                        and _any(c, "Signature", "Ed25519", "RSA.sign",
                                 "serialization.load_pem", "x509.Certificate",
                                 "hashes.SHA256")
                        # Disqualify payload encryption patterns
                        and not _rx(
                    r'Fernet\.encrypt|AES.*encrypt|'
                    r'encrypt.*payload|encrypt.*shellcode', c, re.I
                )
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),
            (
                "jwt_auth",
                lambda c: (
                        _any(c, "import jwt", "from jwt", "import pyjwt")
                        and _any(c, "jwt.encode", "jwt.decode",
                                 "jwt.exceptions", "algorithms=")
                        and _none(c, *_EXEC_FAMILY, *_SHELL_SINK, *_EXFIL)
                ),
            ),
        ]

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def is_safe(self, code: str) -> bool:
        """
        Return True only if a positive pattern matches AND no veto fires.
        Short-circuits on first positive match that passes all veto checks.
        """
        for _name, pattern in self._patterns:
            try:
                if not pattern(code):
                    continue
            except Exception:
                continue
            # Positive match — run veto pass
            for _vname, veto in self._VETO_CHECKS:
                try:
                    if veto(code):
                        return False
                except Exception:
                    pass
            return True
        return False

    def explain(self, code: str) -> dict:
        """
        Debug helper: returns which patterns matched and which vetos fired.
        Always evaluates all patterns regardless of early-exit logic.
        """
        matched: list[str] = []
        vetos: list[str] = []

        for name, pattern in self._patterns:
            try:
                if pattern(code):
                    matched.append(name)
            except Exception:
                pass

        if matched:
            for vname, veto in self._VETO_CHECKS:
                try:
                    if veto(code):
                        vetos.append(vname)
                except Exception:
                    pass

        return {
            "is_safe": bool(matched) and not vetos,
            "matched_patterns": matched,
            "fired_vetos": vetos,
        }


# =============================================================================
# ScriptGuardClassifier
# =============================================================================

class ScriptGuardClassifier:
    """
    Main inference entry point for ScriptGuard.

    Classification pipeline:
      1. TrustEngine allowlist  — immediate benign override for safe patterns
      2. Feature extraction     — 27-dimensional feature vector
      3. Heuristic short-circuit — immediate malicious override for known-bad
         gadget patterns (any single gadget or taint flag is sufficient)
      4. Standard AI path       — fused CodeBERT + feature vector inference
      5. Threshold decision     — configurable malicious probability threshold
    """

    LABEL_MAP: dict[int, str] = {0: "benign", 1: "malicious"}

    def __init__(self, model_path: str, scaler_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise InferenceError(f"Model path does not exist: {model_path}")
        if not Path(scaler_path).exists():
            raise InferenceError(f"Scaler path does not exist: {scaler_path}")

        config_file = path / "inference_config.json"
        if config_file.exists():
            with open(config_file) as f:
                cfg = json.load(f)
            self._max_tokens: int = cfg.get("max_tokens", 512)
            self._chunk_overlap: int = cfg.get("chunk_overlap", 50)
            self._decision_threshold: float = cfg.get("decision_threshold", 0.5)
            # Minimum number of gadget/taint flags required to trigger
            # heuristic short-circuit (default 1 — any single hit blocks).
            self._heuristic_threshold: int = cfg.get("heuristic_threshold", 1)
        else:
            logger.warning(
                "inference_config.json not found; "
                "using defaults max_tokens=512, chunk_overlap=50, threshold=0.5"
            )
            self._max_tokens = 512
            self._chunk_overlap = 50
            self._decision_threshold = 0.5
            self._heuristic_threshold = 1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_fused(path, scaler_path)
        self._trust_engine = TrustEngine()

    def _init_fused(self, path: Path, scaler_path: str) -> None:
        logger.info(f"Loading fused model from {path}")
        self.model, self.tokenizer = load_fused_model(str(path), self.device)
        self.model.eval()
        logger.info(f"Loading feature scaler from {scaler_path}")
        self._scaler = joblib.load(scaler_path)
        self._extractor = FeatureExtractor()
        logger.info("Fused model and scaler loaded successfully")

    def _chunk_script(self, script: str) -> list[dict]:
        return sliding_window_chunks(
            tokenizer=self.tokenizer,
            text=script,
            max_length=self._max_tokens,
            overlap=self._chunk_overlap,
            script_id=0,
            label=0,
        )

    def classify(self, script: str, debug: bool = False) -> tuple[str, float, Optional[float]]:
        if not script or not script.strip():
            raise InferenceError("Cannot classify empty script")

        # 1. TRUST ENGINE (Behavioral Allowlist)
        is_trusted: bool = self._trust_engine.is_safe(script)
        if is_trusted and not debug:
            logger.info("TrustEngine: Script matches strictly safe behavioral profile. Allowed.")
            return "benign", 1.0, None

        # 2. HEURISTIC SHORT-CIRCUIT (Malware Overrides)
        # Parse AST for gadget/taint evaluation; failure is non-fatal —
        # the feature extractor handles its own fallback independently.
        try:
            tree = self._extractor._parse_ast(script)
            aliases = self._extractor._build_alias_map(tree)
        except Exception:
            tree = None
            aliases = {}

        gadget_flags = self._extractor._gadget_features(script)
        taint_flags = self._extractor._taint_features(script, tree, aliases)
        critical_count = sum(gadget_flags) + sum(taint_flags)
        is_heuristic_blocked: bool = critical_count >= self._heuristic_threshold

        if is_heuristic_blocked and not debug:
            logger.info(
                f"Heuristic Block: {int(sum(gadget_flags))} gadgets, "
                f"{int(sum(taint_flags))} taints. Immediate block."
            )
            return "malicious", 1.0, None

        # 3. FEATURE EXTRACTION FOR ML
        features_27d = self._extractor.extract(script)

        # 4. STANDARD AI PATH (CodeBERT)
        scaled_features = self._scaler.transform(np.array([features_27d], dtype=np.float32))
        feature_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(self.device)

        chunks = self._chunk_script(script)
        best_malicious_prob: float = 0.0

        with torch.no_grad():
            for chunk in chunks:
                input_ids = torch.tensor([chunk["input_ids"]], dtype=torch.long).to(self.device)
                attention_mask = torch.tensor([chunk["attention_mask"]], dtype=torch.long).to(self.device)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    feature_vector=feature_tensor,
                )
                probs = torch.softmax(outputs.logits, dim=-1)
                malicious_prob = probs[0][1].item()
                if malicious_prob > best_malicious_prob:
                    best_malicious_prob = malicious_prob

        # 5. FINAL DECISION
        # In debug mode the override flags are still applied so the returned
        # label/confidence faithfully reflects what the pipeline would have
        # decided, while ai_malicious_prob always carries the raw model output.
        if is_trusted:
            logger.info(
                f"TrustEngine [debug]: safe pattern matched; "
                f"AI prob={best_malicious_prob:.4f}. Returning benign override."
            )
            return "benign", 1.0, best_malicious_prob

        if is_heuristic_blocked:
            logger.info(
                f"Heuristic [debug]: {int(sum(gadget_flags))} gadgets, "
                f"{int(sum(taint_flags))} taints; "
                f"AI prob={best_malicious_prob:.4f}. Returning malicious override."
            )
            return "malicious", 1.0, best_malicious_prob

        if best_malicious_prob >= self._decision_threshold:
            return "malicious", best_malicious_prob, best_malicious_prob

        return "benign", 1.0 - best_malicious_prob, best_malicious_prob
