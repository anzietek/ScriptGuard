"""
Feature extractor for ScriptGuard fusion model.
Extracts a 23-dimensional feature vector from Python source code.

Output vector (FEATURE_DIM=23):
  - 22 continuous/count features: AST structure metrics, import risk counts,
    entropy values, network counts, and statistical measures.
  - 1 malware_api_score: integer sum of 39 binary indicator flags covering
    dangerous API imports, obfuscation patterns, persistence mechanisms,
    network C2 indicators, crypto operations, and filesystem recon.

Sub-methods compute an intermediate 61-feature raw vector; extract() post-
processes it into the final 23-dimensional form by separating continuous
features from binary flags and aggregating the latter into malware_api_score.
"""

import ast
import math
import re
import warnings
from collections import deque
from scriptguard.utils.logger import logger


class FeatureExtractor:
    """
    Extracts 23-dimensional feature vector from Python source code.

    Output layout (indices 0-22):
        0-9:   AST counts (tree_depth, n_calls, n_imports, n_funcdefs, n_classdefs,
                           n_for, n_while, n_try, n_exec_nodes, exec_eval_depth)
        10-11: Import counts (total_imports, high_risk_imports)
        12-14: Entropy values (mean_str_entropy, max_str_entropy, high_entropy_count)
        15-16: Network counts (unique_ip_count, unique_url_count)
        17-21: Statistical (total_lines, comment_density, avg_line_len,
                            max_line_len, line_len_cv)
        22:    malware_api_score — sum of 39 binary indicator flags

    Sub-methods produce a 61-feature raw vector which is post-processed in
    extract() to yield this 23-dimensional output.
    """

    FEATURE_DIM = 23
    _RAW_DIM = 61  # intermediate raw vector dimension (internal only)

    _HIGH_RISK_IMPORTS = {
        "socket", "subprocess", "os", "ctypes", "base64",
        "marshal", "pickle", "cryptography", "fernet",
    }

    # Indices into the 61-feature raw vector that are binary (0/1) flags.
    # These are summed into malware_api_score and dropped from the output.
    _BINARY_INDICES: frozenset = frozenset({
        # AST: has_nested_functions(9), has_decode_chain(11), has_dynamic_import(12)
        9, 11, 12,
        # Import: has_socket(13) … has_sock_and_subproc(21)
        13, 14, 15, 16, 17, 18, 19, 20, 21,
        # Entropy: has_hardcoded_key(27)
        27,
        # Obfuscation: indices 28-38 (all 11 are binary)
        28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
        # Network: has_hardcoded_ports(41), has_c2_pattern(42),
        #          has_dns_lookup(43), has_system_recon(44)
        41, 42, 43, 44,
        # Persistence: indices 45-48 (all 4 are binary)
        45, 46, 47, 48,
        # Crypto: indices 49-52 (all 4 are binary)
        49, 50, 51, 52,
        # Recon/FS: indices 53-55 (all 3 are binary)
        53, 54, 55,
    })

    # Indices of continuous/count features kept in output (order preserved).
    _CONTINUOUS_INDICES: tuple = (
        0, 1, 2, 3, 4, 5, 6, 7, 8,  # AST counts (tree_depth … n_exec_nodes)
        10,                           # exec_eval_depth (count, can be > 1)
        22, 23,                       # total_imports, high_risk_imports
        24, 25, 26,                   # mean/max entropy, high_entropy_count
        39, 40,                       # unique_ip_count, unique_url_count
        56, 57, 58, 59, 60,           # statistical (5)
    )

    def extract(self, code: str) -> list[float]:
        """
        Extract 23-dimensional feature vector from Python source code.

        Internally computes 61 raw features, then:
          - keeps the 22 continuous/count features in order
          - sums all 39 binary flags into malware_api_score (index 22)

        Returns [0.0] * 23 on any top-level exception.
        """
        try:
            raw = (
                self._ast_features(code)            # 13  indices 0-12
                + self._import_features(code)       # 11  indices 13-23
                + self._entropy_features(code)      # 4   indices 24-27
                + self._obfuscation_features(code)  # 11  indices 28-38
                + self._network_features(code)      # 6   indices 39-44
                + self._persistence_features(code)  # 4   indices 45-48
                + self._crypto_features(code)       # 4   indices 49-52
                + self._recon_fs_features(code)     # 3   indices 53-55
                + self._statistical_features(code)  # 5   indices 56-60
            )
            assert len(raw) == self._RAW_DIM, (
                f"Raw dimension mismatch: expected {self._RAW_DIM}, got {len(raw)}"
            )
            continuous = [raw[i] for i in self._CONTINUOUS_INDICES]
            malware_api_score = float(sum(raw[i] for i in self._BINARY_INDICES))
            features = continuous + [malware_api_score]
            assert len(features) == self.FEATURE_DIM
            return features
        except Exception as e:
            logger.warning(f"FeatureExtractor.extract failed: {e}")
            return [0.0] * self.FEATURE_DIM

    @staticmethod
    def _parse_ast(code: str) -> ast.AST:
        """Parse code suppressing SyntaxWarning emitted for analyzed samples."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(code)

    # ------------------------------------------------------------------
    # AST features (13)
    # ------------------------------------------------------------------

    def _ast_features(self, code: str) -> list[float]:
        try:
            tree = self._parse_ast(code)
        except SyntaxError:
            return [0.0] * 13

        try:
            # tree_depth: BFS to find maximum depth
            tree_depth = 0
            q: deque = deque([(tree, 0)])
            while q:
                node, depth = q.popleft()
                if depth > tree_depth:
                    tree_depth = depth
                for child in ast.iter_child_nodes(node):
                    q.append((child, depth + 1))

            # node counts
            node_count_call = 0
            node_count_import = 0
            node_count_functiondef = 0
            node_count_classdef = 0
            node_count_for = 0
            node_count_while = 0
            node_count_try = 0
            node_count_exec = 0

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    node_count_call += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    node_count_import += 1
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    node_count_functiondef += 1
                elif isinstance(node, ast.ClassDef):
                    node_count_classdef += 1
                elif isinstance(node, ast.For):
                    node_count_for += 1
                elif isinstance(node, ast.While):
                    node_count_while += 1
                elif isinstance(node, (ast.Try, ast.ExceptHandler)):
                    node_count_try += 1
                elif isinstance(node, ast.Expr):
                    # Check for exec() call
                    if isinstance(getattr(node, 'value', None), ast.Call):
                        func = getattr(node.value, 'func', None)
                        if isinstance(func, ast.Name) and func.id == 'exec':
                            node_count_exec += 1

            # has_nested_functions
            has_nested_functions = 0.0
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for child in ast.walk(node):
                        if child is not node and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            has_nested_functions = 1.0
                            break
                if has_nested_functions:
                    break

            # exec_eval_depth: max nesting depth of exec(eval(...)) patterns
            exec_eval_depth = self._compute_exec_eval_depth(tree)

            # has_decode_chain: chained b64decode/decompress/exec pattern
            has_decode_chain = self._has_decode_chain(code, tree)

            # has_dynamic_import: use of __import__() or importlib.import_module
            has_dynamic_import = 0.0
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = getattr(node, 'func', None)
                    if isinstance(func, ast.Name) and func.id == '__import__':
                        has_dynamic_import = 1.0
                        break
                    if isinstance(func, ast.Attribute) and func.attr == 'import_module':
                        has_dynamic_import = 1.0
                        break

            return [
                float(tree_depth),
                float(node_count_call),
                float(node_count_import),
                float(node_count_functiondef),
                float(node_count_classdef),
                float(node_count_for),
                float(node_count_while),
                float(node_count_try),
                float(node_count_exec),
                has_nested_functions,
                float(exec_eval_depth),
                has_decode_chain,
                has_dynamic_import,
            ]
        except Exception:
            return [0.0] * 13

    def _compute_exec_eval_depth(self, tree: ast.AST) -> float:
        """Compute max nesting depth of exec(eval(...)) call chains."""
        max_depth = 0

        def _depth(node: ast.AST) -> int:
            if not isinstance(node, ast.Call):
                return 0
            func = getattr(node, 'func', None)
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            if name in ('exec', 'eval', 'compile'):
                args = getattr(node, 'args', [])
                if args and isinstance(args[0], ast.Call):
                    return 1 + _depth(args[0])
                return 1
            return 0

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                d = _depth(node)
                if d > max_depth:
                    max_depth = d

        return float(max_depth)

    def _has_decode_chain(self, code: str, tree: ast.AST) -> float:
        """Detect chained decode/decompress/exec patterns."""
        # Regex approach: b64decode(...decompress(...exec(
        if re.search(r'b64decode.+decompress.+exec', code, re.DOTALL):
            return 1.0
        # AST: look for chained calls with decode/decompress in same call chain
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                chain_funcs = self._collect_call_chain_names(node)
                if (
                    any('decode' in n for n in chain_funcs)
                    and any('decompress' in n or 'inflate' in n for n in chain_funcs)
                ):
                    return 1.0
        return 0.0

    def _collect_call_chain_names(self, node: ast.Call) -> list[str]:
        names = []
        current = node
        while isinstance(current, ast.Call):
            func = getattr(current, 'func', None)
            if isinstance(func, ast.Attribute):
                names.append(func.attr)
                current = getattr(func, 'value', None)
                if isinstance(current, ast.Call):
                    continue
                break
            elif isinstance(func, ast.Name):
                names.append(func.id)
                break
            else:
                break
        return names

    # ------------------------------------------------------------------
    # Import features (11)
    # ------------------------------------------------------------------

    def _import_features(self, code: str) -> list[float]:
        try:
            tree = self._parse_ast(code)
        except SyntaxError:
            return [0.0] * 11

        try:
            imported_modules: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_modules.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imported_modules.add(node.module.split('.')[0])

            has_socket = float("socket" in imported_modules)
            has_subprocess = float("subprocess" in imported_modules)
            has_ctypes = float("ctypes" in imported_modules)
            has_base64 = float("base64" in imported_modules)
            has_marshal = float("marshal" in imported_modules)
            has_pickle = float("pickle" in imported_modules or "cPickle" in imported_modules)
            has_cryptography = float("cryptography" in imported_modules)

            # has_os_exec: specific OS execution calls — not just `import os` (fires on everything)
            has_os_exec = 0.0
            _os_exec_attrs = frozenset(
                ('system', 'popen', 'execv', 'execl', 'execvp', 'execve', 'execvpe', 'spawnl', 'spawnle', 'spawnlp')
            )
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = getattr(node, 'func', None)
                    if isinstance(func, ast.Attribute) and func.attr in _os_exec_attrs:
                        val = getattr(func, 'value', None)
                        if isinstance(val, ast.Name) and val.id == 'os':
                            has_os_exec = 1.0
                            break
            if not has_os_exec:
                if re.search(r'os\.(?:system|popen|execv[pe]?|execl[pe]?|spawn)\s*\(', code):
                    has_os_exec = 1.0

            # has_socket_and_subprocess: both imported together — classic reverse shell combo
            has_socket_and_subprocess = float(
                "socket" in imported_modules and "subprocess" in imported_modules
            )

            total_import_count = float(len(imported_modules))
            high_risk_import_count = float(len(imported_modules & self._HIGH_RISK_IMPORTS))

            return [
                has_socket, has_subprocess, has_os_exec, has_ctypes, has_base64,
                has_marshal, has_pickle, has_cryptography, has_socket_and_subprocess,
                total_import_count, high_risk_import_count,
            ]
        except Exception:
            return [0.0] * 11

    # ------------------------------------------------------------------
    # Entropy features (4)
    # ------------------------------------------------------------------

    def _entropy_features(self, code: str) -> list[float]:
        try:
            # Extract string literals using regex (both single and double quoted)
            string_literals = re.findall(
                r'(?:b?)(?:"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')', code
            )

            entropies = []
            for s in string_literals:
                # Strip quotes and prefix
                inner = s
                for prefix in ('b"', "b'", '"', "'"):
                    if inner.startswith(prefix):
                        inner = inner[len(prefix):]
                        break
                for suffix in ('"', "'"):
                    if inner.endswith(suffix):
                        inner = inner[:-1]
                        break
                if len(inner) < 4:
                    continue
                ent = self._shannon_entropy(inner)
                entropies.append((ent, len(inner)))

            if not entropies:
                return [0.0, 0.0, 0.0, 0.0]

            ent_values = [e for e, _ in entropies]
            mean_string_entropy = sum(ent_values) / len(ent_values)
            max_string_entropy = max(ent_values)
            high_entropy_string_count = float(sum(1 for e in ent_values if e > 4.5))

            # has_hardcoded_key: entropy > 4.5 AND length in {16, 32, 64}
            has_hardcoded_key = 0.0
            for ent, length in entropies:
                if ent > 4.5 and length in (16, 32, 64):
                    has_hardcoded_key = 1.0
                    break

            return [mean_string_entropy, max_string_entropy, high_entropy_string_count, has_hardcoded_key]
        except Exception:
            return [0.0] * 4

    def _shannon_entropy(self, s: str) -> float:
        if not s:
            return 0.0
        freq: dict[str, int] = {}
        for c in s:
            freq[c] = freq.get(c, 0) + 1
        n = len(s)
        entropy = -sum((count / n) * math.log2(count / n) for count in freq.values())
        return entropy

    # ------------------------------------------------------------------
    # Obfuscation features (11)
    # ------------------------------------------------------------------

    def _obfuscation_features(self, code: str) -> list[float]:
        try:
            try:
                tree = self._parse_ast(code)
                parse_ok = True
            except SyntaxError:
                tree = None
                parse_ok = False

            # has_exec: exec() call in AST
            has_exec = 0.0
            # has_eval: eval() call
            has_eval = 0.0
            # has_compile: compile() call
            has_compile = 0.0
            # has_dunder_import: __import__() call
            has_dunder_import = 0.0
            lambda_chain_count = 0

            if parse_ok and tree:
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        func = getattr(node, 'func', None)
                        if isinstance(func, ast.Name):
                            if func.id == 'exec':
                                has_exec = 1.0
                            elif func.id == 'eval':
                                has_eval = 1.0
                            elif func.id == 'compile':
                                has_compile = 1.0
                            elif func.id == '__import__':
                                has_dunder_import = 1.0

                # lambda chain: lambda that returns a lambda
                for node in ast.walk(tree):
                    if isinstance(node, ast.Lambda):
                        body = node.body
                        depth = 0
                        while isinstance(body, ast.Lambda):
                            depth += 1
                            body = body.body
                        if depth > 0:
                            lambda_chain_count += 1

            # has_b64decode_call: actual decode operation — not just any alphanumeric string
            # Old `has_base64_pattern` matched ANY 20-char alphanumeric (variable names, URLs, etc.)
            has_b64decode_call = 0.0
            if re.search(
                r'b64decode|b64encode|base64\.b64|\.decode\(["\']base64["\']'
                r'|binascii\.(?:unhexlify|a2b_hex)|bytes\.fromhex|codecs\.decode',
                code,
            ):
                has_b64decode_call = 1.0

            # has_no_comments: malicious code almost never has inline documentation
            # Replaces has_hex_strings (Δ≈-0.001, anti-correlated, useless)
            _ob_lines = code.splitlines()
            _ob_comment_count = sum(1 for l in _ob_lines if l.strip().startswith('#'))
            has_no_comments = float(_ob_comment_count == 0 and len(_ob_lines) > 5)

            # has_encoded_execution: decode + exec/eval in same script (high malware signal)
            has_encoded_execution = 0.0
            if (has_exec or has_eval or has_compile) and has_b64decode_call:
                if re.search(
                    r'(?:exec|eval|compile)\s*\([^)]*(?:b64decode|unhexlify|fromhex|decode)',
                    code, re.DOTALL,
                ):
                    has_encoded_execution = 1.0
                elif re.search(
                    r'(?:b64decode|unhexlify|fromhex)[^;)]*(?:exec|eval)',
                    code, re.DOTALL,
                ):
                    has_encoded_execution = 1.0
                elif re.search(r'(?:exec|eval)\s*\(\s*\w+\s*\(\s*\w+\s*\(', code):
                    has_encoded_execution = 1.0

            # has_very_long_line: single-line blobs = obfuscated/base64 payloads
            # Replaces has_shellcode (Δ=0.000, never fires in practice)
            _ob_line_lens = [len(l) for l in _ob_lines]
            has_very_long_line = float(max(_ob_line_lens) > 500 if _ob_line_lens else False)

            # has_anti_debug
            has_anti_debug = 0.0
            anti_debug_patterns = [
                r'IsDebuggerPresent', r'QueryPerformanceCounter',
                r'CheckRemoteDebuggerPresent', r'NtQueryInformationProcess',
                r'ptrace', r'PTRACE_TRACEME',
            ]
            for pat in anti_debug_patterns:
                if re.search(pat, code):
                    has_anti_debug = 1.0
                    break

            # has_ctypes_windll
            has_ctypes_windll = 0.0
            if re.search(r'ctypes\.windll|windll\.kernel32|windll\.ntdll', code):
                has_ctypes_windll = 1.0

            # has_process_injection
            has_process_injection = 0.0
            injection_patterns = [
                r'VirtualAlloc', r'WriteProcessMemory', r'CreateRemoteThread',
                r'NtWriteVirtualMemory', r'RtlMoveMemory', r'OpenProcess',
            ]
            for pat in injection_patterns:
                if re.search(pat, code):
                    has_process_injection = 1.0
                    break

            return [
                has_exec, has_eval, has_compile, has_dunder_import,
                has_b64decode_call, has_no_comments,
                has_encoded_execution,
                has_very_long_line, has_anti_debug, has_ctypes_windll, has_process_injection,
            ]
        except Exception:
            return [0.0] * 11

    def _detect_shellcode(self, code: str, tree) -> float:
        """Detect shellcode: bytearray/bytes literal > 100 bytes with entropy > 5.5."""
        # Regex for hex-encoded byte arrays
        hex_arrays = re.findall(r'(?:bytearray|bytes)\s*\(\s*\[([^\]]+)\]', code)
        for arr_str in hex_arrays:
            parts = [p.strip() for p in arr_str.split(',') if p.strip()]
            if len(parts) > 100:
                # Compute entropy of the byte values
                try:
                    values = [int(p, 0) for p in parts if p]
                    if len(values) > 100:
                        freq: dict[int, int] = {}
                        for v in values:
                            freq[v] = freq.get(v, 0) + 1
                        n = len(values)
                        ent = -sum((c / n) * math.log2(c / n) for c in freq.values())
                        if ent > 5.5:
                            return 1.0
                except Exception:
                    pass

        # Also check for long \x byte strings
        hex_str_matches = re.findall(r'(?:\\x[0-9a-fA-F]{2}){100,}', code)
        if hex_str_matches:
            return 1.0

        return 0.0

    # ------------------------------------------------------------------
    # Network / C2 features (6)
    # ------------------------------------------------------------------

    def _network_features(self, code: str) -> list[float]:
        try:
            # unique_ip_count
            ips = re.findall(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', code)
            unique_ip_count = float(len(set(ips)))

            # unique_url_count
            urls = re.findall(r'https?://[^\s\'"]+', code)
            unique_url_count = float(len(set(urls)))

            # has_hardcoded_ports: non-well-known port numbers in code
            has_hardcoded_ports = 0.0
            port_matches = re.findall(r'\b(4444|1337|8080|8888|9999|31337|6667|1234|5555|7777)\b', code)
            if port_matches:
                has_hardcoded_ports = 1.0

            # has_c2_pattern: connect + send/recv in same function body (AST scope)
            has_c2_pattern = self._detect_c2_pattern(code)

            # has_dns_lookup
            has_dns_lookup = 0.0
            if re.search(r'socket\.gethostbyname|dns\.resolver|getaddrinfo|nslookup', code):
                has_dns_lookup = 1.0

            # has_system_recon: fingerprinting specific to C2 beacon behavior
            # Removed: os.environ (config in every Flask/Django app), platform.system() (cross-platform libs)
            has_system_recon = 0.0
            recon_patterns = [
                r'platform\.node\(\)',
                r'socket\.gethostname\(\)',
                r'os\.uname\(\)',
                r'getpass\.getuser\(\)',
                r'whoami',
                r'systeminfo',
                r'ipconfig',
                r'net\s+user',
                r'netifaces\.',
                r'psutil\.net_if',
            ]
            for pat in recon_patterns:
                if re.search(pat, code):
                    has_system_recon = 1.0
                    break

            return [
                unique_ip_count, unique_url_count, has_hardcoded_ports,
                has_c2_pattern, has_dns_lookup, has_system_recon,
            ]
        except Exception:
            return [0.0] * 6

    def _detect_c2_pattern(self, code: str) -> float:
        """Detect C2 pattern: connect + send/recv calls in same function body."""
        try:
            tree = self._parse_ast(code)
        except SyntaxError:
            # Fallback to regex
            if re.search(r'\.connect\(', code) and re.search(r'\.(send|recv|sendall|recvfrom)\(', code):
                return 1.0
            return 0.0

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls_in_fn: list[str] = []
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func = getattr(child, 'func', None)
                    if isinstance(func, ast.Attribute):
                        calls_in_fn.append(func.attr)

            has_connect = any(c == 'connect' for c in calls_in_fn)
            has_comm = any(c in ('send', 'recv', 'sendall', 'recvfrom', 'sendto') for c in calls_in_fn)
            if has_connect and has_comm:
                return 1.0

        return 0.0

    # ------------------------------------------------------------------
    # Persistence features (4)
    # ------------------------------------------------------------------

    def _persistence_features(self, code: str) -> list[float]:
        try:
            # has_registry_write
            has_registry_write = 0.0
            if re.search(r'winreg|_winreg|RegSetValue|HKEY_|OpenKey|CreateKey', code):
                has_registry_write = 1.0

            # has_cron_pattern: creating scheduled tasks — not just mentioning cron
            # Removed: crontab|/etc/cron — Ansible, Fabric, SaltStack all manage cron legitimately
            has_cron_pattern = 0.0
            cron_create_patterns = [
                r'schtasks\s+/[Cc]reate',          # Windows schtasks /Create
                r'at\.exe\s+\d',                    # Windows at.exe scheduler
                r'SchTasks\.exe',
                r'open\(["\'][/\\]etc[/\\]cron',    # writing to cron file (not just reading)
                r'crontab\s+-[el]\s',               # editing crontab interactively
                r'TaskScheduler|ITaskScheduler',    # COM interface
            ]
            for pat in cron_create_patterns:
                if re.search(pat, code, re.IGNORECASE):
                    has_cron_pattern = 1.0
                    break

            # has_startup_persistence: actual persistence mechanism installation
            # Removed: Startup, systemd, launchd, autorun, .bashrc, .profile, rc.local
            # — all fired by Ansible, Fabric, SaltStack, Django AppConfig, Flask startup handlers
            has_startup_persistence = 0.0
            startup_patterns = [
                r'\\CurrentVersion\\Run(?:Once)?["\']',  # registry run key (write)
                r'HKCU.*\\Run|HKLM.*\\Run',              # registry persistence
                r'nssm\s+install',                        # Windows service via nssm
                r'sc\.exe\s+create|CreateService\s*\(',  # Windows service creation
                r'open\(["\'][/\\]etc[/\\](?:init\.d|rc\.local)',  # writing init script
                r'(?:plist|launchd).*write|write.*plist', # plist persistence write
            ]
            for pat in startup_patterns:
                if re.search(pat, code, re.IGNORECASE):
                    has_startup_persistence = 1.0
                    break

            # has_creates_executable: actually dropping/writing an executable file
            # Removed: bare .exe/.bat/.cmd mentions (fires on PyInstaller, build scripts, test files)
            has_creates_executable = 0.0
            exec_drop_patterns = [
                r'chmod\s+\+x',                                          # shell chmod +x
                r'os\.chmod\([^,]+,\s*(?:0[oO]?7[0-7]{2}|0o?755|0o?777)',  # os.chmod to make executable
                r'open\([^)]*\.(exe|bat|cmd|vbs|ps1|sh)[^)]*["\'],\s*["\'][wa]b?',  # write to executable
                r'with\s+open\([^)]*\.(exe|bat|cmd|vbs|ps1|sh)',         # context mgr write
                r'HKEY_[A-Z_]+.*(?:\\\\Run|\\\\RunOnce)',                 # registry run key
                r'subprocess.*["\'][^\'"]*\.(exe|bat|cmd)["\']',          # executing dropped file
            ]
            for pat in exec_drop_patterns:
                if re.search(pat, code, re.IGNORECASE):
                    has_creates_executable = 1.0
                    break

            return [has_registry_write, has_cron_pattern, has_startup_persistence, has_creates_executable]
        except Exception:
            return [0.0] * 4

    # ------------------------------------------------------------------
    # Cryptography features (4)
    # ------------------------------------------------------------------

    def _crypto_features(self, code: str) -> list[float]:
        try:
            has_aes_pattern = 0.0
            if re.search(r'AES|Cipher\.new|AESCipher|aes\.encrypt|aes\.decrypt', code):
                has_aes_pattern = 1.0

            has_rc4_pattern = 0.0
            if re.search(r'RC4|arc4|Salsa20|ChaCha|rc4_encrypt', code, re.IGNORECASE):
                has_rc4_pattern = 1.0

            # has_xor_pattern: XOR used as a cipher, not standard ^= assignment
            # Removed: bare `^=` — fires on benign bitwise ops (checksums, flags, hash functions)
            has_xor_pattern = 0.0
            if re.search(r'\bxor\b|\bXOR\b', code):
                has_xor_pattern = 1.0
            # AST: ^= inside a For loop on a subscript — classic XOR cipher loop
            # e.g.  data[i] ^= key[i % len(key)]
            if not has_xor_pattern:
                try:
                    tree = self._parse_ast(code)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.For):
                            for child in ast.walk(node):
                                if (
                                    isinstance(child, ast.AugAssign)
                                    and isinstance(child.op, ast.BitXor)
                                    and isinstance(child.target, ast.Subscript)
                                ):
                                    has_xor_pattern = 1.0
                                    break
                        if has_xor_pattern:
                            break
                except SyntaxError:
                    pass

            has_fernet_usage = 0.0
            if re.search(r'Fernet|fernet\.encrypt|fernet\.decrypt', code):
                has_fernet_usage = 1.0

            return [has_aes_pattern, has_rc4_pattern, has_xor_pattern, has_fernet_usage]
        except Exception:
            return [0.0] * 4

    # ------------------------------------------------------------------
    # Recon / Filesystem features (3)
    # ------------------------------------------------------------------

    def _recon_fs_features(self, code: str) -> list[float]:
        try:
            # has_recursive_traversal: deep/recursive filesystem traversal
            # Removed: os.listdir — simple listing, extremely common in benign code
            # (scikit-learn dataset loaders, scrapy spiders, Ansible file management, etc.)
            has_recursive_traversal = 0.0
            if re.search(r'os\.walk\(|\.rglob\(|glob\.glob[^)]*\*\*', code):
                has_recursive_traversal = 1.0

            # has_mass_file_ops: iteration over multiple file extensions (ransomware pattern)
            has_mass_file_ops = 0.0
            ext_pattern = r'\.(doc|docx|xls|xlsx|pdf|jpg|png|mp4|zip|rar|txt|csv)'
            ext_matches = re.findall(ext_pattern, code, re.IGNORECASE)
            if len(set(ext_matches)) >= 3:
                has_mass_file_ops = 1.0
            # Also check for encrypt/open loops over files
            if re.search(r'for\s+\w+\s+in\s+.*\.walk|encrypt.*\.open|open.*encrypt', code, re.DOTALL):
                has_mass_file_ops = 1.0

            # has_shadow_copy: VSS deletion (ransomware behavior)
            has_shadow_copy = 0.0
            if re.search(r'vssadmin|shadow\s+copy|wmic.*shadowcopy|Win32_ShadowCopy', code, re.IGNORECASE):
                has_shadow_copy = 1.0

            return [has_recursive_traversal, has_mass_file_ops, has_shadow_copy]
        except Exception:
            return [0.0] * 3

    # ------------------------------------------------------------------
    # Statistical features (5)
    # ------------------------------------------------------------------

    def _statistical_features(self, code: str) -> list[float]:
        try:
            lines = code.splitlines()
            total_lines = float(len(lines))

            if total_lines == 0:
                return [0.0, 0.0, 0.0, 0.0, 0.0]

            comment_lines = sum(1 for l in lines if l.strip().startswith('#'))

            # comment_density: fraction of comment lines [0, 1]
            # Replaces code_to_comment_ratio which was unbounded when no comments exist
            # (ratio = code_lines / 1 → same as total_lines → hundreds, pollutes the feature)
            comment_density = float(comment_lines) / total_lines

            line_lengths = [len(l) for l in lines]
            avg_line_length = sum(line_lengths) / max(1, len(line_lengths))
            max_line_length = float(max(line_lengths)) if line_lengths else 0.0

            # line_len_cv: coefficient of variation of line lengths (stdev / mean)
            # High CV = obfuscated code: one very long line among many short lines
            # Replaces blank_line_ratio (Δ=-0.055, anti-correlated, not useful)
            if len(line_lengths) > 1:
                mean = avg_line_length
                variance = sum((l - mean) ** 2 for l in line_lengths) / len(line_lengths)
                line_len_cv = math.sqrt(variance) / max(1.0, mean)
            else:
                line_len_cv = 0.0

            return [
                total_lines,
                comment_density,
                avg_line_length,
                max_line_length,
                line_len_cv,
            ]
        except Exception:
            return [0.0] * 5
