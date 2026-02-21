"""
Feature extractor for ScriptGuard fusion model.
Extracts 61 hand-crafted features from Python source code using AST analysis,
regex patterns, and statistical measures.
"""

import ast
import math
import re
from collections import deque
from scriptguard.utils.logger import logger


class FeatureExtractor:
    """
    Extracts 61-dimensional feature vector from Python source code.

    Feature groups:
        AST (13): tree_depth, node_count_{Call,Import,FunctionDef,ClassDef,For,While,Try,Exec},
                  has_nested_functions, exec_eval_depth, has_decode_chain, has_dynamic_import
        Import (11): has_{socket,subprocess,os,ctypes,base64,marshal,pickle,cryptography,fernet},
                     total_import_count, high_risk_import_count
        Entropy (4): mean_string_entropy, max_string_entropy, high_entropy_string_count, has_hardcoded_key
        Obfuscation (11): has_{exec,eval,compile,dunder_import,base64_pattern,hex_strings},
                          lambda_chain_count, has_{shellcode,anti_debug,ctypes_windll,process_injection}
        Network/C2 (6): unique_ip_count, unique_url_count,
                        has_{hardcoded_ports,c2_pattern,dns_lookup,system_recon}
        Persistence (4): has_{registry_write,cron_pattern,startup_persistence,creates_executable}
        Cryptography (4): has_{aes_pattern,rc4_pattern,xor_pattern,fernet_usage}
        Recon/FS (3): has_{recursive_traversal,mass_file_ops,shadow_copy}
        Statistical (5): total_lines, code_to_comment_ratio, avg_line_length, max_line_length, blank_line_ratio
    """

    FEATURE_DIM = 61

    _HIGH_RISK_IMPORTS = {
        "socket", "subprocess", "os", "ctypes", "base64",
        "marshal", "pickle", "cryptography", "fernet",
    }

    def extract(self, code: str) -> list[float]:
        """
        Extract 61-dimensional feature vector from Python source code.

        Returns [0.0] * 61 on any top-level exception. Individual sub-methods
        return partial zeroes on errors to maintain fixed dimensionality.
        """
        try:
            features = (
                self._ast_features(code)          # 13
                + self._import_features(code)     # 11
                + self._entropy_features(code)    # 4
                + self._obfuscation_features(code)  # 11
                + self._network_features(code)    # 6
                + self._persistence_features(code)  # 4
                + self._crypto_features(code)     # 4
                + self._recon_fs_features(code)   # 3
                + self._statistical_features(code)  # 5
            )
            assert len(features) == self.FEATURE_DIM, (
                f"Feature dimension mismatch: expected {self.FEATURE_DIM}, got {len(features)}"
            )
            return features
        except Exception as e:
            logger.warning(f"FeatureExtractor.extract failed: {e}")
            return [0.0] * self.FEATURE_DIM

    # ------------------------------------------------------------------
    # AST features (13)
    # ------------------------------------------------------------------

    def _ast_features(self, code: str) -> list[float]:
        try:
            tree = ast.parse(code)
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
            tree = ast.parse(code)
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
            has_os = float("os" in imported_modules)
            has_ctypes = float("ctypes" in imported_modules)
            has_base64 = float("base64" in imported_modules)
            has_marshal = float("marshal" in imported_modules)
            has_pickle = float("pickle" in imported_modules or "cPickle" in imported_modules)
            has_cryptography = float("cryptography" in imported_modules)
            has_fernet = float(
                "fernet" in imported_modules
                or "cryptography" in imported_modules  # Fernet is part of cryptography
            )
            total_import_count = float(len(imported_modules))
            high_risk_import_count = float(len(imported_modules & self._HIGH_RISK_IMPORTS))

            return [
                has_socket, has_subprocess, has_os, has_ctypes, has_base64,
                has_marshal, has_pickle, has_cryptography, has_fernet,
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
                tree = ast.parse(code)
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

            # has_base64_pattern: base64 encoded string in code
            has_base64_pattern = 0.0
            if re.search(r'[A-Za-z0-9+/]{20,}={0,2}', code):
                has_base64_pattern = 1.0

            # has_hex_strings: \x-style hex byte sequences
            has_hex_strings = 0.0
            if re.search(r'(?:\\x[0-9a-fA-F]{2}){4,}', code):
                has_hex_strings = 1.0

            # has_shellcode: bytearray/bytes literal > 100 bytes with high entropy
            has_shellcode = self._detect_shellcode(code, tree)

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
                has_base64_pattern, has_hex_strings,
                float(lambda_chain_count),
                has_shellcode, has_anti_debug, has_ctypes_windll, has_process_injection,
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

            # has_system_recon
            has_system_recon = 0.0
            recon_patterns = [
                r'platform\.node\(\)', r'socket\.gethostname\(\)', r'os\.uname\(\)',
                r'getpass\.getuser\(\)', r'whoami', r'systeminfo', r'ipconfig',
                r'os\.environ', r'platform\.system\(\)',
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
            tree = ast.parse(code)
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

            # has_cron_pattern
            has_cron_pattern = 0.0
            if re.search(r'crontab|/etc/cron|at\.exe|schtasks|Task Scheduler', code, re.IGNORECASE):
                has_cron_pattern = 1.0

            # has_startup_persistence
            has_startup_persistence = 0.0
            startup_patterns = [
                r'Startup', r'\\CurrentVersion\\Run', r'launchd', r'systemd',
                r'autorun', r'\.bashrc', r'\.profile', r'rc\.local',
            ]
            for pat in startup_patterns:
                if re.search(pat, code):
                    has_startup_persistence = 1.0
                    break

            # has_creates_executable
            has_creates_executable = 0.0
            if re.search(r'\.exe|\.bat|\.cmd|\.vbs|\.ps1|chmod\s+\+x|os\.chmod', code):
                has_creates_executable = 1.0

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

            has_xor_pattern = 0.0
            if re.search(r'\bxor\b|\^=|\bXOR\b', code, re.IGNORECASE):
                has_xor_pattern = 1.0
            # Also check for XOR loop pattern in AST
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.BitXor):
                        has_xor_pattern = 1.0
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
            # has_recursive_traversal: os.walk or pathlib rglob
            has_recursive_traversal = 0.0
            if re.search(r'os\.walk\(|\.rglob\(|glob\.glob.+\*\*|os\.listdir', code):
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
            code_lines = sum(1 for l in lines if l.strip() and not l.strip().startswith('#'))
            blank_lines = sum(1 for l in lines if not l.strip())

            code_to_comment_ratio = float(code_lines) / max(1.0, float(comment_lines))
            blank_line_ratio = float(blank_lines) / max(1.0, total_lines)

            line_lengths = [len(l) for l in lines]
            avg_line_length = sum(line_lengths) / max(1, len(line_lengths))
            max_line_length = float(max(line_lengths)) if line_lengths else 0.0

            return [
                total_lines,
                code_to_comment_ratio,
                avg_line_length,
                max_line_length,
                blank_line_ratio,
            ]
        except Exception:
            return [0.0] * 5
