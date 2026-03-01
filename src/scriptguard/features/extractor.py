"""
Feature extractor for ScriptGuard fusion model.
Extracts a 27-dimensional feature vector from Python source code.

Output vector (FEATURE_DIM=27):
  - 21 continuous/count features: AST structure metrics, import risk counts,
    entropy values, network counts, and statistical measures.
    NOTE: indices 1, 15, 16, 17 are log1p-scaled (n_calls, total_lines,
    avg_line_len, max_line_len) to prevent high-variance length features
    from drowning out behavioural signals.
  - 1 malware_api_score: integer sum of 113 binary indicator flags covering
    dangerous API imports, obfuscation patterns, persistence mechanisms,
    network C2 indicators, crypto operations, filesystem recon, 5 extended
    API flags (sys.gettrace, ctypes.VirtualAlloc, marshal.loads,
    zlib.decompress, platform.uname), 39 gadget/introspection flags,
    and 30 taint/data-flow flags (source->sink variable tracking).
  - 5 targeted features:
      structural_malware_ratio, bitwise_logic_density,
      data_to_logic_ratio, logic_less_payload_volume,
      lolbin_c2_density -- LOLBin/C2-channel/LDAP-recon density.

Sub-methods compute an intermediate 138-feature raw vector; extract() post-
processes it into the final 27-dimensional form by separating continuous
features from binary flags, applying log1p to 4 heavy features, aggregating
the binary flags into malware_api_score, and appending the 5 targeted features.

Raw vector layout (138 features):
  0-12  : _ast_features           (13)
  13-23 : _import_features        (11)
  24-27 : _entropy_features       (4)
  28-38 : _obfuscation_features   (11)
  39-44 : _network_features       (6)
  45-48 : _persistence_features   (4)
  49-52 : _crypto_features        (4)
  53-55 : _recon_fs_features      (3)
  56-62 : _statistical_features   (7)
  63    : _extra_features          (1)
  64-68 : _extended_api_flags     (5)
  69-107: _gadget_features        (39)
  108-137: _taint_features        (30)
"""

import ast
import math
import re
import warnings
from collections import deque, defaultdict
from typing import Optional
from scriptguard.utils.logger import logger


class FeatureExtractor:
    """
    Extracts 27-dimensional feature vector from Python source code.

    Output layout (indices 0-26):
        0-7  : AST counts (tree_depth, n_calls*, n_imports, n_funcdefs,
                           n_classdefs, n_for, n_while, n_try)  *log1p-scaled
        8-9  : Import counts (total_imports, high_risk_imports)
        10-12: Entropy values (mean_str_entropy, max_str_entropy,
                               high_entropy_count)
        13-14: Network counts (unique_ip_count, unique_url_count)
        15-19: Statistical (total_lines*, avg_line_len*, max_line_len*,
                            line_len_cv, long_line_ratio)  *log1p-scaled
        20   : benign_framework_score
        21   : malware_api_score -- sum of 113 binary flags
        22   : structural_malware_ratio
        23   : bitwise_logic_density
        24   : data_to_logic_ratio
        25   : logic_less_payload_volume
        26   : lolbin_c2_density

    Sub-methods produce a 138-feature raw vector post-processed in extract().
    """

    # -------------------------------------------------------------------------
    # Class-level constants
    # -------------------------------------------------------------------------

    _HIGH_RISK_IMPORTS: frozenset = frozenset({
        "socket", "subprocess", "os", "ctypes", "base64",
        "marshal", "pickle", "cryptography", "fernet",
        "mmap", "ptrace", "capstone", "keystone",
        "pynput", "pyperclip", "win32clipboard", "win32api",
        "win32security", "winreg", "_winreg", "win32com",
        "scapy", "impacket", "ldap3", "rdp3",
        "pycryptodome", "Crypto", "nacl",
    })

    _BINARY_INDICES: frozenset = frozenset({
        # AST binary flags
        9, 11, 12,
        # Import binary flags
        13, 14, 15, 16, 17, 18, 19, 20, 21,
        # Entropy
        27,
        # Obfuscation
        28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
        # Network binary
        41, 42, 43, 44,
        # Persistence
        45, 46, 47, 48,
        # Crypto
        49, 50, 51, 52,
        # Recon/FS
        53, 54, 55,
        # Extended API
        64, 65, 66, 67, 68,
        # Gadget flags
        69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
        90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
        100, 101, 102, 103, 104, 105, 106, 107,
        # Taint/data-flow flags
        108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
        118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
        128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
    })

    _CONTINUOUS_INDICES: tuple = (
        0, 1, 2, 3, 4, 5, 6, 7,
        22, 23,
        24, 25, 26,
        39, 40,
        56, 58, 59, 60, 62,
        63,
    )

    _RAW_DIM: int = 138

    _LOG1P_INDICES: tuple[int, ...] = (1, 15, 16, 17)

    FEATURE_DIM: int = len(_CONTINUOUS_INDICES) + 1 + 5  # 27

    # Taint source/sink classification sets
    _DECODE_SOURCES:  frozenset = frozenset({
        "b64decode", "b32decode", "b16decode", "b85decode", "a85decode",
        "unhexlify", "fromhex", "decodebytes",
    })
    _DECOMP_SOURCES:  frozenset = frozenset({
        "decompress", "inflate", "uncompress", "decompressobj",
    })
    _MARSHAL_SOURCES: frozenset = frozenset({"loads"})
    _NETWORK_SOURCES: frozenset = frozenset({
        "recv", "recvfrom", "recvmsg", "read",
        "urlopen", "urlretrieve", "get", "post", "download", "fetch",
    })
    _CODEC_SOURCES:   frozenset = frozenset({
        "decode", "rot_13", "translate", "encode",
    })
    _STRING_SOURCES:  frozenset = frozenset({
        "replace", "join", "format", "split",
        "strip", "lstrip", "rstrip", "upper", "lower",
    })

    _EXEC_SINKS:   frozenset = frozenset({
        "exec", "eval", "compile", "execfile",
    })
    _SHELL_SINKS:  frozenset = frozenset({
        "system", "popen", "call", "check_output", "run", "Popen",
        "execv", "execve", "execvp", "execvpe", "spawn", "spawnl", "spawnle",
    })
    _LOAD_SINKS:   frozenset = frozenset({
        "load_module", "exec_module", "import_module",
        "find_module", "create_module",
    })
    _WRITE_SINKS:  frozenset = frozenset({
        "write", "writelines", "send", "sendall", "sendto", "upload", "put",
    })
    _DESER_SINKS:  frozenset = frozenset({"loads", "load", "fromstring"})

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    def extract(self, code: str) -> list[float]:
        """
        Extract 27-dimensional feature vector from Python source code.

        Internally computes 138 raw features, then:
          - keeps 21 continuous/count features in order
          - sums all 113 binary flags into malware_api_score (index 21)
          - applies math.log1p to 4 heavy features (indices 1, 15, 16, 17)
          - appends 5 targeted features (indices 22-26)

        Returns [0.0] * 27 on any top-level exception.
        """
        try:
            _aliases: dict[str, str] = {}
            _tree: Optional[ast.AST] = None
            try:
                _tree = self._parse_ast(code)
                _aliases = self._build_alias_map(_tree)
            except SyntaxError:
                pass

            raw = (
                self._ast_features(code, _aliases)             # 13  [0-12]
                + self._import_features(code, _aliases)        # 11  [13-23]
                + self._entropy_features(code)                 # 4   [24-27]
                + self._obfuscation_features(code, _aliases)   # 11  [28-38]
                + self._network_features(code)                 # 6   [39-44]
                + self._persistence_features(code)             # 4   [45-48]
                + self._crypto_features(code)                  # 4   [49-52]
                + self._recon_fs_features(code)                # 3   [53-55]
                + self._statistical_features(code)             # 7   [56-62]
                + self._extra_features(code)                   # 1   [63]
                + self._extended_api_flags(code, _aliases)     # 5   [64-68]
                + self._gadget_features(code)                  # 39  [69-107]
                + self._taint_features(code, _tree, _aliases)  # 30  [108-137]
            )

            assert len(raw) == self._RAW_DIM, (
                f"Raw dimension mismatch: expected {self._RAW_DIM}, got {len(raw)}"
            )

            continuous = [raw[i] for i in self._CONTINUOUS_INDICES]
            malware_api_score = float(sum(raw[i] for i in self._BINARY_INDICES))
            features = continuous + [malware_api_score]

            for _idx in self._LOG1P_INDICES:
                features[_idx] = math.log1p(features[_idx])

            features.append(self._structural_malware_ratio(features))   # 22
            features.append(self._bitwise_logic_density(code))          # 23
            features.append(self._data_to_logic_ratio(code))            # 24
            features.append(self._logic_less_payload_volume(code))      # 25
            features.append(self._lolbin_c2_density(code))              # 26

            assert len(features) == self.FEATURE_DIM  # 27

            for _i in range(len(features)):
                _v = features[_i]
                if not math.isfinite(_v):
                    features[_i] = 0.0
                elif _v > 20.0:
                    features[_i] = 20.0

            return features

        except Exception as e:
            logger.warning(f"FeatureExtractor.extract failed: {e}")
            return [0.0] * self.FEATURE_DIM

    # -------------------------------------------------------------------------
    # AST helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_ast(code: str) -> ast.AST:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(code)

    @staticmethod
    def _build_alias_map(tree: ast.AST) -> dict[str, str]:
        """
        Maps local names to canonical qualified names.
          import os as o          -> {"o": "os"}
          from os import system   -> {"system": "os.system"}
        Star imports are skipped.
        """
        aliases: dict[str, str] = {}
        try:
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        local = alias.asname if alias.asname else alias.name
                        aliases[local] = alias.name
                elif isinstance(node, ast.ImportFrom) and node.module:
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        local = alias.asname if alias.asname else alias.name
                        aliases[local] = f"{node.module}.{alias.name}"
        except Exception:
            pass
        return aliases

    # -------------------------------------------------------------------------
    # AST features (13) -- indices 0-12
    # -------------------------------------------------------------------------

    def _ast_features(self, code: str,
                      aliases: dict[str, str] | None = None) -> list[float]:
        aliases = aliases or {}
        try:
            tree = self._parse_ast(code)
        except SyntaxError:
            return [0.0] * 13

        try:
            tree_depth = 0
            q: deque = deque([(tree, 0)])
            while q:
                node, depth = q.popleft()
                if depth > tree_depth:
                    tree_depth = depth
                for child in ast.iter_child_nodes(node):
                    q.append((child, depth + 1))

            nc_call = nc_import = nc_fdef = nc_cdef = 0
            nc_for = nc_while = nc_try = nc_exec = 0

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    nc_call += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    nc_import += 1
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    nc_fdef += 1
                elif isinstance(node, ast.ClassDef):
                    nc_cdef += 1
                elif isinstance(node, ast.For):
                    nc_for += 1
                elif isinstance(node, ast.While):
                    nc_while += 1
                elif isinstance(node, (ast.Try, ast.ExceptHandler)):
                    nc_try += 1
                elif isinstance(node, ast.Expr):
                    if isinstance(getattr(node, 'value', None), ast.Call):
                        func = getattr(node.value, 'func', None)
                        if isinstance(func, ast.Name):
                            if aliases.get(func.id, func.id) == 'exec':
                                nc_exec += 1

            has_nested = 0.0
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for child in ast.walk(node):
                        if child is not node and isinstance(
                            child, (ast.FunctionDef, ast.AsyncFunctionDef)
                        ):
                            has_nested = 1.0
                            break
                if has_nested:
                    break

            exec_depth    = self._compute_exec_eval_depth(tree, aliases)
            decode_chain  = self._has_decode_chain(code, tree)

            has_dyn_import = 0.0
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = getattr(node, 'func', None)
                    if isinstance(func, ast.Name):
                        r = aliases.get(func.id, func.id)
                        if r == '__import__' or r.endswith('.import_module'):
                            has_dyn_import = 1.0
                            break
                    if isinstance(func, ast.Attribute) and \
                            func.attr == 'import_module':
                        has_dyn_import = 1.0
                        break

            return [
                float(tree_depth),
                float(nc_call),
                float(nc_import),
                float(nc_fdef),
                float(nc_cdef),
                float(nc_for),
                float(nc_while),
                float(nc_try),
                float(nc_exec),      # raw[8] dropped
                has_nested,          # raw[9]  binary
                float(exec_depth),   # raw[10] dropped
                decode_chain,        # raw[11] binary
                has_dyn_import,      # raw[12] binary
            ]
        except Exception:
            return [0.0] * 13

    def _compute_exec_eval_depth(self, tree: ast.AST,
                                  aliases: dict[str, str] | None = None) -> float:
        _a = aliases or {}
        _EN = frozenset({"exec", "eval", "compile"})
        max_d = 0

        def _d(node: ast.AST) -> int:
            if not isinstance(node, ast.Call):
                return 0
            func = getattr(node, 'func', None)
            name = _a.get(func.id, func.id) if isinstance(func, ast.Name) else None
            if name in _EN:
                args = getattr(node, 'args', [])
                if args and isinstance(args[0], ast.Call):
                    return 1 + _d(args[0])
                return 1
            return 0

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                d = _d(node)
                if d > max_d:
                    max_d = d
        return float(max_d)

    def _has_decode_chain(self, code: str, tree: ast.AST) -> float:
        if re.search(r'b64decode.+decompress.+exec', code, re.DOTALL):
            return 1.0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                names = self._collect_call_chain_names(node)
                if (any('decode' in n for n in names)
                        and any('decompress' in n or 'inflate' in n
                                for n in names)):
                    return 1.0
        return 0.0

    def _collect_call_chain_names(self, node: ast.Call) -> list[str]:
        names = []
        cur = node
        while isinstance(cur, ast.Call):
            func = getattr(cur, 'func', None)
            if isinstance(func, ast.Attribute):
                names.append(func.attr)
                cur = getattr(func, 'value', None)
                if isinstance(cur, ast.Call):
                    continue
                break
            elif isinstance(func, ast.Name):
                names.append(func.id)
                break
            else:
                break
        return names

    # -------------------------------------------------------------------------
    # Import features (11) -- indices 13-23
    # -------------------------------------------------------------------------

    def _import_features(self, code: str,
                          aliases: dict[str, str] | None = None) -> list[float]:
        aliases = aliases or {}
        try:
            tree = self._parse_ast(code)
        except SyntaxError:
            return [0.0] * 11

        try:
            mods: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        mods.add(a.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    mods.add(node.module.split('.')[0])

            has_socket      = float("socket"      in mods)
            has_subprocess  = float("subprocess"  in mods)
            has_ctypes      = float("ctypes"       in mods)
            has_base64      = float("base64"       in mods)
            has_marshal     = float("marshal"      in mods)
            has_pickle      = float("pickle" in mods or "cPickle" in mods)
            has_cryptography = float("cryptography" in mods)

            has_os_exec = 0.0
            _OE = frozenset((
                'system','popen','execv','execl','execvp',
                'execve','execvpe','spawnl','spawnle','spawnlp',
            ))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = getattr(node, 'func', None)
                    if isinstance(func, ast.Attribute) and func.attr in _OE:
                        val = getattr(func, 'value', None)
                        if isinstance(val, ast.Name):
                            if aliases.get(val.id, val.id) == 'os':
                                has_os_exec = 1.0
                                break
                    elif isinstance(func, ast.Name):
                        r = aliases.get(func.id, '')
                        if r.startswith('os.') and r[3:] in _OE:
                            has_os_exec = 1.0
                            break
            if not has_os_exec and re.search(
                r'os\.(?:system|popen|execv[pe]?|execl[pe]?|spawn)\s*\(', code
            ):
                has_os_exec = 1.0

            has_sock_sub = float("socket" in mods and "subprocess" in mods)
            total_mods   = float(len(mods))
            risk_mods    = float(len(mods & self._HIGH_RISK_IMPORTS))

            return [
                has_socket, has_subprocess, has_os_exec, has_ctypes, has_base64,
                has_marshal, has_pickle, has_cryptography,
                has_sock_sub, total_mods, risk_mods,
            ]
        except Exception:
            return [0.0] * 11

    # -------------------------------------------------------------------------
    # Entropy features (4) -- indices 24-27
    # -------------------------------------------------------------------------

    def _entropy_features(self, code: str) -> list[float]:
        try:
            lits = re.findall(
                r'(?:b?)(?:"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')', code
            )
            entropies = []
            for s in lits:
                inner = s
                for pfx in ('b"', "b'", '"', "'"):
                    if inner.startswith(pfx):
                        inner = inner[len(pfx):]
                        break
                for sfx in ('"', "'"):
                    if inner.endswith(sfx):
                        inner = inner[:-1]
                        break
                if len(inner) < 4:
                    continue
                entropies.append((self._shannon_entropy(inner), len(inner)))

            if not entropies:
                return [0.0, 0.0, 0.0, 0.0]

            evs = [e for e, _ in entropies]
            mean_ent   = sum(evs) / len(evs)
            max_ent    = max(evs)
            high_count = float(sum(1 for e in evs if e > 4.5))
            has_key    = 0.0
            for e, ln in entropies:
                if e > 4.5 and ln in (16, 32, 64):
                    has_key = 1.0
                    break

            return [mean_ent, max_ent, high_count, has_key]
        except Exception:
            return [0.0] * 4

    def _shannon_entropy(self, s: str) -> float:
        if not s:
            return 0.0
        freq: dict[str, int] = {}
        for c in s:
            freq[c] = freq.get(c, 0) + 1
        n = len(s)
        return -sum((cnt / n) * math.log2(cnt / n) for cnt in freq.values())

    # -------------------------------------------------------------------------
    # Obfuscation features (11) -- indices 28-38
    # -------------------------------------------------------------------------

    def _obfuscation_features(self, code: str,
                               aliases: dict[str, str] | None = None) -> list[float]:
        aliases = aliases or {}
        try:
            try:
                tree = self._parse_ast(code)
                ok = True
            except SyntaxError:
                tree = None
                ok = False

            has_exec = has_eval = has_compile = has_dunder = 0.0
            lambda_chains = 0

            if ok and tree:
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        func = getattr(node, 'func', None)
                        if isinstance(func, ast.Name):
                            r = aliases.get(func.id, func.id)
                            if r == 'exec':        has_exec    = 1.0
                            elif r == 'eval':      has_eval    = 1.0
                            elif r == 'compile':   has_compile = 1.0
                            elif r == '__import__': has_dunder = 1.0
                for node in ast.walk(tree):
                    if isinstance(node, ast.Lambda):
                        b, d = node.body, 0
                        while isinstance(b, ast.Lambda):
                            d += 1; b = b.body
                        if d > 0:
                            lambda_chains += 1

            has_b64 = float(bool(re.search(
                r'b64decode|b64encode|base64\.b64'
                r'|\.decode\(["\']base64["\']'
                r'|binascii\.(?:unhexlify|a2b_hex)'
                r'|bytes\.fromhex|codecs\.decode'
                r'|base64\.b(?:32|16|85|a85)decode',
                code,
            )))

            lines = code.splitlines()
            no_comments = float(
                sum(1 for l in lines if l.strip().startswith('#')) == 0
                and len(lines) > 5
            )

            has_enc_exec = 0.0
            if (has_exec or has_eval or has_compile) and has_b64:
                if re.search(
                    r'(?:exec|eval|compile)\s*\([^)]*'
                    r'(?:b64decode|unhexlify|fromhex|decode)',
                    code, re.DOTALL,
                ) or re.search(
                    r'(?:b64decode|unhexlify|fromhex)[^;)]*(?:exec|eval)',
                    code, re.DOTALL,
                ) or re.search(
                    r'(?:exec|eval)\s*\(\s*\w+\s*\(\s*\w+\s*\(', code
                ):
                    has_enc_exec = 1.0

            line_lens = [len(l) for l in lines]
            has_long = float(max(line_lens) > 500 if line_lens else False)

            has_anti_debug = 0.0
            for pat in [
                r'IsDebuggerPresent', r'QueryPerformanceCounter',
                r'CheckRemoteDebuggerPresent', r'NtQueryInformationProcess',
                r'ptrace', r'PTRACE_TRACEME',
                r'sys\.gettrace\s*\(\)',
                r'ctypes\.windll\.kernel32\.IsDebuggerPresent',
            ]:
                if re.search(pat, code):
                    has_anti_debug = 1.0
                    break

            has_windll = float(bool(re.search(
                r'windll\.(?:kernel32|ntdll|advapi32|user32|shell32)\b', code
            )))

            has_injection = 0.0
            for pat in [
                r'VirtualAlloc', r'WriteProcessMemory', r'CreateRemoteThread',
                r'NtWriteVirtualMemory', r'RtlMoveMemory', r'OpenProcess',
                r'NtCreateThreadEx', r'RtlCreateUserThread',
                r'SetThreadContext', r'GetThreadContext',
            ]:
                if re.search(pat, code):
                    has_injection = 1.0
                    break

            return [
                has_exec, has_eval, has_compile, has_dunder,
                has_b64, no_comments, has_enc_exec,
                has_long, has_anti_debug, has_windll, has_injection,
            ]
        except Exception:
            return [0.0] * 11

    def _detect_shellcode(self, code: str, tree: ast.AST) -> float:
        for arr_str in re.findall(
            r'(?:bytearray|bytes)\s*\(\s*\[([^\]]+)\]', code
        ):
            parts = [p.strip() for p in arr_str.split(',') if p.strip()]
            if len(parts) > 100:
                try:
                    vals = [int(p, 0) for p in parts if p]
                    if len(vals) > 100:
                        freq: dict[int, int] = {}
                        for v in vals:
                            freq[v] = freq.get(v, 0) + 1
                        n = len(vals)
                        ent = -sum(
                            (c/n)*math.log2(c/n) for c in freq.values()
                        )
                        if ent > 5.5:
                            return 1.0
                except Exception:
                    pass
        if re.findall(r'(?:\\x[0-9a-fA-F]{2}){100,}', code):
            return 1.0
        return 0.0

    # -------------------------------------------------------------------------
    # Network / C2 features (6) -- indices 39-44
    # -------------------------------------------------------------------------

    def _network_features(self, code: str) -> list[float]:
        try:
            ips  = re.findall(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', code)
            urls = re.findall(r'https?://[^\s\'"]+', code)

            has_ports = float(bool(re.findall(
                r'\b(4444|1337|8080|8888|9999|31337|6667|1234|5555|7777'
                r'|4242|1111|2222|6666|12345|54321|65535|1024)\b', code
            )))

            has_c2 = self._detect_c2_pattern(code)

            has_dns = float(bool(re.search(
                r'socket\.gethostbyname|dns\.resolver|getaddrinfo'
                r'|nslookup|socket\.getfqdn|dnspython',
                code
            )))

            has_recon = 0.0
            for pat in [
                r'platform\.node\(\)', r'socket\.gethostname\(\)',
                r'os\.uname\(\)', r'getpass\.getuser\(\)',
                r'whoami', r'systeminfo', r'ipconfig',
                r'net\s+user', r'netifaces\.', r'psutil\.net_if',
                r'ifconfig', r'arp\s+-a', r'netstat\s+-',
                r'hostname\s*\(\)',
            ]:
                if re.search(pat, code):
                    has_recon = 1.0
                    break

            return [
                float(len(set(ips))), float(len(set(urls))),
                has_ports, has_c2, has_dns, has_recon,
            ]
        except Exception:
            return [0.0] * 6

    _WS_LIBS: frozenset = frozenset({
        'websockets', 'aiohttp', 'httpx', 'trio', 'anyio',
        'aiofiles', 'tornado',
    })

    def _detect_c2_pattern(self, code: str) -> float:
        try:
            tree = self._parse_ast(code)
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        imported.add(a.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split('.')[0])
            if imported & self._WS_LIBS:
                return 0.0
        except SyntaxError:
            if (re.search(r'\.connect\(', code)
                    and re.search(r'\.(send|recv|sendall|recvfrom)\(', code)):
                return 1.0
            return 0.0

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = []
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func = getattr(child, 'func', None)
                    if isinstance(func, ast.Attribute):
                        calls.append(func.attr)
            if (any(c == 'connect' for c in calls)
                    and any(c in ('send','recv','sendall','recvfrom','sendto')
                            for c in calls)):
                return 1.0
        return 0.0

    # -------------------------------------------------------------------------
    # Persistence features (4) -- indices 45-48
    # -------------------------------------------------------------------------

    def _persistence_features(self, code: str) -> list[float]:
        try:
            has_reg = float(bool(re.search(
                r'winreg|_winreg|RegSetValue|HKEY_|OpenKey|CreateKey'
                r'|SetValueEx|RegCreateKey|RegOpenKey', code
            )))

            has_cron = 0.0
            for pat in [
                r'schtasks\s+/[Cc]reate', r'at\.exe\s+\d',
                r'SchTasks\.exe', r'open\(["\'][/\\]etc[/\\]cron',
                r'crontab\s+-[el]\s', r'TaskScheduler|ITaskScheduler',
                r'crontab\s+-l', r'@reboot',
                r'/etc/cron\.(d|daily|hourly|weekly|monthly)',
                r'launchctl.*plist|plist.*launchctl',
            ]:
                if re.search(pat, code, re.IGNORECASE):
                    has_cron = 1.0
                    break

            has_startup = 0.0
            for pat in [
                r'\\CurrentVersion\\Run(?:Once)?["\']',
                r'HKCU.*\\Run|HKLM.*\\Run',
                r'nssm\s+install',
                r'sc\.exe\s+create|CreateService\s*\(',
                r'open\(["\'][/\\]etc[/\\](?:init\.d|rc\.local)',
                r'(?:plist|launchd).*write|write.*plist',
                r'APPDATA.*startup|startup.*\.lnk',
                r'systemctl\s+enable',
                r'update-rc\.d', r'chkconfig\s+--add',
            ]:
                if re.search(pat, code, re.IGNORECASE):
                    has_startup = 1.0
                    break

            has_drop = 0.0
            for pat in [
                r'chmod\s+\+x',
                r'os\.chmod\([^,]+,\s*(?:0[oO]?7[0-7]{2}|0o?755|0o?777)',
                r'open\([^)]*\.(exe|bat|cmd|vbs|ps1|sh)[^)]*["\'],\s*["\'][wa]b?',
                r'with\s+open\([^)]*\.(exe|bat|cmd|vbs|ps1|sh)',
                r'HKEY_[A-Z_]+.*(?:\\\\Run|\\\\RunOnce)',
                r'subprocess.*["\'][^\'"]*\.(exe|bat|cmd)["\']',
                r'urlretrieve.*\.exe|download.*\.exe',
                r'write.*shellcode|shellcode.*write',
            ]:
                if re.search(pat, code, re.IGNORECASE):
                    has_drop = 1.0
                    break

            return [has_reg, has_cron, has_startup, has_drop]
        except Exception:
            return [0.0] * 4

    # -------------------------------------------------------------------------
    # Cryptography features (4) -- indices 49-52
    # -------------------------------------------------------------------------

    def _crypto_features(self, code: str) -> list[float]:
        try:
            has_aes = float(bool(re.search(
                r'AES|Cipher\.new|AESCipher|aes\.encrypt|aes\.decrypt'
                r'|AES\.MODE_|AES\.new|pyaes\.', code
            )))

            has_rc4 = float(bool(re.search(
                r'RC4|arc4|Salsa20|ChaCha|rc4_encrypt|ChaCha20',
                code, re.IGNORECASE
            )))

            has_xor = 0.0
            if re.search(r'\bxor\b|\bXOR\b', code):
                has_xor = 1.0
            if not has_xor:
                try:
                    tree = self._parse_ast(code)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.For):
                            for child in ast.walk(node):
                                if (isinstance(child, ast.AugAssign)
                                        and isinstance(child.op, ast.BitXor)
                                        and isinstance(child.target,
                                                       ast.Subscript)):
                                    has_xor = 1.0
                                    break
                        if has_xor:
                            break
                    if not has_xor:
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ListComp):
                                src = ast.unparse(node)
                                if '^' in src and 'chr' in src:
                                    has_xor = 1.0
                                    break
                except SyntaxError:
                    pass

            has_fernet = float(bool(
                re.search(r'Fernet|fernet\.encrypt|fernet\.decrypt', code)
            ))

            return [has_aes, has_rc4, has_xor, has_fernet]
        except Exception:
            return [0.0] * 4

    # -------------------------------------------------------------------------
    # Recon / Filesystem features (3) -- indices 53-55
    # -------------------------------------------------------------------------

    def _recon_fs_features(self, code: str) -> list[float]:
        try:
            has_walk = float(bool(
                re.search(r'os\.walk\(|\.rglob\(|glob\.glob[^)]*\*\*', code)
            ))

            has_mass = 0.0
            ext_re = r'\.(doc|docx|xls|xlsx|pdf|jpg|png|mp4|zip|rar|txt|csv|db|sql|pst|ost)'
            if len(set(re.findall(ext_re, code, re.I))) >= 3:
                has_mass = 1.0
            if re.search(
                r'for\s+\w+\s+in\s+.*\.walk|encrypt.*\.open|open.*encrypt',
                code, re.DOTALL
            ):
                has_mass = 1.0

            has_shadow = float(bool(re.search(
                r'vssadmin|shadow\s+copy|wmic.*shadowcopy'
                r'|Win32_ShadowCopy|bcdedit',
                code, re.IGNORECASE
            )))

            return [has_walk, has_mass, has_shadow]
        except Exception:
            return [0.0] * 3

    # -------------------------------------------------------------------------
    # Statistical features (7) -- indices 56-62
    # -------------------------------------------------------------------------

    def _statistical_features(self, code: str) -> list[float]:
        try:
            lines = code.splitlines()
            n = float(len(lines))
            if n == 0:
                return [0.0] * 7

            comment_density = sum(
                1 for l in lines if l.strip().startswith('#')
            ) / n

            ll = [len(l) for l in lines]
            avg_ll = sum(ll) / max(1, len(ll))
            max_ll = float(max(ll)) if ll else 0.0

            if len(ll) > 1:
                var = sum((x - avg_ll) ** 2 for x in ll) / len(ll)
                cv  = math.sqrt(var) / max(1.0, avg_ll)
            else:
                cv = 0.0

            slits = re.findall(
                r'(?:b?)(?:"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')', code
            )
            max_slit = 0.0
            for s in slits:
                stripped = s[1:] if s.startswith("b") else s
                cl = float(len(stripped) - 2)
                if cl > max_slit:
                    max_slit = cl

            long_ratio = sum(1 for x in ll if x > 500) / n

            return [n, comment_density, avg_ll, max_ll, cv, max_slit, long_ratio]
        except Exception:
            return [0.0] * 7

    # -------------------------------------------------------------------------
    # Extra features (1) -- index 63
    # -------------------------------------------------------------------------

    def _extra_features(self, code: str) -> list[float]:
        try:
            tree = self._parse_ast(code)
            ok = True
        except SyntaxError:
            tree = None
            ok = False

        _HARD = frozenset({
            "django", "flask", "fastapi", "sqlalchemy", "celery", "pydantic",
            "aiohttp", "starlette", "tornado", "sanic",
        })
        _EASY = frozenset({
            "logging", "typing", "argparse", "unittest", "pytest",
            "setuptools", "click", "typer", "rich", "textual",
            "structlog", "loguru",
        })
        score = 0.0
        if ok and tree:
            try:
                mods: set[str] = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for a in node.names:
                            mods.add(a.name.split(".")[0])
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        mods.add(node.module.split(".")[0])
                score = (1.0 * sum(1 for m in mods if m in _HARD)
                         + 0.2 * sum(1 for m in mods if m in _EASY))
            except Exception:
                pass
        return [score]

    # -------------------------------------------------------------------------
    # Extended API flags (5) -- indices 64-68
    # -------------------------------------------------------------------------

    def _extended_api_flags(self, code: str,
                             aliases: dict[str, str] | None = None) -> list[float]:
        aliases = aliases or {}
        try:
            # 64: sys.gettrace -- anti-debug
            has_trace = 0.0
            if re.search(r'\bsys\.gettrace\s*\(\)', code):
                has_trace = 1.0
            else:
                try:
                    tree = self._parse_ast(code)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            func = getattr(node, 'func', None)
                            if isinstance(func, ast.Attribute) and \
                                    func.attr == 'gettrace':
                                val = getattr(func, 'value', None)
                                if isinstance(val, ast.Name):
                                    if aliases.get(val.id, val.id) == 'sys':
                                        has_trace = 1.0
                                        break
                except SyntaxError:
                    pass

            # 65: VirtualAlloc
            has_valloc = float(bool(re.search(r'\bVirtualAlloc\b', code)))

            # 66: marshal.loads
            has_marshal = 0.0
            if re.search(r'\bmarshal\.loads\s*\(', code):
                has_marshal = 1.0
            else:
                try:
                    tree = self._parse_ast(code)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            func = getattr(node, 'func', None)
                            if isinstance(func, ast.Attribute) and \
                                    func.attr == 'loads':
                                val = getattr(func, 'value', None)
                                if isinstance(val, ast.Name):
                                    if aliases.get(val.id, val.id) == 'marshal':
                                        has_marshal = 1.0
                                        break
                except SyntaxError:
                    pass

            # 67: zlib/gzip/bz2/lzma decompress
            has_decomp = float(bool(re.search(
                r'(?:zlib|gzip|bz2|lzma)\.decompress\s*\(', code, re.I
            )))

            # 68: platform.uname -- system fingerprint
            has_uname = 0.0
            if re.search(r'\bplatform\.uname\s*\(\)', code):
                has_uname = 1.0
            else:
                try:
                    tree = self._parse_ast(code)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            func = getattr(node, 'func', None)
                            if isinstance(func, ast.Attribute) and \
                                    func.attr == 'uname':
                                val = getattr(func, 'value', None)
                                if isinstance(val, ast.Name):
                                    if aliases.get(val.id, val.id) == 'platform':
                                        has_uname = 1.0
                                        break
                except SyntaxError:
                    pass

            return [has_trace, has_valloc, has_marshal, has_decomp, has_uname]
        except Exception:
            return [0.0] * 5

    # -------------------------------------------------------------------------
    # Gadget / introspection flags (39) -- indices 69-107
    # -------------------------------------------------------------------------

    def _gadget_features(self, code: str) -> list[float]:
        """
        39 binary behavioral flags covering Python sandbox-escape primitives,
        credential/secret theft, container/cloud escapes, OS-level recon,
        persistence, self-replication, advanced module loaders, unsafe
        deserialization, memory manipulation, stealthy exec patterns,
        deferred execution, and C2 communication channels.
        """
        try:
            def hit(pattern: str, flags: int = 0) -> float:
                return float(bool(re.search(pattern, code, flags)))

            I = re.IGNORECASE

            return [
                # ── Namespace & builtins (69-70) ──────────────────────────
                hit(r'__builtins__|globals\s*\(\s*\)\s*\['),
                hit(r'__builtins__\s*\[.*\]\s*=|vars\s*\(\s*__builtins__\s*\)'),

                # ── MRO / inheritance chains (71-72) ─────────────────────
                hit(r'__subclasses__|__mro__|__bases__'),
                hit(r'__subclasses__\s*\(\s*\).*catch_warnings'
                    r'|catch_warnings.*__subclasses__', re.DOTALL),

                # ── Exception hiding (73) ─────────────────────────────────
                hit(r'contextlib\.suppress.*exec'
                    r'|except\s*[:\(].*\bpass\b.*exec', re.DOTALL),

                # ── Credential & secret theft (74-79) ────────────────────
                hit(r'AuthenticationException|ftplib\.FTP'
                    r'|paramiko.*connect.*password', I),
                hit(r'\.git-credentials|known_hosts|id_rsa'
                    r'|authorized_keys|\.netrc\b', I),
                hit(r'os\.environ.*(?:KEY|TOKEN|SECRET|PASSWORD|PASS)'
                    r'|SECRET_PATTERNS|harvest_env'
                    r'|AWS_ACCESS_KEY|AWS_SECRET', I),
                hit(r'169\.254\.169\.254|latest/meta-data'
                    r'|iam/security-credentials', I),
                hit(r'\.env\b.*rglob|rglob.*\.env\b'
                    r'|harvest.*env_files|find.*\.env', I),
                hit(r'ghp_[0-9a-zA-Z]{10}|gho_[0-9a-zA-Z]{10}'
                    r'|xox[baprs]-|sk_live_|AKIA[0-9A-Z]{10}', I),

                # ── Container / cloud escapes (80-81) ────────────────────
                hit(r'docker\.sock|/var/run/docker\.sock'),
                hit(r'kubernetes\.io/serviceaccount'
                    r'|/var/run/secrets/kubernetes'),

                # ── OS-level recon (82-84) ────────────────────────────────
                hit(r'/proc/net/|/proc/self/|/proc/\d+/mem'),
                hit(r'/etc/passwd|/etc/shadow|/etc/sudoers'),
                hit(r'sudo\s+-l|uname\s+-a|id\s*&&\s*uname'
                    r'|GTFOBins|gtfobins', I),

                # ── Persistence (85-89) ───────────────────────────────────
                hit(r'crontab|schtasks.*ONLOGON|/etc/cron\.|at\s+now', I),
                hit(r'HKCU|HKLM|CurrentVersion\\\\Run|winreg\.SetValueEx', I),
                hit(r'win32com\.client|__EventFilter'
                    r'|CommandLineEventConsumer'
                    r'|WQL.*InstanceModification', I),
                hit(r'LD_PRELOAD|__attribute__.*constructor'
                    r'|dlopen|/etc/ld\.so'),
                hit(r'authorized_keys|ssh-rsa.*attacker'
                    r'|inject.*ssh_key', I),

                # ── Self-replication / fileless (90-91) ──────────────────
                hit(r'open\s*\(\s*__file__\s*\)'
                    r'|open\s*\(\s*sys\.argv\s*\[\s*0\s*\]\s*\)'),
                hit(r'APPDATA|TEMP.*svchost|update_helper'
                    r'|WindowsUpdate|msupdate', I),

                # ── Advanced loaders (92-94) ──────────────────────────────
                hit(r'spec_from_loader|exec_module'
                    r'|types\.ModuleType\s*\('),
                hit(r'SourceFileLoader|zipimport\.zipimporter'),
                hit(r'__import__\s*\([^)]+\)\s*\.\s*\w+'
                    r'|getattr\s*\(\s*__import__'),

                # ── Unsafe deserialization (95-96) ────────────────────────
                hit(r'pickle\.loads|marshal\.loads'
                    r'|exec\s*\(\s*marshal\.loads'),
                hit(r'copyreg\.dispatch_table'
                    r'|def\s+__reduce__\s*\(self\)'),

                # ── Memory manipulation (97-99) ───────────────────────────
                hit(r'WriteProcessMemory|VirtualAllocEx'
                    r'|CreateRemoteThread|OpenProcess'),
                hit(r'mmap\.PROT_EXEC|MAP_ANONYMOUS.*PROT_EXEC'
                    r'|PROT_WRITE\s*\|\s*PROT_EXEC'),
                hit(r'ctypes\.CFUNCTYPE|ctypes\.cast.*CFUNCTYPE'
                    r'|from_buffer.*cast'),

                # ── Stealthy exec (100-102) ───────────────────────────────
                hit(r'exec\s*\(\s*compile\s*\('),
                hit(r'eval\s*\(\s*compile\s*\('),
                hit(r'__builtins__\s*\[\s*[\'"]exec[\'"]\s*\]'
                    r'|getattr\s*\(__builtins__.*[\'\"](exec|eval)[\'\"]\)'),

                # ── Deferred / hidden execution (103-104) ─────────────────
                hit(r'def\s+__init_subclass__|def\s+__del__\s*\(self\)'
                    r'|def\s+__format__\s*\(self.*\)\s*->'),
                hit(r'atexit\.register|gc\.callbacks\.append'
                    r'|weakref\.ref\s*\(.*,\s*\w'),

                # ── C2 communication (105-107) ────────────────────────────
                hit(r'api\.telegram\.org|discord\.Client'
                    r'|hooks\.slack\.com', I),
                hit(r'gethostbyname.*base3[26]|exfil.*dns|dns.*exfil', I),
                hit(r'while\s+True.*requests\.post.*sleep'
                    r'|beacon.*C2|jitter.*sleep', re.DOTALL),
            ]
        except Exception:
            return [0.0] * 39

    # -------------------------------------------------------------------------
    # Taint / data-flow features (30) -- indices 108-137
    #
    # Lightweight intra-procedural taint analysis on the AST detecting
    # source->sink variable flows that regex alone cannot catch, e.g.:
    #
    #   x = base64.b64decode(blob)   <- tainted source
    #   y = zlib.decompress(x)       <- taint propagation
    #   exec(y)                      <- sink DETECTED (flag 124)
    #
    # Flags:
    #   108: decode  -> exec/eval
    #   109: decode  -> shell
    #   110: decode  -> load_module
    #   111: decomp  -> exec/eval
    #   112: decomp  -> shell
    #   113: network -> exec/eval
    #   114: network -> shell
    #   115: network -> load_module
    #   116: marshal/pickle -> exec/eval
    #   117: marshal/pickle -> shell
    #   118: codec   -> exec/eval
    #   119: codec   -> shell
    #   120: string_mangling -> exec/eval
    #   121: string_mangling -> shell
    #   122: multi-step chain (>=3 hops source->transform->sink)
    #   123: tainted var reused across multiple functions
    #   124: decode + decompress -> exec (double-layer obfuscation)
    #   125: bytes/bytearray inline literal -> exec
    #   126: chr() list-comp/map -> exec (int-array payload)
    #   127: f-string / format-string -> exec
    #   128: getattr on __import__/importlib used as indirect call
    #   129: tainted data written to tmp file then exec'd (dropper)
    #   130: tainted string passed to compile() then exec
    #   131: tainted bytes to ctypes/mmap (shellcode injection)
    #   132: os.environ used as payload carrier -> exec
    #   133: linecache / tokenize reconstruct payload -> exec
    #   134: weakref / atexit / gc callback with exec
    #   135: __init__ or __new__ contains tainted exec
    #   136: decorator or metaclass with tainted exec
    #   137: summary flag -- any taint flow detected
    # -------------------------------------------------------------------------

    def _taint_features(
        self,
        code: str,
        tree: Optional[ast.AST],
        aliases: dict[str, str] | None = None,
    ) -> list[float]:
        results = [0.0] * 30

        # ── Fallback: regex approximations when AST unavailable ──────────
        # results is a 30-element list (local indices 0-29).
        # Local index = global index - 108.
        if tree is None:
            results[0]  = float(bool(re.search(        # global 108: decode -> exec
                r'(?:b64decode|unhexlify|fromhex)[^\n]{0,200}exec',
                code, re.DOTALL
            )))
            results[5]  = float(bool(re.search(        # global 113: network -> exec
                r'(?:urlopen|recv|requests\.get)[^\n]{0,200}exec',
                code, re.DOTALL
            )))
            results[14] = float(bool(re.search(        # global 122: multi-step chain
                r'(?:b64decode|unhexlify)[^\n]{0,100}'
                r'(?:decompress|zlib)[^\n]{0,100}exec',
                code, re.DOTALL
            )))
            results[29] = float(any(results))          # global 137: summary flag
            return results

        aliases = aliases or {}

        try:
            # ── Step 1: build assign_map and taint_vars ───────────────────
            # assign_map[name] = set of callee-attr names that produced it
            assign_map: dict[str, set[str]] = defaultdict(set)
            taint_vars: set[str] = set()

            _ALL_SOURCES = (
                self._DECODE_SOURCES | self._DECOMP_SOURCES
                | self._NETWORK_SOURCES | self._CODEC_SOURCES
                | self._MARSHAL_SOURCES
            )

            def _callee(call: ast.Call) -> str:
                func = getattr(call, 'func', None)
                if isinstance(func, ast.Name):
                    return aliases.get(func.id, func.id)
                if isinstance(func, ast.Attribute):
                    return func.attr
                return ""

            def _node_tainted(node: ast.expr) -> bool:
                if isinstance(node, ast.Name):
                    return node.id in taint_vars
                if isinstance(node, ast.Call):
                    cn = _callee(node)
                    if cn in _ALL_SOURCES:
                        return True
                    return any(_node_tainted(a)
                               for a in getattr(node, 'args', []))
                if isinstance(node, ast.JoinedStr):
                    return any(_node_tainted(v)
                               for v in getattr(node, 'values', []))
                if isinstance(node, ast.BinOp):
                    return _node_tainted(node.left) or _node_tainted(node.right)
                if isinstance(node, ast.IfExp):
                    return (_node_tainted(node.body)
                            or _node_tainted(node.orelse))
                if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
                    return any(_node_tainted(e)
                               for e in getattr(node, 'elts', []))
                return False

            for stmt in ast.walk(tree):
                if isinstance(stmt, ast.Assign):
                    if not isinstance(stmt.value, ast.Call):
                        continue
                    cn = _callee(stmt.value)
                    for t in stmt.targets:
                        if isinstance(t, ast.Name):
                            assign_map[t.id].add(cn)
                            if cn in _ALL_SOURCES:
                                taint_vars.add(t.id)
                elif isinstance(stmt, ast.AnnAssign):
                    if stmt.value and isinstance(stmt.value, ast.Call):
                        cn = _callee(stmt.value)
                        if isinstance(stmt.target, ast.Name):
                            assign_map[stmt.target.id].add(cn)
                            if cn in _ALL_SOURCES:
                                taint_vars.add(stmt.target.id)
                # Augmented assignment can also propagate taint
                elif isinstance(stmt, ast.AugAssign):
                    if isinstance(stmt.value, ast.Call):
                        cn = _callee(stmt.value)
                        if isinstance(stmt.target, ast.Name):
                            if cn in _ALL_SOURCES or \
                                    stmt.target.id in taint_vars:
                                taint_vars.add(stmt.target.id)

            # ── Step 2: sink detection ────────────────────────────────────
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                sink = _callee(node)
                all_args = (getattr(node, 'args', [])
                            + [kw.value for kw in
                               getattr(node, 'keywords', [])])
                if not any(_node_tainted(a) for a in all_args):
                    continue

                def _src_cats() -> set[str]:
                    cats: set[str] = set()
                    for a in all_args:
                        if isinstance(a, ast.Name) and a.id in taint_vars:
                            p = assign_map.get(a.id, set())
                            if p & self._DECODE_SOURCES:  cats.add('decode')
                            if p & self._DECOMP_SOURCES:  cats.add('decomp')
                            if p & self._NETWORK_SOURCES: cats.add('network')
                            if p & self._MARSHAL_SOURCES: cats.add('marshal')
                            if p & self._CODEC_SOURCES:   cats.add('codec')
                            if p & self._STRING_SOURCES:  cats.add('string')
                        elif isinstance(a, ast.Call):
                            cn2 = _callee(a)
                            if cn2 in self._DECODE_SOURCES:  cats.add('decode')
                            if cn2 in self._DECOMP_SOURCES:  cats.add('decomp')
                            if cn2 in self._NETWORK_SOURCES: cats.add('network')
                            if cn2 in self._MARSHAL_SOURCES: cats.add('marshal')
                            if cn2 in self._CODEC_SOURCES:   cats.add('codec')
                    return cats

                sc = _src_cats()

                if sink in self._EXEC_SINKS:
                    if 'decode'  in sc: results[0] = 1.0
                    if 'decomp'  in sc: results[3] = 1.0
                    if 'network' in sc: results[5] = 1.0
                    if 'marshal' in sc: results[8] = 1.0
                    if 'codec'   in sc: results[10] = 1.0
                    if 'string'  in sc: results[12] = 1.0
                    results[29] = 1.0

                if sink in self._SHELL_SINKS:
                    if 'decode'  in sc: results[1] = 1.0
                    if 'decomp'  in sc: results[4] = 1.0
                    if 'network' in sc: results[6] = 1.0
                    if 'marshal' in sc: results[9] = 1.0
                    if 'codec'   in sc: results[11] = 1.0
                    if 'string'  in sc: results[13] = 1.0
                    results[29] = 1.0

                if sink in self._LOAD_SINKS:
                    if 'decode'  in sc: results[2] = 1.0
                    if 'network' in sc: results[7] = 1.0
                    results[29] = 1.0

                if sink == 'compile' and any(
                    _node_tainted(a) for a in all_args
                ):
                    results[22] = 1.0
                    results[29] = 1.0

            # ── Step 3: pattern-specific flags ────────────────────────────

            # 122: multi-step chain
            if (results[0] or results[1]) and (results[3] or results[4]):
                results[14] = 1.0
            elif re.search(
                r'(?:b64decode|unhexlify|fromhex)[^\n]{0,200}'
                r'(?:decompress|zlib)[^\n]{0,200}exec',
                code, re.DOTALL
            ):
                results[14] = 1.0

            # 123: taint var in multiple functions
            fns = [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if len(fns) >= 2 and taint_vars:
                seen: dict[str, set[int]] = defaultdict(set)
                for i, fn in enumerate(fns):
                    for nd in ast.walk(fn):
                        if isinstance(nd, ast.Name) and nd.id in taint_vars:
                            seen[nd.id].add(i)
                if any(len(v) >= 2 for v in seen.values()):
                    results[15] = 1.0

            # 124: decode + decompress -> exec (double-layer)
            if (results[0] or results[1]) and (results[3] or results[4]):
                results[16] = 1.0

            # 125: bytes/bytearray inline -> exec
            if (re.search(r'(?:bytes|bytearray)\s*\(\s*\[', code)
                    and re.search(r'\bexec\b|\beval\b', code)):
                results[17] = 1.0

            # 126: chr() list-comp/map -> exec
            if (re.search(r'chr\s*\(', code)
                    and re.search(r'(?:map|join|exec|eval)', code)):
                results[18] = 1.0

            # 127: f-string / format -> exec
            if (re.search(r'(?:f["\']|["\'].*%s)', code)
                    and re.search(r'\bexec\b|\beval\b', code)):
                results[19] = 1.0

            # 128: getattr on __import__ as indirect call
            if re.search(
                r'getattr\s*\(\s*(?:__import__|importlib\.import_module)', code
            ):
                results[20] = 1.0

            # 129: tainted data to tmp file then exec (dropper pattern)
            if (re.search(r'tempfile\.|NamedTemporaryFile|/tmp/', code)
                    and re.search(
                        r'\.write\s*\(|os\.chmod|subprocess|Popen',
                        code, re.DOTALL
                    )
                    and taint_vars):
                results[21] = 1.0

            # 131: tainted bytes to ctypes/mmap
            if taint_vars and re.search(
                r'ctypes|mmap|PROT_EXEC|VirtualAlloc', code
            ):
                results[23] = 1.0

            # 132: env variable as exec carrier
            if (re.search(r'os\.environ\s*\[', code)
                    and re.search(r'\bexec\b|\beval\b', code)):
                results[24] = 1.0

            # 133: linecache/tokenize + exec
            if (re.search(r'linecache\.|tokenize\.', code)
                    and re.search(r'\bexec\b|\beval\b', code)):
                results[25] = 1.0

            # 134: weakref/atexit/gc.callbacks + exec
            if (re.search(r'weakref\.|atexit\.|gc\.callbacks', code)
                    and re.search(r'\bexec\b|\beval\b', code)):
                results[26] = 1.0

            # 135: __init__/__new__ with tainted exec
            if (re.search(
                    r'def\s+__(?:init|new)__[^:]*:.*\bexec\b',
                    code, re.DOTALL
                ) and taint_vars):
                results[27] = 1.0

            # 136: decorator/metaclass with tainted exec
            if (re.search(r'@\w+|metaclass\s*=', code)
                    and re.search(r'\bexec\b|\beval\b', code)
                    and taint_vars):
                results[28] = 1.0

            if any(results[:29]):
                results[29] = 1.0

            return results

        except Exception:
            return [0.0] * 30

    # -------------------------------------------------------------------------
    # Targeted features (appended at positions 22-26 of final vector)
    # -------------------------------------------------------------------------

    def _structural_malware_ratio(self, features: list[float]) -> float:
        """
        ln(n_calls) / (1 + ln(n_funcdefs + n_classdefs))
        features[1] already log1p-scaled; features[3], [4] raw counts.
        """
        try:
            return features[1] / (
                1.0 + math.log1p(features[3] + features[4])
            )
        except Exception:
            return 0.0

    _BITWISE_OP_TYPES: tuple = (
        ast.BitXor, ast.BitAnd, ast.BitOr, ast.LShift, ast.RShift,
    )

    def _bitwise_logic_density(self, code: str) -> float:
        """bitwise_ops / log1p(total_chars)"""
        try:
            tc = len(code)
            if tc == 0:
                return 0.0
            try:
                tree = self._parse_ast(code)
                count = sum(
                    1 for n in ast.walk(tree)
                    if isinstance(n, ast.BinOp)
                    and isinstance(n.op, self._BITWISE_OP_TYPES)
                )
            except SyntaxError:
                count = len(re.findall(r'<<|>>|\^|[&|]', code))
            return count / math.log1p(tc)
        except Exception:
            return 0.0

    def _data_to_logic_ratio(self, code: str) -> float:
        """(Expr+Assign) / (1+If+For+While+Try)"""
        _D = (ast.Expr, ast.Assign)
        _L = (ast.If, ast.For, ast.While, ast.Try)
        try:
            try:
                tree = self._parse_ast(code)
            except SyntaxError:
                return 0.0
            nd = sum(1 for n in ast.walk(tree) if isinstance(n, _D))
            nl = sum(1 for n in ast.walk(tree) if isinstance(n, _L))
            return nd / (1.0 + nl)
        except Exception:
            return 0.0

    def _logic_less_payload_volume(self, code: str) -> float:
        """log1p(sum_literal_lengths) / log1p(n_ast_nodes)"""
        try:
            try:
                tree = self._parse_ast(code)
            except SyntaxError:
                return 0.0
            nodes = list(ast.walk(tree))
            denom = math.log1p(len(nodes))
            if denom == 0.0:
                return 0.0
            s = sum(
                len(n.value) for n in nodes
                if isinstance(n, ast.Constant)
                and isinstance(n.value, (str, bytes))
            )
            return math.log1p(s) / denom
        except Exception:
            return 0.0

    _LOLBIN_RE: re.Pattern = re.compile(
        r'\b(certutil|bitsadmin|mshta|msiexec|regsvr32|rundll32|wmic'
        r'|cscript|wscript|installutil|schtasks|regasm|regsvcs|odbcconf'
        r'|msbuild|dnscmd|netsh|appsync|pcalua|forfiles|syncappvpublishing'
        r'|msdeploy|ieexec|cmstp|xwizard|ftp\.exe|esentutl|expand\.exe'
        r'|extrac32|findstr|hh\.exe|makecab|nltest|pcwrun|replace'
        r'|rpcping|runscripthelper|sfc|stordiag|ttdinject|tttracer'
        r'|vbc\.exe|wab\.exe|winrm|wsreset|xpsrchvw|zipfldr)\b'
        r'|api\.telegram\.org'
        r'|discord\.com/api/webhooks'
        r'|\bldap3\b|\bldap://'
        r'|\bsAMAccountName\b'
        r'|\bpyperclip\b|\bwin32clipboard\b'
        r'|\bpynput\b|\brdp3\b'
        r'|\bimpacket\b|\bscapy\b',
        re.IGNORECASE,
    )

    def _lolbin_c2_density(self, code: str) -> float:
        """LOLBin/C2/LDAP match count / log1p(chars). Capped at 5.0."""
        try:
            tc = len(code)
            if tc == 0:
                return 0.0
            return min(
                len(self._LOLBIN_RE.findall(code)) / math.log1p(tc), 5.0
            )
        except Exception:
            return 0.0