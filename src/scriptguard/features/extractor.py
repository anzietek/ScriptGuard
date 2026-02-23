"""
Feature extractor for ScriptGuard fusion model.
Extracts a 45-dimensional feature vector from Python source code.

Output vector (FEATURE_DIM=45):
  - 25 continuous/count features: AST structure metrics, import risk counts,
    entropy values, network counts, statistical measures, and obfuscation
    structural signals (max string literal length, long-line ratio).
    NOTE: indices 1, 17, 19, 20, 22 are log1p-scaled (n_calls, total_lines,
    avg_line_len, max_line_len, max_str_literal_len) to prevent high-variance
    length features from drowning out behavioural signals.
  - 1 malware_api_score: integer sum of 44 binary indicator flags covering
    dangerous API imports, obfuscation patterns, persistence mechanisms,
    network C2 indicators, crypto operations, filesystem recon, and 5 extended
    API flags (sys.gettrace, ctypes.VirtualAlloc, marshal.loads,
    zlib.decompress, platform.uname).
  - 4 targeted features (P1-P4): short_script_malware_score,
    benign_context_ratio, package_infra_score, direct_exec_chain_score.
  - 6 behavioral-chain features (P5-P10): write_exec_chain_score,
    in_memory_exec_score, string_fragmentation_ratio, persistence_indicator,
    anti_debug_score, execution_compactness_score.
  - 4 stealth & structural features (P11-P14): dynamic_attribute_score,
    network_exfil_fuzzy_score, structural_malware_ratio,
    line_length_legitimacy_filter.
  - 1 ghost-script detector (P15): lone_call_in_global_scope.
  - 4 data-density & shadowing features (P16-P19): keyword_to_char_density,
    non_ascii_ratio, dead_string_ratio, suspicious_builtin_shadowing.

Sub-methods compute an intermediate 69-feature raw vector; extract() post-
processes it into the final 45-dimensional form by separating continuous
features from binary flags, applying log1p to 5 heavy features, aggregating
the binary flags into malware_api_score, and appending the 19 targeted features.
"""

import ast
import math
import re
import warnings
from collections import deque
from scriptguard.utils.logger import logger


class FeatureExtractor:
    """
    Extracts 45-dimensional feature vector from Python source code.

    Output layout (indices 0-44):
        0-9:   AST counts (tree_depth, n_calls*, n_imports, n_funcdefs, n_classdefs,
                           n_for, n_while, n_try, n_exec_nodes, exec_eval_depth)
               * log1p-scaled
        10-11: Import counts (total_imports, high_risk_imports)
        12-14: Entropy values (mean_str_entropy, max_str_entropy, high_entropy_count)
        15-16: Network counts (unique_ip_count, unique_url_count)
        17-21: Statistical (total_lines*, comment_density, avg_line_len*,
                            max_line_len*, line_len_cv)  * log1p-scaled
        22:    max_str_literal_len* — log1p-scaled
        23:    long_line_ratio — fraction of lines longer than 500 chars [0, 1]
        24:    benign_framework_score — weighted score of legitimate framework imports
        25:    malware_api_score — sum of 44 binary indicator flags
        26:    short_script_malware_score (P1) — exec+exfil in compact scripts (<100 lines)
        27:    benign_context_ratio (P2) — FP suppressor: benign framework context
        28:    package_infra_score (P3) — pip/stdlib infrastructure code signal
        29:    direct_exec_chain_score (P4) — recon+exfil or ctypes+net chains
        30:    write_exec_chain_score (P5) — net→file→exec workflow
        31:    in_memory_exec_score (P6) — decode+exec within same FunctionDef
        32:    string_fragmentation_ratio (P7) — chr()/concat obfuscation
        33:    persistence_indicator (P8) — crontab/registry/startup patterns [0/1]
        34:    anti_debug_score (P9) — sandbox/debugger evasion signals
        35:    execution_compactness_score (P10) — dropper/stager compactness
        36:    dynamic_attribute_score (P11) — getattr/dunder stealth access
        37:    network_exfil_fuzzy_score (P12) — fuzzy exfil via any HTTP method
        38:    structural_malware_ratio (P13) — ln(n_calls)/(1+ln(n_funcdefs+n_classdefs))
        39:    line_length_legitimacy_filter (P14) — FP suppressor [-2, 0]
        40:    lone_call_in_global_scope (P15) — global-scope exec in <50-line scripts [0/1]
        41:    keyword_to_char_density (P16) — keyword count / log1p(total chars)
        42:    non_ascii_ratio (P17) — fraction of non-ASCII characters [0, 1]
        43:    dead_string_ratio (P18) — log1p(string lens) / log1p(AST nodes)
        44:    suspicious_builtin_shadowing (P19) — assignment to exec/eval/etc. [0/1]

    Sub-methods produce a 69-feature raw vector which is post-processed in
    extract() to yield this 45-dimensional output.
    """

    _HIGH_RISK_IMPORTS = {
        "socket", "subprocess", "os", "ctypes", "base64",
        "marshal", "pickle", "cryptography", "fernet",
    }

    # Indices into the raw vector that are binary (0/1) flags.
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
        # Extended API flags: indices 64-68
        # has_sys_gettrace(64), has_ctypes_virtual_alloc(65), has_marshal_loads(66),
        # has_zlib_decompress(67), has_platform_uname(68)
        64, 65, 66, 67, 68,
    })

    # Indices of continuous/count features kept in output (order preserved).
    _CONTINUOUS_INDICES: tuple = (
        0, 1, 2, 3, 4, 5, 6, 7, 8,  # AST counts (tree_depth … n_exec_nodes)
        10,                           # exec_eval_depth (count, can be > 1)
        22, 23,                       # total_imports, high_risk_imports
        24, 25, 26,                   # mean/max entropy, high_entropy_count
        39, 40,                       # unique_ip_count, unique_url_count
        56, 57, 58, 59, 60,           # statistical (5)
        61, 62,                       # max_str_literal_len, long_line_ratio
        63,                           # benign_framework_score
    )

    # _RAW_DIM auto-computed; _CONTINUOUS_INDICES covers raw indices 0-63,
    # _BINARY_INDICES covers raw indices 0-55 plus 64-68 (extended API flags).
    _RAW_DIM: int = len(_BINARY_INDICES) + len(_CONTINUOUS_INDICES)

    # Indices in the final feature vector where math.log1p() is applied to prevent
    # high-variance length/count features from dominating the behavioural signals.
    # Positions: n_calls=1, total_lines=17, avg_line_len=19, max_line_len=20, max_str_literal_len=22
    _LOG1P_INDICES: tuple[int, ...] = (1, 17, 19, 20, 22)

    # 25 + 1 malware_api_score + 4 (P1-P4) + 6 (P5-P10) + 4 (P11-P14) + 1 (P15) + 4 (P16-P19)
    FEATURE_DIM: int = len(_CONTINUOUS_INDICES) + 1 + 4 + 6 + 4 + 1 + 4

    def extract(self, code: str) -> list[float]:
        """
        Extract 41-dimensional feature vector from Python source code.

        Internally computes 69 raw features, then:
          - keeps the 25 continuous/count features in order
          - sums all 44 binary flags into malware_api_score (index 25)
          - applies math.log1p to 5 heavy features (indices 1,17,19,20,22)
          - appends 15 targeted features P1-P15 (indices 26-40)

        Returns [0.0] * 41 on any top-level exception.
        """
        try:
            _aliases: dict[str, str] = {}
            try:
                _aliases = self._build_alias_map(self._parse_ast(code))
            except SyntaxError:
                pass

            raw = (
                self._ast_features(code, _aliases)            # 13  indices 0-12
                + self._import_features(code, _aliases)       # 11  indices 13-23
                + self._entropy_features(code)                # 4   indices 24-27
                + self._obfuscation_features(code, _aliases)  # 11  indices 28-38
                + self._network_features(code)                # 6   indices 39-44
                + self._persistence_features(code)            # 4   indices 45-48
                + self._crypto_features(code)                 # 4   indices 49-52
                + self._recon_fs_features(code)               # 3   indices 53-55
                + self._statistical_features(code)            # 7   indices 56-62
                + self._extra_features(code)                  # 1   index 63
                + self._extended_api_flags(code, _aliases)    # 5   indices 64-68
            )
            assert len(raw) == self._RAW_DIM, (
                f"Raw dimension mismatch: expected {self._RAW_DIM}, got {len(raw)}"
            )
            continuous = [raw[i] for i in self._CONTINUOUS_INDICES]
            malware_api_score = float(sum(raw[i] for i in self._BINARY_INDICES))
            features = continuous + [malware_api_score]

            # Log-scale heavy count/length features to balance the feature space.
            # Indices: n_calls=1, total_lines=17, avg_line_len=19,
            #          max_line_len=20, max_str_literal_len=22
            for _idx in self._LOG1P_INDICES:
                features[_idx] = math.log1p(features[_idx])

            # Append 15 targeted features (P1-P15)
            features.append(self._short_script_malware_score(code))
            features.append(self._benign_context_ratio(code, _aliases))
            features.append(self._package_infra_score(code))
            features.append(self._direct_exec_chain_score(code))
            features.append(self._write_exec_chain_score(code))
            features.append(self._in_memory_exec_score(code, _aliases))
            features.append(self._string_fragmentation_ratio(code))
            features.append(self._persistence_indicator(code))
            features.append(self._anti_debug_score(code))
            features.append(self._execution_compactness_score(code))
            # P11-P14 read from features[] — sees log1p-scaled values at [1,17,19,20,22]
            features.append(self._dynamic_attribute_score(code))
            features.append(self._network_exfil_fuzzy_score(code))
            features.append(self._structural_malware_ratio(features))
            features.append(self._line_length_legitimacy_filter(features))
            features.append(self._lone_call_in_global_scope(code))
            # P16-P19: data-density and shadowing signals
            features.append(self._keyword_to_char_density(code))
            features.append(self._non_ascii_ratio(code))
            features.append(self._dead_string_ratio(code))
            features.append(self._suspicious_builtin_shadowing(code))

            assert len(features) == self.FEATURE_DIM  # 45

            # Sanity pass: replace non-finite values with 0.0, cap > 20.0
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

    @staticmethod
    def _parse_ast(code: str) -> ast.AST:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(code)

    @staticmethod
    def _build_alias_map(tree: ast.AST) -> dict[str, str]:
        """First-pass alias resolver: maps local names → canonical qualified names.

        Handles:
          import os as o              → {"o": "os"}
          from os import system       → {"system": "os.system"}
          from os import system as s  → {"s": "os.system"}

        Star imports (from x import *) are skipped — cannot be resolved statically.
        Returns empty dict on any error so callers degrade gracefully.
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

    # ------------------------------------------------------------------
    # AST features (13)
    # ------------------------------------------------------------------

    def _ast_features(self, code: str, aliases: dict[str, str] | None = None) -> list[float]:
        aliases = aliases or {}
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
                    if isinstance(getattr(node, 'value', None), ast.Call):
                        func = getattr(node.value, 'func', None)
                        if isinstance(func, ast.Name):
                            resolved = aliases.get(func.id, func.id)
                            if resolved == 'exec':
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

            exec_eval_depth = self._compute_exec_eval_depth(tree, aliases)
            has_decode_chain = self._has_decode_chain(code, tree)

            has_dynamic_import = 0.0
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = getattr(node, 'func', None)
                    if isinstance(func, ast.Name):
                        resolved = aliases.get(func.id, func.id)
                        if resolved == '__import__' or resolved.endswith('.import_module'):
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

    def _compute_exec_eval_depth(
        self, tree: ast.AST, aliases: dict[str, str] | None = None
    ) -> float:
        _aliases = aliases or {}
        _EXEC_NAMES = frozenset({"exec", "eval", "compile"})
        max_depth = 0

        def _depth(node: ast.AST) -> int:
            if not isinstance(node, ast.Call):
                return 0
            func = getattr(node, 'func', None)
            name = None
            if isinstance(func, ast.Name):
                name = _aliases.get(func.id, func.id)
            if name in _EXEC_NAMES:
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
        if re.search(r'b64decode.+decompress.+exec', code, re.DOTALL):
            return 1.0
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

    def _import_features(self, code: str, aliases: dict[str, str] | None = None) -> list[float]:
        aliases = aliases or {}
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

            has_os_exec = 0.0
            _os_exec_attrs = frozenset(
                ('system', 'popen', 'execv', 'execl', 'execvp', 'execve', 'execvpe', 'spawnl', 'spawnle', 'spawnlp')
            )
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = getattr(node, 'func', None)
                    if isinstance(func, ast.Attribute) and func.attr in _os_exec_attrs:
                        val = getattr(func, 'value', None)
                        if isinstance(val, ast.Name):
                            resolved_mod = aliases.get(val.id, val.id)
                            if resolved_mod == 'os':
                                has_os_exec = 1.0
                                break
                    elif isinstance(func, ast.Name):
                        resolved = aliases.get(func.id, '')
                        if resolved.startswith('os.') and resolved[3:] in _os_exec_attrs:
                            has_os_exec = 1.0
                            break
            if not has_os_exec:
                if re.search(r'os\.(?:system|popen|execv[pe]?|execl[pe]?|spawn)\s*\(', code):
                    has_os_exec = 1.0

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
            string_literals = re.findall(
                r'(?:b?)(?:"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')', code
            )

            entropies = []
            for s in string_literals:
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

    def _obfuscation_features(self, code: str, aliases: dict[str, str] | None = None) -> list[float]:
        aliases = aliases or {}
        try:
            try:
                tree = self._parse_ast(code)
                parse_ok = True
            except SyntaxError:
                tree = None
                parse_ok = False

            has_exec = 0.0
            has_eval = 0.0
            has_compile = 0.0
            has_dunder_import = 0.0
            lambda_chain_count = 0

            if parse_ok and tree:
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        func = getattr(node, 'func', None)
                        if isinstance(func, ast.Name):
                            resolved = aliases.get(func.id, func.id)
                            if resolved == 'exec':
                                has_exec = 1.0
                            elif resolved == 'eval':
                                has_eval = 1.0
                            elif resolved == 'compile':
                                has_compile = 1.0
                            elif resolved == '__import__':
                                has_dunder_import = 1.0

                for node in ast.walk(tree):
                    if isinstance(node, ast.Lambda):
                        body = node.body
                        depth = 0
                        while isinstance(body, ast.Lambda):
                            depth += 1
                            body = body.body
                        if depth > 0:
                            lambda_chain_count += 1

            has_b64decode_call = 0.0
            if re.search(
                r'b64decode|b64encode|base64\.b64|\.decode\(["\']base64["\']'
                r'|binascii\.(?:unhexlify|a2b_hex)|bytes\.fromhex|codecs\.decode',
                code,
            ):
                has_b64decode_call = 1.0

            _ob_lines = code.splitlines()
            _ob_comment_count = sum(1 for l in _ob_lines if l.strip().startswith('#'))
            has_no_comments = float(_ob_comment_count == 0 and len(_ob_lines) > 5)

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

            _ob_line_lens = [len(l) for l in _ob_lines]
            has_very_long_line = float(max(_ob_line_lens) > 500 if _ob_line_lens else False)

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

            has_ctypes_windll = 0.0
            if re.search(r'ctypes\.windll|windll\.kernel32|windll\.ntdll', code):
                has_ctypes_windll = 1.0

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
        hex_arrays = re.findall(r'(?:bytearray|bytes)\s*\(\s*\[([^\]]+)\]', code)
        for arr_str in hex_arrays:
            parts = [p.strip() for p in arr_str.split(',') if p.strip()]
            if len(parts) > 100:
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

        hex_str_matches = re.findall(r'(?:\\x[0-9a-fA-F]{2}){100,}', code)
        if hex_str_matches:
            return 1.0

        return 0.0

    # ------------------------------------------------------------------
    # Network / C2 features (6)
    # ------------------------------------------------------------------

    def _network_features(self, code: str) -> list[float]:
        try:
            ips = re.findall(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', code)
            unique_ip_count = float(len(set(ips)))

            urls = re.findall(r'https?://[^\s\'"]+', code)
            unique_url_count = float(len(set(urls)))

            has_hardcoded_ports = 0.0
            port_matches = re.findall(r'\b(4444|1337|8080|8888|9999|31337|6667|1234|5555|7777)\b', code)
            if port_matches:
                has_hardcoded_ports = 1.0

            has_c2_pattern = self._detect_c2_pattern(code)

            has_dns_lookup = 0.0
            if re.search(r'socket\.gethostbyname|dns\.resolver|getaddrinfo|nslookup', code):
                has_dns_lookup = 1.0

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
        try:
            tree = self._parse_ast(code)
        except SyntaxError:
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
            has_registry_write = 0.0
            if re.search(r'winreg|_winreg|RegSetValue|HKEY_|OpenKey|CreateKey', code):
                has_registry_write = 1.0

            has_cron_pattern = 0.0
            cron_create_patterns = [
                r'schtasks\s+/[Cc]reate',
                r'at\.exe\s+\d',
                r'SchTasks\.exe',
                r'open\(["\'][/\\]etc[/\\]cron',
                r'crontab\s+-[el]\s',
                r'TaskScheduler|ITaskScheduler',
            ]
            for pat in cron_create_patterns:
                if re.search(pat, code, re.IGNORECASE):
                    has_cron_pattern = 1.0
                    break

            has_startup_persistence = 0.0
            startup_patterns = [
                r'\\CurrentVersion\\Run(?:Once)?["\']',
                r'HKCU.*\\Run|HKLM.*\\Run',
                r'nssm\s+install',
                r'sc\.exe\s+create|CreateService\s*\(',
                r'open\(["\'][/\\]etc[/\\](?:init\.d|rc\.local)',
                r'(?:plist|launchd).*write|write.*plist',
            ]
            for pat in startup_patterns:
                if re.search(pat, code, re.IGNORECASE):
                    has_startup_persistence = 1.0
                    break

            has_creates_executable = 0.0
            exec_drop_patterns = [
                r'chmod\s+\+x',
                r'os\.chmod\([^,]+,\s*(?:0[oO]?7[0-7]{2}|0o?755|0o?777)',
                r'open\([^)]*\.(exe|bat|cmd|vbs|ps1|sh)[^)]*["\'],\s*["\'][wa]b?',
                r'with\s+open\([^)]*\.(exe|bat|cmd|vbs|ps1|sh)',
                r'HKEY_[A-Z_]+.*(?:\\\\Run|\\\\RunOnce)',
                r'subprocess.*["\'][^\'"]*\.(exe|bat|cmd)["\']',
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

            has_xor_pattern = 0.0
            if re.search(r'\bxor\b|\bXOR\b', code):
                has_xor_pattern = 1.0
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
            has_recursive_traversal = 0.0
            if re.search(r'os\.walk\(|\.rglob\(|glob\.glob[^)]*\*\*', code):
                has_recursive_traversal = 1.0

            has_mass_file_ops = 0.0
            ext_pattern = r'\.(doc|docx|xls|xlsx|pdf|jpg|png|mp4|zip|rar|txt|csv)'
            ext_matches = re.findall(ext_pattern, code, re.IGNORECASE)
            if len(set(ext_matches)) >= 3:
                has_mass_file_ops = 1.0
            if re.search(r'for\s+\w+\s+in\s+.*\.walk|encrypt.*\.open|open.*encrypt', code, re.DOTALL):
                has_mass_file_ops = 1.0

            has_shadow_copy = 0.0
            if re.search(r'vssadmin|shadow\s+copy|wmic.*shadowcopy|Win32_ShadowCopy', code, re.IGNORECASE):
                has_shadow_copy = 1.0

            return [has_recursive_traversal, has_mass_file_ops, has_shadow_copy]
        except Exception:
            return [0.0] * 3

    # ------------------------------------------------------------------
    # Statistical features (7)
    # ------------------------------------------------------------------

    def _statistical_features(self, code: str) -> list[float]:
        try:
            lines = code.splitlines()
            total_lines = float(len(lines))

            if total_lines == 0:
                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            comment_lines = sum(1 for l in lines if l.strip().startswith('#'))
            comment_density = float(comment_lines) / total_lines

            line_lengths = [len(l) for l in lines]
            avg_line_length = sum(line_lengths) / max(1, len(line_lengths))
            max_line_length = float(max(line_lengths)) if line_lengths else 0.0

            if len(line_lengths) > 1:
                mean = avg_line_length
                variance = sum((l - mean) ** 2 for l in line_lengths) / len(line_lengths)
                line_len_cv = math.sqrt(variance) / max(1.0, mean)
            else:
                line_len_cv = 0.0

            string_lits = re.findall(
                r'(?:b?)(?:"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')', code
            )
            max_str_literal_len = 0.0
            for s in string_lits:
                stripped = s[1:] if s.startswith("b") else s
                content_len = float(len(stripped) - 2)
                if content_len > max_str_literal_len:
                    max_str_literal_len = content_len

            long_line_count = sum(1 for l in line_lengths if l > 500)
            long_line_ratio = float(long_line_count) / total_lines

            return [
                total_lines,
                comment_density,
                avg_line_length,
                max_line_length,
                line_len_cv,
                max_str_literal_len,
                long_line_ratio,
            ]
        except Exception:
            return [0.0] * 7

    # ------------------------------------------------------------------
    # Extra features (1) — FP/FN root-cause mitigations
    # ------------------------------------------------------------------

    def _extra_features(self, code: str) -> list[float]:
        try:
            tree = self._parse_ast(code)
            parse_ok = True
        except SyntaxError:
            tree = None
            parse_ok = False

        _HARD = frozenset({"django", "flask", "fastapi", "sqlalchemy", "celery", "pydantic"})
        _EASY = frozenset({"logging", "typing", "argparse", "unittest", "pytest", "setuptools"})
        benign_framework_score = 0.0
        if parse_ok and tree is not None:
            try:
                mods: set[str] = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            mods.add(alias.name.split(".")[0])
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        mods.add(node.module.split(".")[0])
                benign_framework_score = (
                    1.0 * sum(1 for m in mods if m in _HARD)
                    + 0.2 * sum(1 for m in mods if m in _EASY)
                )
            except Exception:
                pass

        return [benign_framework_score]

    # ------------------------------------------------------------------
    # Extended API flags (5) — indices 64-68 in raw vector
    # Contributes to malware_api_score via _BINARY_INDICES.
    # ------------------------------------------------------------------

    def _extended_api_flags(self, code: str, aliases: dict[str, str] | None = None) -> list[float]:
        aliases = aliases or {}
        try:
            # 64: sys.gettrace — anti-debug / sandbox detection
            has_sys_gettrace = 0.0
            if re.search(r'\bsys\.gettrace\s*\(\)', code):
                has_sys_gettrace = 1.0
            else:
                try:
                    tree = self._parse_ast(code)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            func = getattr(node, 'func', None)
                            if isinstance(func, ast.Attribute) and func.attr == 'gettrace':
                                val = getattr(func, 'value', None)
                                if isinstance(val, ast.Name):
                                    resolved = aliases.get(val.id, val.id)
                                    if resolved == 'sys':
                                        has_sys_gettrace = 1.0
                                        break
                except SyntaxError:
                    pass

            # 65: ctypes.VirtualAlloc — process injection primitive
            has_ctypes_virtual_alloc = float(bool(re.search(r'\bVirtualAlloc\b', code)))

            # 66: marshal.loads — bytecode deserialization (code exec vector)
            has_marshal_loads = 0.0
            if re.search(r'\bmarshal\.loads\s*\(', code):
                has_marshal_loads = 1.0
            else:
                try:
                    tree = self._parse_ast(code)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            func = getattr(node, 'func', None)
                            if isinstance(func, ast.Attribute) and func.attr == 'loads':
                                val = getattr(func, 'value', None)
                                if isinstance(val, ast.Name):
                                    resolved = aliases.get(val.id, val.id)
                                    if resolved == 'marshal':
                                        has_marshal_loads = 1.0
                                        break
                except SyntaxError:
                    pass

            # 67: zlib/gzip/bz2/lzma decompress — obfuscated payload unpacking
            has_zlib_decompress = float(
                bool(re.search(r'(?:zlib|gzip|bz2|lzma)\.decompress\s*\(', code, re.I))
            )

            # 68: platform.uname — system fingerprinting / C2 beacon enumeration
            has_platform_uname = 0.0
            if re.search(r'\bplatform\.uname\s*\(\)', code):
                has_platform_uname = 1.0
            else:
                try:
                    tree = self._parse_ast(code)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            func = getattr(node, 'func', None)
                            if isinstance(func, ast.Attribute) and func.attr == 'uname':
                                val = getattr(func, 'value', None)
                                if isinstance(val, ast.Name):
                                    resolved = aliases.get(val.id, val.id)
                                    if resolved == 'platform':
                                        has_platform_uname = 1.0
                                        break
                except SyntaxError:
                    pass

            return [has_sys_gettrace, has_ctypes_virtual_alloc, has_marshal_loads,
                    has_zlib_decompress, has_platform_uname]
        except Exception:
            return [0.0] * 5

    # ------------------------------------------------------------------
    # P1: short_script_malware_score (index 26)
    # ------------------------------------------------------------------

    def _short_script_malware_score(self, code: str) -> float:
        try:
            if len(code.splitlines()) >= 100:
                return 0.0
            score = 0.0
            has_exec = bool(re.search(
                r'\bos\.(?:system|popen)\s*\('
                r'|\bsubprocess\.(?:run|Popen|call|check_output|check_call)\s*\(',
                code,
            ))
            has_exfil = bool(re.search(
                r'\burllib\.request\.urlopen\s*\('
                r'|\brequests\.(?:get|post|put|patch)\s*\('
                r'|\.send\s*\('
                r'|\bsocket\.connect\s*\(',
                code,
            ))
            has_encode = bool(re.search(
                r'\bb64encode\s*\(|\.hex\s*\(\)|\.encode\s*\(',
                code,
            ))
            if has_exec and has_exfil:
                score += 0.8
            if has_encode and has_exfil:
                score += 0.5
            return min(score, 1.0)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # P2: benign_context_ratio (index 27)
    # ------------------------------------------------------------------

    def _benign_context_ratio(self, code: str, aliases: dict[str, str] | None = None) -> float:
        aliases = aliases or {}
        _STRONG = frozenset({'ansible', 'pytest', 'unittest', 'aiohttp', 'flask', 'boto3', 'boto'})
        _UTILS  = frozenset({'logging', 'argparse', 'click', 'pathlib'})
        _OS_EXEC_ATTRS = frozenset({
            'system', 'popen', 'execv', 'execl', 'execvp', 'execve', 'spawnl', 'spawnle', 'spawnlp',
        })
        try:
            score = 0.0
            try:
                tree = self._parse_ast(code)
                imported: set[str] = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imported.add(alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imported.add(node.module.split('.')[0])

                score += 0.4 * sum(1 for m in imported if m in _STRONG)
                score += 0.15 * sum(1 for m in imported if m in _UTILS)

                # +0.3 if a logging call and an os/subprocess exec call share the same Try node
                for try_node in ast.walk(tree):
                    if not isinstance(try_node, ast.Try):
                        continue
                    has_log_call = False
                    has_exec_call = False
                    sections = [try_node.body, try_node.handlers, try_node.orelse]
                    if hasattr(try_node, 'finalbody'):
                        sections.append(try_node.finalbody)
                    for section in sections:
                        for stmt in section:
                            for n in ast.walk(stmt):
                                if not isinstance(n, ast.Call):
                                    continue
                                func = getattr(n, 'func', None)
                                if not isinstance(func, ast.Attribute):
                                    continue
                                val = getattr(func, 'value', None)
                                if not isinstance(val, ast.Name):
                                    continue
                                mod = aliases.get(val.id, val.id)
                                if mod == 'logging':
                                    has_log_call = True
                                elif mod == 'os' and func.attr in _OS_EXEC_ATTRS:
                                    has_exec_call = True
                                elif mod == 'subprocess':
                                    has_exec_call = True
                    if has_log_call and has_exec_call:
                        score += 0.3
                        break

            except SyntaxError:
                # Regex fallback when AST parse fails
                for fw in _STRONG:
                    if re.search(rf'\b{re.escape(fw)}\b', code, re.I):
                        score += 0.4
                for fw in _UTILS:
                    if re.search(rf'\b{re.escape(fw)}\b', code, re.I):
                        score += 0.15

            return min(score, 1.0)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # P3: package_infra_score (index 28)
    # ------------------------------------------------------------------

    def _package_infra_score(self, code: str) -> float:
        try:
            score = 0.0
            # pip._internal / pip._vendor subpackage imports
            if re.search(r'\bfrom pip\._(?:internal|vendor)\b|\bimport pip\._(?:internal|vendor)\b', code):
                score += 0.8
            # importlib.metadata or email.message structural imports
            if re.search(r'\b(?:importlib\.metadata|email\.message)\b', code):
                score += 0.4
            # Infra bonus: defines abstract type with zero IO calls
            has_infra_type = bool(re.search(
                r'\bnamedtuple\s*\(|\bTypedDict\b|\bProtocol\b',
                code,
            ))
            has_io = bool(re.search(
                r'\bopen\s*\(|\bsocket\b|\brequests\b|\burllib\b',
                code,
            ))
            if has_infra_type and not has_io:
                score += 0.3
            return min(score, 1.0)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # P4: direct_exec_chain_score (index 29)
    # ------------------------------------------------------------------

    def _direct_exec_chain_score(self, code: str) -> float:
        try:
            score = 0.0
            has_net = bool(re.search(
                r'\brequests\.(?:get|post)\s*\('
                r'|\burllib\.request\b'
                r'|\.send\s*\('
                r'|\bsocket\.connect\s*\(',
                code,
            ))

            # Rule 1: recon (whoami / hostname / getuser) + outbound network
            has_recon = bool(re.search(
                r'\bwhoami\b|\bhostname\b|\bgetpass\.getuser\s*\(',
                code, re.I,
            ))
            if has_recon and has_net:
                score += 0.6

            # Rule 2: subprocess shell=True with hardcoded URL or IPv4 literal
            has_shell_true = (
                bool(re.search(r'\bsubprocess\.\w+\b', code))
                and bool(re.search(r'\bshell\s*=\s*True\b', code))
            )
            has_url_or_ip = bool(re.search(
                r'https?://\S+|(?:\d{1,3}\.){3}\d{1,3}',
                code,
            ))
            if has_shell_true and has_url_or_ip:
                score += 0.5

            # Rule 3: ctypes import + any network call in same module
            has_ctypes = bool(re.search(r'\bimport ctypes\b|\bfrom ctypes\b', code))
            if has_ctypes and has_net:
                score += 0.5

            return min(score, 1.0)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # P5: write_exec_chain_score (index 30)
    # ------------------------------------------------------------------

    def _write_exec_chain_score(self, code: str) -> float:
        try:
            has_net = bool(re.search(
                r'\brequests\.(?:get|post|put|patch|head|delete)\s*\('
                r'|\burllib\.request\b'
                r'|\bsocket\.(?:connect|send|recv|sendall)\s*\(',
                code,
            ))
            has_file_write = bool(re.search(
                r'\bopen\s*\([^)]*["\'][wa]b?["\']'
                r'|\.write_text\s*\('
                r'|\.write_bytes\s*\(',
                code,
            ))
            has_exec_call = bool(re.search(
                r'\bos\.(?:system|popen)\s*\('
                r'|\bsubprocess\.\w+\s*\('
                r'|\bexec\s*\('
                r'|\beval\s*\(',
                code,
            ))
            return 0.6 if (has_net and has_file_write and has_exec_call) else 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # P6: in_memory_exec_score (index 31)
    # ------------------------------------------------------------------

    def _in_memory_exec_score(self, code: str, aliases: dict[str, str] | None = None) -> float:
        aliases = aliases or {}
        _DECODE_ATTRS: frozenset[str] = frozenset({'b64decode', 'decompress', 'loads'})
        _EXEC_NAMES: frozenset[str] = frozenset({'exec', 'eval', 'compile'})
        try:
            try:
                tree = self._parse_ast(code)
            except SyntaxError:
                return 0.0
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                has_decode = False
                has_exec = False
                for node in ast.walk(fn):
                    if not isinstance(node, ast.Call):
                        continue
                    func = getattr(node, 'func', None)
                    if isinstance(func, ast.Attribute) and func.attr in _DECODE_ATTRS:
                        has_decode = True
                    elif isinstance(func, ast.Name):
                        resolved = aliases.get(func.id, func.id)
                        if resolved in _EXEC_NAMES:
                            has_exec = True
                    if has_decode and has_exec:
                        return 0.7
            return 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # P7: string_fragmentation_ratio (index 32)
    # ------------------------------------------------------------------

    def _string_fragmentation_ratio(self, code: str) -> float:
        try:
            try:
                tree = self._parse_ast(code)
            except SyntaxError:
                return 0.0

            # Rule 1: ≥5 consecutive + operators in a BinOp left-chain
            def _add_depth(node: ast.expr) -> int:
                depth = 0
                cur: ast.expr = node
                while isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.Add):
                    depth += 1
                    cur = cur.left
                return depth

            for node in ast.walk(tree):
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                    if _add_depth(node) >= 5:
                        return 0.6

            # Rule 2: ≥5 chr() calls within the same FunctionDef
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                chr_count = sum(
                    1 for n in ast.walk(fn)
                    if isinstance(n, ast.Call)
                    and isinstance(getattr(n, 'func', None), ast.Name)
                    and n.func.id == 'chr'  # type: ignore[union-attr]
                )
                if chr_count >= 5:
                    return 0.6

            # Rule 3: "".join([...]) with ≥5 literal/chr() elements
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = getattr(node, 'func', None)
                if not isinstance(func, ast.Attribute) or func.attr != 'join':
                    continue
                args = getattr(node, 'args', [])
                if not args or not isinstance(args[0], (ast.List, ast.Tuple)):
                    continue
                elts = args[0].elts
                if len(elts) < 5:
                    continue
                all_simple = all(
                    isinstance(e, ast.Constant)
                    or (
                        isinstance(e, ast.Call)
                        and isinstance(getattr(e, 'func', None), ast.Name)
                        and e.func.id == 'chr'  # type: ignore[union-attr]
                    )
                    for e in elts
                )
                if all_simple:
                    return 0.6

            return 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # P8: persistence_indicator (index 33)
    # ------------------------------------------------------------------

    def _persistence_indicator(self, code: str) -> float:
        try:
            _PATTERNS: list[str] = [
                r'crontab',
                r'\.bashrc',
                r'\bStartup\b',
                r'CurrentVersion\\Run',
                r'HKCU\\',
                r'HKLM\\',
            ]
            for pat in _PATTERNS:
                if re.search(pat, code, re.IGNORECASE):
                    return 1.0
            return 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # P9: anti_debug_score (index 34)
    # ------------------------------------------------------------------

    def _anti_debug_score(self, code: str) -> float:
        try:
            score = 0.0
            if re.search(r'\bsys\.gettrace\s*\(\)', code):
                score += 0.5
            if re.search(r'\bIsDebuggerPresent\b', code):
                score += 0.5
            if re.search(r'\bctypes\.windll\.kernel32\b', code):
                score += 0.5
            if re.search(r'\bwireshark\b|\bprocmon\b|\bvbox\b|\bvmware\b', code, re.IGNORECASE):
                score += 0.4
            return min(score, 1.0)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # P10: execution_compactness_score (index 35)
    # ------------------------------------------------------------------

    def _execution_compactness_score(self, code: str) -> float:
        try:
            try:
                tree = self._parse_ast(code)
            except SyntaxError:
                return 0.0
            func_count = sum(
                1 for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
            has_exec = bool(re.search(
                r'\bexec\s*\(|\beval\s*\(|\bcompile\s*\('
                r'|\bos\.(?:system|popen)\s*\('
                r'|\bsubprocess\.\w+\s*\(',
                code,
            ))
            if not has_exec:
                return 0.0
            if func_count == 0:
                return 0.8
            if func_count <= 2:
                return 0.6
            return 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # P11: dynamic_attribute_score (index 36)
    # ------------------------------------------------------------------

    def _dynamic_attribute_score(self, code: str) -> float:
        _DANGEROUS_ATTRS: frozenset[str] = frozenset({'exec', 'eval', 'system', 'popen', 'spawn'})
        _DUNDER_NAMES: frozenset[str] = frozenset({'__dict__', '__subclasses__', '__builtins__'})
        _INTROSPECT_OBJECTS: frozenset[str] = frozenset({'builtins', 'os'})
        try:
            try:
                tree = self._parse_ast(code)
            except SyntaxError:
                return 0.0

            # Collect node IDs of method bodies where first arg is self/cls
            # (legitimate OOP introspection — skip those entirely)
            method_body_ids: set[int] = set()
            for class_node in ast.walk(tree):
                if not isinstance(class_node, ast.ClassDef):
                    continue
                for item in ast.iter_child_nodes(class_node):
                    if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    fn_args = item.args
                    if fn_args.args and fn_args.args[0].arg in ('self', 'cls'):
                        for child in ast.walk(item):
                            method_body_ids.add(id(child))

            # Collect node IDs inside `if __name__ == "__main__":` blocks
            main_block_ids: set[int] = set()
            for node in ast.walk(tree):
                if not isinstance(node, ast.If):
                    continue
                test = node.test
                if (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == '__name__'
                    and len(test.ops) == 1
                    and isinstance(test.ops[0], ast.Eq)
                    and len(test.comparators) == 1
                    and isinstance(test.comparators[0], ast.Constant)
                    and test.comparators[0].value == '__main__'
                ):
                    for child in ast.walk(node):
                        main_block_ids.add(id(child))

            score = 0.0

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = getattr(node, 'func', None)
                if not isinstance(func, ast.Name) or func.id != 'getattr':
                    continue
                # Skip if inside a self/cls method body
                if id(node) in method_body_ids:
                    continue
                args = node.args
                if len(args) < 2:
                    continue
                obj = args[0]
                attr_expr = args[1]
                # Skip if the accessed object is self or cls
                if isinstance(obj, ast.Name) and obj.id in ('self', 'cls'):
                    continue
                # +0.8 if getattr on builtins/os with a concatenated attribute name
                if (
                    isinstance(obj, ast.Name)
                    and obj.id in _INTROSPECT_OBJECTS
                    and isinstance(attr_expr, ast.BinOp)
                    and isinstance(attr_expr.op, ast.Add)
                ):
                    score += 0.8
                elif not isinstance(attr_expr, ast.Constant) or not isinstance(attr_expr.value, str):
                    score += 0.5
                elif attr_expr.value in _DANGEROUS_ATTRS:
                    score += 0.2

            # +0.3 if any dunder access exists outside __main__ and method bodies (once)
            found_dunder = False
            for node in ast.walk(tree):
                if id(node) in main_block_ids or id(node) in method_body_ids:
                    continue
                if isinstance(node, ast.Attribute) and node.attr in _DUNDER_NAMES:
                    found_dunder = True
                    break
                if isinstance(node, ast.Name) and node.id in _DUNDER_NAMES:
                    found_dunder = True
                    break
            if found_dunder:
                score += 0.3

            return min(score, 1.0)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # P12: network_exfil_fuzzy_score (index 37)
    # ------------------------------------------------------------------

    def _network_exfil_fuzzy_score(self, code: str) -> float:
        _NET_METHODS: frozenset[str] = frozenset({'get', 'post', 'request', 'connect', 'send'})
        _URL_IP_PAT: re.Pattern[str] = re.compile(
            r'https?://|ftps?://|(?:\d{1,3}\.){3}\d{1,3}'
        )
        try:
            try:
                tree = self._parse_ast(code)
            except SyntaxError:
                if re.search(
                    r'\.(?:get|post|request|connect|send)\s*\([^)]*'
                    r'(?:https?://|ftps?://|\d{1,3}(?:\.\d{1,3}){3})',
                    code,
                ):
                    return 0.7
                return 0.0

            # Collect locally-defined function names (to exclude as caller objects)
            local_fn_names: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    local_fn_names.add(node.name)

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = getattr(node, 'func', None)
                if not isinstance(func, ast.Attribute) or func.attr not in _NET_METHODS:
                    continue
                # Skip calls on locally-defined function names
                caller = getattr(func, 'value', None)
                if isinstance(caller, ast.Name) and caller.id in local_fn_names:
                    continue
                # Check all positional + keyword arguments for URL/IP string literals
                kw_values = [kw.value for kw in node.keywords]
                for arg in list(node.args) + kw_values:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        if _URL_IP_PAT.search(arg.value):
                            return 0.7

            return 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # P13: structural_malware_ratio (index 38)
    # Formula: log1p(n_calls) / (1 + log1p(n_funcdefs + n_classdefs))
    # features[1] is already log1p-scaled (n_calls); features[3,4] are raw counts.
    # Amplifies flat structure: many calls with few abstractions → high ratio.
    # ------------------------------------------------------------------

    def _structural_malware_ratio(self, features: list[float]) -> float:
        try:
            ln_n_calls = features[1]        # already log1p-scaled by extract()
            n_funcdefs = features[3]
            n_classdefs = features[4]
            return ln_n_calls / (1.0 + math.log1p(n_funcdefs + n_classdefs))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # P14: line_length_legitimacy_filter (index 39)
    # Reads log1p-scaled features[20] (max_line_len), features[4] (n_classdefs),
    # features[18] (comment_density), features[24] (benign_framework_score).
    # Threshold is log1p(1000) ≈ 6.908 because max_line_len is already log-scaled.
    # Returns -2.0 (strong suppressor) for long lines in legitimate libraries.
    # ------------------------------------------------------------------

    def _line_length_legitimacy_filter(self, features: list[float]) -> float:
        try:
            log_max_line = features[20]  # already log1p-scaled in extract()
            if log_max_line <= math.log1p(1000):
                return 0.0
            n_classdefs = features[4]
            comment_density = features[18]
            benign_framework_score = features[24]
            if n_classdefs > 2 and comment_density > 0.05 and benign_framework_score > 0.15:
                return -2.0
            return 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # P15: lone_call_in_global_scope (index 40)
    # ------------------------------------------------------------------

    def _lone_call_in_global_scope(self, code: str) -> float:
        _EXEC_NAMES: frozenset[str] = frozenset({'exec', 'eval', 'compile'})
        try:
            if len(code.splitlines()) >= 50:
                return 0.0
            try:
                tree = self._parse_ast(code)
            except SyntaxError:
                return 0.0

            def _has_exec_at_scope(node: ast.AST) -> bool:
                """BFS that does NOT descend into nested FunctionDef/ClassDef."""
                stack = [node]
                while stack:
                    cur = stack.pop()
                    if isinstance(cur, ast.Call):
                        func = getattr(cur, 'func', None)
                        if isinstance(func, ast.Name) and func.id in _EXEC_NAMES:
                            return True
                        if isinstance(func, ast.Attribute):
                            val = getattr(func, 'value', None)
                            if func.attr in ('system', 'popen') and isinstance(val, ast.Name):
                                return True
                            if isinstance(val, ast.Name) and val.id == 'subprocess':
                                return True
                    for child in ast.iter_child_nodes(cur):
                        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            stack.append(child)
                return False

            for stmt in ast.iter_child_nodes(tree):
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if _has_exec_at_scope(stmt):
                    return 1.0

            return 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # P16: keyword_to_char_density (index 41)
    # ------------------------------------------------------------------

    _KW_PATTERN: re.Pattern[str] = re.compile(
        r'\b(?:if|else|elif|for|while|import|from|exec|def|class|return|with|'
        r'try|except|finally|raise|pass|break|continue|and|or|not|in|is|'
        r'lambda|yield|global|nonlocal|del|assert|True|False|None|as|print)\b'
    )

    def _keyword_to_char_density(self, code: str) -> float:
        try:
            total_chars = len(code)
            if total_chars == 0:
                return 0.0
            kw_count = len(self._KW_PATTERN.findall(code))
            return kw_count / math.log1p(total_chars)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # P17: non_ascii_ratio (index 42)
    # ------------------------------------------------------------------

    def _non_ascii_ratio(self, code: str) -> float:
        try:
            total = len(code)
            if total == 0:
                return 0.0
            return sum(1 for c in code if ord(c) > 127) / total
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # P18: dead_string_ratio (index 43)
    # ------------------------------------------------------------------

    def _dead_string_ratio(self, code: str) -> float:
        try:
            string_lits = re.findall(
                r'(?:"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')', code
            )
            sum_lens = sum(len(s) - 2 for s in string_lits if len(s) >= 2)
            try:
                tree = self._parse_ast(code)
                n_ast_nodes = sum(1 for _ in ast.walk(tree))
            except SyntaxError:
                n_ast_nodes = max(1, len(code.splitlines()))
            return math.log1p(max(0, sum_lens)) / math.log1p(max(1, n_ast_nodes))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # P19: suspicious_builtin_shadowing (index 44)
    # ------------------------------------------------------------------

    def _suspicious_builtin_shadowing(self, code: str) -> float:
        _CORE_BUILTINS: frozenset[str] = frozenset({
            'exec', 'eval', '__import__', 'compile', 'open',
            'getattr', 'setattr', 'delattr', 'hasattr',
            'globals', 'locals', 'vars',
        })
        try:
            try:
                tree = self._parse_ast(code)
            except SyntaxError:
                return 0.0
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in _CORE_BUILTINS:
                            return 1.0
                elif isinstance(node, ast.AnnAssign):
                    if isinstance(node.target, ast.Name) and node.target.id in _CORE_BUILTINS:
                        return 1.0
                elif isinstance(node, ast.Subscript):
                    val = getattr(node, 'value', None)
                    if isinstance(val, ast.Name) and val.id == '__builtins__':
                        return 1.0
            return 0.0
        except Exception:
            return 0.0
