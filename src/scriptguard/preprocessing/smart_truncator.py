"""
Smart Code Truncation
Intelligently truncates long code samples while preserving critical sections.
"""

import ast
import re
from typing import List, Tuple
from scriptguard.utils.logger import logger


# Security-related keywords that indicate malicious patterns
SECURITY_KEYWORDS = [
    "socket", "subprocess", "os.system", "eval", "exec",
    "base64.b64decode", "pickle.loads", "__import__",
    "compile", "execfile", "input", "raw_input",
    "urllib", "requests", "http", "ftp",
    "crypto", "cipher", "encrypt", "decrypt"
]


def extract_imports(code: str) -> str:
    """Extract all import statements from code."""
    try:
        tree = ast.parse(code)
        lines = code.split('\n')
        import_lines = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if hasattr(node, 'lineno'):
                    # Get the line (1-indexed in AST)
                    line_idx = node.lineno - 1
                    if line_idx < len(lines):
                        import_lines.append(lines[line_idx])

        return '\n'.join(import_lines)
    except:
        # Fallback: regex-based extraction
        import_pattern = re.compile(r'^(?:from\s+\S+\s+)?import\s+.+$', re.MULTILINE)
        imports = import_pattern.findall(code)
        return '\n'.join(imports)


def extract_security_relevant_functions(code: str, keywords: List[str]) -> List[Tuple[str, int, int]]:
    """
    Extract functions that contain security-related keywords.

    Returns:
        List of (function_code, start_line, end_line) tuples
    """
    try:
        tree = ast.parse(code)
        lines = code.split('\n')
        relevant_functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Get function source lines
                start_line = node.lineno - 1  # Convert to 0-indexed
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10

                # Extract function code
                func_lines = lines[start_line:end_line]
                func_code = '\n'.join(func_lines)

                # Check if function contains security keywords
                for keyword in keywords:
                    if keyword in func_code:
                        relevant_functions.append((func_code, start_line, end_line))
                        break

        return relevant_functions
    except:
        return []


def extract_main_entry(code: str) -> str:
    """Extract if __name__ == '__main__' section."""
    try:
        tree = ast.parse(code)
        lines = code.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check if this is the main guard
                if isinstance(node.test, ast.Compare):
                    try:
                        # Look for __name__ == '__main__' pattern
                        if hasattr(node.test.left, 'id') and node.test.left.id == '__name__':
                            start_line = node.lineno - 1
                            end_line = node.end_lineno if hasattr(node, 'end_lineno') else len(lines)
                            return '\n'.join(lines[start_line:end_line])
                    except:
                        continue
    except:
        pass

    # Fallback: regex-based extraction
    main_pattern = re.compile(
        r'if\s+__name__\s*==\s*["\']__main__["\']\s*:.*',
        re.DOTALL | re.MULTILINE
    )
    match = main_pattern.search(code)
    if match:
        return match.group(0)

    return ""


def smart_truncate(code: str, max_chars: int) -> str:
    """
    Intelligently truncate code preserving HEAD (imports), TAIL (payloads), and SECURITY keywords.
    """
    if len(code) <= max_chars:
        return code

    # Zwiększamy bufor bezpieczeństwa dla promptu
    effective_max = max_chars

    try:
        # Strategy:
        # 1. Always keep the last 25% of budget for the TAIL (common place for payloads)
        # 2. Keep Imports (Head)
        # 3. Keep Security Keywords
        # 4. Fill the rest with body from top

        tail_size = int(effective_max * 0.3)  # Increased to 30% for tail
        head_budget = effective_max - tail_size

        # Extract components
        tail_code = code[-tail_size:]
        remaining_code = code[:-tail_size]  # Work with the rest

        imports = extract_imports(remaining_code)
        security_funcs = extract_security_relevant_functions(remaining_code, SECURITY_KEYWORDS)

        sections = []
        current_size = 0

        # A. Add Imports (Context)
        if imports and len(imports) < head_budget * 0.3:
            sections.append(("# Imports", imports))
            current_size += len(imports)

        # B. Add Security Functions (Malicious Logic)
        if security_funcs:
            for func_code, _, _ in security_funcs:
                if current_size + len(func_code) < head_budget * 0.7:
                    sections.append(("# Security Function", func_code))
                    current_size += len(func_code)
                else:
                    break

        # C. Add Main/Body if space allows (Context)
        remaining_space = head_budget - current_size - 100  # Buffer
        if remaining_space > 200:
            # Take from the start of the remaining code (skipping what might be imports if duplicated)
            body_sample = remaining_code[:remaining_space]
            sections.append(("# Code Body", body_sample))

        # Reconstruct
        reconstructed = []
        for header, content in sections:
            reconstructed.append(header)
            reconstructed.append(content)

        # Add truncation marker
        reconstructed.append("\n# ... [TRUNCATED] ...\n")

        # Add Tail (CRITICAL for Malware)
        reconstructed.append("# End of file (Payloads often here)")
        reconstructed.append(tail_code)

        result = '\n'.join(reconstructed)

        # Hard safety cut just in case logic overflowed
        if len(result) > max_chars:
            return simple_truncate(code, max_chars)  # Fallback to Head+Tail

        return result

    except Exception as e:
        logger.warning(f"Smart truncation failed: {e}. Using robust fallback.")
        return simple_truncate(code, max_chars)


def simple_truncate(code: str, max_chars: int) -> str:
    """Robust truncation: HEAD + TAIL strategy (Vital for malware detection)."""
    if len(code) <= max_chars:
        return code

    keep_part = (max_chars // 2) - 50  # Reserve space for marker
    head = code[:keep_part]
    tail = code[-keep_part:]

    return f"{head}\n\n# ... [Truncated {len(code) - max_chars} chars] ...\n\n{tail}"
