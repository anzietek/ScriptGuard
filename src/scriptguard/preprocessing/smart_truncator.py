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
    Intelligently truncate code while preserving critical sections.

    Strategy:
    1. If code fits, return as-is
    2. Extract and prioritize:
       - Imports (top of file context)
       - Security-related functions
       - Main entry point
    3. Reconstruct within char limit

    Args:
        code: Source code to truncate
        max_chars: Maximum character limit

    Returns:
        Truncated code with preserved critical sections
    """
    if len(code) <= max_chars:
        return code

    logger.debug(f"Smart truncating code from {len(code)} to {max_chars} chars")

    try:
        # Extract critical sections
        imports = extract_imports(code)
        security_funcs = extract_security_relevant_functions(code, SECURITY_KEYWORDS)
        main_entry = extract_main_entry(code)

        # Build truncated version with priorities
        sections = []
        current_size = 0
        budget = max_chars

        # 1. Always include imports (usually small)
        if imports and len(imports) < budget * 0.2:  # Max 20% for imports
            sections.append(("# Imports", imports))
            current_size += len(imports) + 15  # +15 for header

        # 2. Include security-relevant functions
        if security_funcs:
            sections.append(("# Security-relevant functions", ""))
            current_size += 30

            for func_code, _, _ in security_funcs:
                if current_size + len(func_code) < budget * 0.6:  # Max 60% for functions
                    sections.append(("", func_code))
                    current_size += len(func_code) + 2  # +2 for newlines
                else:
                    break

        # 3. Include main entry if space allows
        if main_entry and current_size + len(main_entry) < budget * 0.8:  # Max 80% total
            sections.append(("# Main entry point", main_entry))
            current_size += len(main_entry) + 20

        # 4. Fill remaining space with beginning of code if we have room
        if current_size < budget * 0.5:  # We haven't used much space
            remaining = budget - current_size - 50  # Reserve space for truncation marker
            if remaining > 100:
                sections.insert(1, ("# Beginning of code", code[:remaining]))

        # Reconstruct
        reconstructed = []
        for header, content in sections:
            if header:
                reconstructed.append(header)
            if content:
                reconstructed.append(content)

        result = '\n\n'.join(reconstructed)

        # Add truncation marker
        if len(result) < len(code):
            result += "\n\n# ... [Code truncated by smart truncation] ..."

        # Final safety check
        if len(result) > max_chars:
            result = result[:max_chars - 50] + "\n\n# ... [Truncated] ..."

        logger.debug(f"Smart truncation: {len(code)} -> {len(result)} chars")
        return result

    except Exception as e:
        logger.warning(f"Smart truncation failed: {e}. Falling back to simple truncation.")
        # Fallback: simple head + tail
        half = max_chars // 2
        return code[:half] + "\n\n# ... [Truncated] ...\n\n" + code[-half:]


def simple_truncate(code: str, max_chars: int) -> str:
    """Simple truncation: just take first N characters."""
    if len(code) <= max_chars:
        return code
    return code[:max_chars]
