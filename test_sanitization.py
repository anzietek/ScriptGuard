"""
Test Code Sanitization and Context Injection
Validates the new ingestion pipeline enhancements.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from scriptguard.rag.code_sanitization import CodeSanitizer, ContextEnricher, create_sanitizer, create_enricher
from scriptguard.utils.logger import logger


def test_sanitization():
    """Test code sanitization with various inputs."""
    logger.info("=" * 60)
    logger.info("TEST 1: Code Sanitization")
    logger.info("=" * 60)

    sanitizer = CodeSanitizer()

    # Test 1: Valid Python code
    valid_code = """
import os
import sys

def hello():
    print("Hello, World!")
    
if __name__ == "__main__":
    hello()
"""

    cleaned, report = sanitizer.sanitize(valid_code, language="python")
    logger.info(f"\n✅ Valid Python Code:")
    logger.info(f"  Result: {'PASSED' if report['valid'] else 'REJECTED'}")
    logger.info(f"  Entropy: {report.get('entropy', 0):.2f}")
    logger.info(f"  Lines: {report.get('original_lines')} → {report.get('cleaned_lines')}")

    # Test 2: Binary data (base64)
    binary_code = "A" * 100 + "=" * 2  # Fake base64
    cleaned, report = sanitizer.sanitize(binary_code, language="python")
    logger.info(f"\n❌ Binary/Base64 Data:")
    logger.info(f"  Result: {'PASSED' if report['valid'] else 'REJECTED'}")
    logger.info(f"  Reason: {report.get('reason', 'N/A')}")

    # Test 3: Low entropy (repeated pattern)
    low_entropy_code = "print('a')\n" * 100
    cleaned, report = sanitizer.sanitize(low_entropy_code, language="python")
    logger.info(f"\n⚠️  Low Entropy Code:")
    logger.info(f"  Result: {'PASSED' if report['valid'] else 'REJECTED'}")
    logger.info(f"  Entropy: {report.get('entropy', 0):.2f}")
    logger.info(f"  Reason: {report.get('reason', 'N/A')}")

    # Test 4: Code with license header
    licensed_code = """
# Copyright (c) 2024 Company Name
# Licensed under MIT License
# Permission is hereby granted...

import requests

def fetch_data():
    return requests.get('https://api.example.com')
"""

    cleaned, report = sanitizer.sanitize(licensed_code, language="python")
    logger.info(f"\n🔧 Code with License Header:")
    logger.info(f"  Result: {'PASSED' if report['valid'] else 'REJECTED'}")
    logger.info(f"  Original length: {report.get('original_length')}")
    logger.info(f"  Cleaned length: {report.get('cleaned_length')}")
    logger.info(f"  Compression: {report.get('compression_ratio', 0):.1%}")

    # Test 5: Invalid syntax (strict mode)
    invalid_code = """
def broken_function(
    print("Missing closing paren"
"""

    sanitizer_strict = CodeSanitizer(strict_mode=True)
    cleaned, report = sanitizer_strict.sanitize(invalid_code, language="python")
    logger.info(f"\n❌ Invalid Syntax (Strict Mode):")
    logger.info(f"  Result: {'PASSED' if report['valid'] else 'REJECTED'}")
    logger.info(f"  Reason: {report.get('reason', 'N/A')}")
    if 'syntax_report' in report:
        logger.info(f"  Syntax error: {report['syntax_report'].get('error', 'N/A')}")


def test_context_injection():
    """Test context enrichment."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Context Injection")
    logger.info("=" * 60)

    code = """
import socket

def reverse_shell(host, port):
    s = socket.socket()
    s.connect((host, port))
    os.dup2(s.fileno(), 0)
"""

    metadata = {
        "file_path": "malware/reverse_shell.py",
        "repository": "attacker/toolkit",
        "language": "python",
        "source": "github",
        "label": "malicious"
    }

    # Test structured format
    enricher = ContextEnricher(injection_format="structured")
    enriched = enricher.enrich(code, metadata)

    logger.info("\n📋 Structured Format:")
    logger.info("─" * 60)
    logger.info(enriched[:300] + "...")

    # Test inline format
    enricher = ContextEnricher(injection_format="inline")
    enriched = enricher.enrich(code, metadata)

    logger.info("\n📝 Inline Format:")
    logger.info("─" * 60)
    logger.info(enriched[:200] + "...")

    # Test minimal format
    enricher = ContextEnricher(injection_format="minimal")
    enriched = enricher.enrich(code, metadata)

    logger.info("\n🔖 Minimal Format:")
    logger.info("─" * 60)
    logger.info(enriched[:150] + "...")


def test_factory_functions():
    """Test factory functions with config."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Factory Functions (Config-based)")
    logger.info("=" * 60)

    config = {
        "sanitization": {
            "enabled": True,
            "min_entropy": 4.0,
            "strict_mode": True
        },
        "context_injection": {
            "enabled": True,
            "injection_format": "inline"
        }
    }

    sanitizer = create_sanitizer(config["sanitization"])
    enricher = create_enricher(config["context_injection"])

    logger.info(f"✅ Sanitizer created: min_entropy={sanitizer.min_entropy}")
    logger.info(f"✅ Enricher created: format={enricher.injection_format}")

    # Test with sample code
    code = "import sys\nprint('Hello')\n"
    cleaned, report = sanitizer.sanitize(code)

    if cleaned:
        enriched = enricher.enrich(cleaned, {"file_path": "test.py"})
        logger.info(f"\n📦 Pipeline result:")
        logger.info(f"  Valid: {report['valid']}")
        logger.info(f"  Enriched: {len(enriched)} chars")


def test_integration():
    """Test full integration: sanitize + enrich."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Full Integration (Sanitize → Enrich)")
    logger.info("=" * 60)

    # Simulate real code sample
    raw_code = """
# MIT License
# Copyright (c) 2024

import os
import subprocess


def execute_command(cmd):
    '''Execute system command'''
    result = subprocess.run(cmd, shell=True, capture_output=True)
    return result.stdout.decode()


if __name__ == '__main__':
    print(execute_command('ls -la'))
"""

    metadata = {
        "file_path": "src/utils/executor.py",
        "repository": "security-tools/scanner",
        "language": "python",
        "source": "github",
        "label": "benign"
    }

    # Step 1: Sanitize
    sanitizer = CodeSanitizer(remove_license_headers=True)
    cleaned, report = sanitizer.sanitize(raw_code, language="python", metadata=metadata)

    logger.info(f"\n🔍 Sanitization:")
    logger.info(f"  Valid: {report['valid']}")
    logger.info(f"  Entropy: {report.get('entropy', 0):.2f}")
    logger.info(f"  Size: {report.get('original_length')} → {report.get('cleaned_length')} bytes")
    logger.info(f"  Warnings: {len(report.get('warnings', []))}")

    if cleaned:
        # Step 2: Enrich
        enricher = ContextEnricher(injection_format="structured")
        enriched = enricher.enrich(cleaned, metadata)

        logger.info(f"\n📦 Context Injection:")
        logger.info(f"  Enriched size: {len(enriched)} bytes")
        logger.info(f"\n  Preview:")
        logger.info("  " + "─" * 56)
        for line in enriched.split('\n')[:12]:
            logger.info(f"  {line}")
        logger.info("  " + "─" * 56)


if __name__ == "__main__":
    logger.info("🚀 Starting Code Sanitization & Context Injection Tests\n")

    try:
        test_sanitization()
        test_context_injection()
        test_factory_functions()
        test_integration()

        logger.info("\n" + "=" * 60)
        logger.info("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {e}", exc_info=True)
        sys.exit(1)
