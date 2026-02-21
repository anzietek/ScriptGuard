"""
Test script to verify early data quality filter works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from scriptguard.utils.data_quality_filter import is_valid_source_code, quick_binary_check


def test_valid_python_code():
    """Test that valid Python code passes."""
    code = """
import os
import sys

def hello_world():
    print("Hello, world!")

if __name__ == "__main__":
    hello_world()
"""
    is_valid, reason = is_valid_source_code(code, ".py")
    assert is_valid, f"Valid Python code should pass, but got: {reason}"
    print("OK: Valid Python code passes")


def test_reject_binary_data():
    """Test that binary data is rejected."""
    binary_data = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff".decode('latin-1', errors='ignore')
    is_valid, reason = is_valid_source_code(binary_data, ".py")
    assert not is_valid, f"Binary data should be rejected, got: {reason}"
    # Accept any rejection reason (windows_executable, non_printable, low_ascii_ratio, etc.)
    print(f"OK: Binary data rejected ({reason})")


def test_reject_null_bytes():
    """Test that null bytes are rejected."""
    code_with_null = "import os\x00\x00\x00"
    is_valid, reason = is_valid_source_code(code_with_null, ".py")
    assert not is_valid, "Code with null bytes should be rejected"
    assert reason == "null_bytes_detected"
    print(f"OK: Null bytes rejected ({reason})")


def test_reject_excessive_base64():
    """Test that excessive base64 is rejected."""
    base64_spam = "A" * 100 + "B" * 100 + "C" * 100 + "D" * 100 + "E" * 100
    is_valid, reason = is_valid_source_code(base64_spam, ".py")
    assert not is_valid, "Excessive base64 should be rejected"
    assert "excessive_base64" in reason
    print(f"OK: Excessive base64 rejected ({reason})")


def test_reject_image_files():
    """Test that image files are rejected."""
    png_header = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR".decode('latin-1', errors='ignore')
    is_valid, reason = is_valid_source_code(png_header, ".png")
    assert not is_valid, f"PNG image should be rejected, got: {reason}"
    # Accept any rejection reason (png_image, invalid_extension, non_printable, etc.)
    print(f"OK: PNG image rejected ({reason})")


def test_reject_invalid_extension():
    """Test that invalid extensions are rejected."""
    code = "print('hello')"
    is_valid, reason = is_valid_source_code(code, ".exe")
    assert not is_valid, "Invalid extension should be rejected"
    assert "invalid_extension" in reason
    print(f"OK: Invalid extension rejected ({reason})")


def test_code_pattern_override():
    """Test that code patterns allow marginal quality."""
    # This has lowish ASCII ratio but clear Python code
    code_with_unicode = "# -*- coding: utf-8 -*-\nimport sys\ndef main():\n    print('test')\n"
    is_valid, reason = is_valid_source_code(code_with_unicode, ".py")
    # Should pass because it has clear code patterns (import, def)
    assert is_valid or reason == "", f"Code with patterns should pass, got: {reason}"
    print("OK: Code pattern override works")


def test_quick_binary_check():
    """Test quick binary check on raw bytes."""
    # Windows PE file
    pe_bytes = b"MZ\x90\x00\x03\x00\x00\x00"
    assert quick_binary_check(pe_bytes), "PE file should be detected as binary"
    print("OK: Quick binary check detects PE files")

    # Linux ELF file
    elf_bytes = b"\x7fELF\x02\x01\x01\x00"
    assert quick_binary_check(elf_bytes), "ELF file should be detected as binary"
    print("OK: Quick binary check detects ELF files")

    # ZIP file
    zip_bytes = b"PK\x03\x04\x14\x00\x00\x00"
    assert quick_binary_check(zip_bytes), "ZIP file should be detected as binary"
    print("OK: Quick binary check detects ZIP files")

    # Valid Python code (should NOT be detected as binary)
    python_bytes = b"import os\nimport sys\n"
    assert not quick_binary_check(python_bytes), "Python code should NOT be detected as binary"
    print("OK: Quick binary check allows Python code")


def test_empty_content():
    """Test that empty content is rejected."""
    is_valid, reason = is_valid_source_code("", ".py")
    assert not is_valid, "Empty content should be rejected"
    assert reason == "empty_content"
    print(f"OK: Empty content rejected ({reason})")


def test_too_short():
    """Test that too short content is rejected."""
    is_valid, reason = is_valid_source_code("abc", ".py")
    assert not is_valid, "Too short content should be rejected"
    assert reason == "too_short"
    print(f"OK: Too short content rejected ({reason})")


def test_obfuscated_but_valid():
    """Test that obfuscated but valid code passes."""
    # Obfuscated malware-like code (but still valid Python)
    obfuscated = """
import base64
exec(base64.b64decode(b'cHJpbnQoIkhlbGxvIik='))
"""
    is_valid, reason = is_valid_source_code(obfuscated, ".py")
    # Should pass because it has 'import' and 'exec' patterns
    assert is_valid or reason == "", f"Obfuscated code should pass if it has code patterns, got: {reason}"
    print("OK: Obfuscated but valid code passes")


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Data Quality Filter")
    print("=" * 80)

    try:
        test_valid_python_code()
        test_reject_binary_data()
        test_reject_null_bytes()
        test_reject_excessive_base64()
        test_reject_image_files()
        test_reject_invalid_extension()
        test_code_pattern_override()
        test_quick_binary_check()
        test_empty_content()
        test_too_short()
        test_obfuscated_but_valid()

        print("\n" + "=" * 80)
        print("OK ALL TESTS PASSED!")
        print("=" * 80)
        print("\nData quality filter is working correctly!")
        print("\nNext steps:")
        print("1. Run the full pipeline: python src/main.py --config config.yaml")
        print("2. Check logs for rejection statistics")
        print("3. Verify sanitization rejection rate is <5%")
        print("=" * 80)

    except AssertionError as e:
        print(f"\nFAIL TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\nFAIL UNEXPECTED ERROR: {e}")
        raise
