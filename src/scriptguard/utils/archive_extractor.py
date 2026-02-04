"""Archive extraction utilities for malware samples."""

try:
    import pyzipper
    HAS_PYZIPPER = True
except ImportError:
    pyzipper = None
    HAS_PYZIPPER = False
    import zipfile

import tarfile
import io
from typing import List, Tuple
from scriptguard.utils.logger import logger

try:
    import py7zr
    HAS_7Z = True
except ImportError:
    HAS_7Z = False
    py7zr = None
    logger.warning("py7zr not available - .7z archives will not be supported")

COMMON_PASSWORDS = [
    b"infected",
    b"malware",
    b"virus",
    b"password",
    b"",
]

SCRIPT_EXTENSIONS = {'.py', '.ps1', '.js', '.vbs', '.bat', '.sh', '.cmd'}

BINARY_EXTENSIONS = {'.exe', '.dll', '.so', '.dylib', '.bin', '.com', '.scr', '.pif',
                     '.msi', '.cab', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz',
                     '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.jar'}

BINARY_MAGIC_BYTES = [
    b'MZ',      # PE executables (.exe, .dll)
    b'PK\x03\x04',  # ZIP
    b'PK\x05\x06',  # Empty ZIP
    b'\x7fELF',     # ELF (Linux)
    b'\xcf\xfa\xed\xfe',  # Mach-O (macOS)
    b'Rar!',    # RAR
    b'7z\xbc\xaf\x27\x1c',  # 7Z
    b'\x1f\x8b',  # GZIP
    b'BZh',     # BZIP2
]

def is_binary_content(data: bytes) -> bool:
    """Check if data contains binary content."""
    if len(data) < 4:
        return False

    for magic in BINARY_MAGIC_BYTES:
        if data[:len(magic)] == magic:
            return True

    null_count = data[:1024].count(b'\x00')
    if null_count > 10:
        return True

    return False

def extract_from_zip(content: bytes) -> List[Tuple[str, str]]:
    """Extract script files from ZIP archive with AES-256 support."""
    results = []

    if HAS_PYZIPPER:
        for password in COMMON_PASSWORDS:
            try:
                with pyzipper.AESZipFile(io.BytesIO(content)) as zf:
                    for name in zf.namelist():
                        name_lower = name.lower()

                        if any(name_lower.endswith(ext) for ext in BINARY_EXTENSIONS):
                            continue

                        if any(name_lower.endswith(ext) for ext in SCRIPT_EXTENSIONS):
                            try:
                                data = zf.read(name, pwd=password if password else None)

                                if is_binary_content(data):
                                    logger.debug(f"Rejected binary file: {name}")
                                    continue

                                text = data.decode('utf-8', errors='ignore')

                                if len(text) > 50 and '\x00' not in text:
                                    results.append((name, text))
                            except Exception as e:
                                logger.debug(f"Failed to extract {name}: {e}")
                                continue
                    if results:
                        return results
            except:
                continue
    else:
        for password in COMMON_PASSWORDS:
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    for name in zf.namelist():
                        name_lower = name.lower()

                        if any(name_lower.endswith(ext) for ext in BINARY_EXTENSIONS):
                            continue

                        if any(name_lower.endswith(ext) for ext in SCRIPT_EXTENSIONS):
                            try:
                                try:
                                    data = zf.read(name, pwd=password if password else None)
                                except NotImplementedError:
                                    logger.debug(f"Skipping {name} - unsupported compression method")
                                    continue
                                except RuntimeError:
                                    try:
                                        data = zf.read(name)
                                    except:
                                        continue

                                if is_binary_content(data):
                                    logger.debug(f"Rejected binary file: {name}")
                                    continue

                                text = data.decode('utf-8', errors='ignore')

                                if len(text) > 50 and '\x00' not in text:
                                    results.append((name, text))
                            except Exception as e:
                                logger.debug(f"Failed to extract {name}: {e}")
                                continue
                    if results:
                        return results
            except:
                continue

    return results

def extract_from_tar(content: bytes) -> List[Tuple[str, str]]:
    """Extract script files from TAR/TAR.GZ archive."""
    results = []

    for compression in ['', 'gz', 'bz2', 'xz']:
        try:
            mode = 'r:' + compression if compression else 'r'
            with tarfile.open(fileobj=io.BytesIO(content), mode=mode) as tf:
                for member in tf.getmembers():
                    if member.isfile():
                        name_lower = member.name.lower()

                        if any(name_lower.endswith(ext) for ext in BINARY_EXTENSIONS):
                            continue

                        if any(name_lower.endswith(ext) for ext in SCRIPT_EXTENSIONS):
                            try:
                                f = tf.extractfile(member)
                                if f:
                                    data = f.read()

                                    if is_binary_content(data):
                                        logger.debug(f"Rejected binary file: {member.name}")
                                        continue

                                    text = data.decode('utf-8', errors='ignore')
                                    if len(text) > 50 and '\x00' not in text:
                                        results.append((member.name, text))
                            except:
                                continue
            if results:
                return results
        except:
            continue

    return results

def extract_from_7z(content: bytes) -> List[Tuple[str, str]]:
    """Extract script files from 7Z archive."""
    if not HAS_7Z:
        return []

    results = []

    for password in [pwd.decode('utf-8') if pwd else None for pwd in COMMON_PASSWORDS]:
        try:
            with py7zr.SevenZipFile(io.BytesIO(content), mode='r', password=password) as sz:
                for name, bio in sz.read().items():
                    name_lower = name.lower()

                    if any(name_lower.endswith(ext) for ext in BINARY_EXTENSIONS):
                        continue

                    if any(name_lower.endswith(ext) for ext in SCRIPT_EXTENSIONS):
                        try:
                            data = bio.read()

                            if is_binary_content(data):
                                logger.debug(f"Rejected binary file: {name}")
                                continue

                            text = data.decode('utf-8', errors='ignore')
                            if len(text) > 50 and '\x00' not in text:
                                results.append((name, text))
                        except:
                            continue
            if results:
                return results
        except:
            continue

    return results

def is_archive(content: bytes) -> bool:
    """Check if content is an archive."""
    if len(content) < 4:
        return False

    if content[:2] == b'PK':
        return True
    if content[:4] == b'\x1f\x8b\x08':
        return True
    if content[:3] == b'\x42\x5a\x68':
        return True
    if content[:6] == b'7z\xbc\xaf\x27\x1c':
        return True
    if content[:5] == b'ustar':
        return True

    return False

def extract_scripts_from_archive(content: bytes, filename: str = "") -> List[Tuple[str, str]]:
    """Extract all script files from any supported archive format."""
    results = []

    if content[:2] == b'PK':
        results = extract_from_zip(content)
        if results:
            logger.info(f"Extracted {len(results)} scripts from ZIP archive")
            return results

    if content[:4] == b'\x1f\x8b\x08' or content[:3] == b'\x42\x5a\x68' or b'ustar' in content[:512]:
        results = extract_from_tar(content)
        if results:
            logger.info(f"Extracted {len(results)} scripts from TAR archive")
            return results

    if HAS_7Z and content[:6] == b'7z\xbc\xaf\x27\x1c':
        results = extract_from_7z(content)
        if results:
            logger.info(f"Extracted {len(results)} scripts from 7Z archive")
            return results

    try:
        text = content.decode('utf-8', errors='ignore')
        if len(text) > 50 and any(ext in filename.lower() for ext in SCRIPT_EXTENSIONS):
            return [(filename, text)]
    except:
        pass

    return []
