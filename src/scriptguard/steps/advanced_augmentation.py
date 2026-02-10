"""
Advanced Augmentation Step
Code obfuscation, mutation, and polymorphic variant generation.
"""

import base64
import random
import warnings
from scriptguard.utils.logger import logger
import ast
import re
from typing import Dict, List
from zenml import step

def obfuscate_base64(code: str) -> str:
    """
    Obfuscate code using base64 encoding.

    Args:
        code: Python code string

    Returns:
        Obfuscated code that decodes and executes original
    """
    encoded = base64.b64encode(code.encode()).decode()
    obfuscated = f"import base64; exec(base64.b64decode('{encoded}').decode())"
    return obfuscated

def obfuscate_hex(code: str) -> str:
    """
    Obfuscate code using hex encoding.

    Args:
        code: Python code string

    Returns:
        Hex-obfuscated code
    """
    hex_encoded = code.encode().hex()
    obfuscated = f"exec(bytes.fromhex('{hex_encoded}').decode())"
    return obfuscated

def obfuscate_rot13(code: str) -> str:
    """
    Obfuscate code using ROT13.

    Args:
        code: Python code string

    Returns:
        ROT13-obfuscated code
    """
    import codecs
    rot13 = codecs.encode(code, 'rot_13')
    obfuscated = f"import codecs; exec(codecs.decode('''{rot13}''', 'rot_13'))"
    return obfuscated

def rename_variables(code: str) -> str:
    """
    Rename variables in code to random names.

    Args:
        code: Python code string

    Returns:
        Code with renamed variables
    """
    try:
        # Suppress SyntaxWarning for invalid escape sequences in analyzed code
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=SyntaxWarning)
            tree = ast.parse(code)
    except SyntaxError:
        return code

    # Find all variable names
    var_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            var_names.add(node.id)

    # Filter out builtins and common names
    builtins = {'print', 'len', 'range', 'str', 'int', 'list', 'dict', 'open', 'exec', 'eval'}
    var_names = var_names - builtins

    # Create random mappings
    mapping = {}
    for var in var_names:
        new_name = f"var_{random.randint(1000, 9999)}"
        mapping[var] = new_name

    # Replace in code
    modified_code = code
    for old_name, new_name in mapping.items():
        # Use word boundaries to avoid partial replacements
        modified_code = re.sub(rf'\b{re.escape(old_name)}\b', new_name, modified_code)

    return modified_code

def split_strings(code: str) -> str:
    """
    Split string literals in code.

    Args:
        code: Python code string

    Returns:
        Code with split strings
    """
    # Find string literals and split them
    def split_string(match):
        string_content = match.group(1)
        if len(string_content) > 4:
            mid = len(string_content) // 2
            return f"'{string_content[:mid]}' + '{string_content[mid:]}'"
        return match.group(0)

    modified = re.sub(r"'([^']{4,})'", split_string, code)
    modified = re.sub(r'"([^"]{4,})"', lambda m: split_string(m).replace("'", '"'), modified)

    return modified

def add_junk_code(code: str) -> str:
    """
    Add junk code (dead code) to obfuscate.

    Args:
        code: Python code string

    Returns:
        Code with junk added
    """
    junk_lines = [
        "_ = 1 + 1",
        "__ = 'junk'",
        "___ = [1, 2, 3]",
        "____ = {'a': 1}",
        "_____ = None"
    ]

    lines = code.split("\n")
    result = []

    for line in lines:
        result.append(line)
        if random.random() < 0.1:  # 10% chance to add junk
            result.append(random.choice(junk_lines))

    return "\n".join(result)

def obfuscate_code(code: str, technique: str) -> str:
    """
    Obfuscate code using specified technique.

    Args:
        code: Python code string
        technique: Obfuscation technique name

    Returns:
        Obfuscated code
    """
    techniques = {
        "base64": obfuscate_base64,
        "hex": obfuscate_hex,
        "rot13": obfuscate_rot13,
        "rename_vars": rename_variables,
        "split_strings": split_strings,
        "add_junk": add_junk_code
    }

    if technique in techniques:
        try:
            return techniques[technique](code)
        except Exception as e:
            logger.warning(f"Obfuscation failed for technique {technique}: {e}")
            return code
    else:
        logger.warning(f"Unknown obfuscation technique: {technique}")
        return code

def generate_polymorphic_variant(sample: Dict) -> Dict:
    """
    Generate a polymorphic variant of a malicious sample.

    Args:
        sample: Original sample dictionary

    Returns:
        New variant sample
    """
    code = sample.get("content", "")

    # Choose random obfuscation technique
    techniques = ["base64", "hex", "rot13", "rename_vars", "split_strings", "add_junk"]
    technique = random.choice(techniques)

    obfuscated = obfuscate_code(code, technique)

    # Create new sample
    variant = {
        "id": None,  # CRITICAL: Augmented samples are synthetic (no DB ID)
        "content": obfuscated,
        "label": sample.get("label"),
        "source": sample.get("source") + "_augmented",
        "url": sample.get("url"),
        "metadata": {
            **sample.get("metadata", {}),
            "augmentation": technique,
            "original_sample": sample.get("content_hash"),
            "parent_id": sample.get("id")  # Track original DB ID if present
        }
    }

    return variant

@step
def augment_malicious_samples(
    data: List[Dict],
    variants_per_sample: int = 2
) -> List[Dict]:
    """
    Generate polymorphic variants of malicious samples.

    Args:
        data: List of code samples
        variants_per_sample: Number of variants to generate per malicious sample

    Returns:
        Original data plus augmented variants
    """
    logger.info(f"Augmenting malicious samples with {variants_per_sample} variants each...")

    malicious_samples = [s for s in data if s.get("label") == "malicious"]
    augmented_samples = []

    logger.info(f"Found {len(malicious_samples)} malicious samples to augment")

    for sample in malicious_samples:
        for _ in range(variants_per_sample):
            variant = generate_polymorphic_variant(sample)
            augmented_samples.append(variant)

    logger.info(f"Generated {len(augmented_samples)} augmented samples")

    # Combine original and augmented
    combined = data + augmented_samples

    logger.info(f"Total samples after augmentation: {len(combined)}")

    return combined

@step
def apply_obfuscation_techniques(
    data: List[Dict],
    techniques: List[str],
    apply_to_label: str = "malicious"
) -> List[Dict]:
    """
    Apply specific obfuscation techniques to samples.

    Args:
        data: List of code samples
        techniques: List of technique names
        apply_to_label: Which label to augment

    Returns:
        Augmented dataset
    """
    logger.info(f"Applying obfuscation techniques: {techniques}")

    target_samples = [s for s in data if s.get("label") == apply_to_label]
    augmented = []

    for sample in target_samples:
        for technique in techniques:
            obfuscated = obfuscate_code(sample.get("content", ""), technique)

            if obfuscated != sample.get("content"):  # Only add if changed
                augmented.append({
                    "id": None,  # CRITICAL: Augmented samples are synthetic (no DB ID)
                    "content": obfuscated,
                    "label": sample.get("label"),
                    "source": sample.get("source") + f"_{technique}",
                    "url": sample.get("url"),
                    "metadata": {
                        **sample.get("metadata", {}),
                        "obfuscation": technique,
                        "parent_id": sample.get("id")  # Track original DB ID if present
                    }
                })

    logger.info(f"Created {len(augmented)} obfuscated variants")

    return data + augmented

@step
def balance_dataset(
    data: List[Dict],
    target_ratio: float = 1.0,
    method: str = "undersample"
) -> List[Dict]:
    """
    Balance dataset by adjusting class distribution.

    Args:
        data: List of code samples
        target_ratio: Target ratio of malicious to benign (1.0 = equal)
        method: "undersample" or "oversample"

    Returns:
        Balanced dataset
    """
    malicious = [s for s in data if s.get("label") == "malicious"]
    benign = [s for s in data if s.get("label") == "benign"]

    logger.info(f"Original distribution: {len(malicious)} malicious, {len(benign)} benign")

    if method == "hybrid":
        # Hybrid: Undersample majority + augment minority intelligently
        target_size = min(len(malicious), len(benign))

        # Undersample majority class to match minority
        if len(malicious) > target_size:
            malicious = random.sample(malicious, target_size)
        if len(benign) > target_size:
            benign = random.sample(benign, target_size)

        logger.info(f"After undersampling: {len(malicious)} malicious, {len(benign)} benign")

        # Augment to reach target balance ratio (e.g., 1.0 for 1:1)
        target_malicious = int(target_size * target_ratio)

        if len(malicious) < target_malicious:
            augment_count = target_malicious - len(malicious)
            logger.info(f"Augmenting {augment_count} malicious variants to reach target")

            for _ in range(augment_count):
                original = random.choice(malicious[:target_size])  # Only augment original samples
                variant = generate_polymorphic_variant(original)
                malicious.append(variant)

    elif method == "undersample":
        # Reduce majority class
        if len(malicious) > len(benign) * target_ratio:
            # Too many malicious
            target_malicious = int(len(benign) * target_ratio)
            malicious = random.sample(malicious, target_malicious)
        elif len(benign) > len(malicious) / target_ratio:
            # Too many benign
            target_benign = int(len(malicious) / target_ratio)
            benign = random.sample(benign, target_benign)

    elif method == "oversample":
        # Intelligent oversampling - create augmented variants instead of simple duplication
        target_malicious = int(len(benign) * target_ratio)
        target_benign = int(len(malicious) / target_ratio)

        # For malicious samples, create augmented variants
        while len(malicious) < target_malicious:
            original = random.choice(malicious)
            # Create augmented variant instead of duplicate
            variant = generate_polymorphic_variant(original)
            malicious.append(variant)

        # For benign samples, just duplicate (we don't want to obfuscate benign code)
        while len(benign) < target_benign:
            benign.append(random.choice(benign))

    balanced = malicious + benign
    random.shuffle(balanced)

    logger.info(f"Balanced distribution: {len(malicious)} malicious, {len(benign)} benign")

    return balanced
