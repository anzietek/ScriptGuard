import os
from typing import List, Dict
from datetime import datetime

def save_samples_to_disk(samples: List[Dict], output_dir: str = "extracted_samples") -> List[str]:
    """
    Save extracted samples to disk.

    Args:
        samples: List of sample dicts with 'content' and 'metadata'
        output_dir: Directory to save files

    Returns:
        List of saved file paths
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saved_paths = []

    for i, sample in enumerate(samples):
        source = sample.get('source', 'unknown')
        file_name = sample.get('metadata', {}).get('file_name', f'sample_{i}')

        safe_name = file_name.replace('/', '_').replace('\\', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f"{source}_{timestamp}_{safe_name}")

        with open(output_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(sample['content'])

        saved_paths.append(output_path)

    return saved_paths
