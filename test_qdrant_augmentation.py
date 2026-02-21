"""
Test script to verify Qdrant augmentation works correctly.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import yaml
from scriptguard.steps.qdrant_augmentation import augment_with_qdrant_patterns
from scriptguard.utils.logger import logger

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create minimal test data
test_data = [
    {
        'code': 'import os\nprint("Hello")',
        'label': 'benign',
        'source': 'test',
        'id': 1
    },
    {
        'code': 'import subprocess\nsubprocess.call(["rm", "-rf", "/"])',
        'label': 'malicious',
        'source': 'test',
        'id': 2
    }
]

logger.info(f"Starting test with {len(test_data)} samples")
logger.info(f"Config use_qdrant_patterns: {config.get('augmentation', {}).get('use_qdrant_patterns', False)}")

# Call augmentation step
try:
    augmented_data = augment_with_qdrant_patterns(
        data=test_data,
        config=config
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ AUGMENTATION SUCCESSFUL")
    logger.info(f"{'='*60}")
    logger.info(f"Original samples: {len(test_data)}")
    logger.info(f"Augmented samples: {len(augmented_data)}")
    logger.info(f"Added samples: {len(augmented_data) - len(test_data)}")

    # Count by source
    sources = {}
    for sample in augmented_data:
        source = sample.get('source', 'unknown')
        sources[source] = sources.get(source, 0) + 1

    logger.info(f"\nSample distribution by source:")
    for source, count in sources.items():
        logger.info(f"  {source}: {count}")

except Exception as e:
    logger.error(f"❌ AUGMENTATION FAILED: {e}", exc_info=True)
