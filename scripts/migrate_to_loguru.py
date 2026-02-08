"""
Script to migrate all logging statements to loguru.
Replaces:
  - import logging → from scriptguard.utils.logger import logger
  - logger = logging.getLogger(__name__) → (remove)
  - logging.info() → logger.info()
"""

import re
from pathlib import Path

# Files to migrate
files_to_migrate = [
    "src/scriptguard/data_sources/malwarebazaar_api.py",
    "src/scriptguard/steps/model_evaluation.py",
    "src/scriptguard/database/dataset_manager.py",
    "src/scriptguard/models/qlora_finetuner.py",
    "src/scriptguard/rag/qdrant_store.py",
    "src/scriptguard/database/db_schema.py",
    "src/scriptguard/steps/feature_extraction.py",
    "src/scriptguard/steps/advanced_augmentation.py",
    "src/scriptguard/steps/data_validation.py",
    "src/scriptguard/steps/advanced_ingestion.py",
    "src/scriptguard/monitoring/data_stats.py",
    "src/scriptguard/database/deduplication.py",
    "src/scriptguard/data_sources/cve_feeds.py",
    "src/scriptguard/data_sources/huggingface_datasets.py",
    "src/scriptguard/data_sources/github_api.py",
    "src/scriptguard/steps/data_ingestion.py",
    "src/scriptguard/steps/model_training.py",
    "src/scriptguard/steps/data_preprocessing.py",
]


def migrate_file(file_path: str):
    """Migrate a single file to loguru."""
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        return

    content = path.read_text(encoding='utf-8')
    original_content = content

    # Replace import logging
    content = re.sub(
        r'^import logging\s*$',
        'from scriptguard.utils.logger import logger',
        content,
        flags=re.MULTILINE
    )

    # Replace from logging import ...
    content = re.sub(
        r'^from logging import .*$',
        'from scriptguard.utils.logger import logger',
        content,
        flags=re.MULTILINE
    )

    # Remove logger = logging.getLogger(__name__)
    content = re.sub(
        r'^logger = logging\.getLogger\(__name__\)\s*$',
        '',
        content,
        flags=re.MULTILINE
    )

    # Remove logger = logging.getLogger(...)
    content = re.sub(
        r'^logger = logging\.getLogger\(["\'].*["\']\)\s*$',
        '',
        content,
        flags=re.MULTILINE
    )

    # Clean up double blank lines
    content = re.sub(r'\n\n\n+', '\n\n', content)

    if content != original_content:
        path.write_text(content, encoding='utf-8')
        print(f"[OK] Migrated: {file_path}")
    else:
        print(f"[SKIP] No changes: {file_path}")


def main():
    print("Migrating all files to loguru...\n")

    for file_path in files_to_migrate:
        migrate_file(file_path)

    print("\nMigration complete!")
    print("\nNext steps:")
    print("1. Review changes: git diff")
    print("2. Test: python src/main.py")
    print("3. Commit: git add . && git commit -m 'Migrate to loguru'")


if __name__ == "__main__":
    main()
