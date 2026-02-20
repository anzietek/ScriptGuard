from typing import Any, Dict, List
from zenml import step, ArtifactConfig
from typing import Annotated
from scriptguard.database.dataset_manager import DatasetManager
from scriptguard.database.deduplication import deduplicate_samples
from scriptguard.exceptions import DataIngestionError
from scriptguard.utils.logger import logger


@step
def ingest_data(
    config: Dict[str, Any],
) -> Annotated[List[Dict[str, Any]], ArtifactConfig(name="clean_data")]:
    manager = DatasetManager()
    raw = manager.get_all_samples()

    if not raw:
        raise DataIngestionError("No samples returned from PostgreSQL")

    valid = []
    skipped = 0
    for row in raw:
        content = row.get("content", "")
        label = row.get("label", "")
        if not content or not content.strip():
            skipped += 1
            continue
        if label not in ("malicious", "benign"):
            skipped += 1
            continue
        valid.append(row)

    logger.info(f"Fetched {len(raw)} rows; {skipped} skipped (empty/invalid); {len(valid)} valid")

    if not valid:
        raise DataIngestionError("All fetched samples were invalid (empty content or bad label)")

    validation_cfg = config.get("validation", {})
    dedup_threshold = validation_cfg.get("dedup_threshold", 0.92)
    dedup_method = validation_cfg.get("dedup_method", "auto")

    deduped = deduplicate_samples(
        samples=valid,
        threshold=dedup_threshold,
        method=dedup_method,
        enable_exact=validation_cfg.get("dedup_exact_first", True),
    )

    label_counts: dict[str, int] = {}
    for s in deduped:
        lbl = s.get("label", "unknown")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    logger.info(f"After deduplication: {len(deduped)} samples")
    for lbl, cnt in sorted(label_counts.items()):
        logger.info(f"  {lbl}: {cnt} ({cnt / len(deduped) * 100:.1f}%)")

    if not deduped:
        raise DataIngestionError("All samples were deduplicated away")

    return deduped
