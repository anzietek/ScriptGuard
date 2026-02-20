from collections import Counter
from typing import Any, Dict, List, Tuple
from zenml import step, ArtifactConfig
from typing import Annotated
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from scriptguard.utils.logger import logger

# Strata with fewer than this many samples are merged into a "{label}|_rare" bucket
# to prevent sklearn from failing on single-sample strata.
_RARE_STRATUM_THRESHOLD = 2


def _label_to_int(label: str) -> int:
    return 1 if label == "malicious" else 0


def _build_strata(data: List[Dict[str, Any]]) -> np.ndarray:
    """
    Build a composite stratification key of the form ``"<label>|<keyword>"``.

    The keyword comes from ``sample["metadata"]["keyword"]`` when available.
    Strata that contain fewer than ``_RARE_STRATUM_THRESHOLD`` samples are
    collapsed into ``"<label>|_rare"`` so that sklearn never receives a stratum
    it cannot split.
    """
    raw: list[str] = []
    for s in data:
        label = s.get("label", "benign")
        meta = s.get("metadata")
        keyword = (meta.get("keyword") or "").strip() if isinstance(meta, dict) else ""
        raw.append(f"{label}|{keyword}" if keyword else f"{label}|")

    counts = Counter(raw)
    strata: list[str] = []
    for key in raw:
        label_part = key.split("|", 1)[0]
        if counts[key] < _RARE_STRATUM_THRESHOLD:
            strata.append(f"{label_part}|_rare")
        else:
            strata.append(key)

    return np.array(strata)


@step
def split_data(
    data: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Tuple[
    Annotated[List[Dict[str, Any]], ArtifactConfig(name="train_data")],
    Annotated[List[Dict[str, Any]], ArtifactConfig(name="val_data")],
    Annotated[List[Dict[str, Any]], ArtifactConfig(name="test_data")],
]:
    training_cfg = config.get("training", {})
    test_size: float = training_cfg.get("test_size", 0.15)
    val_size: float = training_cfg.get("val_size", 0.15)
    seed: int = training_cfg.get("seed", 42)

    indices = np.arange(len(data))

    families = [s.get("metadata", {}).get("family") if isinstance(s.get("metadata"), dict) else None for s in data]
    has_families = any(f is not None for f in families)

    if has_families:
        # Group-aware split: keep entire malware families in a single partition.
        labels = np.array([_label_to_int(s.get("label", "benign")) for s in data])
        groups = np.array([f if f is not None else f"_no_family_{i}" for i, f in enumerate(families)])
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_val_idx, test_idx = next(splitter.split(indices, labels, groups=groups))

        remaining_labels = labels[train_val_idx]
        remaining_groups = groups[train_val_idx]
        adjusted_val_size = val_size / (1.0 - test_size)

        val_splitter = GroupShuffleSplit(n_splits=1, test_size=adjusted_val_size, random_state=seed)
        rel_train_idx, rel_val_idx = next(
            val_splitter.split(train_val_idx, remaining_labels, groups=remaining_groups)
        )
        train_idx = train_val_idx[rel_train_idx]
        val_idx = train_val_idx[rel_val_idx]
    else:
        # Stratified split by composite (label, keyword) key so each malware
        # type is proportionally represented across all three partitions.
        strata = _build_strata(data)

        unique, counts = np.unique(strata, return_counts=True)
        logger.info(f"Stratification key distribution ({len(unique)} strata):")
        for key, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
            logger.info(f"  {key!r}: {cnt}")

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_val_idx, test_idx = next(splitter.split(indices, strata))

        remaining_strata = strata[train_val_idx]
        adjusted_val_size = val_size / (1.0 - test_size)
        val_splitter = StratifiedShuffleSplit(n_splits=1, test_size=adjusted_val_size, random_state=seed)
        rel_train_idx, rel_val_idx = next(val_splitter.split(train_val_idx, remaining_strata))
        train_idx = train_val_idx[rel_train_idx]
        val_idx = train_val_idx[rel_val_idx]

    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]
    test_data = [data[i] for i in test_idx]

    logger.info(
        f"Split: {len(train_data)} train / {len(val_data)} val / {len(test_data)} test "
        f"(group-aware={has_families})"
    )

    # Log per-keyword breakdown for each partition
    for split_name, split in [("train", train_data), ("val", val_data), ("test", test_data)]:
        keyword_counts: dict[str, int] = {}
        for s in split:
            label = s.get("label", "unknown")
            meta = s.get("metadata")
            keyword = (meta.get("keyword") or "").strip() if isinstance(meta, dict) else ""
            key = f"{label}|{keyword}" if keyword else label
            keyword_counts[key] = keyword_counts.get(key, 0) + 1
        top = sorted(keyword_counts.items(), key=lambda x: -x[1])
        logger.info(f"  {split_name} ({len(split)} samples): { {k: v for k, v in top[:10]} }")

    return train_data, val_data, test_data
