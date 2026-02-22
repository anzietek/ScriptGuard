"""
extract_features ZenML steps.

Steps:
  cache_features   — runs BEFORE split_data; computes and persists features for
                     ALL ingested samples to PostgreSQL so that the post-split
                     extract_features step is always a pure DB cache hit.

  extract_features — runs AFTER tokenize_data; loads features from DB (cache
                     hit after cache_features ran), fits StandardScaler on train
                     split only, and attaches a `feature_vector` column to each
                     HuggingFace Dataset split.

The `feature_vector` is keyed by `script_id` so every chunk of the same script
gets the same feature vector (parent-level signal, not chunk-level signal).
"""

import os
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from datasets import Dataset
from sklearn.preprocessing import StandardScaler
from zenml import step, ArtifactConfig
from typing import Annotated

from scriptguard.database.feature_store import load_features_from_db, save_features_to_db
from scriptguard.features.extractor import FeatureExtractor
from scriptguard.materializers.dataset_materializer import HuggingFaceDatasetMaterializer
from scriptguard.utils.logger import logger


# ---------------------------------------------------------------------------
# Pre-split step: cache features for ALL ingested samples
# ---------------------------------------------------------------------------

@step(enable_cache=False)
def cache_features(
    all_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Pre-split step: compute 61-dim feature vectors for every ingested sample
    that doesn't already have them in PostgreSQL, then persist them to DB.

    Runs BEFORE split_data so that the post-split extract_features step is
    always a pure DB cache hit regardless of the split configuration.

    Returns all_data unchanged — output is wired to split_data to enforce
    correct DAG ordering.
    """
    extractor = FeatureExtractor()

    db_ids: list[int] = [d["id"] for d in all_data if d.get("id") is not None]
    logger.info(f"cache_features: {len(db_ids)} / {len(all_data)} samples have DB IDs")

    cached: dict[int, list[float]] = load_features_from_db(db_ids) if db_ids else {}
    logger.info(f"cache_features: {len(cached)} already in DB")

    newly_computed: dict[int, list[float]] = {}
    for sample in all_data:
        sid = sample.get("id")
        if sid is None or sid in cached:
            continue
        newly_computed[sid] = extractor.extract(sample.get("content") or "")

    if newly_computed:
        save_features_to_db(newly_computed)
        logger.info(f"cache_features: computed and saved {len(newly_computed)} new feature vectors")
    else:
        logger.info("cache_features: all features already cached — no computation needed")

    # ------------------------------------------------------------------
    # Diagnostic: per-feature discrimination report (benign vs malicious mean)
    # ------------------------------------------------------------------
    _FEATURE_NAMES = [
        # AST (13)
        "tree_depth", "n_calls", "n_imports", "n_funcdefs", "n_classdefs",
        "n_for", "n_while", "n_try", "n_exec_nodes", "has_nested_funcs",
        "exec_eval_depth", "has_decode_chain", "has_dynamic_import",
        # Import (11)
        "has_socket", "has_subprocess", "has_os_exec", "has_ctypes", "has_base64_import",
        "has_marshal", "has_pickle", "has_cryptography", "has_sock_and_subproc",
        "total_imports", "high_risk_imports",
        # Entropy (4)
        "mean_str_entropy", "max_str_entropy", "high_entropy_count", "has_hardcoded_key",
        # Obfuscation (11)
        "has_exec", "has_eval", "has_compile", "has_dunder_import",
        "has_b64decode_call", "has_no_comments", "has_encoded_exec",
        "has_very_long_line", "has_anti_debug", "has_ctypes_windll", "has_proc_injection",
        # Network/C2 (6)
        "unique_ip_count", "unique_url_count", "has_hardcoded_ports",
        "has_c2_pattern", "has_dns_lookup", "has_system_recon",
        # Persistence (4)
        "has_registry_write", "has_cron_pattern", "has_startup_persist", "has_creates_exec",
        # Crypto (4)
        "has_aes", "has_rc4", "has_xor_cipher", "has_fernet",
        # Recon/FS (3)
        "has_recursive_trav", "has_mass_file_ops", "has_shadow_copy",
        # Statistical (5)
        "total_lines", "comment_density", "avg_line_len", "max_line_len", "line_len_cv",
    ]

    all_vectors = {**cached, **newly_computed}
    if all_vectors:
        def _to_int_label(raw) -> int:
            if isinstance(raw, int):
                return raw
            if isinstance(raw, str):
                return 1 if raw.lower() == "malicious" else 0
            return -1

        label_by_id: dict[int, int] = {
            d["id"]: _to_int_label(d.get("label", -1))
            for d in all_data
            if d.get("id") is not None
        }

        benign_vecs    = [v for sid, v in all_vectors.items() if label_by_id.get(sid) == 0]
        malicious_vecs = [v for sid, v in all_vectors.items() if label_by_id.get(sid) == 1]

        n_b, n_m = len(benign_vecs), len(malicious_vecs)
        n_all_zero_b = sum(1 for v in benign_vecs if all(x == 0.0 for x in v))
        n_all_zero_m = sum(1 for v in malicious_vecs if all(x == 0.0 for x in v))

        logger.info(
            f"cache_features DIAGNOSTICS  benign={n_b}  malicious={n_m}  "
            f"all-zero: benign={n_all_zero_b} malicious={n_all_zero_m}"
        )

        if benign_vecs and malicious_vecs:
            dim = len(_FEATURE_NAMES)
            b_means = [sum(v[i] for v in benign_vecs) / n_b for i in range(dim)]
            m_means = [sum(v[i] for v in malicious_vecs) / n_m for i in range(dim)]

            # Sort by |delta| descending — most discriminative first
            deltas = [(m_means[i] - b_means[i], i) for i in range(dim)]
            deltas.sort(key=lambda x: abs(x[0]), reverse=True)

            lines = ["  Per-feature means (benign → malicious, Δ=mal-ben), sorted by |Δ|:"]
            lines.append(f"  {'Feature':<22} {'Benign':>8} {'Malicious':>10} {'Δ':>8}  Direction")
            lines.append("  " + "-" * 62)
            for delta, i in deltas:
                name = _FEATURE_NAMES[i]
                direction = "✓ MAL>" if delta > 0.05 else ("✗ BEN>" if delta < -0.05 else "  ~same")
                lines.append(
                    f"  {name:<22} {b_means[i]:>8.3f} {m_means[i]:>10.3f} {delta:>+8.3f}  {direction}"
                )
            logger.info("\n".join(lines))

    return all_data


def _gather_ids_and_content(
    data: List[Dict[str, Any]],
) -> tuple[list[int | None], list[str]]:
    """Return parallel lists of (id_or_None, content) from raw data dicts."""
    ids: list[int | None] = []
    contents: list[str] = []
    for sample in data:
        ids.append(sample.get("id"))
        contents.append(sample.get("content") or "")
    return ids, contents


@step(
    output_materializers={
        "train_tokens_feat": HuggingFaceDatasetMaterializer,
        "val_tokens_feat": HuggingFaceDatasetMaterializer,
        "test_tokens_feat": HuggingFaceDatasetMaterializer,
    }
)
def extract_features(
    train_data: List[Dict[str, Any]],
    val_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    train_tokens: Dataset,
    val_tokens: Dataset,
    test_tokens: Dataset,
    config: Dict[str, Any],
) -> Tuple[
    Annotated[Dataset, ArtifactConfig(name="train_tokens_feat")],
    Annotated[Dataset, ArtifactConfig(name="val_tokens_feat")],
    Annotated[Dataset, ArtifactConfig(name="test_tokens_feat")],
    Annotated[str, ArtifactConfig(name="scaler_path")],
]:
    """
    Extract 61-dimensional feature vectors for all splits and attach them to
    each HuggingFace Dataset as a `feature_vector` column.

    Returns:
        (train_tokens_feat, val_tokens_feat, test_tokens_feat, scaler_path)
    """
    codebert_cfg = config.get("codebert", {})
    output_dir: str = codebert_cfg.get("output_dir", "/workspace/models/codebert")
    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, "feature_scaler.joblib")

    extractor = FeatureExtractor()

    # ------------------------------------------------------------------
    # 1. Collect all DB sample IDs across all three splits
    # ------------------------------------------------------------------
    train_ids, train_contents = _gather_ids_and_content(train_data)
    val_ids, val_contents = _gather_ids_and_content(val_data)
    test_ids, test_contents = _gather_ids_and_content(test_data)

    all_db_ids = [sid for sid in (train_ids + val_ids + test_ids) if sid is not None]
    logger.info(f"extract_features: {len(all_db_ids)} samples have DB IDs across all splits")

    # ------------------------------------------------------------------
    # 2. Load cached features from DB
    # ------------------------------------------------------------------
    cached: dict[int, list[float]] = load_features_from_db(all_db_ids) if all_db_ids else {}
    logger.info(f"Loaded {len(cached)} feature vectors from DB cache")

    # ------------------------------------------------------------------
    # 3. Compute missing features and persist to DB
    # ------------------------------------------------------------------
    def _compute_for_split(
        ids: list[int | None],
        contents: list[str],
        split_name: str,
    ) -> list[list[float]]:
        """Compute and return feature vectors for one split (n_samples × 61)."""
        newly_computed: dict[int, list[float]] = {}
        result: list[list[float]] = []

        for sid, content in zip(ids, contents):
            if sid is not None and sid in cached:
                result.append(cached[sid])
            else:
                fvec = extractor.extract(content)
                result.append(fvec)
                if sid is not None:
                    newly_computed[sid] = fvec

        if newly_computed:
            logger.info(
                f"extract_features [{split_name}]: computed {len(newly_computed)} new feature vectors"
            )
            save_features_to_db(newly_computed)
            cached.update(newly_computed)
        else:
            logger.info(f"extract_features [{split_name}]: all features loaded from cache")

        return result

    train_raw = _compute_for_split(train_ids, train_contents, "train")  # [n_train × 61]
    val_raw = _compute_for_split(val_ids, val_contents, "val")
    test_raw = _compute_for_split(test_ids, test_contents, "test")

    # ------------------------------------------------------------------
    # 4. Fit StandardScaler on TRAIN features only
    # ------------------------------------------------------------------
    train_arr = np.array(train_raw, dtype=np.float32)  # [n_train, 61]
    scaler = StandardScaler()
    scaler.fit(train_arr)
    joblib.dump(scaler, scaler_path)
    logger.info(f"StandardScaler fitted on {len(train_arr)} training samples; saved to {scaler_path}")

    # ------------------------------------------------------------------
    # 5. Scale all splits using the train-fitted scaler
    # ------------------------------------------------------------------
    train_scaled = scaler.transform(train_arr).tolist()  # list of lists
    val_scaled = scaler.transform(np.array(val_raw, dtype=np.float32)).tolist()
    test_scaled = scaler.transform(np.array(test_raw, dtype=np.float32)).tolist()

    # ------------------------------------------------------------------
    # 6. Build script_id → scaled_feature lookup and attach to Datasets
    #
    #    tokenize_data assigns script_ids independently per split, all starting
    #    from 0 (script_id_offset=0 default in _tokenize_split).  So:
    #      train: script_ids 0 … len(train_data)-1
    #      val:   script_ids 0 … len(val_data)-1
    #      test:  script_ids 0 … len(test_data)-1
    #
    #    _attach_feature_column uses offset = min(script_ids) = 0, so
    #    scaled_features[script_id] maps directly to the i-th sample in the split.
    # ------------------------------------------------------------------
    def _attach_feature_column(
        dataset: Dataset,
        scaled_features: list[list[float]],
        split_label: str,
    ) -> Dataset:
        """Map dataset rows → scaled feature vectors via script_id offset."""
        if "script_id" not in dataset.column_names:
            # Fallback: repeat a zero vector for all chunks
            logger.warning(
                f"extract_features [{split_label}]: 'script_id' column missing; "
                "attaching zero feature vectors"
            )
            col = [[0.0] * FeatureExtractor.FEATURE_DIM for _ in range(len(dataset))]
            return dataset.add_column("feature_vector", col)

        script_ids: list[int] = dataset["script_id"]
        if not script_ids:
            return dataset.add_column("feature_vector", [])

        offset = min(script_ids)
        n_samples = len(scaled_features)

        col: list[list[float]] = []
        out_of_range = 0
        for sid in script_ids:
            idx = sid - offset
            if 0 <= idx < n_samples:
                col.append(scaled_features[idx])
            else:
                out_of_range += 1
                col.append([0.0] * FeatureExtractor.FEATURE_DIM)

        if out_of_range:
            logger.warning(
                f"extract_features [{split_label}]: {out_of_range} chunks had "
                "out-of-range script_id; used zero feature vector"
            )

        return dataset.add_column("feature_vector", col)

    train_tokens_feat = _attach_feature_column(train_tokens, train_scaled, "train")
    val_tokens_feat = _attach_feature_column(val_tokens, val_scaled, "val")
    test_tokens_feat = _attach_feature_column(test_tokens, test_scaled, "test")

    logger.info(
        f"extract_features: attached feature_vector column to "
        f"{len(train_tokens_feat)} train / {len(val_tokens_feat)} val / {len(test_tokens_feat)} test chunks"
    )

    return train_tokens_feat, val_tokens_feat, test_tokens_feat, scaler_path
