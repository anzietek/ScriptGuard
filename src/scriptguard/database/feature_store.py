"""
Feature store for ScriptGuard fusion model.
Provides load/save operations for the 26-dimensional feature vectors stored in
the `samples.features` JSONB column of PostgreSQL.

Schema version history:
  v1 — 61-dim raw vector (deprecated, caused by pre-refactor dimension)
  v2 — 23-dim output (deprecated, FEATURE_DIM=23 refactor)
  v3 — 25-dim output (=25: added max_str_literal_len, long_line_ratio)
  v4 — 33-dim output (FEATURE_DIM=33: added 8 FP/FN mitigation features)
  v5 — 27-dim output (FEATURE_DIM=27: removed 6 zero-delta features, kept benign_framework_score + repetitive_identifier_ratio)
  v6 — 26-dim output (FEATURE_DIM=26: removed repetitive_identifier_ratio, Δ=-0.018 wrong direction)
  v7 — 27-dim output (FEATURE_DIM=27: malware_api_score now sums 50 flags; added 6 gadget/introspection binary flags at raw indices 69-74; narrowed has_ctypes_windll; C2 pattern excludes websocket libs)
"""

import json
from typing import Optional
from psycopg2.extras import execute_values
from scriptguard.database.db_schema import get_connection, return_connection
from scriptguard.features.extractor import FeatureExtractor
from scriptguard.utils.logger import logger

_SCHEMA_VERSION = 7


def load_features_from_db(sample_ids: list[int]) -> dict[int, list[float]]:
    """
    Load pre-computed feature vectors from the database.

    Args:
        sample_ids: List of sample IDs to look up.

    Returns:
        Dict mapping sample_id → feature_list for samples that have a valid,
        non-null features field with matching schema version.  Missing or
        version-mismatched entries are omitted (caller must recompute them).
    """
    if not sample_ids:
        return {}

    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, features FROM samples WHERE id = ANY(%s) AND features IS NOT NULL",
                (sample_ids,),
            )
            rows = cur.fetchall()

        result: dict[int, list[float]] = {}
        for row in rows:
            sid = row["id"]
            raw = row["features"]
            if raw is None:
                continue
            # raw may be a dict (psycopg2 auto-deserialises JSONB) or a string
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    continue
            if not isinstance(raw, dict):
                continue
            if raw.get("v") != _SCHEMA_VERSION:
                continue
            values = raw.get("values")
            if isinstance(values, list) and len(values) == FeatureExtractor.FEATURE_DIM:
                result[sid] = [float(v) for v in values]

        logger.info(f"Loaded {len(result)} feature vectors from DB (of {len(sample_ids)} requested)")
        return result

    except Exception as e:
        logger.error(f"feature_store.load_features_from_db failed: {e}")
        return {}
    finally:
        if conn is not None:
            return_connection(conn)


def save_features_to_db(features_by_id: dict[int, list[float]]) -> None:
    """
    Persist feature vectors to the database.

    Args:
        features_by_id: Dict mapping sample_id → 27-float feature list.

    Stores as JSONB: {"v": 7, "values": [27 floats]} for schema versioning.
    Uses a single batch UPDATE for efficiency.
    """
    if not features_by_id:
        return

    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            data = [
                (json.dumps({"v": _SCHEMA_VERSION, "values": fvec}), sid)
                for sid, fvec in features_by_id.items()
            ]
            execute_values(
                cur,
                "UPDATE samples SET features = data.features::jsonb "
                "FROM (VALUES %s) AS data(features, id) "
                "WHERE samples.id = data.id::int",
                data,
            )
        conn.commit()
        logger.info(f"Saved {len(features_by_id)} feature vectors to DB")

    except Exception as e:
        logger.error(f"feature_store.save_features_to_db failed: {e}")
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
    finally:
        if conn is not None:
            return_connection(conn)
