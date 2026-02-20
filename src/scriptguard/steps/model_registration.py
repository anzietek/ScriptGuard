import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict
from zenml import step, ArtifactConfig
from typing import Annotated
from scriptguard.exceptions import ModelRegistrationError
from scriptguard.utils.logger import logger


@step
def register_model(
    metrics: Dict[str, Any],
    model_path: str,
    config: Dict[str, Any],
) -> Annotated[bool, ArtifactConfig(name="registered")]:
    codebert_cfg = config.get("codebert", {})
    threshold: float = codebert_cfg.get("eval_threshold", 0.92)

    if "malicious_recall" not in metrics:
        raise ModelRegistrationError(
            f"Cannot register model: 'malicious_recall' key missing from metrics. "
            f"Available keys: {list(metrics.keys())}"
        )

    recall: float = metrics["malicious_recall"]

    if recall >= threshold:
        latest_dir = os.path.join(os.path.dirname(model_path), "latest")
        if os.path.exists(latest_dir):
            shutil.rmtree(latest_dir)
        shutil.copytree(model_path, latest_dir)

        registration_meta = {
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "model_path": model_path,
            "latest_path": latest_dir,
            "metrics": metrics,
            "threshold": threshold,
        }
        with open(os.path.join(latest_dir, "registration_metadata.json"), "w") as f:
            json.dump(registration_meta, f, indent=2)

        logger.info(
            f"Model registered: malicious_recall={recall:.4f} >= threshold={threshold:.4f}. "
            f"Saved to {latest_dir}"
        )
        return True
    else:
        logger.warning(
            f"Model NOT registered: malicious_recall={recall:.4f} < threshold={threshold:.4f}. "
            f"Improve the model before deploying."
        )
        return False
