"""
ScriptGuard CLI

Usage:
    scriptguard classify <file>         [--model PATH] [--threshold FLOAT]
    scriptguard classify -               [--model PATH] [--threshold FLOAT]  # stdin
    python -m scriptguard.cli classify script.py --model /workspace/models/codebert
"""

import argparse
import os
import sys

from scriptguard.config_loader import load_config
from scriptguard.inference.classifier import ScriptGuardClassifier
from scriptguard.utils.logger import logger


def _resolve_model_path(args_model: str | None) -> str:
    if args_model:
        return args_model
    env = os.environ.get("SCRIPTGUARD_MODEL_PATH")
    if env:
        return env
    cfg = load_config()
    path = cfg.get("codebert", {}).get("output_dir")
    if path:
        return path
    print("ERROR: model path not specified. Use --model, SCRIPTGUARD_MODEL_PATH env var, "
          "or set codebert.output_dir in config.yaml", file=sys.stderr)
    sys.exit(2)


def cmd_classify(args: argparse.Namespace) -> None:
    model_path = _resolve_model_path(args.model)

    if args.file == "-":
        code = sys.stdin.read()
        source = "<stdin>"
    else:
        if not os.path.isfile(args.file):
            print(f"ERROR: file not found: {args.file}", file=sys.stderr)
            sys.exit(2)
        with open(args.file, "r", errors="replace") as f:
            code = f.read()
        source = args.file

    classifier = ScriptGuardClassifier(model_path)

    if args.threshold is not None:
        classifier._decision_threshold = args.threshold

    label, confidence = classifier.classify(code)

    print(f"{source}: {label.upper()}  (confidence={confidence:.4f})")

    sys.exit(1 if label == "malicious" else 0)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="scriptguard",
        description="ScriptGuard — malicious Python script detector",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    classify_parser = sub.add_parser("classify", help="Classify a Python script")
    classify_parser.add_argument(
        "file",
        help="Path to Python script, or '-' to read from stdin",
    )
    classify_parser.add_argument(
        "--model", "-m",
        default=None,
        help="Path to trained model directory (overrides config / env var)",
    )
    classify_parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=None,
        help="Decision threshold override (default: from inference_config.json)",
    )

    args = parser.parse_args()

    if args.command == "classify":
        cmd_classify(args)


if __name__ == "__main__":
    main()
