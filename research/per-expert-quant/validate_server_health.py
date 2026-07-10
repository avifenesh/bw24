#!/usr/bin/env python3
"""Validate that a bw24 health response serves the expected model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("health", type=Path)
    parser.add_argument("model")
    parser.add_argument(
        "--exact",
        action="store_true",
        help="require the expected model to be the server's only model",
    )
    args = parser.parse_args()

    health = json.loads(args.health.read_text())
    if health.get("status") != "ok":
        raise SystemExit(f"{args.health}: server status is not ok")
    models = health.get("models")
    if not isinstance(models, list) or not all(isinstance(model, str) for model in models):
        raise SystemExit(f"{args.health}: models must be a list of strings")
    if args.model not in models:
        raise SystemExit(f"{args.health}: expected model {args.model!r}, got {models!r}")
    if args.exact and models != [args.model]:
        raise SystemExit(f"{args.health}: expected only model {args.model!r}, got {models!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
