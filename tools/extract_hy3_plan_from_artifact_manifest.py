#!/usr/bin/env python3
"""Recover and verify an embedded immutable allocation plan from an artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical_sha256(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--legacy-reference-plan",
        action="store_true",
        help=(
            "allow an older reference-only plan that predates canonical-hash and "
            "logical-byte receipts; the serialized plan hash is still required"
        ),
    )
    parser.add_argument(
        "--comparison-plan",
        action="store_true",
        help="allow a hash-locked plan above the 100GB candidate gate for delta analysis only",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    plan = manifest.get("plan")
    if not isinstance(plan, dict) or plan.get("format") != "bw24-expert-tier-plan-v2":
        raise SystemExit("manifest lacks a v2 allocation plan")
    if manifest.get("plan_sha256") != args.expected_plan_sha256:
        raise SystemExit("manifest plan receipt does not match the expected frozen hash")
    embedded_canonical = manifest.get("plan_canonical_sha256")
    actual_canonical = canonical_sha256(plan)
    if embedded_canonical is None and not args.legacy_reference_plan:
        raise SystemExit("manifest lacks a canonical plan receipt")
    if embedded_canonical is not None and embedded_canonical != actual_canonical:
        raise SystemExit(
            f"embedded plan canonical hash mismatch: {actual_canonical} != {embedded_canonical}"
        )
    logical_bytes = plan.get("policy", {}).get("result_logical_bytes")
    if logical_bytes is None and not args.legacy_reference_plan:
        raise SystemExit("manifest lacks a logical-byte receipt")
    if (
        logical_bytes is not None
        and int(logical_bytes) > 100_000_000_000
        and not args.comparison_plan
    ):
        raise SystemExit("embedded base plan exceeds the 100GB user gate")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    if sha256(args.out) != args.expected_plan_sha256:
        args.out.unlink()
        raise SystemExit("serialized plan hash does not match the frozen receipt")
    print(
        f"recovered {args.out} serialized_sha256={sha256(args.out)} "
        f"receipt_sha256={args.expected_plan_sha256} canonical_sha256={actual_canonical}"
    )


if __name__ == "__main__":
    main()
