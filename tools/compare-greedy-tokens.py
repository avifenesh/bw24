#!/usr/bin/env python3
"""Compare an hf-greedy-reference JSON record with a run-gen raw log."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


def first_mismatch(reference: list[int], candidate: list[int]) -> int | None:
    limit = min(len(reference), len(candidate))
    mismatch = next(
        (index for index in range(limit) if reference[index] != candidate[index]),
        None,
    )
    if mismatch is None and len(reference) != len(candidate):
        return limit
    return mismatch


def describe_mismatch(label: str, reference: list[int], candidate: list[int]) -> str | None:
    mismatch = first_mismatch(reference, candidate)
    if mismatch is None:
        return None
    ref_value = reference[mismatch] if mismatch < len(reference) else "<end>"
    memra_value = candidate[mismatch] if mismatch < len(candidate) else "<end>"
    return (
        f"{label} token {mismatch}: HF={ref_value} memra={memra_value}; "
        f"lengths {len(reference)} vs {len(candidate)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("memra_log", type=Path)
    args = parser.parse_args()

    record = json.loads(args.reference.read_text())
    log = args.memra_log.read_text()
    prompt_matches = re.findall(r"^prompt tokens:\s*(\[[^\n]*\])\s*$", log, re.MULTILINE)
    generated_matches = re.findall(r"^tokens:\s*(\[[^\n]*\])\s*$", log, re.MULTILINE)
    if not prompt_matches:
        print("HF-ARGMAX FAIL: run-gen log has no 'prompt tokens:' line")
        return 1
    if not generated_matches:
        print("HF-ARGMAX FAIL: run-gen log has no generated 'tokens:' line")
        return 1

    prompt = ast.literal_eval(prompt_matches[-1])
    generated = ast.literal_eval(generated_matches[-1])
    prompt_error = describe_mismatch("prompt", record["input_tokens"], prompt)
    if prompt_error:
        print(f"HF-ARGMAX FAIL: first mismatch at {prompt_error}")
        return 1
    generated_error = describe_mismatch("generated", record["tokens"], generated)
    if generated_error:
        print(f"HF-ARGMAX FAIL: first mismatch at {generated_error}")
        return 1
    print(
        f"HF-ARGMAX PASS: {len(prompt)} prompt tokens and "
        f"{len(generated)} generated tokens are identical"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
