#!/usr/bin/env python3
"""Run the frozen hourish summarizer for task-isolated endpoint identities.

The bounded Hy3 screen gives every task its own server endpoint, named
``<arm>-<task>``.  The frozen summarizer assumes one endpoint named exactly
``<arm>``.  This postprocess-only adapter verifies the exact frozen source,
changes only those three identity expectations in a generated copy, and then
executes that copy with otherwise unchanged arguments.  It never launches a
model and never rewrites generation or scoring evidence.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import sys


EXPECTED_SOURCE_SHA256 = (
    "263b8070ee1afa7dc494b80a1ff58cb28bce692087f8a16d516ebf7250bf0b36"
)


def main() -> None:
    if len(sys.argv) < 5 or sys.argv[1] != "--source" or sys.argv[3] != "--generated":
        raise SystemExit(
            "usage: recover_hy3_110gb_bounded_summary.py --source SOURCE "
            "--generated GENERATED [summarizer args...]"
        )
    source = pathlib.Path(sys.argv[2])
    generated = pathlib.Path(sys.argv[4])
    original = source.read_bytes()
    actual = hashlib.sha256(original).hexdigest()
    if actual != EXPECTED_SOURCE_SHA256:
        raise SystemExit(f"frozen summarizer SHA mismatch: {actual}")

    text = original.decode()
    needle = '"model": arm,'
    if text.count(needle) != 2:
        raise SystemExit("unexpected endpoint-model expectation count")
    text = text.replace(needle, '"model": f"{arm}-{task}",')
    needle = 'result.get("model_name") != arm'
    if text.count(needle) != 1:
        raise SystemExit("unexpected result-model expectation count")
    text = text.replace(needle, 'result.get("model_name") != f"{arm}-{task}"')

    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text(text)
    os.execv(sys.executable, [sys.executable, str(generated), *sys.argv[5:]])


if __name__ == "__main__":
    main()
