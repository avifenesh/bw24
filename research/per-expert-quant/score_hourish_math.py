#!/usr/bin/env python3
"""Score frozen MATH-500 generations with a strict first-answer policy."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import re
import signal
from typing import Any

from math_verify import parse, verify


TASK = "hendrycks_math500"
MAX_ANSWER_CHARS = 4096
VERIFY_TIMEOUT_SECONDS = 5
ANSWER_CLAUSE = re.compile(r"(?i)\banswer\s*:\s*")
UNIT_SUFFIX = re.compile(r"\\text\{\s*[^{}]+\s*\}\s*$")
PLAIN_FRACTION = re.compile(r"^(-?\d+)/(-?\d+)$")
VERSIONS = {
    name: importlib.metadata.version(name)
    for name in (
        "antlr4-python3-runtime",
        "latex2sympy2-extended",
        "math-verify",
        "mpmath",
        "sympy",
    )
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_from(row: dict[str, Any]) -> str:
    filtered = row.get("filtered_resps")
    if not isinstance(filtered, list) or len(filtered) != 1:
        raise ValueError("expected exactly one filtered candidate")
    candidate = filtered[0]
    if isinstance(candidate, list) and len(candidate) == 1:
        candidate = candidate[0]
    if not isinstance(candidate, str):
        raise ValueError("expected exactly one filtered candidate")
    return candidate


def first_answer_line(generation: str) -> str:
    """Honor the Answer: prompt without crediting later self-corrections."""
    line = next((line.strip() for line in generation.splitlines() if line.strip()), "")
    clauses = list(ANSWER_CLAUSE.finditer(line))
    if clauses:
        line = line[clauses[-1].end() :].strip()
    line = line.rstrip().removesuffix(".").rstrip()
    if len(line) > MAX_ANSWER_CHARS:
        raise ValueError("first answer line is too long")
    return line


def unwrap_outer(value: str, prefix: str) -> str:
    if value.startswith(prefix) and value.endswith("}"):
        return value[len(prefix) : -1].strip()
    return value


def fix_fracs(value: str) -> str:
    parts = value.split("\\frac")
    output = parts[0]
    for part in parts[1:]:
        output += "\\frac"
        if not part or part[0] == "{":
            output += part
            continue
        if len(part) < 2:
            return value
        numerator, rest = part[0], part[1:]
        if rest[0] == "{":
            output += "{" + numerator + "}" + rest
        else:
            output += "{" + numerator + "}{" + rest[0] + "}" + rest[1:]
    return output


def literal_normalize(value: str) -> str:
    value = value.strip().removesuffix(".").rstrip()
    if value.startswith("$$") and value.endswith("$$") and len(value) >= 4:
        value = value[2:-2].strip()
    elif value.startswith("$") and value.endswith("$") and len(value) >= 2:
        value = value[1:-1].strip()
    value = unwrap_outer(value, "\\boxed{")
    if value.startswith("\\text{") and value.endswith("}"):
        value = unwrap_outer(value, "\\text{")
    else:
        value = UNIT_SUFFIX.sub("", value).strip()
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\!", "").replace("\\,", "")
    value = re.sub(r"\s+", "", value)
    value = fix_fracs(value)
    fraction = PLAIN_FRACTION.fullmatch(value)
    if fraction:
        value = f"\\frac{{{fraction.group(1)}}}{{{fraction.group(2)}}}"
    if value.startswith("."):
        value = "0" + value
    return value


class VerificationTimeout:
    def __enter__(self) -> None:
        signal.signal(signal.SIGALRM, self._raise)
        signal.alarm(VERIFY_TIMEOUT_SECONDS)

    def __exit__(self, *_: object) -> None:
        signal.alarm(0)

    @staticmethod
    def _raise(*_: object) -> None:
        raise TimeoutError("math verification timed out")


def parse_complete_expression(value: str) -> list[Any]:
    expression = value if value.startswith("$") and value.endswith("$") else f"${value}$"
    try:
        return parse(expression)
    except Exception:
        return []


def equivalent(candidate: str, target: str) -> tuple[bool, str]:
    normalized_candidate = literal_normalize(candidate)
    normalized_target = literal_normalize(target)
    if normalized_candidate == normalized_target or (
        normalized_candidate.isalpha()
        and normalized_target.isalpha()
        and normalized_candidate.casefold() == normalized_target.casefold()
    ):
        return True, "normalized_literal"
    if not candidate:
        return False, "none"
    try:
        with VerificationTimeout():
            gold = parse_complete_expression(target)
            prediction = parse_complete_expression(candidate)
            if gold and prediction and verify(gold=gold, target=prediction):
                return True, "math_verify"
    except Exception:
        pass
    return False, "none"


def score(paths: list[pathlib.Path], output_format: str = "bw24-hourish-math-score-v1") -> dict[str, Any]:
    if len(paths) != 1:
        raise ValueError(f"expected exactly one sample file, found {len(paths)}")
    path = paths[0]
    if not path.name.startswith(f"samples_{TASK}_") or path.suffix != ".jsonl":
        raise ValueError(f"unexpected sample file: {path}")
    samples = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            doc_id = row.get("doc_id")
            target = row.get("target")
            if not isinstance(doc_id, int) or not isinstance(target, str):
                raise ValueError(f"invalid sample identity at {path}:{line_number}")
            if doc_id in seen:
                raise ValueError(f"duplicate sample {doc_id}")
            seen.add(doc_id)
            answer = first_answer_line(candidate_from(row))
            passed, method = equivalent(answer, target)
            samples.append(
                {
                    "task": TASK,
                    "doc_id": doc_id,
                    "doc_hash": row.get("doc_hash"),
                    "prompt_hash": row.get("prompt_hash"),
                    "target_hash": row.get("target_hash"),
                    "answer": answer,
                    "normalized_answer": literal_normalize(answer),
                    "passed": passed,
                    "method": method,
                }
            )
    passed = sum(int(row["passed"]) for row in samples)
    return {
        "format": output_format,
        "policy": {
            "answer_selection": "first_nonempty_line_then_same_line_answer_clause",
            "later_lines_ignored": True,
            "max_answer_chars": MAX_ANSWER_CHARS,
            "verification_timeout_seconds": VERIFY_TIMEOUT_SECONDS,
        },
        "versions": VERSIONS,
        "input_files": [{"path": str(path), "sha256": sha256(path)}],
        "by_task": {TASK: {"passed": passed, "total": len(samples)}},
        "passed": passed,
        "total": len(samples),
        "samples": samples,
    }


def self_test() -> None:
    assert first_answer_line(" 28\n\nProblem:") == "28"
    assert first_answer_line("wrong\nExplanation ending in the correct answer") == "wrong"
    assert first_answer_line("Reasoning. So answer: east.\nProblem:") == "east"
    assert equivalent("east", "\\text{east}")[0]
    assert equivalent("1/4", "\\frac14")[0]
    assert equivalent("$(15,-29)$", "(15,-29)")[0]
    assert equivalent("7!/(2!2!)=1260", "1260")[0]
    assert not equivalent("1/(137)-i", "1+274i")[0]
    assert not equivalent("6*sqrt(3)", "3 \\sqrt{5}")[0]
    print("hourish math scorer self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", nargs="*", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--format",
        choices=("bw24-hourish-math-score-v1", "bw24-promoted-math-score-v1"),
        default="bw24-hourish-math-score-v1",
    )
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    print(json.dumps(score(args.samples, args.format), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
