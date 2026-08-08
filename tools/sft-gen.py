#!/usr/bin/env python3
"""Generate verified SFT coding traces through opencode and a pinned DeepSeek model.

Each task runs in an isolated temporary git repository containing a small Python
fixture derived from a real memra tool shape. opencode emits raw JSON events; the
generator also exports the completed session, verifies the workspace outcome,
scans prompt material for secrets, and writes one normalized JSONL record.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import Any


DEFAULT_MODEL = "openrouter/deepseek/deepseek-v4-flash-0731"
DEFAULT_OPENCODE_CONFIG = Path("~/.config/opencode/opencode.json")
SCHEMA_VERSION = "memra.sft.trace.v1"
PROVIDER_ONLY = ("novita", "deepseek", "deepinfra", "fireworks")
FORBIDDEN_COMMAND_RE = re.compile(
    r"(^|[;&|()\s])(?:rustup|cargo|nvidia-smi|nvcc|rocminfo|rocm-smi)"
    r"(?=$|[;&|()\s])"
)
VOLATILE_KEYS = {
    "id",
    "sessionID",
    "messageID",
    "time",
    "timestamp",
    "snapshot",
    "cost",
    "tokens",
}
IGNORED_CONTENT_PARTS = {"step-start", "step-finish"}

SECRET_PATTERNS = (
    ("private-key", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----")),
    ("openai-style-key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("github-token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")),
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    (
        "assigned-secret",
        re.compile(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)"
            r"\s*[:=]\s*[\"'][^\"'\n]{8,}[\"']"
        ),
    ),
)

WORKSPACE_INSTRUCTIONS = """\
# Isolated SFT task instructions

- Work only inside this directory. Do not read parent directories or the user's home.
- Do not use the network.
- CPU only: do not invoke CUDA, ROCm, GPU utilities, or accelerator libraries.
- Never run rustup or cargo. These fixtures use only the Python standard library.
- Inspect the local files before editing and preserve the stated behavioral contract.
- Run `python3 -m unittest -v` before finishing.
- Do not edit this AGENTS.md file.
- Report the files changed and the verification command in the final response.
"""


def dedent(value: str) -> str:
    return textwrap.dedent(value).lstrip()


@dataclass(frozen=True)
class Scenario:
    slug: str
    module_name: str
    source_paths: tuple[str, ...]
    good_module: str
    buggy_module: str
    base_tests: str
    bug_tests: str
    bug_request: str
    refactor_request: str
    test_request: str
    explain_request: str


@dataclass(frozen=True)
class TaskTemplate:
    template_id: str
    category: str
    scenario: str
    source_paths: tuple[str, ...]
    prompt: str
    files: dict[str, str]
    module_name: str
    test_name: str


SCENARIOS = (
    Scenario(
        slug="cache-economics",
        module_name="cache_economics.py",
        source_paths=("tools/cache_economics.py",),
        good_module=dedent(
            """
            def _counter(metrics, key, default=0):
                value = metrics.get(key, default)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"{key} must be a non-negative integer")
                return value


            def row_from_metrics(metrics, factor):
                if not 0.0 <= factor <= 1.0:
                    raise ValueError("factor must be between 0 and 1")
                prompt = _counter(metrics, "prompt_tokens_in")
                cached = _counter(metrics, "cached_tokens_in")
                computed = _counter(metrics, "computed_tokens_in", prompt - cached)
                if prompt <= 0:
                    raise ValueError("no prompt tokens")
                if cached > prompt:
                    raise ValueError("cached tokens exceed prompt tokens")
                if computed != prompt - cached:
                    raise ValueError("computed tokens do not equal prompt minus cached")
                billed = computed + factor * cached
                return {
                    "prompt_tokens_in": prompt,
                    "cached_tokens_in": cached,
                    "computed_tokens_in": computed,
                    "cache_hit_token_ratio": cached / prompt,
                    "revenue_multiplier": billed / computed if computed else None,
                }
            """
        ),
        buggy_module=dedent(
            """
            def _counter(metrics, key, default=0):
                value = metrics.get(key, default)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"{key} must be a non-negative integer")
                return value


            def row_from_metrics(metrics, factor):
                if not 0.0 <= factor <= 1.0:
                    raise ValueError("factor must be between 0 and 1")
                prompt = _counter(metrics, "prompt_tokens_in")
                cached = _counter(metrics, "cached_tokens_in")
                computed = _counter(metrics, "computed_tokens_in", prompt - cached)
                if prompt <= 0:
                    raise ValueError("no prompt tokens")
                if cached > prompt:
                    raise ValueError("cached tokens exceed prompt tokens")
                billed = computed + factor * cached
                return {
                    "prompt_tokens_in": prompt,
                    "cached_tokens_in": cached,
                    "computed_tokens_in": computed,
                    "cache_hit_token_ratio": cached / prompt,
                    "revenue_multiplier": billed / computed if computed else None,
                }
            """
        ),
        base_tests=dedent(
            """
            import unittest

            from cache_economics import row_from_metrics


            class CacheEconomicsTests(unittest.TestCase):
                def test_mixed_cache_row(self):
                    row = row_from_metrics({
                        "prompt_tokens_in": 100,
                        "cached_tokens_in": 40,
                        "computed_tokens_in": 60,
                    }, 0.25)
                    self.assertAlmostEqual(row["cache_hit_token_ratio"], 0.4)
                    self.assertAlmostEqual(row["revenue_multiplier"], 70 / 60)


            if __name__ == "__main__":
                unittest.main()
            """
        ),
        bug_tests=dedent(
            """
            import unittest

            from cache_economics import row_from_metrics


            class CacheEconomicsTests(unittest.TestCase):
                def test_mixed_cache_row(self):
                    row = row_from_metrics({
                        "prompt_tokens_in": 100,
                        "cached_tokens_in": 40,
                        "computed_tokens_in": 60,
                    }, 0.25)
                    self.assertAlmostEqual(row["revenue_multiplier"], 70 / 60)

                def test_rejects_inconsistent_computed_counter(self):
                    with self.assertRaisesRegex(ValueError, "computed"):
                        row_from_metrics({
                            "prompt_tokens_in": 100,
                            "cached_tokens_in": 40,
                            "computed_tokens_in": 59,
                        }, 1.0)


            if __name__ == "__main__":
                unittest.main()
            """
        ),
        bug_request=(
            "A cumulative metrics scrape can carry an inconsistent computed-token counter. "
            "The failing test demonstrates that the row builder currently trusts it. Fix the "
            "smallest production-code bug without weakening the tests."
        ),
        refactor_request=(
            "Refactor validation and arithmetic so the counter invariants are clear at the "
            "boundary while preserving the exact returned fields and values."
        ),
        test_request=(
            "Add focused unittest coverage for zero prompt tokens, a fully cached prompt "
            "(unbounded multiplier represented as None), and a 0.25 billing factor."
        ),
        explain_request=(
            "Explain the difference between prompt, cached, computed, and billed tokens; state "
            "the invariants and work one numeric example through the multiplier."
        ),
    ),
    Scenario(
        slug="fleet-deltas",
        module_name="fleet_deltas.py",
        source_paths=("tools/fleet-report.py", "tools/cache_economics.py"),
        good_module=dedent(
            """
            COUNTER_KEYS = (
                "prompt_tokens_in",
                "cached_tokens_in",
                "computed_tokens_in",
            )


            def counters_regressed(previous, current):
                return any(current[key] < previous[key] for key in COUNTER_KEYS)


            def delta_rows(rows):
                result = []
                previous = None
                for row in rows:
                    reset = (
                        previous is None
                        or bool(row.get("restart"))
                        or counters_regressed(previous, row)
                    )
                    delta = {
                        key: row[key] if reset else row[key] - previous[key]
                        for key in COUNTER_KEYS
                    }
                    delta["restart"] = previous is not None and reset
                    result.append(delta)
                    previous = row
                return result
            """
        ),
        buggy_module=dedent(
            """
            COUNTER_KEYS = (
                "prompt_tokens_in",
                "cached_tokens_in",
            )


            def counters_regressed(previous, current):
                return any(current[key] < previous[key] for key in COUNTER_KEYS)


            def delta_rows(rows):
                result = []
                previous = None
                for row in rows:
                    reset = (
                        previous is None
                        or bool(row.get("restart"))
                        or counters_regressed(previous, row)
                    )
                    delta = {
                        key: row[key] if reset else row[key] - previous[key]
                        for key in (
                            "prompt_tokens_in",
                            "cached_tokens_in",
                            "computed_tokens_in",
                        )
                    }
                    delta["restart"] = previous is not None and reset
                    result.append(delta)
                    previous = row
                return result
            """
        ),
        base_tests=dedent(
            """
            import unittest

            from fleet_deltas import delta_rows


            class FleetDeltaTests(unittest.TestCase):
                def test_monotonic_counters_are_differenced(self):
                    rows = [
                        {"prompt_tokens_in": 100, "cached_tokens_in": 20, "computed_tokens_in": 80},
                        {"prompt_tokens_in": 140, "cached_tokens_in": 35, "computed_tokens_in": 105},
                    ]
                    self.assertEqual(delta_rows(rows)[1], {
                        "prompt_tokens_in": 40,
                        "cached_tokens_in": 15,
                        "computed_tokens_in": 25,
                        "restart": False,
                    })


            if __name__ == "__main__":
                unittest.main()
            """
        ),
        bug_tests=dedent(
            """
            import unittest

            from fleet_deltas import delta_rows


            class FleetDeltaTests(unittest.TestCase):
                def test_computed_counter_regression_infers_restart(self):
                    rows = [
                        {"prompt_tokens_in": 100, "cached_tokens_in": 20, "computed_tokens_in": 80},
                        {"prompt_tokens_in": 110, "cached_tokens_in": 40, "computed_tokens_in": 70},
                    ]
                    self.assertEqual(delta_rows(rows)[1], {
                        "prompt_tokens_in": 110,
                        "cached_tokens_in": 40,
                        "computed_tokens_in": 70,
                        "restart": True,
                    })


            if __name__ == "__main__":
                unittest.main()
            """
        ),
        bug_request=(
            "A worker restart can be visible only in computed_tokens_in while the other "
            "cumulative counters still increase. The implementation misses that reset. Fix "
            "the counter contract and keep delta behavior unchanged otherwise."
        ),
        refactor_request=(
            "Refactor reset detection and counter differencing into small named helpers. Do "
            "not change the output shape or mutate input rows."
        ),
        test_request=(
            "Add tests for an explicit restart marker, an inferred restart, the first snapshot, "
            "and input immutability."
        ),
        explain_request=(
            "Explain why cumulative snapshots require restart-aware differencing and how a "
            "missed reset corrupts daily cache economics."
        ),
    ),
    Scenario(
        slug="acceptance-parser",
        module_name="acceptance_parser.py",
        source_paths=("tools/acceptance_parse.py",),
        good_module=dedent(
            """
            import re

            ACCEPTANCE_RE = re.compile(
                r"acceptance:\\s*(\\d+)/(\\d+)\\s*=\\s*([\\d.]+)%"
            )
            CONSISTENCY_RE = re.compile(r"self-consistency:\\s*(PASS|FAIL)")


            def parse_run(text):
                row = {
                    "accepted": None,
                    "drafted": None,
                    "acc_rate": None,
                    "self_consistency": None,
                }
                match = ACCEPTANCE_RE.search(text)
                if match:
                    row["accepted"] = int(match.group(1))
                    row["drafted"] = int(match.group(2))
                    row["acc_rate"] = float(match.group(3)) / 100.0
                else:
                    row["error"] = "no acceptance line parsed"
                    row["tail"] = "\\n".join(text.strip().splitlines()[-4:])
                match = CONSISTENCY_RE.search(text)
                if match:
                    row["self_consistency"] = match.group(1)
                return row
            """
        ),
        buggy_module=dedent(
            """
            import re

            ACCEPTANCE_RE = re.compile(
                r"acceptance:\\s*(\\d+)/(\\d+)\\s*=\\s*([\\d.]+)%"
            )
            CONSISTENCY_RE = re.compile(r"self-consistency:\\s*(PASS|FAIL)")


            def parse_run(text):
                row = {
                    "accepted": None,
                    "drafted": None,
                    "acc_rate": None,
                    "self_consistency": None,
                }
                match = ACCEPTANCE_RE.search(text)
                if match:
                    row["accepted"] = int(match.group(1))
                    row["drafted"] = int(match.group(2))
                    row["acc_rate"] = float(match.group(3))
                else:
                    row["error"] = "no acceptance line parsed"
                    row["tail"] = "\\n".join(text.strip().splitlines()[-4:])
                match = CONSISTENCY_RE.search(text)
                if match:
                    row["self_consistency"] = match.group(1)
                return row
            """
        ),
        base_tests=dedent(
            """
            import unittest

            from acceptance_parser import parse_run


            class AcceptanceParserTests(unittest.TestCase):
                def test_parses_acceptance_and_consistency(self):
                    row = parse_run(
                        "acceptance: 27/40 = 67.5%   self-consistency: PASS"
                    )
                    self.assertEqual(row["accepted"], 27)
                    self.assertEqual(row["drafted"], 40)
                    self.assertEqual(row["self_consistency"], "PASS")


            if __name__ == "__main__":
                unittest.main()
            """
        ),
        bug_tests=dedent(
            """
            import unittest

            from acceptance_parser import parse_run


            class AcceptanceParserTests(unittest.TestCase):
                def test_percentage_is_normalized_to_fraction(self):
                    row = parse_run(
                        "acceptance: 27/40 = 67.5%   self-consistency: PASS"
                    )
                    self.assertAlmostEqual(row["acc_rate"], 0.675)


            if __name__ == "__main__":
                unittest.main()
            """
        ),
        bug_request=(
            "The parser stores a displayed percentage as though it were already a fraction. "
            "Fix the normalization while preserving the raw accepted/drafted counts."
        ),
        refactor_request=(
            "Refactor the parser so field extraction and missing-evidence handling are easy "
            "to extend, without changing current keys or failure text."
        ),
        test_request=(
            "Add tests for 100% acceptance, missing acceptance output with the exact last-four-"
            "line tail, and a FAIL self-consistency verdict."
        ),
        explain_request=(
            "Explain why the parser emits an auditable error row instead of dropping a failed "
            "run, and distinguish displayed percent from stored fractional rate."
        ),
    ),
    Scenario(
        slug="perf-markers",
        module_name="perf_markers.py",
        source_paths=("tools/update-perf-board.py",),
        good_module=dedent(
            """
            import re


            def replace_block(text, tag, body):
                pattern = re.compile(
                    rf"(<!-- {re.escape(tag)}:START[^>]*-->\\n).*?"
                    rf"(\\n<!-- {re.escape(tag)}:END -->)",
                    re.DOTALL,
                )
                matches = list(pattern.finditer(text))
                if len(matches) != 1:
                    raise ValueError(f"expected one marker block for {tag}, found {len(matches)}")
                return pattern.sub(lambda match: match.group(1) + body + match.group(2), text)
            """
        ),
        buggy_module=dedent(
            """
            import re


            def replace_block(text, tag, body):
                pattern = re.compile(
                    rf"(<!-- {re.escape(tag)}:START[^>]*-->\\n).*?"
                    rf"(\\n<!-- {re.escape(tag)}:END -->)",
                    re.DOTALL,
                )
                matches = list(pattern.finditer(text))
                if len(matches) != 1:
                    raise ValueError(f"expected one marker block for {tag}, found {len(matches)}")
                return pattern.sub(rf"\\1{body}\\2", text)
            """
        ),
        base_tests=dedent(
            """
            import unittest

            from perf_markers import replace_block


            class PerfMarkerTests(unittest.TestCase):
                def test_replaces_one_block(self):
                    original = "a\\n<!-- PERF:START -->\\nold\\n<!-- PERF:END -->\\nz\\n"
                    updated = replace_block(original, "PERF", "new")
                    self.assertIn("\\nnew\\n", updated)
                    self.assertNotIn("\\nold\\n", updated)


            if __name__ == "__main__":
                unittest.main()
            """
        ),
        bug_tests=dedent(
            """
            import unittest

            from perf_markers import replace_block


            class PerfMarkerTests(unittest.TestCase):
                def test_body_backslashes_are_literal_content(self):
                    original = "<!-- PERF:START -->\\nold\\n<!-- PERF:END -->"
                    body = r"path=C:\\bench\\1"
                    self.assertEqual(
                        replace_block(original, "PERF", body),
                        "<!-- PERF:START -->\\n" + body + "\\n<!-- PERF:END -->",
                    )


            if __name__ == "__main__":
                unittest.main()
            """
        ),
        bug_request=(
            "Generated block bodies can contain backslashes. The current replacement string "
            "lets the regex engine reinterpret them as group references. Preserve body bytes "
            "literally and keep marker validation strict."
        ),
        refactor_request=(
            "Refactor marker matching and replacement for readability while preserving the "
            "one-block-only contract and exact surrounding text."
        ),
        test_request=(
            "Add tests for a missing block, duplicate blocks, tags containing regex punctuation, "
            "and body text containing backslashes."
        ),
        explain_request=(
            "Explain why generated marker blocks use a single source of truth, why duplicate "
            "markers are an error, and why callable regex replacement matters."
        ),
    ),
    Scenario(
        slug="batch-divergence",
        module_name="batch_divergence.py",
        source_paths=("tools/check-batch-exact.py",),
        good_module=dedent(
            """
            import hashlib


            def first_divergence(reference, output):
                limit = min(len(reference), len(output))
                for index in range(limit):
                    if reference[index] != output[index]:
                        return index
                if len(reference) != len(output):
                    return limit
                return None


            def mismatch_record(reference, output):
                point = first_divergence(reference, output)
                if point is None:
                    return None
                return {
                    "diverge_at_char": point,
                    "ref_sha": hashlib.sha256(reference.encode()).hexdigest()[:12],
                    "out_sha": hashlib.sha256(output.encode()).hexdigest()[:12],
                    "ref_tail": reference[point:point + 40],
                    "out_tail": output[point:point + 40],
                }
            """
        ),
        buggy_module=dedent(
            """
            import hashlib


            def first_divergence(reference, output):
                return next(
                    (
                        index
                        for index, (left, right) in enumerate(zip(reference, output))
                        if left != right
                    ),
                    None,
                )


            def mismatch_record(reference, output):
                point = first_divergence(reference, output)
                if point is None:
                    return None
                return {
                    "diverge_at_char": point,
                    "ref_sha": hashlib.sha256(reference.encode()).hexdigest()[:12],
                    "out_sha": hashlib.sha256(output.encode()).hexdigest()[:12],
                    "ref_tail": reference[point:point + 40],
                    "out_tail": output[point:point + 40],
                }
            """
        ),
        base_tests=dedent(
            """
            import unittest

            from batch_divergence import first_divergence, mismatch_record


            class BatchDivergenceTests(unittest.TestCase):
                def test_equal_outputs_have_no_mismatch(self):
                    self.assertIsNone(first_divergence("same", "same"))
                    self.assertIsNone(mismatch_record("same", "same"))

                def test_middle_character_differs(self):
                    self.assertEqual(first_divergence("abc", "axc"), 1)


            if __name__ == "__main__":
                unittest.main()
            """
        ),
        bug_tests=dedent(
            """
            import unittest

            from batch_divergence import first_divergence


            class BatchDivergenceTests(unittest.TestCase):
                def test_prefix_length_difference_is_a_mismatch(self):
                    self.assertEqual(first_divergence("answer", "answer extra"), 6)
                    self.assertEqual(first_divergence("answer extra", "answer"), 6)


            if __name__ == "__main__":
                unittest.main()
            """
        ),
        bug_request=(
            "The exactness checker reports no mismatch when one completion is a strict prefix "
            "of the other because zip truncates. Return the correct first divergence for both "
            "length directions."
        ),
        refactor_request=(
            "Refactor mismatch construction so divergence detection, digesting, and evidence "
            "tails are separately readable while preserving the record schema."
        ),
        test_request=(
            "Add tests for empty strings, prefix differences in both directions, Unicode text, "
            "and a mismatch at character zero."
        ),
        explain_request=(
            "Explain the serving-level exactness contract, why byte-identical outputs matter, "
            "and how the first-divergence evidence narrows a regression."
        ),
    ),
    Scenario(
        slug="tier-plan",
        module_name="tier_plan.py",
        source_paths=(
            "tools/build_expert_tier_plan.py",
            "tools/prepare_mixed_expert_repack.py",
            "CLAUDE.md",
        ),
        good_module=dedent(
            """
            ALLOWED_QTYPES = {"Q2_K", "Q3_K", "NVFP4"}


            def active_experts(expert_count, pruned):
                pruned = set(pruned)
                if any(expert < 0 or expert >= expert_count for expert in pruned):
                    raise ValueError("pruned expert id out of range")
                return [expert for expert in range(expert_count) if expert not in pruned]


            def validate_plan(expert_count, assignments, pruned=()):
                active = active_experts(expert_count, pruned)
                active_set = set(active)
                assigned_set = set(assignments)
                missing = sorted(active_set - assigned_set)
                extra = sorted(assigned_set - active_set)
                if missing:
                    raise ValueError(f"missing assignments for active experts: {missing}")
                if extra:
                    raise ValueError(f"assignments include inactive experts: {extra}")
                invalid = {
                    expert: assignments[expert]
                    for expert in active
                    if assignments[expert] not in ALLOWED_QTYPES
                }
                if invalid:
                    raise ValueError(f"invalid qtypes: {invalid}")
                return {expert: assignments[expert] for expert in active}
            """
        ),
        buggy_module=dedent(
            """
            ALLOWED_QTYPES = {"Q2_K", "Q3_K", "NVFP4", "BF16"}


            def active_experts(expert_count, pruned):
                pruned = set(pruned)
                if any(expert < 0 or expert >= expert_count for expert in pruned):
                    raise ValueError("pruned expert id out of range")
                return [expert for expert in range(expert_count) if expert not in pruned]


            def validate_plan(expert_count, assignments, pruned=()):
                active = active_experts(expert_count, pruned)
                return {
                    expert: assignments.get(expert, "BF16")
                    for expert in active
                }
            """
        ),
        base_tests=dedent(
            """
            import unittest

            from tier_plan import validate_plan


            class TierPlanTests(unittest.TestCase):
                def test_complete_plan_preserves_original_ids(self):
                    plan = validate_plan(
                        4,
                        {0: "NVFP4", 2: "Q3_K", 3: "Q2_K"},
                        pruned={1},
                    )
                    self.assertEqual(plan, {0: "NVFP4", 2: "Q3_K", 3: "Q2_K"})


            if __name__ == "__main__":
                unittest.main()
            """
        ),
        bug_tests=dedent(
            """
            import unittest

            from tier_plan import validate_plan


            class TierPlanTests(unittest.TestCase):
                def test_missing_active_assignment_is_an_error(self):
                    with self.assertRaisesRegex(ValueError, "missing assignments"):
                        validate_plan(
                            4,
                            {0: "NVFP4", 2: "Q3_K"},
                            pruned={1},
                        )


            if __name__ == "__main__":
                unittest.main()
            """
        ),
        bug_request=(
            "The v2 plan silently fabricates BF16 for an unassigned retained expert. Missing "
            "active assignments must be a hard error and only Q2_K, Q3_K, or NVFP4 are legal."
        ),
        refactor_request=(
            "Refactor plan validation into explicit range, coverage, and qtype checks while "
            "preserving original router ids for retained experts."
        ),
        test_request=(
            "Add tests for pruned ids being absent, assignments to pruned ids being rejected, "
            "out-of-range ids, invalid qtypes, and original-id preservation."
        ),
        explain_request=(
            "Explain why pruned expert ids retain their router positions, why every active "
            "expert needs an explicit tier, and why BF16 fallback is unsafe."
        ),
    ),
)


def test_filename(module_name: str) -> str:
    return f"test_{Path(module_name).stem}.py"


def task_prompt(category: str, request: str) -> str:
    lead = {
        "bug_fix": "Fix the demonstrated bug.",
        "refactor": "Perform the requested behavior-preserving refactor.",
        "test_writing": "Write the requested focused tests.",
        "explain": "Analyze the local code and explain it without editing files.",
    }[category]
    final = (
        "Run `python3 -m unittest -v` and make the smallest justified change."
        if category != "explain"
        else "Do not edit files. You may run `python3 -m unittest -v` to confirm the current behavior."
    )
    return (
        f"{lead}\n\n{request}\n\n"
        "This is an isolated CPU-only fixture adapted from a real memra repository shape. "
        "Inspect AGENTS.md and the local Python files. Do not access the network or any path "
        "outside the current directory. "
        f"{final}"
    )


def build_templates() -> list[TaskTemplate]:
    templates = []
    requests = {
        "bug_fix": "bug_request",
        "refactor": "refactor_request",
        "test_writing": "test_request",
        "explain": "explain_request",
    }
    for scenario in SCENARIOS:
        test_name = test_filename(scenario.module_name)
        for category, request_attr in requests.items():
            files = {
                scenario.module_name: (
                    scenario.buggy_module if category == "bug_fix" else scenario.good_module
                ),
                test_name: (
                    scenario.bug_tests if category == "bug_fix" else scenario.base_tests
                ),
            }
            templates.append(
                TaskTemplate(
                    template_id=f"{category.replace('_', '-')}-{scenario.slug}",
                    category=category,
                    scenario=scenario.slug,
                    source_paths=scenario.source_paths,
                    prompt=task_prompt(category, getattr(scenario, request_attr)),
                    files=files,
                    module_name=scenario.module_name,
                    test_name=test_name,
                )
            )
    return templates


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def scan_text(label: str, value: str) -> list[str]:
    return [f"{label}: {name}" for name, pattern in SECRET_PATTERNS if pattern.search(value)]


def scan_template(template: TaskTemplate) -> list[str]:
    findings = scan_text(f"{template.template_id}:prompt", template.prompt)
    for path, content in template.files.items():
        findings.extend(scan_text(f"{template.template_id}:{path}", content))
    findings.extend(scan_text(f"{template.template_id}:AGENTS.md", WORKSPACE_INSTRUCTIONS))
    return findings


def scan_templates(templates: list[TaskTemplate]) -> None:
    findings = []
    for template in templates:
        findings.extend(scan_template(template))
    if findings:
        raise ValueError("template secret scan failed:\n" + "\n".join(findings))


def minimal_environment() -> dict[str, str]:
    keep = (
        "HOME",
        "PATH",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "LC_ALL",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "ALL_PROXY",
    )
    env = {key: os.environ[key] for key in keep if key in os.environ}
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "void",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "NO_COLOR": "1",
        }
    )
    return env


def run_command(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 120.0,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=check,
    )


def run_with_file_stdout(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+",
            encoding="utf-8",
            prefix="memra-sft-stdout-",
            delete=False,
        ) as target:
            path = Path(target.name)
            result = subprocess.run(
                argv,
                cwd=cwd,
                env=env,
                text=True,
                stdout=target,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
            target.flush()
            target.seek(0)
            stdout = target.read()
        return subprocess.CompletedProcess(
            args=result.args,
            returncode=result.returncode,
            stdout=stdout,
            stderr=result.stderr,
        )
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def git_revision(repo_root: Path) -> str:
    result = run_command(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True)
    return result.stdout.strip()


def opencode_version(opencode: str, env: dict[str, str]) -> str:
    result = run_command([opencode, "--version"], env=env, check=True)
    return result.stdout.strip()


def validate_opencode_config(config_path: Path, model: str) -> dict[str, Any]:
    path = config_path.expanduser().resolve()
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"opencode config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"opencode config is not valid JSON: {path}: {exc}") from exc

    prefix = "openrouter/"
    if not model.startswith(prefix):
        raise ValueError(f"model must use the pinned openrouter entry: {model}")
    model_id = model.removeprefix(prefix)

    try:
        entry = config["provider"]["openrouter"]["models"][model_id]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"missing pinned opencode model entry: {model_id}") from exc
    if not isinstance(entry, dict):
        raise ValueError(f"invalid pinned opencode model entry: {model_id}")
    options = entry.get("options")
    provider = options.get("provider") if isinstance(options, dict) else None
    usage = options.get("usage") if isinstance(options, dict) else None
    if not isinstance(provider, dict) or not isinstance(usage, dict):
        raise ValueError(f"missing pinned opencode model policy: {model_id}")

    expected = list(PROVIDER_ONLY)
    problems = []
    if provider.get("order") != expected:
        problems.append(f"provider.order must equal {expected}")
    if provider.get("only") != expected:
        problems.append(f"provider.only must equal {expected}")
    if provider.get("allow_fallbacks") is not False:
        problems.append("provider.allow_fallbacks must be false")
    if usage.get("include") is not True:
        problems.append("usage.include must be true")
    if problems:
        raise ValueError("unsafe opencode model policy: " + "; ".join(problems))

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "model_id": model_id,
        "name": entry.get("name"),
        "provider": {
            "order": expected,
            "only": expected,
            "allow_fallbacks": False,
        },
        "usage": {"include": True},
    }


def prepare_workspace(template: TaskTemplate, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError(f"workspace is not empty: {root}")
    for relative, content in {
        **template.files,
        "AGENTS.md": WORKSPACE_INSTRUCTIONS,
    }.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe fixture path: {relative}")
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")

    run_command(["git", "init", "-q"], cwd=root, check=True)
    run_command(["git", "add", "."], cwd=root, check=True)
    run_command(
        [
            "git",
            "-c",
            "user.name=memra-sft",
            "-c",
            "user.email=memra-sft@localhost",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )


def test_result(workspace: Path, env: dict[str, str], timeout: float = 60.0) -> dict[str, Any]:
    result = run_command(
        [sys.executable, "-m", "unittest", "-v"],
        cwd=workspace,
        env=env,
        timeout=timeout,
    )
    return {
        "command": [sys.executable, "-m", "unittest", "-v"],
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def parse_json_events(stdout: str) -> list[dict[str, Any]]:
    events = []
    for lineno, line in enumerate(stdout.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"opencode stdout line {lineno} is not JSON: {exc}") from exc
        if not isinstance(event, dict):
            raise ValueError(f"opencode stdout line {lineno} is not a JSON object")
        events.append(event)
    if not events:
        raise ValueError("opencode emitted no JSON events")
    return events


def session_id_from_events(events: list[dict[str, Any]]) -> str:
    ids = {event.get("sessionID") for event in events if event.get("sessionID")}
    if len(ids) != 1:
        raise ValueError(f"expected one opencode session id, found {sorted(ids)}")
    return next(iter(ids))


def tool_parts(session_export: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for message in session_export.get("messages", []):
        if not isinstance(message, dict):
            continue
        for part in message.get("parts", []):
            if isinstance(part, dict) and part.get("type") == "tool":
                result.append(part)
    return result


def audit_tool_commands(session_export: dict[str, Any]) -> list[str]:
    violations = []
    for part in tool_parts(session_export):
        tool = str(part.get("tool", ""))
        state = part.get("state") if isinstance(part.get("state"), dict) else {}
        input_value = state.get("input") if isinstance(state, dict) else None
        if tool not in {"bash", "shell", "run_command"} or not isinstance(input_value, dict):
            continue
        command = input_value.get("command")
        if isinstance(command, str) and FORBIDDEN_COMMAND_RE.search(command):
            violations.append(f"{tool}: forbidden CPU/GPU command requested")
    return violations


def strip_volatile(value: Any, workspace: str | None = None) -> Any:
    if isinstance(value, list):
        return [strip_volatile(item, workspace) for item in value]
    if not isinstance(value, dict):
        if isinstance(value, str) and workspace:
            return value.replace(workspace, "<workspace>")
        return value
    if value.get("type") in IGNORED_CONTENT_PARTS:
        return None
    cleaned = {}
    for key, item in value.items():
        if key in VOLATILE_KEYS:
            continue
        normalized = strip_volatile(item, workspace)
        if normalized is not None:
            cleaned[key] = normalized
    return cleaned


def canonical_transcript(session_export: dict[str, Any]) -> list[dict[str, Any]]:
    info = session_export.get("info") if isinstance(session_export.get("info"), dict) else {}
    workspace = info.get("directory") if isinstance(info.get("directory"), str) else None
    messages = []
    for message in session_export.get("messages", []):
        if not isinstance(message, dict):
            continue
        info = message.get("info") if isinstance(message.get("info"), dict) else {}
        parts = []
        for part in message.get("parts", []):
            normalized = strip_volatile(part, workspace)
            if normalized is not None:
                parts.append(normalized)
        messages.append({"role": info.get("role"), "parts": parts})
    return messages


def content_hash(template: TaskTemplate, session_export: dict[str, Any]) -> str:
    payload = {
        "template_id": template.template_id,
        "prompt": template.prompt,
        "messages": canonical_transcript(session_export),
    }
    return sha256_text(canonical_json(payload))


def summarize_usage(session_export: dict[str, Any]) -> dict[str, Any]:
    info = session_export.get("info") if isinstance(session_export.get("info"), dict) else {}
    tokens = info.get("tokens") if isinstance(info.get("tokens"), dict) else {}
    cache = tokens.get("cache") if isinstance(tokens.get("cache"), dict) else {}
    return {
        "input_tokens": int(tokens.get("input", 0) or 0),
        "output_tokens": int(tokens.get("output", 0) or 0),
        "reasoning_tokens": int(tokens.get("reasoning", 0) or 0),
        "cache_read_tokens": int(cache.get("read", 0) or 0),
        "cache_write_tokens": int(cache.get("write", 0) or 0),
        "cost_usd": float(info.get("cost", 0.0) or 0.0),
    }


def workspace_diff(workspace: Path) -> tuple[list[str], str]:
    names = run_command(
        ["git", "diff", "--name-only"],
        cwd=workspace,
        check=True,
    ).stdout.splitlines()
    diff = run_command(
        ["git", "diff", "--no-ext-diff", "--binary"],
        cwd=workspace,
        check=True,
    ).stdout
    return names, diff


def verify_workspace(
    template: TaskTemplate,
    workspace: Path,
    env: dict[str, str],
    initial: dict[str, Any],
) -> dict[str, Any]:
    final = test_result(workspace, env)
    changed_files, diff = workspace_diff(workspace)
    reasons = []

    if template.category == "bug_fix":
        if initial["exit_code"] == 0:
            reasons.append("bug-fix fixture did not fail before generation")
        if template.module_name not in changed_files:
            reasons.append("bug-fix task did not change the production module")
    elif initial["exit_code"] != 0:
        reasons.append(f"{template.category} fixture did not start green")

    if final["exit_code"] != 0:
        reasons.append("final unittest verification failed")
    if "AGENTS.md" in changed_files:
        reasons.append("agent edited AGENTS.md")
    if template.category == "refactor" and template.module_name not in changed_files:
        reasons.append("refactor task did not change the production module")
    if template.category == "test_writing":
        if template.test_name not in changed_files:
            reasons.append("test-writing task did not change the test module")
        if template.module_name in changed_files:
            reasons.append("test-writing task changed already-correct production code")
    if template.category == "explain" and changed_files:
        reasons.append(f"explain task changed files: {changed_files}")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "initial_tests": initial,
        "final_tests": final,
        "changed_files": changed_files,
        "git_diff": diff,
    }


def receipt_entry(repo_root: Path, relative: str, note: str) -> dict[str, Any]:
    path = repo_root / relative
    entry = {"path": f"memra/{relative}", "note": note}
    if path.is_file():
        entry["sha256"] = sha256_file(path)
    else:
        entry["sha256"] = None
    return entry


def header_record(
    repo_root: Path,
    *,
    model: str,
    config_receipt: dict[str, Any],
    opencode_version_value: str,
    template_count: int,
) -> dict[str, Any]:
    return {
        "record_type": "corpus_header",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "generator": {
            "path": "memra/tools/sft-gen.py",
            "repo_revision": git_revision(repo_root),
            "opencode_version": opencode_version_value,
        },
        "model": model,
        "provider_policy": {
            "only": list(PROVIDER_ONLY),
            "allow_fallbacks": False,
            "source": config_receipt["path"],
            "config_sha256": config_receipt["sha256"],
        },
        "task_templates": {
            "count": template_count,
            "categories": ["bug_fix", "refactor", "test_writing", "explain"],
            "source_material": "owner-authorized reductions of memra tool and invariant shapes",
        },
        "license_tos": {
            "note": (
                "Owner GO task #56 authorizes this trace-generation lane. OpenRouter, model, "
                "and endpoint-provider terms remain binding; this header records engineering "
                "provenance and is not an independent legal opinion."
            ),
            "pilot_clearance_receipts": [
                receipt_entry(
                    repo_root,
                    "research/finetune-sku-20260802/REPORT.md",
                    "recommended opencode -> OpenRouter -> DeepSeek V4-Flash pilot path",
                ),
                receipt_entry(
                    repo_root,
                    "research/finetune-sku-20260802/openrouter-tos-20260802.html",
                    "stored OpenRouter terms receipt displayed as last updated 2026-07-27",
                ),
                receipt_entry(
                    repo_root,
                    "research/finetune-sku-20260802/deepseek-open-platform-tos-20260802.html",
                    "DeepSeek platform terms receipt; section 4.2 names model distillation",
                ),
            ],
            "live_recheck": {
                "checked_at": "2026-08-08",
                "openrouter_terms_displayed_update": "2026-07-29",
                "note": (
                    "The competing-service sentence was already present in the stored "
                    "2026-07-27 receipt; no broader clearance is inferred from the date change."
                ),
            },
        },
        "hygiene": {
            "dedup": "sha256 over template id, prompt, and volatile-field-stripped transcript",
            "template_secret_scan": [name for name, _ in SECRET_PATTERNS],
            "runtime": "isolated temporary git repos; CPU-only environment; no rustup",
        },
    }


def read_existing_hashes(path: Path) -> set[str]:
    hashes = set()
    if not path.exists():
        return hashes
    with path.open(encoding="utf-8") as source:
        for lineno, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid existing JSONL: {exc}") from exc
            value = row.get("content_sha256") if isinstance(row, dict) else None
            if isinstance(value, str):
                hashes.add(value)
    return hashes


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as target:
        target.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        target.flush()
        os.fsync(target.fileno())


class GenerationError(RuntimeError):
    def __init__(self, message: str, failure: dict[str, Any]):
        super().__init__(message)
        self.failure = failure


def generate_one(
    template: TaskTemplate,
    *,
    repo_root: Path,
    opencode: str,
    model: str,
    env: dict[str, str],
    timeout: float,
    work_parent: Path | None,
    keep_workspace: bool,
    auto_approve: bool,
) -> tuple[dict[str, Any], Path | None]:
    prefix = f"memra-sft-{template.template_id}-"
    workspace = Path(
        tempfile.mkdtemp(prefix=prefix, dir=str(work_parent) if work_parent else None)
    )
    retained_workspace = workspace if keep_workspace else None
    started = time.monotonic()
    try:
        prepare_workspace(template, workspace)
        initial = test_result(workspace, env)

        command = [
            opencode,
            "run",
            "--pure",
            "--format",
            "json",
            "--model",
            model,
            "--dir",
            str(workspace),
            "--title",
            template.template_id,
        ]
        if auto_approve:
            command.append("--auto")
        command.append(template.prompt)

        try:
            run = run_command(command, env=env, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            failure = {
                "record_type": "generation_failure",
                "schema_version": SCHEMA_VERSION,
                "template_id": template.template_id,
                "timestamp": utc_now(),
                "stage": "opencode_run",
                "error": f"timeout after {timeout}s",
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
            }
            raise GenerationError(f"{template.template_id}: opencode timed out", failure) from exc

        try:
            events = parse_json_events(run.stdout)
        except ValueError as exc:
            failure = {
                "record_type": "generation_failure",
                "schema_version": SCHEMA_VERSION,
                "template_id": template.template_id,
                "timestamp": utc_now(),
                "stage": "event_parse",
                "exit_code": run.returncode,
                "error": str(exc),
                "stdout": run.stdout,
                "stderr": run.stderr,
            }
            raise GenerationError(f"{template.template_id}: invalid opencode event stream", failure) from exc

        if run.returncode != 0:
            failure = {
                "record_type": "generation_failure",
                "schema_version": SCHEMA_VERSION,
                "template_id": template.template_id,
                "timestamp": utc_now(),
                "stage": "opencode_run",
                "exit_code": run.returncode,
                "error": "opencode returned non-zero",
                "stdout": run.stdout,
                "stderr": run.stderr,
                "events": events,
            }
            raise GenerationError(
                f"{template.template_id}: opencode failed: {run.stderr.strip() or 'no stderr'}",
                failure,
            )

        session_id = session_id_from_events(events)
        # opencode 1.18.13 can truncate a large `export` JSON document near 64 KiB
        # when its single stdout write targets a pipe. A regular temporary file
        # preserves the complete document; raw run events remain streamed via PIPE.
        exported = run_with_file_stdout(
            [opencode, "export", session_id],
            env=env,
            timeout=min(timeout, 120.0),
        )
        if exported.returncode != 0:
            failure = {
                "record_type": "generation_failure",
                "schema_version": SCHEMA_VERSION,
                "template_id": template.template_id,
                "timestamp": utc_now(),
                "stage": "opencode_export",
                "exit_code": exported.returncode,
                "error": "opencode export returned non-zero",
                "stdout": exported.stdout,
                "stderr": exported.stderr,
                "events": events,
            }
            raise GenerationError(f"{template.template_id}: session export failed", failure)
        try:
            session_export = json.loads(exported.stdout)
        except json.JSONDecodeError as exc:
            failure = {
                "record_type": "generation_failure",
                "schema_version": SCHEMA_VERSION,
                "template_id": template.template_id,
                "timestamp": utc_now(),
                "stage": "opencode_export_parse",
                "error": str(exc),
                "stdout": exported.stdout,
                "stderr": exported.stderr,
            }
            raise GenerationError(f"{template.template_id}: invalid session export", failure) from exc

        command_violations = audit_tool_commands(session_export)
        verification = verify_workspace(template, workspace, env, initial)
        if command_violations:
            verification["reasons"].extend(command_violations)
            verification["passed"] = False

        content_sha256 = content_hash(template, session_export)
        usage = summarize_usage(session_export)
        record = {
            "record_type": "trace",
            "schema_version": SCHEMA_VERSION,
            "template_id": template.template_id,
            "task_kind": template.category,
            "scenario": template.scenario,
            "timestamp": utc_now(),
            "model": model,
            "provider_policy": {
                "only": list(PROVIDER_ONLY),
                "allow_fallbacks": False,
            },
            "source": {
                "repo": "memra",
                "repo_revision": git_revision(repo_root),
                "shape_paths": list(template.source_paths),
                "prompt_sha256": sha256_text(template.prompt),
            },
            "prompt": template.prompt,
            "opencode": {
                "session_id": session_id,
                "version": (
                    session_export.get("info", {}).get("version")
                    if isinstance(session_export.get("info"), dict)
                    else None
                ),
                "exit_code": run.returncode,
                "stderr": run.stderr,
                "duration_ms": round((time.monotonic() - started) * 1000),
            },
            "usage": usage,
            "raw_events": events,
            "session_export": session_export,
            "verification": verification,
            "content_sha256": content_sha256,
            "hygiene": {
                "template_secret_scan": "pass",
                "forbidden_command_audit": "pass" if not command_violations else "fail",
            },
        }
        output_findings = scan_text(
            f"{template.template_id}:record",
            json.dumps(record, ensure_ascii=False),
        )
        if output_findings:
            verification["passed"] = False
            verification["reasons"].extend(output_findings)

        if not verification["passed"]:
            failure = {
                "record_type": "generation_failure",
                "schema_version": SCHEMA_VERSION,
                "template_id": template.template_id,
                "timestamp": utc_now(),
                "stage": "verification",
                "error": "; ".join(verification["reasons"]),
                "candidate_trace": record,
            }
            raise GenerationError(f"{template.template_id}: verification failed", failure)
        return record, retained_workspace
    finally:
        if not keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)


def select_templates(args: argparse.Namespace, templates: list[TaskTemplate]) -> list[TaskTemplate]:
    selected = templates
    if args.category:
        wanted = set(args.category)
        selected = [template for template in selected if template.category in wanted]
    if args.template:
        wanted = set(args.template)
        missing = wanted - {template.template_id for template in templates}
        if missing:
            raise ValueError(f"unknown template ids: {sorted(missing)}")
        selected = [template for template in selected if template.template_id in wanted]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("no templates selected")
    return selected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_output = (
        Path.home()
        / "projects"
        / "sft-traces"
        / "corpus"
        / f"deepseek-v4-flash-{datetime.now().strftime('%Y%m%d')}.jsonl"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--opencode", default="opencode")
    parser.add_argument(
        "--opencode-config",
        type=Path,
        default=DEFAULT_OPENCODE_CONFIG,
        help="config whose model/provider pin must pass before API calls",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--category",
        action="append",
        choices=["bug_fix", "refactor", "test_writing", "explain"],
    )
    parser.add_argument("--template", action="append")
    parser.add_argument("--list", action="store_true", help="list selected templates and exit")
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="run the prompt/fixture secret scan and exit",
    )
    parser.add_argument("--keep-workspaces", action="store_true")
    parser.add_argument("--work-root", type=Path)
    parser.add_argument(
        "--no-auto",
        action="store_true",
        help="do not pass opencode --auto (tool tasks may stop for permission)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="record a failure and continue with later templates",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")

    repo_root = Path(__file__).resolve().parent.parent
    templates = build_templates()
    scan_templates(templates)
    selected = select_templates(args, templates)

    if args.list:
        for template in selected:
            print(
                f"{template.template_id}\t{template.category}\t"
                f"{','.join(template.source_paths)}"
            )
        return 0
    if args.scan_only:
        print(f"template secret scan PASS: {len(templates)} templates")
        return 0

    try:
        config_receipt = validate_opencode_config(args.opencode_config, args.model)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    opencode = shutil.which(args.opencode) or args.opencode
    env = minimal_environment()
    version = opencode_version(opencode, env)
    args.output = args.output.expanduser().resolve()
    if args.work_root:
        args.work_root = args.work_root.expanduser().resolve()
        args.work_root.mkdir(parents=True, exist_ok=True)

    existing_hashes = read_existing_hashes(args.output)
    if not args.output.exists() or args.output.stat().st_size == 0:
        append_jsonl(
            args.output,
            header_record(
                repo_root,
                model=args.model,
                config_receipt=config_receipt,
                opencode_version_value=version,
                template_count=len(templates),
            ),
        )

    failure_path = args.output.with_name(args.output.stem + "-failures.jsonl")
    written = 0
    duplicates = 0
    failed = 0
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "cost_usd": 0.0,
    }

    for index, template in enumerate(selected, 1):
        print(
            f"[{index}/{len(selected)}] {template.template_id}",
            file=sys.stderr,
            flush=True,
        )
        try:
            record, retained = generate_one(
                template,
                repo_root=repo_root,
                opencode=opencode,
                model=args.model,
                env=env,
                timeout=args.timeout,
                work_parent=args.work_root,
                keep_workspace=args.keep_workspaces,
                auto_approve=not args.no_auto,
            )
        except GenerationError as exc:
            failed += 1
            append_jsonl(failure_path, exc.failure)
            print(f"FAIL {exc}", file=sys.stderr)
            if not args.continue_on_error:
                return 1
            continue

        if record["content_sha256"] in existing_hashes:
            duplicates += 1
            print(f"DEDUP {template.template_id}", file=sys.stderr)
            continue
        append_jsonl(args.output, record)
        existing_hashes.add(record["content_sha256"])
        written += 1
        for key in totals:
            totals[key] += record["usage"][key]
        if retained:
            print(f"workspace retained: {retained}", file=sys.stderr)

    summary = {
        "selected": len(selected),
        "written": written,
        "duplicates": duplicates,
        "failed": failed,
        **totals,
        "output": str(args.output),
        "failure_output": str(failure_path) if failure_path.exists() else None,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
