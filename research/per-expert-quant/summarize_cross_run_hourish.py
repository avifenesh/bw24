#!/usr/bin/env python3
"""Strictly compare matched expanded-panel arms produced under different run IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import sys
from typing import Any

from summarize_hourish_results import (
    TASKS,
    bootstrap_domain_macro_delta,
    exact_sign_p,
    load_arm,
)


FORMAT = "bw24-cross-run-expanded-capability-frontier-v1"
COMPARABLE_CONFIG_KEYS = (
    "bw24_commit",
    "lm_eval_commit",
    "num_concurrent",
    "declared_spill_io",
    "declared_spill_pread_depth",
    "declared_spill_stats",
    "declared_serve_spec",
    "server_binary_sha256",
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_arm_spec(raw: str) -> tuple[str, pathlib.Path, str]:
    try:
        arm, location = raw.split("=", 1)
        out_root, run_id = location.split("::", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "arm specs must be ARM=OUT_ROOT::RUN_ID"
        ) from exc
    if not arm or not out_root or not run_id:
        raise argparse.ArgumentTypeError("arm specs may not contain empty fields")
    if any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in arm):
        raise argparse.ArgumentTypeError(f"invalid arm name {arm!r}")
    return arm, pathlib.Path(out_root).resolve(), run_id


def parse_commit_pair(raw: str) -> tuple[str, str]:
    try:
        reference, candidate = raw.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "compatible commits must be REFERENCE=CANDIDATE"
        ) from exc
    if not all(re.fullmatch(r"[0-9a-f]{40}", value) for value in (reference, candidate)):
        raise argparse.ArgumentTypeError("compatible commits must be full lowercase Git SHAs")
    if reference == candidate:
        raise argparse.ArgumentTypeError("compatible commits must differ")
    return reference, candidate


def comparable_config(value: dict[str, Any]) -> dict[str, Any]:
    return {key: value.get(key) for key in COMPARABLE_CONFIG_KEYS}


def config_compatibility_override(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    compatible_commit_pairs: set[tuple[str, str]],
) -> dict[str, str] | None:
    differing = sorted(key for key in reference if reference[key] != candidate[key])
    if not differing:
        return None
    pair = (reference.get("bw24_commit"), candidate.get("bw24_commit"))
    if differing != ["bw24_commit"] or pair not in compatible_commit_pairs:
        raise ValueError("run configuration differs from baseline")
    return {
        "key": "bw24_commit",
        "reference": pair[0],
        "candidate": pair[1],
    }


def pareto_arms(loaded: dict[str, dict[str, Any]]) -> list[str]:
    result = []
    for arm, data in loaded.items():
        dominated = any(
            other != arm
            and candidate["logical_model_bytes"] <= data["logical_model_bytes"]
            and candidate["domain_macro"] >= data["domain_macro"]
            and (
                candidate["logical_model_bytes"] < data["logical_model_bytes"]
                or candidate["domain_macro"] > data["domain_macro"]
            )
            for other, candidate in loaded.items()
        )
        if not dominated:
            result.append(arm)
    return sorted(
        result,
        key=lambda arm: (loaded[arm]["logical_model_bytes"], -loaded[arm]["domain_macro"], arm),
    )


def compare_arms(
    reference: dict[str, Any], candidate: dict[str, Any], same_arm: bool
) -> dict[str, Any]:
    wins = losses = ties = 0
    if not same_arm:
        for task in TASKS:
            for identity, base_value in reference["values"][task].items():
                delta = candidate["values"][task][identity] - base_value
                wins += delta > 0
                losses += delta < 0
                ties += delta == 0
    saved = reference["logical_model_bytes"] - candidate["logical_model_bytes"]
    macro_delta = candidate["domain_macro"] - reference["domain_macro"]
    return {
        "bytes_saved": saved,
        "decimal_gb_saved": saved / 1e9,
        "compression_ratio": reference["logical_model_bytes"] / candidate["logical_model_bytes"],
        "retained_domain_macro_fraction": candidate["domain_macro"] / reference["domain_macro"],
        "retained_question_weighted_fraction": candidate["question_weighted"] / reference["question_weighted"],
        "domain_macro_delta": macro_delta,
        "domain_macro_delta_bootstrap_ci95": (
            [0.0, 0.0]
            if same_arm
            else bootstrap_domain_macro_delta(reference["values"], candidate["values"])
        ),
        "question_weighted_delta": candidate["question_weighted"] - reference["question_weighted"],
        "domain_macro_points_lost_per_10gb_saved": (
            None if saved <= 0 else -macro_delta * 10e9 / saved
        ),
        "paired_wins": wins,
        "paired_losses": losses,
        "paired_ties": ties,
        "paired_exact_sign_p": exact_sign_p(wins, losses),
    }


def summarize(
    panel_lock: pathlib.Path,
    expected_server_sha: str,
    arm_specs: list[tuple[str, pathlib.Path, str]],
    baseline: str,
    compatible_commit_pairs: set[tuple[str, str]],
) -> dict[str, Any]:
    if len({arm for arm, _, _ in arm_specs}) != len(arm_specs):
        raise ValueError("arm specs contain duplicate arm names")
    if baseline not in {arm for arm, _, _ in arm_specs}:
        raise ValueError(f"baseline {baseline!r} is not present")
    panel = json.loads(panel_lock.read_text())
    if panel.get("format") != "bw24-expanded-capability-panel-v1":
        raise ValueError("cross-run comparison requires the expanded capability panel")
    panel_sha = sha256(panel_lock)
    loaded = {
        arm: load_arm(out_root, run_id, arm, panel, panel_sha, expected_server_sha)
        for arm, out_root, run_id in arm_specs
    }
    reference = loaded[baseline]
    reference_config = comparable_config(reference["shared_config"])
    compatibility_overrides = []
    for arm, data in loaded.items():
        try:
            override = config_compatibility_override(
                reference_config,
                comparable_config(data["shared_config"]),
                compatible_commit_pairs,
            )
        except ValueError as exc:
            raise ValueError(f"{arm}: {exc}") from exc
        if override is not None:
            compatibility_overrides.append({"arm": arm, **override})
        if data["task_hashes"] != reference["task_hashes"]:
            raise ValueError(f"{arm}: task definitions differ from baseline")
        if data["suite_lock_sha256"] != reference["suite_lock_sha256"]:
            raise ValueError(f"{arm}: suite lock differs from baseline")
        if data["code_scorer"] != reference["code_scorer"]:
            raise ValueError(f"{arm}: code scorer identity differs from baseline")
        if data["math_scorer"] != reference["math_scorer"]:
            raise ValueError(f"{arm}: math scorer identity differs from baseline")
        for task in TASKS:
            if set(data["values"][task]) != set(reference["values"][task]):
                raise ValueError(f"{arm}/{task}: paired sample identities differ")

    pairwise = {
        candidate_arm: {
            reference_arm: compare_arms(
                reference_data,
                candidate_data,
                candidate_arm == reference_arm,
            )
            for reference_arm, reference_data in loaded.items()
        }
        for candidate_arm, candidate_data in loaded.items()
    }
    comparisons = {
        arm: pairwise[arm][baseline]
        for arm in loaded
    }

    sources = []
    for arm, out_root, run_id in arm_specs:
        sources.append({
            "arm": arm,
            "out_root": str(out_root),
            "run_id": run_id,
            "run_dir": loaded[arm]["run_dir"],
        })
    for data in loaded.values():
        data.pop("values")
    return {
        "format": FORMAT,
        "purpose": "cross-run directional compression frontier; not a final capability claim",
        "panel_lock": {"path": str(panel_lock.resolve()), "sha256": panel_sha},
        "server_binary_sha256": expected_server_sha,
        "baseline": baseline,
        "source_runs": sources,
        "configuration_compatibility_overrides": compatibility_overrides,
        "arms": loaded,
        "comparisons_to_baseline": comparisons,
        "pairwise_comparisons": pairwise,
        "point_estimate_pareto": pareto_arms(loaded),
    }


def self_test() -> None:
    assert parse_arm_spec("a=/tmp/x::run") == ("a", pathlib.Path("/tmp/x"), "run")
    old = "a" * 40
    new = "b" * 40
    assert parse_commit_pair(f"{old}={new}") == (old, new)
    reference_config = {key: "same" for key in COMPARABLE_CONFIG_KEYS}
    reference_config["bw24_commit"] = old
    candidate_config = dict(reference_config, bw24_commit=new)
    assert config_compatibility_override(
        reference_config, candidate_config, {(old, new)}
    ) == {"key": "bw24_commit", "reference": old, "candidate": new}
    for pairs in (set(), {(new, old)}):
        try:
            config_compatibility_override(reference_config, candidate_config, pairs)
        except ValueError:
            pass
        else:
            raise AssertionError("unaudited commit mismatch was accepted")
    rejected_config = dict(candidate_config, num_concurrent=2)
    try:
        config_compatibility_override(reference_config, rejected_config, {(old, new)})
    except ValueError:
        pass
    else:
        raise AssertionError("non-commit configuration mismatch was accepted")
    loaded = {
        "small": {"logical_model_bytes": 100, "domain_macro": 0.5},
        "large": {"logical_model_bytes": 200, "domain_macro": 0.7},
        "dominated": {"logical_model_bytes": 200, "domain_macro": 0.4},
    }
    assert pareto_arms(loaded) == ["small", "large"]
    assert math.isclose(exact_sign_p(3, 0), 0.25)
    values = {task: {"x": 1.0} for task in TASKS}
    arm = {
        "logical_model_bytes": 100,
        "domain_macro": 0.5,
        "question_weighted": 0.5,
        "values": values,
    }
    assert compare_arms(arm, arm, True)["paired_exact_sign_p"] == 1.0


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("cross-run hourish summarizer self-test: PASS")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-lock", type=pathlib.Path, required=True)
    parser.add_argument("--server-sha256", required=True)
    parser.add_argument("--arm", action="append", type=parse_arm_spec, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument(
        "--compatible-bw24-commits",
        action="append",
        default=[],
        type=parse_commit_pair,
        help="explicitly audited REFERENCE=CANDIDATE orchestration commit pair",
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if len(args.server_sha256) != 64:
        raise SystemExit("--server-sha256 must be a SHA-256 digest")
    report = summarize(
        args.panel_lock.resolve(), args.server_sha256, args.arm, args.baseline,
        set(args.compatible_bw24_commits),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output} pareto={report['point_estimate_pareto']}")


if __name__ == "__main__":
    main()
