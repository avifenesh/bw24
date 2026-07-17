#!/usr/bin/env python3
"""Build a bounded Layer100+donor-delta plan using only private route coverage.

Every Layer100 retained expert and its three qtypes are immutable.  The only legal mutation is to
restore an expert that is retained by the frozen donor plan but pruned by Layer100, inheriting the
donor's exact projection qtypes.  Public capability scores are neither accepted nor parsed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


PLAN_FORMAT = "bw24-expert-tier-plan-v2"
OUTPUT_FORMAT = "bw24-layer100-delta-restore-v1"
PROJECTIONS = ("gate", "up", "down")
TENSOR_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.(\d+)\.weight$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text())
    if plan.get("format") != PLAN_FORMAT:
        raise ValueError(f"unsupported plan format: {path}")
    return plan


def state_map(plan: dict[str, Any]) -> dict[tuple[int, int, str], str]:
    result: dict[tuple[int, int, str], str] = {}
    for assignment in plan["assignments"]:
        layer = int(assignment["layer"])
        qtype = str(assignment["qtype"])
        for projection in assignment["projections"]:
            for expert in assignment["experts"]:
                key = (layer, int(expert), str(projection))
                if key in result:
                    raise ValueError(f"duplicate assignment: {key}")
                result[key] = qtype
    return result


def retained(plan: dict[str, Any]) -> set[tuple[int, int]]:
    pruned = {
        (int(layer), int(expert))
        for layer, experts in plan.get("pruned_experts", {}).items()
        for expert in experts
    }
    layers = [int(layer) for layer in plan["model"]["moe_layers"]]
    expert_count = int(plan["model"]["expert_count"])
    return {(layer, expert) for layer in layers for expert in range(expert_count)} - pruned


def load_private_routes(
    requests_path: Path, routes_path: Path
) -> tuple[dict[tuple[int, int], dict[str, float]], dict[str, int]]:
    requests: list[dict[str, Any]] = []
    for line in requests_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row["prompt_tokens"]) != len(row["prompt_ids"]):
            raise ValueError("private request token coverage is inconsistent")
        requests.append(row)
    token_strata = [
        str(row["stratum"])
        for row in requests
        for _ in range(int(row["prompt_tokens"]))
    ]
    if not token_strata:
        raise ValueError("private request corpus is empty")
    layers = list(range(1, 80))
    expected_rows = len(token_strata) * len(layers)
    masses: dict[tuple[int, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    row_count = 0
    for line_no, line in enumerate(routes_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        if row_count >= expected_rows:
            raise ValueError("private route trace has extra rows")
        fields = line.split()
        if len(fields) != 3:
            raise ValueError("invalid private route row")
        layer, row_tokens = map(int, fields[:2])
        expected_layer = layers[row_count % len(layers)]
        if layer != expected_layer or row_tokens != 1:
            raise ValueError(
                f"private route row {line_no} has layer/tokens={layer}/{row_tokens}, "
                f"expected {expected_layer}/1"
            )
        token = row_count // len(layers)
        stratum = token_strata[token]
        for item in fields[2].split(","):
            expert, weight = item.split(":", 1)
            masses[(layer, int(expert))][stratum] += abs(float(weight))
        row_count += 1
    if row_count != expected_rows:
        raise ValueError(f"private route rows={row_count}, expected {expected_rows}")
    return {key: dict(value) for key, value in masses.items()}, {
        "requests": len(requests),
        "tokens": len(token_strata),
        "route_rows": row_count,
    }


def donor_tensor_bytes(manifest: dict[str, Any]) -> dict[tuple[int, int, str], int]:
    result: dict[tuple[int, int, str], int] = {}
    for name, receipt in manifest.get("tensors", {}).items():
        match = TENSOR_RE.match(name)
        if match:
            layer, projection, expert = match.groups()
            result[(int(layer), int(expert), projection)] = int(receipt["bytes"])
    return result


def score_candidates(
    candidates: set[tuple[int, int]],
    masses: dict[tuple[int, int], dict[str, float]],
) -> dict[tuple[int, int], dict[str, Any]]:
    by_layer: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for key in candidates:
        by_layer[key[0]].append(key)
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for layer, keys in by_layer.items():
        strata = sorted({name for key in keys for name in masses.get(key, {})})
        maxima = {
            name: max((masses.get(key, {}).get(name, 0.0) for key in keys), default=0.0)
            for name in strata
        }
        max_total = max((sum(masses.get(key, {}).values()) for key in keys), default=0.0)
        for key in keys:
            raw = masses.get(key, {})
            relative = sorted(
                (raw.get(name, 0.0) / maxima[name] for name in strata if maxima[name] > 0),
                reverse=True,
            )
            tail = sum(relative[:2]) / min(2, len(relative)) if relative else 0.0
            total = sum(raw.values())
            specialization = max(raw.values(), default=0.0) / total if total else 0.0
            mass_scaled = math.log1p(total) / math.log1p(max_total) if max_total else 0.0
            score = (0.55 * tail + 0.25 * specialization + 0.20 * mass_scaled) * (
                0.5 + 0.5 * specialization
            )
            result[key] = {
                "score": score,
                "tail_top2_mean": tail,
                "specialization": specialization,
                "mass_scaled": mass_scaled,
                "total_router_mass": total,
                "stratum_router_mass": raw,
            }
    return result


def group_assignments(states: dict[tuple[int, int, str], str]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    for (layer, expert, projection), qtype in states.items():
        groups[(layer, projection, qtype)].append(expert)
    projection_order = {name: index for index, name in enumerate(PROJECTIONS)}
    return [
        {"layer": layer, "projections": [projection], "qtype": qtype, "experts": sorted(experts)}
        for (layer, projection, qtype), experts in sorted(
            groups.items(), key=lambda item: (item[0][0], projection_order[item[0][1]], item[0][2])
        )
    ]


def routed_summary(
    identities: set[tuple[int, int]], masses: dict[tuple[int, int], dict[str, float]]
) -> dict[str, Any]:
    by_stratum: dict[str, float] = defaultdict(float)
    active = 0
    for key in identities:
        raw = masses.get(key, {})
        if sum(raw.values()) > 0:
            active += 1
        for name, value in raw.items():
            by_stratum[name] += value
    return {
        "experts": len(identities),
        "experts_seen_in_private_top8": active,
        "router_mass": sum(by_stratum.values()),
        "stratum_router_mass": dict(sorted(by_stratum.items())),
    }


def build(
    base: dict[str, Any],
    donor: dict[str, Any],
    manifest: dict[str, Any],
    masses: dict[tuple[int, int], dict[str, float]],
    target_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_states = state_map(base)
    donor_states = state_map(donor)
    base_retained = retained(base)
    donor_retained = retained(donor)
    donor_only = donor_retained - base_retained
    base_only = base_retained - donor_retained
    common = base_retained & donor_retained
    tensor_bytes = donor_tensor_bytes(manifest)
    scores = score_candidates(donor_only, masses)
    base_result = int(base["policy"]["result_logical_bytes"])
    if target_bytes <= base_result:
        raise ValueError("target must exceed the frozen base size")
    budget = target_bytes - base_result
    candidates: list[dict[str, Any]] = []
    for layer, expert in sorted(donor_only):
        projection_states = {p: donor_states[(layer, expert, p)] for p in PROJECTIONS}
        cost = sum(tensor_bytes[(layer, expert, p)] for p in PROJECTIONS)
        row = dict(scores[(layer, expert)])
        row.update(
            {
                "layer": layer,
                "expert": expert,
                "bytes": cost,
                "qtypes": projection_states,
                "priority_per_sqrt_byte": row["score"] / math.sqrt(cost),
            }
        )
        candidates.append(row)

    q2_count: dict[int, int] = defaultdict(int)
    for (layer, _expert, _projection), qtype in base_states.items():
        q2_count[layer] += qtype == "Q2_K"
    q2_limits = {
        int(layer): int(row["max_q2_projections"])
        for layer, row in base["layer_summary"].items()
    }
    selected: list[dict[str, Any]] = []
    used = 0

    def can_add(row: dict[str, Any]) -> bool:
        layer = int(row["layer"])
        added_q2 = sum(qtype == "Q2_K" for qtype in row["qtypes"].values())
        return used + int(row["bytes"]) <= budget and q2_count[layer] + added_q2 <= q2_limits[layer]

    def add(row: dict[str, Any]) -> None:
        nonlocal used
        selected.append(row)
        used += int(row["bytes"])
        layer = int(row["layer"])
        q2_count[layer] += sum(qtype == "Q2_K" for qtype in row["qtypes"].values())

    # One best private-coverage donor per represented layer prevents a global-mass stratum from
    # consuming the entire budget.  The remaining budget is a score/sqrt(bytes) greedy fill.
    selected_ids: set[tuple[int, int]] = set()
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_layer[int(row["layer"])].append(row)
    for layer in sorted(by_layer):
        for row in sorted(
            by_layer[layer],
            key=lambda item: (-float(item["score"]), int(item["bytes"]), int(item["expert"])),
        ):
            if row["score"] > 0 and can_add(row):
                add(row)
                selected_ids.add((int(row["layer"]), int(row["expert"])))
                break
    for row in sorted(
        candidates,
        key=lambda item: (
            -float(item["priority_per_sqrt_byte"]),
            -float(item["score"]),
            int(item["bytes"]),
            int(item["layer"]),
            int(item["expert"]),
        ),
    ):
        identity = (int(row["layer"]), int(row["expert"]))
        if identity not in selected_ids and row["score"] > 0 and can_add(row):
            add(row)
            selected_ids.add(identity)

    states = dict(base_states)
    for row in selected:
        layer, expert = int(row["layer"]), int(row["expert"])
        for projection, qtype in row["qtypes"].items():
            states[(layer, expert, projection)] = qtype
    final_retained = base_retained | selected_ids
    layers = [int(layer) for layer in base["model"]["moe_layers"]]
    expert_count = int(base["model"]["expert_count"])
    pruned = {
        str(layer): [expert for expert in range(expert_count) if (layer, expert) not in final_retained]
        for layer in layers
    }
    result_bytes = base_result + used
    output = copy.deepcopy(base)
    output["description"] = "Layer100-preserving private-route complement restored from frozen Layer137"
    output["recipe"] = "layer100-plus-donor-delta-private-route-coverage"
    output["assignments"] = group_assignments(states)
    output["pruned_experts"] = pruned
    output["policy"] = copy.deepcopy(base["policy"])
    output["policy"].update(
        {
            "target_logical_bytes": target_bytes,
            "result_logical_bytes": result_bytes,
            "expert_byte_budget": target_bytes - int(base["policy"]["fixed_non_expert_bytes"]),
            "headroom_bytes": target_bytes - result_bytes,
            "base_preservation": {
                "mode": "restore-only",
                "retained_experts_may_be_pruned": False,
                "retained_projection_qtypes_may_change": False,
            },
            "donor_policy": {
                "eligible_experts": "donor retained AND base pruned",
                "restored_projection_qtypes": "exact frozen donor states",
                "selection_signal": "private route strata only",
                "public_capability_results_used": False,
            },
        }
    )
    for layer in layers:
        output["layer_summary"][str(layer)]["retained"] = sum(
            (layer, expert) in final_retained for expert in range(expert_count)
        )
        output["layer_summary"][str(layer)]["pruned"] = expert_count - output["layer_summary"][str(layer)]["retained"]
    output["selection"] = {
        "retained_experts": len(final_retained),
        "pruned_experts": len(layers) * expert_count - len(final_retained),
        "restored_experts": len(selected_ids),
        "restored_bytes": used,
        "selection_metric": "private_route_coverage_v1",
        "public_capability_results_used": False,
    }

    qtype_changes = sum(
        base_states[(layer, expert, projection)] != donor_states[(layer, expert, projection)]
        for layer, expert in common
        for projection in PROJECTIONS
    )
    analysis = {
        "format": OUTPUT_FORMAT,
        "sets": {
            "base_only": routed_summary(base_only, masses),
            "donor_only": routed_summary(donor_only, masses),
            "common": routed_summary(common, masses),
        },
        "common_projection_qtype_changes": qtype_changes,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "selected_bytes": used,
        "target_bytes": target_bytes,
        "result_bytes": result_bytes,
        "headroom_bytes": target_bytes - result_bytes,
        "selected": sorted(selected, key=lambda row: (int(row["layer"]), int(row["expert"]))),
        "public_capability_results_used": False,
    }
    return output, analysis


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-delta-restore-"):
        def plan(kept: set[int], qtype: str, size: int) -> dict[str, Any]:
            return {
                "format": PLAN_FORMAT,
                "model": {"moe_layers": [1], "expert_count": 4, "expert_used_count": 2},
                "assignments": [
                    {"layer": 1, "projections": [projection], "qtype": qtype, "experts": sorted(kept)}
                    for projection in PROJECTIONS
                ],
                "pruned_experts": {"1": sorted(set(range(4)) - kept)},
                "policy": {
                    "result_logical_bytes": size,
                    "fixed_non_expert_bytes": 10,
                    "target_logical_bytes": size,
                    "expert_byte_budget": size - 10,
                    "headroom_bytes": 0,
                },
                "layer_summary": {"1": {"retained": len(kept), "pruned": 4-len(kept), "max_q2_projections": 12}},
                "selection": {},
            }
        base = plan({0, 1}, "Q2_K", 100)
        donor = plan({0, 2, 3}, "IQ3_S", 160)
        manifest = {"tensors": {}}
        for expert in (0, 2, 3):
            for projection in PROJECTIONS:
                manifest["tensors"][f"blk.1.ffn_{projection}_exps.{expert}.weight"] = {"bytes": 5}
        masses = {(1, 2): {"code": 9.0, "math": 0.1}, (1, 3): {"code": 1.0, "math": 1.0}}
        output, analysis = build(base, donor, manifest, masses, 116)
        assert output["policy"]["result_logical_bytes"] == 115
        assert retained(output) == {(1, 0), (1, 1), (1, 2)}
        assert all(state_map(output)[(1, expert, p)] == "Q2_K" for expert in (0, 1) for p in PROJECTIONS)
        assert analysis["selected"][0]["expert"] == 2
        assert analysis["common_projection_qtype_changes"] == 3


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("Hy3 delta restore plan self-test: PASS")
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--donor-plan", type=Path, required=True)
    parser.add_argument("--donor-artifact-manifest", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--target-logical-bytes", type=int, default=110_000_000_000)
    parser.add_argument("--out-plan", type=Path, required=True)
    parser.add_argument("--out-analysis", type=Path, required=True)
    args = parser.parse_args()
    base = load_plan(args.base_plan)
    donor = load_plan(args.donor_plan)
    manifest = json.loads(args.donor_artifact_manifest.read_text())
    if manifest.get("plan_sha256") != sha256(args.donor_plan):
        raise SystemExit("donor manifest and donor plan receipts disagree")
    masses, coverage = load_private_routes(args.requests, args.routes)
    output, analysis = build(base, donor, manifest, masses, args.target_logical_bytes)
    provenance = {
        "base_plan": {"path": str(args.base_plan.resolve()), "sha256": sha256(args.base_plan)},
        "donor_plan": {"path": str(args.donor_plan.resolve()), "sha256": sha256(args.donor_plan)},
        "donor_artifact_manifest": {
            "path": str(args.donor_artifact_manifest.resolve()),
            "sha256": sha256(args.donor_artifact_manifest),
            "fields_used": ["plan_sha256", "tensors.*.bytes"],
            "public_capability_fields_used": False,
        },
        "private_requests": {"path": str(args.requests.resolve()), "sha256": sha256(args.requests)},
        "private_routes": {"path": str(args.routes.resolve()), "sha256": sha256(args.routes)},
        "private_coverage": coverage,
    }
    output["calibration"] = {"provenance": provenance, "public_eval_data_used_for_selection": False}
    output["policy"]["base_preservation"]["base_plan"] = provenance["base_plan"]
    output["policy"]["donor_policy"]["donor_plan"] = provenance["donor_plan"]
    analysis["provenance"] = provenance
    args.out_plan.parent.mkdir(parents=True, exist_ok=True)
    args.out_analysis.parent.mkdir(parents=True, exist_ok=True)
    args.out_plan.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    args.out_analysis.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote plan={args.out_plan} sha256={sha256(args.out_plan)} "
        f"analysis={args.out_analysis} sha256={sha256(args.out_analysis)} "
        f"restored={analysis['selected_count']} bytes={analysis['result_bytes']}"
    )


if __name__ == "__main__":
    main()
