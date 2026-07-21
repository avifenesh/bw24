#!/usr/bin/env python3
"""Build a Layer100-preserving late-layer complement from a frozen delta arm.

The construction is public-score blind.  It keeps only restored experts whose
maximum downstream exposure is structurally bounded by ``max_downstream_layers``.
All Layer100 experts and qtypes remain exact, and selected restored experts keep
their exact frozen delta qtypes.
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib

from build_hy3_delta_restore_plan import (
    PROJECTIONS,
    group_assignments,
    load_plan,
    retained,
    sha256,
    state_map,
)


OUTPUT_FORMAT = "bw24-layer100-late-restore-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-plan", type=pathlib.Path, required=True)
    parser.add_argument("--delta-plan", type=pathlib.Path, required=True)
    parser.add_argument("--delta-analysis", type=pathlib.Path, required=True)
    parser.add_argument("--displacement", type=pathlib.Path, required=True)
    parser.add_argument("--max-downstream-layers", type=int, default=20)
    parser.add_argument("--out-plan", type=pathlib.Path, required=True)
    parser.add_argument("--out-analysis", type=pathlib.Path, required=True)
    args = parser.parse_args()

    if not 1 <= args.max_downstream_layers <= 79:
        raise SystemExit("max downstream layers must be in 1..79")
    minimum_layer = 80 - args.max_downstream_layers
    base = load_plan(args.base_plan)
    delta = load_plan(args.delta_plan)
    delta_analysis = json.loads(args.delta_analysis.read_text())
    displacement = json.loads(args.displacement.read_text())
    if delta_analysis.get("public_capability_results_used") is not False:
        raise SystemExit("delta analysis is not explicitly public-score blind")
    if displacement.get("public_capability_results_used") is not False:
        raise SystemExit("displacement evidence is not explicitly private-only")

    base_retained = retained(base)
    delta_retained = retained(delta)
    restored = delta_retained - base_retained
    receipt_rows = {
        (int(row["layer"]), int(row["expert"])): row
        for row in delta_analysis["selected"]
    }
    if set(receipt_rows) != restored:
        raise SystemExit("delta plan and delta selection receipt disagree")
    churn_rows = {
        (int(row["layer"]), int(row["expert"])): row
        for row in displacement["restored_expert_stats"]
    }
    if set(churn_rows) != restored:
        raise SystemExit("private displacement coverage is not exact")

    selected = {identity for identity in restored if identity[0] >= minimum_layer}
    if not selected:
        raise SystemExit("late-layer rule selected no restored experts")
    base_states = state_map(base)
    delta_states = state_map(delta)
    states = dict(base_states)
    for layer, expert in selected:
        for projection in PROJECTIONS:
            states[(layer, expert, projection)] = delta_states[(layer, expert, projection)]

    layers = [int(layer) for layer in base["model"]["moe_layers"]]
    expert_count = int(base["model"]["expert_count"])
    final_retained = base_retained | selected
    pruned = {
        str(layer): [
            expert for expert in range(expert_count) if (layer, expert) not in final_retained
        ]
        for layer in layers
    }
    restored_bytes = sum(int(receipt_rows[key]["bytes"]) for key in selected)
    base_bytes = int(base["policy"]["result_logical_bytes"])
    result_bytes = base_bytes + restored_bytes

    output = copy.deepcopy(base)
    output["description"] = (
        "Layer100-preserving late-layer complement with structurally bounded downstream churn"
    )
    output["recipe"] = "layer100-plus-private-late-restore"
    output["assignments"] = group_assignments(states)
    output["pruned_experts"] = pruned
    output["policy"] = copy.deepcopy(base["policy"])
    output["policy"].update(
        {
            "target_logical_bytes": result_bytes,
            "result_logical_bytes": result_bytes,
            "expert_byte_budget": result_bytes
            - int(base["policy"]["fixed_non_expert_bytes"]),
            "headroom_bytes": 0,
            "base_preservation": {
                "mode": "restore-only",
                "retained_experts_may_be_pruned": False,
                "retained_projection_qtypes_may_change": False,
            },
            "donor_policy": {
                "eligible_experts": "frozen delta restored experts",
                "restored_projection_qtypes": "exact frozen delta states",
                "selection_signal": "private route displacement and layer position only",
                "public_capability_results_used": False,
            },
        }
    )
    for layer in layers:
        kept = sum((layer, expert) in final_retained for expert in range(expert_count))
        output["layer_summary"][str(layer)]["retained"] = kept
        output["layer_summary"][str(layer)]["pruned"] = expert_count - kept
    output["selection"] = {
        "retained_experts": len(final_retained),
        "pruned_experts": len(layers) * expert_count - len(final_retained),
        "restored_experts": len(selected),
        "restored_bytes": restored_bytes,
        "minimum_layer": minimum_layer,
        "max_downstream_layers": args.max_downstream_layers,
        "selection_metric": "private_late_layer_churn_bound_v1",
        "public_capability_results_used": False,
    }
    provenance = {
        "base_plan": {"path": str(args.base_plan.resolve()), "sha256": sha256(args.base_plan)},
        "delta_plan": {
            "path": str(args.delta_plan.resolve()),
            "sha256": sha256(args.delta_plan),
        },
        "delta_analysis": {
            "path": str(args.delta_analysis.resolve()),
            "sha256": sha256(args.delta_analysis),
        },
        "private_displacement": {
            "path": str(args.displacement.resolve()),
            "sha256": sha256(args.displacement),
        },
        "public_capability_results_used": False,
    }
    output["calibration"] = {
        "provenance": provenance,
        "public_eval_data_used_for_selection": False,
    }

    selected_rows = [
        {
            **receipt_rows[key],
            "private_churn": churn_rows[key],
        }
        for key in sorted(selected)
    ]
    analysis = {
        "format": OUTPUT_FORMAT,
        "public_capability_results_used": False,
        "minimum_layer": minimum_layer,
        "max_downstream_layers": args.max_downstream_layers,
        "base_bytes": base_bytes,
        "restored_bytes": restored_bytes,
        "result_bytes": result_bytes,
        "selected_count": len(selected),
        "predicted_direct_entries": sum(churn_rows[key]["entries"] for key in selected),
        "predicted_downstream_exposure": sum(
            churn_rows[key]["downstream_exposure"] for key in selected
        ),
        "predicted_paired_base_exit_weight": sum(
            churn_rows[key]["paired_base_exit_weight"] for key in selected
        ),
        "selected": selected_rows,
        "provenance": provenance,
    }
    args.out_plan.parent.mkdir(parents=True, exist_ok=True)
    args.out_analysis.parent.mkdir(parents=True, exist_ok=True)
    args.out_plan.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    args.out_analysis.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(
        f"plan={args.out_plan} sha256={sha256(args.out_plan)} "
        f"restored={len(selected)} result_bytes={result_bytes}"
    )


if __name__ == "__main__":
    main()
