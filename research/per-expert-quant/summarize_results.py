#!/usr/bin/env python3
"""Compare lm-eval arms and compute paired-bootstrap confidence intervals from logged samples."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def find_results(path: Path) -> tuple[Path, dict[str, Any]]:
    candidates = [path] if path.is_file() else list(path.rglob("results_*.json"))
    valid: list[tuple[Path, dict[str, Any]]] = []
    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "results" in data:
            valid.append((candidate, data))
    if not valid:
        raise FileNotFoundError(f"no lm-eval results JSON under {path}")
    return max(valid, key=lambda item: item[0].stat().st_mtime_ns)


def aggregate_value(results: dict[str, Any], spec: dict[str, str]) -> tuple[float, float | None] | None:
    section = results.get(spec["result_section"], {})
    task = section.get(spec["result_task"])
    if not isinstance(task, dict):
        return None
    key = f"{spec['metric']},{spec['filter']}"
    if key not in task:
        return None
    stderr = task.get(f"{spec['metric']}_stderr,{spec['filter']}")
    return float(task[key]), float(stderr) if isinstance(stderr, (int, float)) else None


def sample_values(root: Path, spec: dict[str, str]) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], float] = {}
    for path in root.rglob(spec["sample_glob"]):
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("filter") != spec["filter"]:
                    continue
                value = row.get(spec["metric"])
                if not isinstance(value, (int, float, bool)):
                    continue
                key = (str(row.get("doc_hash", "")), str(row.get("target_hash", "")))
                if key in values:
                    raise ValueError(f"duplicate sample identity {key} under {root}")
                values[key] = float(value)
    return values


def paired_bootstrap(
    baseline: dict[tuple[str, str], float],
    candidate: dict[tuple[str, str], float],
    iterations: int,
) -> dict[str, float | int] | None:
    if not baseline or baseline.keys() != candidate.keys():
        return None
    keys = sorted(baseline)
    delta = np.asarray([candidate[key] - baseline[key] for key in keys], dtype=np.float64)
    rng = np.random.default_rng(42)
    means: list[np.ndarray] = []
    batch = max(1, min(256, 2_000_000 // len(delta)))
    for start in range(0, iterations, batch):
        count = min(batch, iterations - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        means.append(delta[indices].mean(axis=1))
    draws = np.concatenate(means)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "n": len(delta),
        "mean_delta": float(delta.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "iterations": iterations,
    }


def compare(
    lock: dict[str, Any],
    baseline_path: Path,
    candidates: list[tuple[str, Path]],
    iterations: int,
) -> dict[str, Any]:
    baseline_file, baseline = find_results(baseline_path)
    output: dict[str, Any] = {
        "baseline": {"path": str(baseline_file)},
        "candidates": {},
        "metrics": [],
    }
    loaded = {name: (*find_results(path), path) for name, path in candidates}
    for spec in lock["primary_metrics"]:
        base_value = aggregate_value(baseline, spec)
        if base_value is None:
            continue
        row: dict[str, Any] = {
            "label": spec["label"],
            "baseline": {"value": base_value[0], "stderr": base_value[1]},
            "candidates": {},
        }
        base_samples = sample_values(baseline_path, spec)
        for name, (result_file, result, root) in loaded.items():
            value = aggregate_value(result, spec)
            if value is None:
                continue
            entry: dict[str, Any] = {
                "path": str(result_file),
                "value": value[0],
                "stderr": value[1],
                "delta": value[0] - base_value[0],
            }
            entry["paired_bootstrap"] = paired_bootstrap(
                base_samples, sample_values(root, spec), iterations
            )
            row["candidates"][name] = entry
        output["metrics"].append(row)
    for name, (_, _, root) in loaded.items():
        deltas = [row["candidates"][name]["delta"] for row in output["metrics"]
                  if name in row["candidates"]]
        output["candidates"][name] = {
            "root": str(root),
            "macro_delta": sum(deltas) / len(deltas) if deltas else None,
            "worst_delta": min(deltas) if deltas else None,
        }
    return output


def markdown(report: dict[str, Any]) -> str:
    names = list(report["candidates"])
    lines = ["# Per-expert quantization public-eval comparison", ""]
    for name in names:
        summary = report["candidates"][name]
        if summary["macro_delta"] is None:
            lines.append(f"- **{name}**: no matching primary metrics")
        else:
            lines.append(
                f"- **{name}**: macro delta {summary['macro_delta']:+.4f}; "
                f"worst task delta {summary['worst_delta']:+.4f}"
            )
    lines += ["", "| Metric | BF16 reference | " + " | ".join(names) + " |",
              "|---|---:|" + "---:|" * len(names)]
    for row in report["metrics"]:
        cells = [row["label"], f"{row['baseline']['value']:.4f}"]
        for name in names:
            entry = row["candidates"].get(name)
            if entry is None:
                cells.append("n/a")
                continue
            boot = entry["paired_bootstrap"]
            ci = ""
            if boot:
                ci = f"; 95% CI [{boot['ci95_low']:+.4f}, {boot['ci95_high']:+.4f}]"
            cells.append(f"{entry['value']:.4f} ({entry['delta']:+.4f}{ci})")
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Deltas are candidate minus BF16. Confidence intervals are paired by lm-eval document hashes",
        "and bootstrap the per-document metric difference with seed 42.",
    ]
    return "\n".join(lines) + "\n"


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-eval-summary-") as tmp:
        root = Path(tmp)
        base, candidate = root / "base", root / "candidate"
        base.mkdir(); candidate.mkdir()
        for directory, score, values in [(base, 0.5, [0, 1]), (candidate, 1.0, [1, 1])]:
            (directory / "results_test.json").write_text(json.dumps({
                "results": {"toy": {"acc,none": score, "acc_stderr,none": 0.1}}
            }))
            with (directory / "samples_toy_test.jsonl").open("w") as handle:
                for i, value in enumerate(values):
                    handle.write(json.dumps({
                        "doc_hash": f"d{i}", "target_hash": f"t{i}", "filter": "none", "acc": value,
                    }) + "\n")
        lock = {"primary_metrics": [{
            "label": "toy", "result_task": "toy", "result_section": "results",
            "metric": "acc", "filter": "none", "sample_glob": "samples_toy_*.jsonl",
        }]}
        report = compare(lock, base, [("candidate", candidate)], 200)
        entry = report["metrics"][0]["candidates"]["candidate"]
        assert entry["delta"] == 0.5 and entry["paired_bootstrap"]["n"] == 2
        print("public eval summarizer self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("suite.lock.json"))
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--out", type=Path, default=Path("public-eval-comparison.md"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.baseline is None or not args.candidate:
        parser.error("--baseline and at least one --candidate NAME=PATH are required")
    candidates = []
    for item in args.candidate:
        name, sep, raw_path = item.partition("=")
        if not sep or not name:
            parser.error(f"invalid --candidate {item!r}; expected NAME=PATH")
        candidates.append((name, Path(raw_path)))
    report = compare(json.loads(args.lock.read_text()), args.baseline, candidates, args.iterations)
    args.out.write_text(markdown(report))
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} and {args.out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
