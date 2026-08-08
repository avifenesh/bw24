#!/usr/bin/env python3
"""Report daily fleet cache economics from cumulative /metrics snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import cache_economics


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEDGER = ROOT / "research/fleet-meter/rig5090-fleet.jsonl"
COUNTER_KEYS = ("prompt_tokens_in", "cached_tokens_in", "computed_tokens_in")


def parse_timestamp(value: object, where: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{where}: ts must be a string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{where}: invalid timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{where}: timestamp must carry a timezone")
    return parsed.astimezone(timezone.utc)


def counter(row: dict, key: str, where: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where}: {key} must be a non-negative integer")
    return value


def validate_histogram(row: dict, where: str) -> None:
    histogram = row.get("lcp_histogram")
    if not isinstance(histogram, dict):
        raise ValueError(f"{where}: lcp_histogram must be an object")
    edges = histogram.get("edges")
    counts = histogram.get("counts")
    if not isinstance(edges, list) or not isinstance(counts, list):
        raise ValueError(f"{where}: lcp_histogram edges/counts must be arrays")
    if len(edges) != len(counts):
        raise ValueError(f"{where}: lcp_histogram edges/counts length mismatch")
    for i, value in enumerate(edges):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{where}: lcp_histogram.edges[{i}] is invalid")
    for i, value in enumerate(counts):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{where}: lcp_histogram.counts[{i}] is invalid")


def validate_tenants(row: dict, where: str) -> None:
    tenants = row.get("tenants")
    if not isinstance(tenants, dict):
        raise ValueError(f"{where}: tenants must be an object")
    for name, tenant in tenants.items():
        if not isinstance(tenant, dict):
            raise ValueError(f"{where}: tenant {name!r} must be an object")
        prompt = counter(tenant, "prompt_tokens_in", f"{where} tenant {name!r}")
        cached = counter(tenant, "cached_tokens_in", f"{where} tenant {name!r}")
        if cached > prompt:
            raise ValueError(f"{where}: tenant {name!r} cached exceeds prompt")


def validate_snapshot(row: dict, where: str) -> datetime:
    if not isinstance(row, dict):
        raise ValueError(f"{where}: snapshot must be an object")
    timestamp = parse_timestamp(row.get("ts"), where)
    prompt = counter(row, "prompt_tokens_in", where)
    cached = counter(row, "cached_tokens_in", where)
    computed = counter(row, "computed_tokens_in", where)
    if cached > prompt:
        raise ValueError(f"{where}: cached_tokens_in exceeds prompt_tokens_in")
    if computed != prompt - cached:
        raise ValueError(
            f"{where}: computed_tokens_in is {computed}, expected {prompt - cached}"
        )
    validate_histogram(row, where)
    validate_tenants(row, where)
    return timestamp


def load_snapshots(path: Path) -> list[dict]:
    snapshots = []
    previous_ts = None
    with path.open(encoding="utf-8") as source:
        for lineno, line in enumerate(source, 1):
            if not line.strip():
                continue
            where = f"{path}:{lineno}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{where}: invalid JSON: {exc}") from exc
            timestamp = validate_snapshot(row, where)
            if previous_ts is not None and timestamp < previous_ts:
                raise ValueError(f"{where}: timestamps are not append-ordered")
            previous_ts = timestamp
            row["_timestamp"] = timestamp
            snapshots.append(row)
    if not snapshots:
        raise ValueError(f"{path}: no snapshots")
    return snapshots


def counters_regressed(previous: dict, current: dict) -> bool:
    if any(current[key] < previous[key] for key in COUNTER_KEYS):
        return True

    old_hist = previous["lcp_histogram"]
    new_hist = current["lcp_histogram"]
    if old_hist["edges"] == new_hist["edges"] and any(
        new < old for old, new in zip(old_hist["counts"], new_hist["counts"])
    ):
        return True

    old_tenants = previous["tenants"]
    new_tenants = current["tenants"]
    for name, old in old_tenants.items():
        if name not in new_tenants:
            if old["prompt_tokens_in"] or old["cached_tokens_in"]:
                return True
            continue
        new = new_tenants[name]
        if (
            new["prompt_tokens_in"] < old["prompt_tokens_in"]
            or new["cached_tokens_in"] < old["cached_tokens_in"]
        ):
            return True
    return False


def histogram_delta(previous: dict | None, current: dict, reset: bool) -> dict:
    histogram = current["lcp_histogram"]
    if previous is None or reset:
        return {
            "edges": list(histogram["edges"]),
            "counts": list(histogram["counts"]),
        }
    old = previous["lcp_histogram"]
    if old["edges"] != histogram["edges"]:
        raise ValueError("lcp_histogram edges changed without a restart")
    counts = [new - old for old, new in zip(old["counts"], histogram["counts"])]
    if any(value < 0 for value in counts):
        raise ValueError("lcp_histogram regressed without a restart")
    return {"edges": list(histogram["edges"]), "counts": counts}


def tenant_deltas(previous: dict | None, current: dict, reset: bool) -> dict:
    if previous is None or reset:
        return {
            name: {
                "prompt_tokens_in": tenant["prompt_tokens_in"],
                "cached_tokens_in": tenant["cached_tokens_in"],
            }
            for name, tenant in current["tenants"].items()
        }

    deltas = {}
    old_tenants = previous["tenants"]
    for name, tenant in current["tenants"].items():
        old = old_tenants.get(
            name, {"prompt_tokens_in": 0, "cached_tokens_in": 0}
        )
        prompt = tenant["prompt_tokens_in"] - old["prompt_tokens_in"]
        cached = tenant["cached_tokens_in"] - old["cached_tokens_in"]
        if prompt < 0 or cached < 0 or cached > prompt:
            raise ValueError(f"tenant {name!r} counters regressed without a restart")
        if prompt or cached:
            deltas[name] = {
                "prompt_tokens_in": prompt,
                "cached_tokens_in": cached,
            }
    return deltas


def daily_deltas(snapshots: list[dict]) -> list[dict]:
    days: OrderedDict[str, dict] = OrderedDict()
    previous = None
    for row in snapshots:
        inferred_restart = previous is not None and counters_regressed(previous, row)
        reset = previous is None or bool(row.get("restart")) or inferred_restart
        date = row["_timestamp"].date().isoformat()
        day = days.setdefault(
            date,
            {
                "date": date,
                "prompt_tokens_in": 0,
                "cached_tokens_in": 0,
                "computed_tokens_in": 0,
                "lcp_histogram": None,
                "tenants": {},
                "snapshots": 0,
                "restarts": 0,
            },
        )
        day["snapshots"] += 1
        if previous is not None and reset:
            day["restarts"] += 1

        for key in COUNTER_KEYS:
            delta = row[key] if reset else row[key] - previous[key]
            if delta < 0:
                raise ValueError(f"{key} regressed without a restart")
            day[key] += delta

        histogram = histogram_delta(previous, row, reset)
        if day["lcp_histogram"] is None:
            day["lcp_histogram"] = histogram
        else:
            aggregate = day["lcp_histogram"]
            if aggregate["edges"] != histogram["edges"]:
                raise ValueError("lcp_histogram edges changed within a UTC day")
            aggregate["counts"] = [
                total + delta
                for total, delta in zip(aggregate["counts"], histogram["counts"])
            ]

        for name, delta in tenant_deltas(previous, row, reset).items():
            tenant = day["tenants"].setdefault(
                name, {"prompt_tokens_in": 0, "cached_tokens_in": 0}
            )
            tenant["prompt_tokens_in"] += delta["prompt_tokens_in"]
            tenant["cached_tokens_in"] += delta["cached_tokens_in"]

        previous = row

    return list(days.values())


def economics(day: dict, factor: float) -> dict | None:
    if day["prompt_tokens_in"] == 0:
        return None
    metrics = {
        key: day[key] for key in COUNTER_KEYS
    }
    metrics["lcp_histogram"] = day["lcp_histogram"]
    metrics["tenants"] = day["tenants"]
    return cache_economics.row_from_metrics(metrics, factor)


def format_multiplier(value: object) -> str:
    return "unbounded" if value is None else f"{float(value):.4f}x"


def render_report(snapshots: list[dict], days: list[dict]) -> str:
    rows = []
    previous_ratio = None
    for day in days:
        low = economics(day, 0.25)
        high = economics(day, 1.0)
        if high is None:
            ratio_text = "n/a"
            trend_text = "-"
            revenue_text = "n/a"
            window_text = "n/a"
        else:
            ratio = high["cache_hit_token_ratio"]
            ratio_text = f"{ratio:.1%}"
            trend_text = (
                "-" if previous_ratio is None else f"{(ratio - previous_ratio) * 100:+.2f}pp"
            )
            previous_ratio = ratio
            revenue_text = (
                f"{format_multiplier(low['revenue_multiplier'])}"
                f"..{format_multiplier(high['revenue_multiplier'])}"
            )
            window = high.get("lcp_window_64_512_share")
            window_text = "n/a" if window is None else f"{window:.1%}"
        rows.append(
            [
                day["date"],
                f"{day['prompt_tokens_in']:,}",
                f"{day['cached_tokens_in']:,}",
                f"{day['computed_tokens_in']:,}",
                ratio_text,
                trend_text,
                revenue_text,
                window_text,
                str(day["restarts"]),
            ]
        )

    headers = [
        "UTC day",
        "tokens/day",
        "cached",
        "computed",
        "hit-token",
        "trend",
        "revenue x @0.25..1.0",
        "tick-seg",
        "restarts",
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    lines = [
        (
            f"fleet-report: {len(snapshots)} snapshots, {len(days)} UTC day(s); "
            "first snapshot and restart rows count from zero"
        ),
        "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(value.ljust(widths[i]) for i, value in enumerate(row)).rstrip()
        for row in rows
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ledger",
        nargs="?",
        type=Path,
        default=DEFAULT_LEDGER,
        help=f"fleet JSONL ledger (default {DEFAULT_LEDGER})",
    )
    parser.add_argument(
        "--days",
        type=int,
        help="show only the newest N UTC days (for example, --days 7)",
    )
    args = parser.parse_args()
    if args.days is not None and args.days <= 0:
        parser.error("--days must be positive")

    try:
        snapshots = load_snapshots(args.ledger)
        days = daily_deltas(snapshots)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"fleet-report: {exc}") from exc
    if args.days is not None:
        days = days[-args.days :]
    print(render_report(snapshots, days))


if __name__ == "__main__":
    main()
