#!/usr/bin/env python3
"""corpus_stats.py — corpus report for SFT trace JSONL (sft-corpus-20260802).

Reads one or more trace JSONL files (raw/verified) and optionally the rejects file,
emits a Markdown report:

  - counts by seed_source, outcome.method, outcome.verified
  - turn-length histogram (messages per trace)
  - tool-call stats (traces with calls, total calls, calls-per-trace histogram, top tools)
  - near-dup check: duplicate (task_id) and duplicate (task_id-stem + first-user-message
    normalized hash) groups are listed
  - keep-rate when --rejects is given: kept / (kept + rejected)

Usage: corpus_stats.py verified/*.jsonl --rejects rejects/rejects.jsonl -o report.md
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict


def _norm(text):
    """Normalization for near-dup hashing: lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _first_user(trace):
    for m in trace.get("messages", []):
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def _hist_rows(counter, bucket=None):
    """Counter -> sorted (label, count) rows; optional int bucketing."""
    if bucket:
        b = Counter()
        for k, v in counter.items():
            lo = (k // bucket) * bucket
            b[f"{lo}-{lo + bucket - 1}"] += v
        return sorted(b.items(), key=lambda kv: int(kv[0].split("-")[0]))
    return sorted(counter.items(), key=lambda kv: (isinstance(kv[0], str), kv[0]))


def _md_table(rows, headers):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def load_traces(paths):
    traces, errors = [], []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as fh:
                for ln, line in enumerate(fh, 1):
                    if not line.strip():
                        continue
                    try:
                        traces.append((path, ln, json.loads(line)))
                    except json.JSONDecodeError as e:
                        errors.append(f"{path}:{ln}: {e}")
        except OSError as e:
            errors.append(f"{path}: {e}")
    return traces, errors


def count_rejects(path):
    n = 0
    reasons = Counter()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            n += 1
            try:
                obj = json.loads(line)
                reasons[(obj.get("reject") or {}).get("reason", "unknown")] += 1
            except json.JSONDecodeError:
                reasons["unparseable reject line"] += 1
    return n, reasons


def build_report(traces, errors, rejects=None):
    by_seed = Counter()
    by_method = Counter()
    by_verified = Counter()
    turn_hist = Counter()
    calls_hist = Counter()
    tool_names = Counter()
    total_calls = 0
    traces_with_calls = 0
    task_ids = Counter()
    dup_groups = defaultdict(list)

    for path, ln, t in traces:
        by_seed[t.get("seed_source", "(missing)")] += 1
        outcome = t.get("outcome") or {}
        by_method[outcome.get("method", "(missing)")] += 1
        by_verified[str(outcome.get("verified", "(missing)"))] += 1
        msgs = t.get("messages", [])
        turn_hist[len(msgs)] += 1
        n_calls = 0
        for m in msgs:
            for c in m.get("tool_calls") or []:
                n_calls += 1
                name = ((c.get("function") or {}).get("name")) or "(unnamed)"
                tool_names[name] += 1
        calls_hist[n_calls] += 1
        total_calls += n_calls
        if n_calls:
            traces_with_calls += 1
        tid = t.get("task_id", "(missing)")
        task_ids[tid] += 1
        stem = re.sub(r"[-_.]\d+$", "", tid)  # task-001/task-002 share a stem
        key = (stem, hashlib.sha256(_norm(_first_user(t)).encode()).hexdigest()[:16])
        dup_groups[key].append(f"{tid} ({path}:{ln})")

    n = len(traces)
    lines = ["# SFT corpus stats", ""]
    lines.append(f"Traces: **{n}**"
                 + (f" | parse errors: **{len(errors)}**" if errors else ""))
    if rejects is not None:
        n_rej, rej_reasons = rejects
        denom = n + n_rej
        rate = f"{n / denom:.1%}" if denom else "n/a"
        lines += ["", f"Keep-rate: **{n}/{denom} = {rate}** "
                      f"(kept / kept+rejected)"]
    lines += ["", "## By seed_source", "",
              _md_table(_hist_rows(by_seed), ["seed_source", "traces"]),
              "", "## By outcome.method", "",
              _md_table(_hist_rows(by_method), ["method", "traces"]),
              "", "## By outcome.verified", "",
              _md_table(_hist_rows(by_verified), ["verified", "traces"])]

    lines += ["", "## Turn-length histogram (messages per trace)", "",
              _md_table(_hist_rows(turn_hist, bucket=4), ["turns", "traces"])]

    avg = f"{total_calls / n:.2f}" if n else "n/a"
    lines += ["", "## Tool calls", "",
              f"- traces with tool calls: {traces_with_calls}/{n}",
              f"- total tool calls: {total_calls} (avg {avg}/trace)", "",
              "### Calls-per-trace histogram", "",
              _md_table(_hist_rows(calls_hist), ["calls", "traces"])]
    if tool_names:
        lines += ["", "### Top tools", "",
                  _md_table(tool_names.most_common(15), ["tool", "calls"])]

    exact_dups = {tid: c for tid, c in task_ids.items() if c > 1}
    near_dups = {k: v for k, v in dup_groups.items() if len(v) > 1}
    lines += ["", "## Dedup check", "",
              f"- exact task_id duplicates: {len(exact_dups)}",
              f"- near-dup groups (task-id stem + first-user-message hash): {len(near_dups)}"]
    for tid, c in sorted(exact_dups.items()):
        lines.append(f"  - task_id `{tid}` appears {c} times")
    for (stem, h), members in sorted(near_dups.items()):
        lines.append(f"  - stem `{stem}` / hash `{h}`: " + ", ".join(members))

    if rejects is not None and rejects[1]:
        lines += ["", "## Reject reasons", "",
                  _md_table(rejects[1].most_common(), ["reason", "count"])]
    if errors:
        lines += ["", "## Parse errors", ""]
        lines += [f"- {e}" for e in errors]
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", help="trace JSONL files")
    ap.add_argument("--rejects", help="rejects.jsonl (enables keep-rate + reasons)")
    ap.add_argument("-o", "--output", default="-", help="Markdown report (default stdout)")
    args = ap.parse_args(argv)

    traces, errors = load_traces(args.files)
    rejects = count_rejects(args.rejects) if args.rejects else None
    report = build_report(traces, errors, rejects)
    if args.output == "-":
        sys.stdout.write(report)
    else:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(f"wrote {args.output}", file=sys.stderr)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
