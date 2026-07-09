#!/usr/bin/env python3
"""Parse rebaseline-2026-07-09 logs and generate JSONL rows + summary table."""
import re
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

LOGDIR = Path("/home/avifenesh/projects/bw24/research/tune-data/rebaseline-logs")

def parse_bw24_plain(log_text: str) -> float:
    """Extract tok/s from bw24 decode-bench output."""
    m = re.search(r'decode tg128 @ctx\d+: EAGER ([\d.]+) tok/s', log_text)
    return float(m.group(1)) if m else 0.0

def parse_llama_plain(log_text: str) -> float:
    """Extract tg128 tok/s from llama-bench output."""
    m = re.search(r'tg128 @ d\d+\s+\|\s+([\d.]+)', log_text)
    return float(m.group(1)) if m else 0.0

def parse_bw24_spec(log_text: str) -> Tuple[float, float]:
    """Extract tok/s and acceptance from bw24 run-spec output."""
    # Look for: generate] K=3] 239.4 tok/s (4.18 ms/tok, median accept 63.7%)
    m = re.search(r'generate\].*?([\d.]+) tok/s.*?median accept ([\d.]+)%', log_text)
    if m:
        return float(m.group(1)), float(m.group(2))
    return 0.0, 0.0

def parse_llama_spec(log_text: str, prompt_name: str) -> Tuple[float, int]:
    """Extract tok/s and token count from llama spec output."""
    # Look for line like: gen: 256 tok @ 186.23 tok/s
    pattern = rf'=== llama \w+ {prompt_name} ===.*?gen: (\d+) tok @ ([\d.]+) tok/s'
    m = re.search(pattern, log_text, re.DOTALL)
    if m:
        return float(m.group(2)), int(m.group(1))
    return 0.0, 0

def median_of_runs(model: str, cell: str, engine: str, parser, *args) -> float:
    """Get median of N runs for a cell."""
    values = []
    for run in [1, 2]:
        log_path = LOGDIR / f"{model}{cell}-{engine}-*.run{run}.log"
        matches = list(LOGDIR.glob(f"{model}{cell}-{engine}-*run{run}.log"))
        if not matches:
            continue
        log_text = matches[0].read_text()
        val = parser(log_text, *args) if args else parser(log_text)
        if isinstance(val, tuple):
            val = val[0]  # tok/s only for median
        if val > 0:
            values.append(val)
    return statistics.median(values) if values else 0.0

def check_text_audit(log_path: Path) -> str:
    """Check text-audit log for loops/degeneration."""
    if not log_path.exists():
        return "no-log"
    text = log_path.read_text()
    # Look for the generated block after "generate]"
    m = re.search(r'\[generate\](.*?)(?:\[|$)', text, re.DOTALL)
    if not m:
        return "no-output"
    gen = m.group(1).strip()
    # Simple heuristics: repeated lines or very short output
    lines = gen.split('\n')
    if len(lines) < 10:
        return "short"
    # Check for loops: same line repeated 3+ times
    for i, line in enumerate(lines):
        if line and lines[i:i+3].count(line) >= 3:
            return "LOOP-DETECTED"
    return "clean"

# Parse all models
models = {
    "A": ("qwen35-9b-nvfp4-gguf", "9B GGUF"),
    "B": ("qwen35-9b-nvfp4-st", "9B ST"),
    "C": ("qwen36-27b-nvfp4-gguf", "27B GGUF"),
    "D": ("qwen36-27b-nvfp4-st", "27B ST"),
    "E": ("qwen36-35b-a3b-iq4xs", "35B GGUF"),
}

results = {}
for model_key, (model_id, model_name) in models.items():
    print(f"Parsing {model_name}...")
    results[model_key] = {
        "model": model_id,
        "name": model_name,
        "plain_d512_bw24": median_of_runs(model_key, "1", "bw24", parse_bw24_plain),
        "plain_d512_llama": median_of_runs(model_key, "1", "llama", parse_llama_plain),
        "plain_d6257_bw24": median_of_runs(model_key, "2", "bw24", parse_bw24_plain),
        "plain_d6257_llama": median_of_runs(model_key, "2", "llama", parse_llama_plain),
    }

    # Spec cells: p1 and p2 (p3 excluded per rules)
    spec_p1_logs = list(LOGDIR.glob(f"{model_key}3-bw24-spec-p1-run*.log"))
    if spec_p1_logs:
        p1_vals = []
        for log in sorted(spec_p1_logs):
            tps, acc = parse_bw24_spec(log.read_text())
            if tps > 0:
                p1_vals.append((tps, acc))
        if p1_vals:
            results[model_key]["spec_p1_bw24"] = statistics.median([v[0] for v in p1_vals])
            results[model_key]["spec_p1_bw24_acc"] = statistics.median([v[1] for v in p1_vals])

    spec_p2_logs = list(LOGDIR.glob(f"{model_key}3-bw24-spec-p2-run*.log"))
    if spec_p2_logs:
        p2_vals = []
        for log in sorted(spec_p2_logs):
            tps, acc = parse_bw24_spec(log.read_text())
            if tps > 0:
                p2_vals.append((tps, acc))
        if p2_vals:
            results[model_key]["spec_p2_bw24"] = statistics.median([v[0] for v in p2_vals])
            results[model_key]["spec_p2_bw24_acc"] = statistics.median([v[1] for v in p2_vals])

    # Llama spec: extract from the combined log
    llama_spec_logs = list(LOGDIR.glob(f"{model_key}3-llama-spec-run*.log"))
    if llama_spec_logs:
        p1_llama = []
        p2_llama = []
        for log in sorted(llama_spec_logs):
            text = log.read_text()
            p1_tps, _ = parse_llama_spec(text, "p1-code-short")
            p2_tps, _ = parse_llama_spec(text, "p2-code-medium")
            if p1_tps > 0:
                p1_llama.append(p1_tps)
            if p2_tps > 0:
                p2_llama.append(p2_tps)
        if p1_llama:
            results[model_key]["spec_p1_llama"] = statistics.median(p1_llama)
        if p2_llama:
            results[model_key]["spec_p2_llama"] = statistics.median(p2_llama)

    # Text audit
    audit_log = LOGDIR / f"{model_key}-audit-p2.log"
    results[model_key]["text_audit"] = check_text_audit(audit_log)

# Generate JSONL rows
jsonl_path = Path("/home/avifenesh/projects/bw24/research/tune-data/rig5090.jsonl")
print("\nGenerating JSONL rows...")
for model_key, data in results.items():
    row = {
        "date": "2026-07-09",
        "machine": "rig5090",
        "model": data["model"],
        "tag": f"rebaseline-2026-07-09-{model_key}",
        "plain_d512": {
            "bw24": data.get("plain_d512_bw24", 0),
            "llama": data.get("plain_d512_llama", 0),
            "ratio": round(data.get("plain_d512_bw24", 0) / data.get("plain_d512_llama", 1), 3) if data.get("plain_d512_llama") else 0,
        },
        "plain_d6257": {
            "bw24": data.get("plain_d6257_bw24", 0),
            "llama": data.get("plain_d6257_llama", 0),
            "ratio": round(data.get("plain_d6257_bw24", 0) / data.get("plain_d6257_llama", 1), 3) if data.get("plain_d6257_llama") else 0,
        },
        "spec_p1": {
            "bw24": data.get("spec_p1_bw24", 0),
            "bw24_acc": data.get("spec_p1_bw24_acc", 0),
            "llama": data.get("spec_p1_llama", 0),
            "ratio": round(data.get("spec_p1_bw24", 0) / data.get("spec_p1_llama", 1), 3) if data.get("spec_p1_llama") else 0,
        },
        "spec_p2": {
            "bw24": data.get("spec_p2_bw24", 0),
            "bw24_acc": data.get("spec_p2_bw24_acc", 0),
            "llama": data.get("spec_p2_llama", 0),
            "ratio": round(data.get("spec_p2_bw24", 0) / data.get("spec_p2_llama", 1), 3) if data.get("spec_p2_llama") else 0,
        },
        "text_audit": data.get("text_audit", "unknown"),
    }
    print(json.dumps(row))

# Generate summary table markdown
print("\n\nGenerating summary table...")
table_path = Path("/home/avifenesh/projects/bw24/research/tune-data/rebaseline-2026-07-09.md")
with open(table_path, "w") as f:
    f.write("# Rebaseline 2026-07-09: Full Board\n\n")
    f.write("N=2 medians, both engines, interleaved per model, idle-gated.\n\n")
    f.write("## Plain Decode tg128 @d512\n\n")
    f.write("| Model | bw24 | llama | ratio |\n")
    f.write("|-------|------|-------|-------|\n")
    for key in ["A", "B", "C", "D", "E"]:
        data = results[key]
        bw = data.get("plain_d512_bw24", 0)
        ll = data.get("plain_d512_llama", 0)
        ratio = bw / ll if ll else 0
        f.write(f"| {data['name']} | {bw:.1f} | {ll:.1f} | {ratio:.2f}x |\n")

    f.write("\n## Plain Decode tg128 @d6257\n\n")
    f.write("| Model | bw24 | llama | ratio |\n")
    f.write("|-------|------|-------|-------|\n")
    for key in ["A", "B", "C", "D", "E"]:
        data = results[key]
        bw = data.get("plain_d6257_bw24", 0)
        ll = data.get("plain_d6257_llama", 0)
        ratio = bw / ll if ll else 0
        f.write(f"| {data['name']} | {bw:.1f} | {ll:.1f} | {ratio:.2f}x |\n")

    f.write("\n## Spec p1 (K=3, 256 tokens)\n\n")
    f.write("| Model | bw24 | acc% | llama | ratio |\n")
    f.write("|-------|------|------|-------|-------|\n")
    for key in ["A", "B", "C", "D", "E"]:
        data = results[key]
        bw = data.get("spec_p1_bw24", 0)
        acc = data.get("spec_p1_bw24_acc", 0)
        ll = data.get("spec_p1_llama", 0)
        ratio = bw / ll if ll else 0
        f.write(f"| {data['name']} | {bw:.1f} | {acc:.1f} | {ll:.1f} | {ratio:.2f}x |\n")

    f.write("\n## Spec p2 (K=3, 256 tokens)\n\n")
    f.write("| Model | bw24 | acc% | llama | ratio |\n")
    f.write("|-------|------|------|-------|-------|\n")
    for key in ["A", "B", "C", "D", "E"]:
        data = results[key]
        bw = data.get("spec_p2_bw24", 0)
        acc = data.get("spec_p2_bw24_acc", 0)
        ll = data.get("spec_p2_llama", 0)
        ratio = bw / ll if ll else 0
        f.write(f"| {data['name']} | {bw:.1f} | {acc:.1f} | {ll:.1f} | {ratio:.2f}x |\n")

    f.write("\n## Text Audit (p2)\n\n")
    for key in ["A", "B", "C", "D", "E"]:
        data = results[key]
        audit = data.get("text_audit", "unknown")
        f.write(f"- **{data['name']}**: {audit}\n")

    f.write("\n## Anomalies\n\n")
    f.write("(To be filled after manual review of logs)\n")

print(f"\nSummary table written to: {table_path}")
print("NEXT: Review logs, run this parser, append JSONL rows, commit with prefix 'data:'")
