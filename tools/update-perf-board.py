#!/usr/bin/env python3
"""Regenerate the perf surfaces from research/tune-data/current-board.json.

Generated surfaces:
    - README.md PERF-SAMPLES block          (representative sample comparisons, from "samples")
    - README.md PERF-MODELS block           (supported-models table, from "supported_models")
    - docs/PERFORMANCE.md PERF-DATE / PERF-PLAIN / PERF-SPEC blocks  (full 5090 boards)
    - docs/PERFORMANCE.md PERF-H100 block   (full H100 board, from "h100_board")
    - docs/perf-card.svg                    (5090 card)
    - docs/perf-card-h100.svg               (H100 card, same visual style)

Usage:
    tools/update-perf-board.py          # regenerate all surfaces in place
    tools/update-perf-board.py --check  # exit 1 if any surface would change (for a pre-push hook / CI)

The board JSON is the single source of truth for published numbers. Never hand-edit the
generated regions in README.md / docs/PERFORMANCE.md (marked with PERF-*:START/END comments)
or the SVG cards; edit research/tune-data/current-board.json and rerun this script instead.

The README carries only sample rows ("samples": each entry names a source section and a row's
"model" key, plus a display "label") and the numbers-free supported-models table
("supported_models"). The full boards render into docs/PERFORMANCE.md.

Note: 5090 ratios are computed from the display-rounded tok/s values stored in the board JSON,
not from raw unrounded measurements. This can shift a borderline ratio by ~0.01x versus a value
computed from full-precision logs (e.g. 162/155 rounds to 1.05x here even where the underlying
raw measurement was 1.04x) — store more decimal places in current-board.json if a row is close
to the bold_ratio_threshold and this matters. H100 rows instead carry an explicit receipted
"ratio" field (the board publishes the ratio measured from full-precision logs, which can
differ by 0.01x from the rounded-e2e quotient); rows render sorted by that ratio, descending.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOARD_PATH = ROOT / "research" / "tune-data" / "current-board.json"
README_PATH = ROOT / "README.md"
PERFORMANCE_PATH = ROOT / "docs" / "PERFORMANCE.md"
SVG_PATH = ROOT / "docs" / "perf-card.svg"
H100_SVG_PATH = ROOT / "docs" / "perf-card-h100.svg"


def load_board():
    return json.loads(BOARD_PATH.read_text())


def fmt_ratio(memra, llama, threshold):
    ratio = memra / llama
    text = f"{ratio:.2f}x"
    return f"**{text}**" if ratio >= threshold else text


def render_date_block(board):
    return (
        f"Measured {board['updated']} on the tracked measuring rig ({board['rig']}, {board['protocol']}) "
        "against llama.cpp built on the same machine, same exact prompts, both engines "
        "re-baselined the same day. The llama.cpp columns are frozen reference points "
        "recorded through 2026-08-03, when head-to-head benching stopped (owner call) — "
        "regression anchors, not a live scoreboard. Boards move with the tuning campaign — "
        "`research/tune-data/rig5090.jsonl` is the running record; the generated boards "
        "(README samples + this document) are refreshed with every board-moving merge."
    )


def render_plain_table(board):
    threshold = board["bold_ratio_threshold"]
    lines = ["| Model | memra plain | llama.cpp plain | Ratio |", "|---|---|---|---|"]
    for row in board["plain_decode"]["rows"]:
        ratio = fmt_ratio(row["memra"], row["llama"], threshold)
        lines.append(f"| {row['model']} | {row['memra']} | {row['llama']} | {ratio} |")
    return "\n".join(lines)


def render_spec_table(board):
    threshold = board["bold_ratio_threshold"]
    lines = ["| Model | memra spec | llama.cpp spec-best | Ratio |", "|---|---|---|---|"]
    for row in board["speculative"]["rows"]:
        memra_cells = " / ".join(str(v) for v in row["memra"])
        llama_cells = " / ".join(str(v) for v in row["llama"])
        ratios = " / ".join(
            fmt_ratio(b, l, threshold) for b, l in zip(row["memra"], row["llama"])
        )
        lines.append(f"| {row['model']} | {memra_cells} | {llama_cells} | {ratios} |")
    return "\n".join(lines)


def resolve_sample_row(board, sample):
    rows = board[sample["source"]]["rows"]
    for row in rows:
        if row["model"] == sample["model"]:
            return row
    raise SystemExit(
        f"sample references model {sample['model']!r} not found in {sample['source']!r}"
    )


def render_samples_block(board):
    """README sample table: a few representative rows resolved out of the full boards."""
    threshold = board["bold_ratio_threshold"]
    lines = [
        "| Model / scenario | memra tok/s | llama.cpp tok/s | Ratio |",
        "|---|---|---|---|",
    ]
    for sample in board["samples"]:
        row = resolve_sample_row(board, sample)
        if isinstance(row["memra"], list):
            memra_cells = " / ".join(str(v) for v in row["memra"])
            llama_cells = " / ".join(str(v) for v in row["llama"])
            ratios = " / ".join(
                fmt_ratio(b, l, threshold) for b, l in zip(row["memra"], row["llama"])
            )
        else:
            memra_cells = str(row["memra"])
            llama_cells = str(row["llama"])
            ratios = fmt_ratio(row["memra"], row["llama"], threshold)
        lines.append(f"| {sample['label']} | {memra_cells} | {llama_cells} | {ratios} |")
    lines.append("")
    lines.append(
        f"*Measured {board['updated']} on the {board['rig']} — same-session interleaved "
        "medians, same exact prompts; memra at its naked defaults, llama.cpp at its swept "
        "best ([docs/COMPETITOR-SETUP.md](docs/COMPETITOR-SETUP.md)). The llama.cpp column "
        "is a frozen reference recorded through 2026-08-03 (benching stopped that day). N, "
        "thermal regime, and the full boards: [docs/PERFORMANCE.md](docs/PERFORMANCE.md); "
        "raw per-run logs: [research/tune-data/](research/tune-data/).*"
    )
    return "\n".join(lines)


def render_models_block(board):
    lines = [
        "| Model | Class | Quant | Drafter | Supported since |",
        "|---|---|---|---|---|",
    ]
    for row in board["supported_models"]:
        lines.append(
            f"| {row['model']} | {row['class']} | {row['quants']} | {row['drafter']} "
            f"| {row['since']} |"
        )
    return "\n".join(lines)


def h100_rows_sorted(board):
    return sorted(board["h100_board"]["rows"], key=lambda r: r["ratio"], reverse=True)


def render_h100_block(board):
    h = board["h100_board"]
    lines = [
        f"Measured {h['updated']} on {h['rig']} against {h['competitor']} — {h['protocol']}",
        "",
        f"| Model | memra e2e | {h['competitor']} e2e (artifact) | Ratio |",
        "|---|---:|---:|---:|",
    ]
    for row in h100_rows_sorted(board):
        ratio = row["ratio"]
        memra = f"**{row['memra_e2e']}**" if ratio > 1.0 else str(row["memra_e2e"])
        vllm = f"**{row['vllm_e2e']}**" if ratio < 1.0 else str(row["vllm_e2e"])
        ratio_txt = f"{ratio:.2f}x"
        if ratio > 1.0:
            ratio_txt = f"**{ratio_txt}**"
        lines.append(
            f"| {row['model']} | {memra} | {vllm} ({row['vllm_artifact']}) | {ratio_txt} |"
        )
    return "\n".join(lines)


def replace_block(text, tag, body, path):
    pattern = re.compile(
        rf"(<!-- {tag}:START[^>]*-->\n).*?(\n<!-- {tag}:END -->)", re.DOTALL
    )
    if not pattern.search(text):
        raise SystemExit(f"marker block {tag} not found in {path.name}")
    return pattern.sub(lambda m: m.group(1) + body + m.group(2), text)


def render_readme(board, original):
    text = original
    text = replace_block(text, "PERF-SAMPLES", render_samples_block(board), README_PATH)
    text = replace_block(text, "PERF-MODELS", render_models_block(board), README_PATH)
    return text


def render_performance_doc(board, original):
    text = original
    text = replace_block(text, "PERF-DATE", render_date_block(board), PERFORMANCE_PATH)
    text = replace_block(text, "PERF-PLAIN", render_plain_table(board), PERFORMANCE_PATH)
    text = replace_block(text, "PERF-SPEC", render_spec_table(board), PERFORMANCE_PATH)
    text = replace_block(text, "PERF-H100", render_h100_block(board), PERFORMANCE_PATH)
    return text


def render_card(rig_label, right_label, rows, is_win):
    """Shared perf-card renderer: rows = [(label, ratio)], is_win(ratio) -> bold/teal."""
    row_height = 34
    top = 168
    height = top + row_height * len(rows) + 40

    row_svg = []
    for i, (name, ratio) in enumerate(rows):
        y = top + i * row_height
        color = "#3fa79c" if is_win(ratio) else "#978f80"
        weight = "700" if is_win(ratio) else "500"
        row_svg.append(
            f'<line x1="64" y1="{y - 12}" x2="1216" y2="{y - 12}" stroke="#3a352c" stroke-width="1"/>'
            f'<text x="64" y="{y}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" '
            f'font-size="15" fill="#eee7da">{name}</text>'
            f'<text x="1216" y="{y}" text-anchor="end" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" '
            f'font-size="17" font-weight="{weight}" fill="{color}">{ratio:.2f}x</text>'
        )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="{height}" viewBox="0 0 1280 {height}">
  <rect width="1280" height="{height}" fill="#171613"/>
  <text x="64" y="72" font-family="ui-sans-serif,Arial,sans-serif" font-weight="800" font-size="52" fill="#eee7da">memra</text>
  <text x="64" y="104" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" letter-spacing="0.06em" fill="#978f80">{rig_label}</text>
  <text x="1216" y="104" text-anchor="end" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" fill="#978f80">{right_label}</text>
  {"".join(row_svg)}
</svg>
'''


def render_svg(board):
    threshold = board["bold_ratio_threshold"]
    plain_rows = board["plain_decode"]["rows"]
    spec_rows = board["speculative"]["rows"]

    def ratio_of(row):
        return row["memra"][0] / row["llama"][0] if isinstance(row["memra"], list) else row["memra"] / row["llama"]

    all_rows = [(r["model"], ratio_of(r)) for r in plain_rows] + [
        (r["model"] + " (spec)", ratio_of(r)) for r in spec_rows
    ]
    # card-only rows (families whose table lives outside the generated blocks, e.g. Gemma):
    # {model, memra, llama, spec?: true} — appended in order, "(spec)" suffix when spec.
    for r in board.get("extra_card_rows", []):
        label = r["model"] + (" (spec)" if r.get("spec") else "")
        all_rows.append((label, ratio_of(r)))

    return render_card(
        board["rig"],
        f'vs llama.cpp · updated {board["updated"]}',
        all_rows,
        lambda ratio: ratio >= threshold,
    )


def render_svg_h100(board):
    h = board["h100_board"]
    rows = [
        (f'{r["model"]} (vs {r["vllm_artifact"]})', r["ratio"])
        for r in h100_rows_sorted(board)
    ]
    return render_card(
        h["rig"],
        f'vs {h["competitor"]} · updated {h["updated"]}',
        rows,
        lambda ratio: ratio > 1.0,
    )


def main():
    check_only = "--check" in sys.argv
    board = load_board()

    original_readme = README_PATH.read_text()
    original_perf = PERFORMANCE_PATH.read_text()
    new_readme = render_readme(board, original_readme)
    new_perf = render_performance_doc(board, original_perf)
    new_svg = render_svg(board)
    new_h100_svg = render_svg_h100(board)

    old_svg = SVG_PATH.read_text() if SVG_PATH.exists() else ""
    old_h100_svg = H100_SVG_PATH.read_text() if H100_SVG_PATH.exists() else ""

    changed = (
        new_readme != original_readme
        or new_perf != original_perf
        or new_svg != old_svg
        or new_h100_svg != old_h100_svg
    )

    if check_only:
        if changed:
            print("perf board is stale — run tools/update-perf-board.py and commit the result")
            sys.exit(1)
        print("perf board is up to date")
        return

    README_PATH.write_text(new_readme)
    PERFORMANCE_PATH.write_text(new_perf)
    SVG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text(new_svg)
    H100_SVG_PATH.write_text(new_h100_svg)
    if changed:
        print(
            "regenerated README.md + docs/PERFORMANCE.md perf tables"
            " + docs/perf-card.svg + docs/perf-card-h100.svg"
        )
    else:
        print("no changes (already up to date)")


if __name__ == "__main__":
    main()
