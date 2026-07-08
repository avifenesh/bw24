#!/usr/bin/env python3
"""Regenerate README.md's perf tables and docs/perf-card.svg from research/tune-data/current-board.json.

Usage:
    tools/update-perf-board.py          # regenerate README.md + docs/perf-card.svg in place
    tools/update-perf-board.py --check  # exit 1 if either file would change (for a pre-push hook / CI)

The board JSON is the single source of truth for published numbers. Never hand-edit the
generated regions in README.md (marked with PERF-*:START/END comments) or docs/perf-card.svg;
edit research/tune-data/current-board.json and rerun this script instead.

Note: ratios are computed from the display-rounded tok/s values stored in the board JSON, not
from raw unrounded measurements. This can shift a borderline ratio by ~0.01x versus a value
computed from full-precision logs (e.g. 162/155 rounds to 1.05x here even where the underlying
raw measurement was 1.04x) — store more decimal places in current-board.json if a row is close
to the bold_ratio_threshold and this matters.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOARD_PATH = ROOT / "research" / "tune-data" / "current-board.json"
README_PATH = ROOT / "README.md"
SVG_PATH = ROOT / "docs" / "perf-card.svg"


def load_board():
    return json.loads(BOARD_PATH.read_text())


def fmt_ratio(bw24, llama, threshold):
    ratio = bw24 / llama
    text = f"{ratio:.2f}x"
    return f"**{text}**" if ratio >= threshold else text


def render_date_block(board):
    return (
        f"Measured {board['updated']} on the target rig ({board['rig']}, {board['protocol']}) "
        "against llama.cpp built on the same machine, same exact prompts, both engines "
        "re-baselined the same day. Boards move with the tuning campaign — "
        "`research/tune-data/rig5090.jsonl` is the running record; the README is refreshed "
        "with every board-moving merge."
    )


def render_plain_table(board):
    threshold = board["bold_ratio_threshold"]
    lines = ["| Model | bw24 plain | llama.cpp plain | Ratio |", "|---|---|---|---|"]
    for row in board["plain_decode"]["rows"]:
        ratio = fmt_ratio(row["bw24"], row["llama"], threshold)
        lines.append(f"| {row['model']} | {row['bw24']} | {row['llama']} | {ratio} |")
    return "\n".join(lines)


def render_spec_table(board):
    threshold = board["bold_ratio_threshold"]
    lines = ["| Model | bw24 spec | llama.cpp spec-best | Ratio |", "|---|---|---|---|"]
    for row in board["speculative"]["rows"]:
        bw24_cells = " / ".join(str(v) for v in row["bw24"])
        llama_cells = " / ".join(str(v) for v in row["llama"])
        ratios = " / ".join(
            fmt_ratio(b, l, threshold) for b, l in zip(row["bw24"], row["llama"])
        )
        lines.append(f"| {row['model']} | {bw24_cells} | {llama_cells} | {ratios} |")
    return "\n".join(lines)


def replace_block(text, tag, body):
    pattern = re.compile(
        rf"(<!-- {tag}:START[^>]*-->\n).*?(\n<!-- {tag}:END -->)", re.DOTALL
    )
    if not pattern.search(text):
        raise SystemExit(f"marker block {tag} not found in README.md")
    return pattern.sub(lambda m: m.group(1) + body + m.group(2), text)


def render_readme(board, original):
    text = original
    text = replace_block(text, "PERF-DATE", render_date_block(board))
    text = replace_block(text, "PERF-PLAIN", render_plain_table(board))
    text = replace_block(text, "PERF-SPEC", render_spec_table(board))
    return text


def render_svg(board):
    threshold = board["bold_ratio_threshold"]
    plain_rows = board["plain_decode"]["rows"]
    spec_rows = board["speculative"]["rows"]

    def ratio_of(row):
        return row["bw24"][0] / row["llama"][0] if isinstance(row["bw24"], list) else row["bw24"] / row["llama"]

    row_height = 34
    top = 168
    all_rows = [(r["model"], ratio_of(r)) for r in plain_rows] + [
        (r["model"] + " (spec)", ratio_of(r)) for r in spec_rows
    ]
    height = top + row_height * len(all_rows) + 40

    row_svg = []
    for i, (name, ratio) in enumerate(all_rows):
        y = top + i * row_height
        color = "#3fa79c" if ratio >= threshold else "#978f80"
        weight = "700" if ratio >= threshold else "500"
        row_svg.append(
            f'<line x1="64" y1="{y - 12}" x2="1216" y2="{y - 12}" stroke="#3a352c" stroke-width="1"/>'
            f'<text x="64" y="{y}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" '
            f'font-size="15" fill="#eee7da">{name}</text>'
            f'<text x="1216" y="{y}" text-anchor="end" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" '
            f'font-size="17" font-weight="{weight}" fill="{color}">{ratio:.2f}x</text>'
        )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="{height}" viewBox="0 0 1280 {height}">
  <rect width="1280" height="{height}" fill="#171613"/>
  <text x="64" y="72" font-family="ui-sans-serif,Arial,sans-serif" font-weight="800" font-size="52" fill="#eee7da">bw24</text>
  <text x="64" y="104" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" letter-spacing="0.06em" fill="#978f80">{board["rig"]}</text>
  <text x="1216" y="104" text-anchor="end" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" fill="#978f80">vs llama.cpp · updated {board["updated"]}</text>
  {"".join(row_svg)}
</svg>
'''


def main():
    check_only = "--check" in sys.argv
    board = load_board()

    original_readme = README_PATH.read_text()
    new_readme = render_readme(board, original_readme)
    new_svg = render_svg(board)

    old_svg = SVG_PATH.read_text() if SVG_PATH.exists() else ""

    changed = new_readme != original_readme or new_svg != old_svg

    if check_only:
        if changed:
            print("perf board is stale — run tools/update-perf-board.py and commit the result")
            sys.exit(1)
        print("perf board is up to date")
        return

    README_PATH.write_text(new_readme)
    SVG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text(new_svg)
    if changed:
        print("regenerated README.md perf tables + docs/perf-card.svg")
    else:
        print("no changes (already up to date)")


if __name__ == "__main__":
    main()
