#!/usr/bin/env python3
"""cache_economics.py — turn a /metrics scrape into the earning-model row
(lane/cache-metering, 2026-08-07).

THE QUESTION THIS ANSWERS: cache-hit prompt tokens bill at (a fraction of) full
price but cost ~zero compute. At the measured hit rate, how many billed prompt
tokens does one computed prompt token carry?

    effective revenue multiplier = billed_prompt_tokens / computed_prompt_tokens

where billed = computed + cache_billing_factor * cached (factor 1.0 = cached
tokens bill at full input price — the default listing shape; 0.25 = the
OpenRouter cached-input discount tier, research/or-provider-20260802/REPORT.md).

INPUT (any one):
  cache_economics.py http://127.0.0.1:8181/metrics      # live scrape
  cache_economics.py metrics.json                       # saved scrape
  cache_economics.py --log server.log                   # [meter]/[prefix-cache] lines only
                                                        # (no usage totals — scrape preferred)

The /metrics fields consumed (all worker-truth, cumulative since process start):
  prompt_tokens_in   every prompt token admitted (cached + computed)
  cached_tokens_in   tokens served from ANY cache tier (continuation pool,
                     spec resume, cross-request prefix cache)
  computed_tokens_in prompt_tokens_in - cached_tokens_in (recomputed here if absent)
  tenants            optional per-tenant {prompt_tokens_in, cached_tokens_in} rows
  lcp_histogram      optional probe-depth histogram; buckets [64,512) = the
                     tick-seg segmentation window

OUTPUT: one JSON row on stdout (append it to the earning ledger), plus a
human-readable table on stderr. Exit 1 if the scrape carries no prompt tokens
(no traffic = no receipt; never report a multiplier from zeros).

First-listed-week receipt query (the whole point of the lane):
    python3 tools/cache_economics.py http://<serve-host>/metrics \
        --cache-billing-factor 1.0 >> research/cache-meter-<date>/economics.jsonl
"""
import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone


def load_metrics(src: str) -> dict:
    if src.startswith("http://") or src.startswith("https://"):
        with urllib.request.urlopen(src, timeout=10) as f:
            return json.load(f)
    with open(src) as f:
        return json.load(f)


def row_from_metrics(m: dict, factor: float) -> dict:
    prompt = int(m.get("prompt_tokens_in", 0))
    cached = int(m.get("cached_tokens_in", 0))
    computed = int(m.get("computed_tokens_in", prompt - cached))
    if prompt <= 0:
        sys.exit("cache_economics: no prompt tokens in the scrape — no traffic, no receipt")
    if not 0 <= cached <= prompt:
        sys.exit(f"cache_economics: inconsistent scrape (cached {cached} vs prompt {prompt})")
    hit_ratio = cached / prompt
    billed = computed + factor * cached
    row = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prompt_tokens_in": prompt,
        "cached_tokens_in": cached,
        "computed_tokens_in": computed,
        "cache_hit_token_ratio": round(hit_ratio, 6),
        "cache_billing_factor": factor,
        "billed_prompt_tokens": round(billed, 1),
        # THE ROW: billed prompt tokens carried per computed prompt token.
        # computed == 0 (a pure-replay workload) is honest infinity, reported as null
        # with the note — never a fabricated large number.
        "revenue_multiplier": round(billed / computed, 4) if computed > 0 else None,
        "completed_requests": m.get("completed"),
        "tokens_out": m.get("tokens_out"),
    }
    if computed == 0:
        row["note"] = "every prompt token cached (computed=0): multiplier unbounded"
    # per-tenant rows (PC-ISO composition): same arithmetic per tenant.
    tenants = m.get("tenants") or {}
    if tenants:
        trows = {}
        for name, t in sorted(tenants.items()):
            tp, tc = int(t.get("prompt_tokens_in", 0)), int(t.get("cached_tokens_in", 0))
            tcomp = tp - tc
            trows[name] = {
                "prompt_tokens_in": tp,
                "cached_tokens_in": tc,
                "hit_ratio": round(tc / tp, 6) if tp else 0.0,
                "revenue_multiplier": round((tcomp + factor * tc) / tcomp, 4)
                if tcomp > 0 else None,
            }
        row["tenants"] = trows
    # LCP histogram: how much probe traffic lands in the tick-seg [64,512) window.
    h = m.get("lcp_histogram") or {}
    edges, counts = h.get("edges") or [], h.get("counts") or []
    if edges and counts and sum(counts) > 0:
        total = sum(counts)
        # buckets whose lower edge is in [64, 512) — matches worker::LCP_HIST_EDGES.
        window = sum(c for e, c in zip(edges, counts) if 64 <= e < 512)
        row["lcp_probes"] = total
        row["lcp_window_64_512_share"] = round(window / total, 6)
    return row


# ---- log fallback: [prefix-cache] hit/insert lines (no usage totals in the log —
# per-request cached/prompt splits ride the HTTP usage block, not stderr; this mode
# only recovers prefix-cache hit-token mass for a scrape-less post-mortem). ----
HIT_RE = re.compile(r"\[prefix-cache\] hit: (\d+) of (\d+) prompt tokens")


def row_from_log(path: str, factor: float) -> dict:
    hits, hit_toks, prompt_toks = 0, 0, 0
    with open(path, errors="replace") as f:
        for line in f:
            mm = HIT_RE.search(line)
            if mm:
                hits += 1
                hit_toks += int(mm.group(1))
                prompt_toks += int(mm.group(2))
    if hits == 0:
        sys.exit("cache_economics: no [prefix-cache] hit lines in the log — "
                 "scrape /metrics for the full picture (log mode sees only prefix-cache "
                 "hits, and only their own requests' prompts)")
    computed = prompt_toks - hit_toks
    return {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "log-partial (prefix-cache hit requests only — NOT the full traffic split)",
        "prefix_hits": hits,
        "prompt_tokens_in_hit_requests": prompt_toks,
        "cached_tokens_in_hit_requests": hit_toks,
        "hit_ratio_within_hit_requests": round(hit_toks / prompt_toks, 6),
        "revenue_multiplier_within_hit_requests":
            round((computed + factor * hit_toks) / computed, 4) if computed > 0 else None,
        "cache_billing_factor": factor,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", help="/metrics URL or saved JSON scrape")
    ap.add_argument("--log", help="server stderr log (partial fallback; scrape preferred)")
    ap.add_argument("--cache-billing-factor", type=float, default=1.0,
                    help="fraction of input price a cached token bills at "
                         "(1.0 = full price, 0.25 = the OR cached-input tier)")
    args = ap.parse_args()
    if bool(args.source) == bool(args.log):
        ap.error("exactly one of <source> or --log")
    if args.source:
        row = row_from_metrics(load_metrics(args.source), args.cache_billing_factor)
    else:
        row = row_from_log(args.log, args.cache_billing_factor)
    print(json.dumps(row))
    # human summary to stderr so the JSON row stays pipe-clean.
    mult = row.get("revenue_multiplier",
                   row.get("revenue_multiplier_within_hit_requests"))
    hit = row.get("cache_hit_token_ratio",
                  row.get("hit_ratio_within_hit_requests", 0.0))
    print(f"[cache-economics] hit-token ratio {hit:.1%}, "
          f"billing factor {args.cache_billing_factor}, "
          f"revenue multiplier {mult if mult is not None else 'unbounded'}"
          f" (billed prompt tokens per computed prompt token)", file=sys.stderr)
    if "lcp_window_64_512_share" in row:
        print(f"[cache-economics] tick-seg window [64,512): "
              f"{row['lcp_window_64_512_share']:.1%} of {row['lcp_probes']} probes",
              file=sys.stderr)


if __name__ == "__main__":
    main()
