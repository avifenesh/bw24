# Task #56: SFT trace corpus progress

Status: **GREEN** for the bounded CPU/API proof batch.

Branch: `lane/sft-corpus`

Corpus repository: `~/projects/sft-traces`

Corpus commit: `e6b8de8` (`data: add DeepSeek SFT trace proof batch`)

## Steering and prior clearance

- Read and followed this worktree's `CLAUDE.md`.
- `~/.lanectl/inbox/sft.md` did not exist when checked on 2026-08-08. The
  inbox directory contained only `cx-fleet.md`; no substitute steering file was
  treated as task #56 guidance.
- Reviewed the prior pilot in
  `research/finetune-sku-20260802/REPORT.md`, including the recommended
  opencode -> OpenRouter -> pinned DeepSeek V4-Flash path, the endpoint
  allowlist, and the requirement to disable provider fallbacks.
- Pinned receipt hashes embedded in the corpus header:
  - report:
    `6f49abe08c49d0a24dd8fc759a3996ed7228110bca0281e722715d225664dbbf`
  - OpenRouter terms receipt:
    `108c5bed4357f8d6549040ea994caacc069dbdbc4f78f574e4c3fbb1e5a4f2a7`
  - DeepSeek platform terms receipt:
    `2b433c53cbac75491959025eba5a27908f9f1886352dce0a05f3d5a936d87790`
- Rechecked the live OpenRouter and DeepSeek terms on 2026-08-08. The
  OpenRouter page displayed a 2026-07-29 update date; its competing-service
  sentence was already present in the stored 2026-07-27 receipt. The corpus
  header records this as provenance and does not broaden the pilot's legal
  conclusion.

## Runtime validation

- Installed opencode: `1.18.13`.
- Configured model:
  `openrouter/deepseek/deepseek-v4-flash-0731`.
- Configured provider policy:
  `only=["novita","deepseek","deepinfra","fireworks"]`,
  `allow_fallbacks=false`.
- Live config SHA-256:
  `c7216cd234c6b4b6a931300e16d16fe770984a38d7668b27bd04451f7f1769bd`.
- A non-interactive auth smoke returned exactly `TRACE_PILOT_OK`, with empty
  stderr and provider-reported usage of 37,257 input, 7 output, 1,920
  cache-read tokens, and `$0.00338895`.
- `opencode run --format json` produced the expected event stream.
  `opencode export` produced the complete prompt, assistant messages, tool
  calls/results, model/provider ids, token accounting, cost, and file-change
  summary used by the normalized record.

No credential value was printed or copied into a prompt, receipt, or corpus
record.

## Implementation

- `tools/sft-gen.py`
  - 24 templates across bug-fix, refactor, test-writing, and explain.
  - Six reduced fixtures based on real memra shapes: cache economics, fleet
    counter deltas, acceptance parsing, perf markers, batch divergence, and
    mixed-expert tier plans.
  - Isolated temporary git repositories with CPU-only `AGENTS.md` constraints.
  - Full raw opencode events plus exported sessions in one JSONL trace record.
  - Token/cost metadata, prompt and content hashes, diffs, and before/after
    unittest receipts.
  - Content-hash dedup, prompt/record secret scans, forbidden-command audit,
    and fail-closed verification.
  - Fail-closed live-config validation for the exact provider order/allowlist,
    disabled fallbacks, and usage accounting before API calls.
  - Failure records written separately and never promoted to corpus traces.
- `tools/test_sft_gen.py`
  - Nine CPU-only tests for battery shape, fixture compilation, secret
    scanning, stable dedup hashes, event parsing, workspace setup, and large
    export capture, exact opencode provider-policy validation, and forbidden
    Rust/accelerator command auditing.
- `docs/SFT-TRACE-GENERATION.md`
  - Runbook, record schema, hygiene contract, and authorization provenance.

Corpus data is absent from the memra worktree. It lives only in
`~/projects/sft-traces`.

## Proof batch

File: `~/projects/sft-traces/corpus/deepseek-v4-flash-20260808.jsonl`

| Measure | Result |
|---|---:|
| Corpus header rows | 1 |
| Verified trace rows | 24 |
| Unique template ids | 24 |
| Unique content hashes | 24 |
| Task mix | 6 per task kind |
| Captured tool parts | 154 |
| Input tokens | 938,687 |
| Output tokens | 23,263 |
| Reasoning tokens | 32,717 |
| Cache-read tokens | 3,878,656 |
| Cache-write tokens | 0 |
| Provider-reported cost | $0.164374038 |

Trace timestamps span `2026-08-08T02:26:45.908218Z` through
`2026-08-08T02:43:03.569751Z`.

Corpus SHA-256:
`9f2ee4b90995f8060f3740213f82b0583afda71e4c6b33a0ed210d5c748f7814`

## Export failure and recovery

The first pass admitted 21 verified traces and excluded three refactor
templates at `opencode_export_parse`:

- `refactor-cache-economics`: `Unterminated string starting at: line 813
  column 22 (char 65056)`
- `refactor-perf-markers`: `Unterminated string starting at: line 741 column
  26 (char 64811)`
- `refactor-tier-plan`: `Unterminated string starting at: line 571 column 27
  (char 50202)`

The affected large `opencode export` payloads were truncated when captured
through a stdout pipe. Capturing the same export into a regular temporary file
produced valid JSON. The generator now uses file-backed stdout capture for
exports, and an 80,000-byte single-write regression test covers the path.

All three templates were rerun after the fix and appended as verified traces,
bringing the corpus to 24/24. The original failures remain in
`corpus/deepseek-v4-flash-20260808-failures.jsonl` as raw evidence; they are not
training records.

Failure-ledger SHA-256:
`8a2b64f76269045d33a66a9c5c1a1b1a771733403c858cfd9605a8f029bb01d4`

## Verification

```text
python3 -m py_compile tools/sft-gen.py tools/test_sft_gen.py
PASS

python3 tools/sft-gen.py --scan-only
PASS: 24 templates scanned

python3 -m unittest -v tools/test_sft_gen.py
Ran 9 tests in 0.049s ... OK

git diff --check
PASS

corpus invariant query
1 header; 24 traces; 24 verified; 24 unique templates; 24 unique hashes;
6 traces per task kind; provider/model policy PASS; hygiene PASS

sha256sum -c
corpus/deepseek-v4-flash-20260808.jsonl: OK
corpus/deepseek-v4-flash-20260808-failures.jsonl: OK
```

## Boundaries

- No GPU or accelerator job was started.
- No `rustup` or Rust toolchain command was run.
- No training job was started.
- No merge, tag, or origin push was performed.
