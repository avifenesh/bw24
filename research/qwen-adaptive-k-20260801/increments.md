# qwen adaptive-K port — lane 3 (2026-08-01, H100 darklanes-8x GPU 3)

Mission: port gemma_spec.rs's adaptive draft length (MEMRA_SPEC_ADAPT — accepted-run law) to
the qwen MTP path (spec.rs), gate it, and answer whether it beats lane 1's fixed K=3 on q27.

**Verdict: REFUTED on the tuned qwen configs — fixed-K stays.** The law engages exactly as
designed (draft-length histograms below), holds exactness everywhere (K=1..8 self-consistency
PASS on both models with the law ON; every measurement run PASS), and still loses: it buys
acceptance-rate by adding rounds, and the round's fixed draft+verify cost wins. Same verdict
class as the retired 2026-07-07 EMA arm — different law, same conclusion.

## The port (spec.rs, generate_spec_inner2)

- Law: next round's draft depth `kc = (n_acc + 1).clamp(floor(pos), k_cap)` — gemma_spec.rs
  verbatim. Signal = the round's EXISTING accept readback (zero new syncs).
- Same envs/semantics as gemma: `MEMRA_SPEC_ADAPT`, `MEMRA_SPEC_ADAPT_FLOOR` (per-model
  default: n_embd>=3500 -> 4, >=2500 -> 2, else 1), `MEMRA_SPEC_FLOOR_CTX` (default 1024;
  past it floors >=4 relax to 1), `MEMRA_SPEC_CAPMAX` (default 7).
- qwen difference: OPT-IN (`=1`); fixed-K default untouched (gemma's is default-on). qwen's
  in-round p-min cut already shortens chains mid-round, so gemma's one-round-late p-min fold
  into kc is unnecessary — the law sees the cut through n_acc.
- Structurally free: the qwen draft graph is a SINGLE-STEP capture replayed per drafted
  token (no per-depth re-capture, unlike gemma's whole-chain graphs), and the verify/accept/
  rollback path already handles variable rounds (p-min). Burst (MEMRA_SPEC_STREAM) rounds
  draft the captured fixed depth and skip adaptation, like gemma's burst arm.

## Gates (GPU 3)

| gate | result |
|---|---|
| q27 run-spec K=1..8, ADAPT=1 | 8/8 PASS (identical to generate) |
| q35 run-spec K=1..8, ADAPT=1 (+ MEMRA_MTP_DRAFT sidecar head) | 8/8 PASS |
| all 42 measurement runs (both arms) | self-consistency PASS each |

q35 note: the box's `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` carries NO nextn tensors (`nextn=0`,
captured in the first gate attempt inside gate-q35-adapt-k1-8.log's predecessor — error line:
"model has no MTP/NextN head"). Its spec config is the board recipe
(tools/full-board-bench.sh q35-spec cell): the own-gen trimmed NVFP4 head sidecar
`draft-35b-owntrim-nvfp4head-q4blk.gguf` via `MEMRA_MTP_DRAFT`, K=2. The sidecar was staged
from the 5090 rig to `~/models/` on the box (944MB, byte-copied via scp).

## Measurements (interleaved pairs x3, NGEN=256, MEMRA_SPEC_STATS=1; spec tok/s from the
## same invocation as its plain-generate denominator; N=3 medians; warm box, lane GPU idle)

q27 = Qwen3.6-27B-Q4_K_M (dense, embedded nextn head), lane-1 flags
`MEMRA_SPEC_K=3 MEMRA_SPEC_HPOST=1 MEMRA_SPEC_PMIN=0.3`:

| cell | adaptive (median) | fixed (median) | delta | acceptance a/f |
|---|---|---|---|---|
| short (p1-code-short) | 128.59 | 127.57 | +0.8% (noise — law idles, len_hist identical) | 77.7% / 76.3% |
| board-2048 | 102.87 | 104.87 | **-1.9%** | 71.6% / 66.1% |
| agentic-long-v3 (~6k tok) | 134.05 | 134.68 | -0.5% | 91.5% / 91.2% |
| short, ADAPT_FLOOR=1 probe | 124.12 | 127.75 | **-2.8%** (gemma's floor-collapse, reproduced) | 79.9% / 76.3% |
| board, PMIN=0 probe | 99.23 | 101.34 | **-2.1%** (the law itself loses, not p-min shadowing) | 61.3% / 53.2% |
| board, K=6 + ADAPT_FLOOR=4 probe | 101.04 | 99.54 (fixed K=6) | +1.5% vs fixed K=6, **-3.7% vs fixed K=3** | 59.8% / 56.0% |

q35 = Qwen3.6-35B-A3B-UD-IQ4_XS (MoE) + owntrim sidecar head, its config K=2, board-2048:

| cell | adaptive (median) | fixed (median) | delta |
|---|---|---|---|
| board-2048, K=2 | 218.44 | 233.48 | **-6.4%** |

## Mechanism receipts (len_hist = rounds by draft depth, rep1 logs; identical across reps)

| run | rounds | full_accept | len_hist |
|---|---|---|---|
| q27 board adapt | 111 | 72 | [0, 45, 39, 27] — depths spread, law live |
| q27 board fixed | 103 | 57 | [0, 31, 14, 58] — p-min cuts only |
| q27 short adapt/fixed | 79 | 46 | [0, 1, 3, 75] BOTH — law never binds at short ctx |
| q27 board pmin0 adapt | — | — | [0, 24, 42, 44] vs fixed [0, 0, 0, 99] |
| q35 board adapt | 136 | 63 | [0, 52, 84] — 38% of rounds shrink to depth 1 |
| q35 board fixed | 122 | 55 | [0, 0, 122] |

The pattern everywhere: adaptive raises the full-accept share and acceptance-% but needs MORE
rounds for the same 256 tokens (q27 board 111 vs 103; q35 136 vs 122). Each round pays a
fixed draft-chain + verify + commit cost; at qwen's tuned shallow K (2-3) the law can only
remove draft opportunities, never add them.

## Why gemma won (+7-20%) and qwen doesn't

1. Gemma's measured wins ride the FLOOR at K=4-6 (floor=4 keeps drafts deep after misses:
   31B chat +15.7%, 12B +20%) — a mechanism for deep-K configs. Lane 1's q27 optimum is
   K=3 and q35's is K=2: there is no depth for a floor to preserve. The K=6 probe shows the
   gemma direction is real on qwen too (+1.5% over fixed K=6) — but the static per-class
   optimum (K=3) beats both.
2. qwen's tuned config already carries the in-round p-min cut (PMIN=0.3, worth ~+3.5% on
   board: fixed 104.87 vs 101.34 at PMIN=0), which owns the "stop drafting when unsure"
   value at per-step confidence granularity — finer than a one-round-late run-length proxy.

## Disposition

Per the flags doctrine (negative/flat experiments: kill the flag and dispatch arm — the
record is the ledger, not dead code): do NOT merge this arm as a default; the branch holds
the port + evidence for the owner's integration call. spec.rs's in-code ledger comment and
docs/FLAGS.md carry the verdict either way.

## Files

- crates/memra-engine/src/spec.rs — the port (setup block above the round loop; `k_this`
  dispatch; end-of-round kc update) + measured-verdict ledger comment
- docs/FLAGS.md — §3 qwen entry with verdict; graveyard cross-reference updated
- this dir: gate logs (gate-q27-adapt-k1-8.log, gate-q35-adapt-k1-8.log), 42 measurement
  logs (q27-*, q35-*), gates-adaptive.sh, bench-adaptive-ab.sh, ab-summary.log
