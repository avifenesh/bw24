# ladder-3072: the sp8→sp64 rung re-sweep under the deep kernel (2026-08-02)

Lane `lane/ladder-3072` (from `restructure/public-split` b8ca4e2e). Priced follow-up to
`research/fa-decode-deep-20260802/RESULTS.md` §8: the 3072 rung of the nkv≤4 split ladder
was calibrated 2026-07-08 on the conflicted v4 core; under the always-on deep kernel the
combine (19 µs at d2048) exceeded the vec kernel (11.5 µs) — the stale-verdict law says
re-sweep. This is also KAT's weakest vs-llama region (0.96–1.02x, floor at d2048).

Rig: RTX 5090 Laptop 24463 MiB sm_120a, 82 SMs, `performance` profile, every GPU run under
`flock /tmp/gpu5090.lock`. POWER NOTE: the owner cut wall power once this session
(~16:35–16:37Z, `outage-note.md` + `power-log.txt`); no timed cell ran during or spanned
the window — kernel cells (regime A) ended 16:34:40Z, all e2e cells (regime B) started
after re-plug verification (AC=1, idle clocks). The sweep harness carries a per-rep ADP0
guard; zero reps quarantined.

## 1. Current rungs (the config under test)

`fa_split_keys` (crates/memra-engine/src/lib.rs), 5090 branch (`fa_sm_count() < 128`):

- `n_head_kv <= 4` (q35/KAT/o35b class): **`t_kv <= 3072 → 8`**, `<= 16384 → 64`, else 128.
- `n_head_kv > 4`: `t_kv <= 8192 → 32`, `<= 16384 → 64`, else 128.
- big-rig (≥128 SMs) and gemma (FA_SP_GEMMA) ladders untouched by this lane.

The lane arbitrates only the nkv≤4 sp8→sp64 boundary (candidates: keep 3072, move earlier,
move later). `MEMRA_FA_SPLIT` (OnceLock, one split per process) is the sweep seam — at a
fixed depth each forced split reproduces a candidate boundary's behavior exactly, no code
churn per arm.

## 2. Kernel-level receipts (regime A, pre-outage, quiet rig 56–60 °C)

(a) `fa_deep_bench 200` per forced split (dc call form incl. memsets + combine, 3
interleaved rounds, medians — `kernel-sweep-split.log`); (b) nsys per-kernel medians
(N=10/kernel, `nsys-sp{8,32,64}-d{1024..4096}.txt`), deep vec + combine_f32 sums:

| depth | sp8 vec+combine | sp32 vec+combine | sp64 vec+combine | best |
|---|---|---|---|---|
| 1024 | 7.5+9.7 = **17.2 µs** | 5.3+3.0 = 8.3 µs | 8.7+1.9 = 10.6 µs | sp32 |
| 2048 | 11.4+19.6 = **31.0 µs** | 7.8+5.3 = 13.2 µs | 9.2+3.1 = 12.2 µs | sp64 |
| 3072 | 15.7+28.4 = **44.1 µs** | 11.1+8.0 = 19.0 µs | 13.9+4.4 = 18.3 µs | sp64 |
| 4096 | 20.6+36.2 = **56.8 µs** | 14.3+10.6 = 24.9 µs | 14.2+5.7 = 19.9 µs | sp64 |

The mechanism is exactly §8's pricing: combine + partial-buffer cost scales with n_splits
(sp8 at d2048 = 256 splits → combine 19.6 µs vs sp64's 3.1 µs), while the deep vec kernel
lost so little from coarser splits that sp64's vec penalty never catches the combine bill.
sp8's *total* is 2.4–2.9x the sp64 total across d2048–4096. The `fa_deep_bench` wall
timings agree (sp8 d2048 36.8 µs vs sp64 16.8 µs, deep arm).

## 3. e2e arbitration (regime B, N=3 interleaved, run-gen tg128, argmax gate in every run)

`run-ladder-sweep.sh` + `run-ladder-d512.sh` → `ladder-sweep.jsonl` (per-rep values +
temps), `sweep-console.log`, `mem-*.log`. Prompts = the depth-decode lane's document
prefixes (d512/2048/4096 reused; d1024/3072 cut fresh by the same tokenizer-exact recipe,
`ladder-prompts-manifest.jsonl`). Arms adjacent per (model, depth), order alternated per
rep; llama denominators fresh this session (`llama-*-rep*.log`). Temps 58–77 °C
(fan-limited laptop regime; pairing, not absolute level, is the evidence).

KAT (tg128 tok/s, medians of 3):

| arm | d512 | d1024 | d2048 | d3072 | d4096 |
|---|---|---|---|---|---|
| old ladder (B=3072: sp8 to 3072) | 195.2 | 186.0 | 182.6 | 175.9 | 182.2 |
| new ladder (B=512: sp8 to 512) | 195.2 | 190.3 | 188.0 | 186.4 | 182.2 |
| sp32 probe | 196.2 | 189.4 | 188.1 | 185.6 | 180.5 |
| **new/old** | 1.000x | **1.023x** | **1.029x** | **1.060x** | 1.000x |
| new/llama | — | 1.014x | 0.996x | 1.008x | 1.000x |

q35 (tg128 tok/s, medians of 3):

| arm | d512 | d1024 | d2048 | d3072 | d4096 |
|---|---|---|---|---|---|
| old ladder | 191.9 | 185.2 | 180.2 | 174.0 | 169.2 |
| new ladder | 191.9 | 188.5 | 187.6 | 183.8 | 182.6 |
| sp32 probe | 192.9 | 190.4 | 187.7 | 183.6 | 180.3 |
| **new/old** | 1.000x | **1.018x** | **1.041x** | **1.056x** | **1.080x** |
| new/llama | — | 1.138x | 1.133x | 1.139x | 1.168x |

- sp8 loses at EVERY depth ≥1024, growing with depth (kat d3072 −5.6 %, q35 d4096 −7.4 %).
- d512: sp8/sp64 within ±0.2 % (noise) — sp8 keeps only the short band it was validated
  on (its 2026-07-08 win was ctx128–512). Boundary **3072 → 512**.
- sp32: ties sp64 in the low band (d512–1024, ≤+1 %, inside spread), loses at d4096
  (q35 180.3 vs 182.6, kat 180.5 vs 182.2) — no third rung earns its config; one boundary
  move is the whole change.
- KAT's weak cell (d2048, was 0.960x vs llama in the fa-deep table): now 0.996x same-session
  (llama read 188.7 here); d1024/3072/4096 all ≥1.000x. q35 widens every cell (min 1.133x).
- q35 d4096 old-arm note: the old ladder at d4096 already ran sp64 (3072 < 4096), but the
  decode window 4096..4224 sits above the boundary in BOTH ladders, so new/old = 1.080x
  there restates sp8-forced vs sp64 (the sweep's forced-arm map); the published ladder
  cells (d≤3072) are the boundary-sensitive evidence.
- FOUND IN PASSING (forced-env diagnostic, NOT a production cell): kat + forced
  `MEMRA_FA_SPLIT=8` at d4096 fails run-gen's prefill-vs-decode argmax gate
  deterministically ×3 (near-tie: prefill l[20]=16.4378 vs l[19]=16.3688, margin 0.069;
  512 splits' combine fold order flips it). No ladder — old or new — dispatches sp8 at
  d4096; the cells are null in the JSONL with the gate line as the recorded cause
  (`mem-sp8-kat-d4096-rep*.log`).

## 4. The adopted change

`fa_split_keys`, nkv≤4 branch: `t_kv <= 3072 { 8 }` → **`t_kv <= 512 { 8 }`** (sp64 above,
16384/128 rung unchanged). Kernel-check straddle depths follow the rung (511/512/513 added
in `kernel_check.rs` FA-DEEP pin + `fa_deep_bench` bit depths; 3071/3073 kept as deep-region
coverage).

### The graph-segment bug the new rung EXPOSED (fix included, decode.rs)

First battery run: `graph-decode-gate kat P=400 N=160` FAILED 97/160 (eager crosses
t_kv 512→513 inside the session; buckets show ns 51..64 then 9, captures=1). Mechanism:
the exec-update dc kernels derive their in-kernel partition from the CAPTURED `split_keys`
argument (ONE-PARTITION law: `ns_eff = ceil(T_kv/split_keys)`), and `fa_apply` retunes
only `n_splits`/grid — it cannot rewrite `split_keys`. But `fa_class_of` (the round-45
segment fingerprint) tracked only (fa_vec, v4, fa512) — a capture whose segment straddled
a LADDER RUNG replayed the far side's partition against eager's near side: same math,
different FP fold order, first near-tie flips the stream. This was LATENT at the old 3072
rung (kat P=3000 crossed 3072 and passed on logit margins — the fa-deep battery's green
was luck, not law); the 512 rung exposed it at a near-tie two steps in. Fix: the ladder
value `fa_split_keys(t_kv, nkv)` joins the fingerprint tuple, so a segment never straddles
a rung and the captured `split_keys` equals the live ladder on every replay — kat P=400
now PASS 160/160 BIT-IDENTICAL, captures=2 (one per side of the rung), and rung crossings
cost one recapture per session, not per token.

## 5. The battery (NEW NUMERIC CONFIG — full gates on the final binary)

`run-battery-ladder.sh` → `battery-console2.log` + `gate-*.log`, `kernel-check-full.log`
(first battery: `battery-console.log`, identical result except the P=400 find above —
superseded by this run on the fixed binary):

- **kernel-check**: ALL GREEN (436 OK, 0 FAIL) — incl. the FA-DEEP byte pin at the new
  511/512/513 rung straddle + both geometries, and every pre-existing pin.
- **fa_deep_bench bit gate**: 36/36 OK at the new bit-depth grid incl. 511/512/513 eager +
  dc + bucketed replay (`fa-deep-bench-newrung.log`).
- **run-gen argmax**: MATCH ×3 model classes (kat/q35/o35b) at d4096, plus kat/q35 at
  d1024 — the band the rung move re-assigns (sp8→sp64) (`gate-argmax-*.log`).
- **run-spec K=1..8**: PASS 8/8 on q35 (own-trim drafter, d2048 prompt = new sp64 band)
  (`gate-spec-q35.log`).
- **decode-batch**: config (steps 32, B=8) ALL GREEN; strict equalized (`MEMRA_MMVQ=0
  MEMRA_NO_FUSE_NORMQ=1`, steps 16, B=4) ALL GREEN — bucketed rows group consecutive rows
  by ladder value, so the rung move changes grouping boundaries (`gate-decode-batch-*.log`).
- **graph-decode**: PASS ×4 — kat P=400 N=160 (crosses the NEW rung in-session, the cell
  that failed pre-fix), q35 P=500 N=96, kat P=3000 N=160, q35 P=6000 N=160 — all
  BIT-IDENTICAL (`gate-graph-decode-*.log`).
- **graph-session**: q35 step-lift PASS (`gate-graph-session.log`).

## 6. Verdict

**ADOPTED**: sp8→sp64 boundary moves 3072 → 512 in the nkv≤4 5090 ladder. e2e
+1.8–6.0 % (KAT) and +1.8–8.0 % (q35) across d1024–4096, no losing cell (d512 flat), the
full new-numeric-config battery green, and the stale rung's root cause (combine scaling
with n_splits under a faster vec kernel) receipted at kernel level. KAT's weakest vs-llama
cell (d2048) moves 0.960x → 0.996x same-session. The segment-fingerprint fix rides along
as a correctness prerequisite (a ladder-straddling capture is wrong under ANY ladder —
the old rung just never got caught).

Not adopted: an sp32 mid-rung (ties sp64 low, loses deep — no config earned); any change
to the nkv>4, big-rig, gemma, or ≥16384 rungs (out of scope, unmeasured here).

## Files

`kernel-sweep-split.log` (fa_deep_bench × forced splits), `nsys-sp*-d*.txt` (per-kernel
medians; `.nsys-rep` local-only per .gitignore), `make-ladder-prompts.py` +
`ladder-prompts-manifest.jsonl` + `depth-{1024,3072}-{kat,q35}.txt`,
`run-ladder-sweep.sh` / `run-ladder-d512.sh` → `ladder-sweep.jsonl`, `sweep-console.log`,
`mem-sp*-*.log`, `llama-*-rep*.log`, `summarize-ladder.py`, `outage-note.md` +
`power-log.txt`, `fa-deep-bench-newrung.log`, `run-battery-ladder.sh` →
`battery-console.log` (pre-fix, the P=400 find), `battery-console2.log` (final, ALL
GREEN) + `gate-*.log`, `kernel-check-full.log`,
`gate-graph-decode-kat-p400-oldladder-sp8.log` (the forced-sp8 control that passed —
isolating the rung crossing as the trigger).
