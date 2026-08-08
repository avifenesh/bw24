# Serving — the OpenAI surface and the replica fleet

This is the serve-surface doc: fleet topology and measured throughput, the isolation
contract, the OpenAI tools surface, the gateway listing surface (`/v1/models` schema,
rate-limit headers, graceful drain), safetensors/FP8 checkpoint serving, cross-request
prompt caching with per-tenant `cache_salt` isolation, and the honestly-stated numeric
edges of batched serving.

> Numbers here are engineering receipts, each labeled with its rig — see
> [Rigs](PERFORMANCE.md#rigs--what-was-measured-on-what) for what each label is. A number
> without its rig label is not a number: the same cell moves 5-12% between two pods of the
> same SKU and ~2x between a 188-SM and an 82-SM board. The open gaps stated below travel
> with the wins.

Multi-GPU serving has two shapes, and which one applies is decided by whether the model fits
on one card:

- **Replica fleet** (the default for a model that fits): N independent `memra-server`
  processes, one engine per GPU (`Engine::new(0)`; `CUDA_VISIBLE_DEVICES` is the placement
  mechanism), fronted by an admission proxy. This is the throughput shape — see
  [Fleet tooling](#fleet-tooling).
- **Pipeline-parallel PP-2** (for a model that fits only across the pair): ONE engine
  process, the layer trunk cut into stages, each stage's weights and KV resident on its own
  card. Opt-in via `MEMRA_PP_STAGES` / `MEMRA_PP_DEVICES`; see
  [Pipeline-parallel serving](#pipeline-parallel-pp-2-serving) below for what is gated and
  what is refused.

Tensor parallelism is neither — it is a separate in-progress build (M0 comms floor measured
— ARCHITECTURE-H100.md).

## Fleet tooling

(Not to be confused with the OpenAI `tools` API surface — that is
"[OpenAI tools surface](#openai-tools-surface-serve-tools-lane-2026-08-02)" below.)

| tool | what it does |
|---|---|
| `tools/serve-fleet.sh start\|stop\|status\|restart` | declarative fleet supervisor: brings up `REPLICAS_PER_GPU` replicas per GPU in `GPUS`, fronts them with the proxy, health-loop restarts anything that dies. systemd-free; pidfiles under `$FLEET_RUN` |
| `tools/serve-proxy.py` | least-outstanding reverse proxy with per-backend admission cap (default 8 = the engine's exactness-tier batch width and the two-replicas-per-GPU anti-thrash bound). Bounded FIFO queue with deadline → 429 + Retry-After; `/health` + `/metrics` JSON |
| `tools/load-serve.py` | concurrent OpenAI-format load harness: aggregate output tok/s, p50/p95 latency, JSONL per load point |
| `tools/serve-smoke.sh` | OpenAI-surface smoke gate for a single server |
| `deploy/systemd/memra-server.service` | example unit for a **single supervised instance** (the other deployment shape — `serve-fleet.sh` is the systemd-free multi-replica path). `Type=notify` with `READY=1` after the models load and the socket binds, `WATCHDOG=1` pings only while inference is live, `STOPPING=1` + `EXTEND_TIMEOUT_USEC` so a drain is not SIGKILLed, and exit 70 (unrecoverable GPU) distinguished from exit 1 (bad config). Copy, do not symlink: every path is site-specific and the value is the supervision contract in the directive choices, each commented with the failure it prevents |

## Measured numbers (Qwen3.5-9B Q8_0; receipts in `research/`)

- **Single replica (H100, rented pod):** temp-0.7 c=8/16/32 medians **654/657/659 tok/s** after the
  batched decode tick (z-batched FA + KV append, device sampling, lean logits — +25-36%
  over the pre-batched tick; N=4, `research/batched-tick-inc2-20260801/`; chunk-8 era —
  see the exact-16 tier below).
- **Managed fleet, 3 rented H100s x 2 replicas (v0.60-validated):** **1,477 tok/s** through the
  admission proxy at c=96 (N=2 interleaved passes: 1477.0/1473.1), zero 429s/5xx —
  managed now matches the v0.59-era 1,480 direct number (the ~7% admission-overhead gap
  closed at the fleet level). Chaos-tested: SIGKILL a replica mid-load, breaker DOWN the
  same second, supervisor restart +2s, backend UP +9s, 8/768 requests lost (exactly the
  victim's in-flight cap), aggregate across the kill 1,487 tok/s; greedy hash identical
  on all 6 replicas in every condition, 18/18 (`research/fleet-v060-20260801/SUMMARY.md`).
  The proxy cap (8) was calibrated on the v0.59 core — the cap re-sweep is pending the
  next box window (stale-verdict risk flagged in the validation summary).
- **Single replica (RTX 5090 Laptop — the local rig, exact-16 tier):** with the Q8_0 split-plane mirror
  (`MEMRA_Q8RP=1` on 24GB; Hopper default), the worker auto-selects decode chunk 16 —
  c=16 median **494.5 tok/s vs 416.4** at chunk 8, same mirror, interleaved N=4
  (**+18.8%**; +33.8% vs the mirror-less baseline); c=32 at `MEMRA_CTX=2048` runs
  **502.1** with 128/128 ok (single run; `research/batched-tick-inc3-20260801/`).
- **Solo (c=1) decode rides the naked m=1 program** (serve-path phase 2, 2026-08-05): a lone
  session's tick runs `decode_layers_eager` verbatim instead of the batched body, inheriting
  the whole m=1 fusion chain. Order-paired N=5 on the 5090 (82 SM), decode-only `step_p50`:
  **+8.33%** on the 9B (123.7 → 134.1 tok/s, 5/5 wins) and **+5.19%** on the 27B (43.6 → 45.8,
  5/5); c=8 saturation flat (−0.00% / −0.18%). This closes the class of c=1 gap phase 1
  measured against naked decode — serve c=1 now sits level with the same-board `run-gen`
  denominator (134.8/134.5/134.0). Notably it also **retires the solo graph door as a win**:
  `GraphSession` replay amortized the same launch overhead this removes outright, so with the
  fast path in place the door is a net loss at every length measured out to mt=1024 and
  `MEMRA_GS_MIN=384` must NOT be lowered (FLAGS §serve; `research/servepath-p2-20260805/`).
- **Spec fast lane, now CONCURRENCY-GATED inside one process** (lane/spec-gate, 2026-08-07 —
  this supersedes the "run spec and bulk as separate server processes" guidance): MTP
  speculative serving is a single-stream latency tier — 1.82x plain serving at c=1 on the 27B
  (131.8 vs 72.5 tok/s) — and plain batching overtakes between c=2 and c=4 because the spec
  path is a serial burst QUEUE, not a contended one (phase (a) steps each spec session's whole
  burst in a host loop; phase (c) excludes spec rows from batched decode). Pooling the verify
  is REFUTED at a 16-column exact-kernel width ceiling (`research/spec-scaling-20260806/`), so
  the answer is scheduling policy: **one server now admits spec only while `active+1 <= 2` and
  DEMOTES live spec sessions into the batched phase at `active >= 4`**, with `active==3` a
  hysteresis band and demotion one-way per session. The handoff is a real cache transfer
  (`(cache, next_pred)` into the session's cache + `device_next`, a carried pending flushed
  first) and is byte-exact for greedy: a session demoted mid-generation emits a stream
  byte-identical to one batched from the start. Measured q9 on the 5090, N=5 interleaved: the
  gated curve tracks spec at c=1-2 (251.2 tok/s, 1.81x over batched) and batched at c=4-8
  (504.7 tok/s, 2.03x over always-spec), with per-stream p50 at c=8 equal to batched's 1.963s
  rather than spec's 3.973s. Sampled and constrained spec sessions do not demote (their
  `next_pred` is a greedy/unmasked argmax) and stay on the serial path, bounded by the admit
  ceiling. One residual, disclosed: a first-wave TTFT p95 transient (0.423s vs never-spec's
  0.017s at c=4) confined to the at-most-`LOW` sessions admitted before a load ramp — p50
  matches never-spec; set `MEMRA_SPEC_GATE_LOW=0` to never admit spec if cold-ramp p95
  outweighs c=1 throughput. Flags: `MEMRA_SPEC_GATE` (rollback seam),
  `MEMRA_SPEC_GATE_LOW`/`_HIGH`. Receipts: `research/spec-serving-20260801/`,
  `research/spec-gate-20260806/`.
- **The plain-serve c=1 gap (task #70) is closed by the fast path above — with one cell
  pending re-measure and one still open.** Phase 1 measured serve c=1 trailing the naked
  CLI **−11.74%** on a Q8_0 27B cell (`memra-server` 46.09 tok/s, N=3 median, vs `run-gen`
  naked 52.22, single run; rig `pro6000wk-runpod-community`, same commit and prompt); the
  measured cause — B=1 ran the batched body and missed the m=1 fusion chain — is exactly
  what `MEMRA_SERVE_B1FAST` fixes, and on the 82-SM 5090 serve c=1 now sits level with the
  same-board `run-gen` denominator. The −11.74% number itself is **pre-fix** and the 188-SM
  cell has not been re-measured since — do not quote it as current. Still open: the NVFP4
  **spec** serve path at **−8.66%** (serve 170.55 vs bare 186.72, rig `pro6000wk-runpod`,
  also a pre-H3 measurement) — the spec tier runs its own burst loop that the `b_n==1`
  fast path does not touch. Receipts: `research/q27-deepdive-20260805/RESULTS.md` §4
  (phase 1), `research/servepath-p2-20260805/RESULTS.md` (the fix + the H1 refutation).

## The isolation contract

Greedy serving is **isolated-identical under concurrent load at defaults**: a request's
output tokens are byte-identical whether it arrives alone or inside a full batch. This is
gated, not assumed — the serve gate replays the same prompts at c=1 and c=16 and
byte-compares every stream.

**Read the gate's exact scope before quoting the contract as unconditional.** The gate runs
16 prompts at **96 max_tokens** with all sessions arriving together, i.e. at *equal* depth.
Outside that shape a 768-token greedy request diverged from its own solo reference at byte
1347 (≈ token **331**) when it shared batched decode with sessions **staggered to different
depths**, and on a second run the divergence moved to byte 2379 (lane/spec-gate receipt,
`research/spec-gate-20260806/logs/exact/`, arm `REF_LOAD`).

**lane/iso-gap (task #91, 2026-08-07) reproduced that receipt on demand and attributed it —
the two mechanisms this paragraph used to name are both innocent** (receipts
`research/iso-gap-20260807/`):

- *Depth staggering moves nothing.* At the engine tick, with the program family held fixed,
  a co-resident session at ANY other depth — including across a `fa_split_keys` ladder-rung
  boundary, B=2..8, three rungs, 300-step horizons — changes **zero bits** of a session's
  logits (`iso-gap-probe`, 8 arms + canary). `decode_step_batch`'s rung guard is
  per-session-correct: every row either shares one rung (the seqs kernel then derives each
  session's split partition from its OWN `t_kv` — the ONE-PARTITION law) or all rows take the
  per-seq eager loop. The property is now pinned by the `isogap` fast-gate arm, which places
  the straddle per-rig.
- *The real carrier is the solo↔batched **program flip at the co-residence boundary**.* A solo
  session runs the m=1 fused trunk (`MEMRA_SERVE_B1FAST`) or GraphSession replay
  (`MEMRA_SERVE_GS`); the moment a second session arrives mid-stream, its ticks flip to the
  batched body — a *different documented FP composition* (`decode-batch-gate` gate1's config
  jurisdiction), and a near-tie can flip. Measured: solo-vs-loaded diverges at byte 659 under
  defaults, while with the program family pinned (`MEMRA_SERVE_B1FAST=0 MEMRA_SERVE_GS=0`)
  solo and loaded streams are **byte-identical** — and the loaded default stream equals the
  pinned stream byte-for-byte, so the flip accounts for the *entire* divergence. The moving
  byte (1347 → 2379; reproduced 1248 → 1361 at a fixed 2 s co-arrival delay) is the
  **arrival-tick jitter** of the flip boundary, and a co-resident arriving after the stream
  finished (6 s delay) leaves it byte-identical.

So the honest statement of the contract today: **byte-identical at equal depth (gated to 96
tokens) and depth-isolation-clean within a program family (the `isogap` gate); a session whose
co-residency CHANGES mid-stream crosses the solo↔batched config boundary, and its stream may
legally differ from its solo twin from that tick on — the token-level cross-config gap this
doc already documents for `MEMRA_SERVE_B1FAST`.** Deployments that need solo-vs-loaded
byte-equality pin one program family (`MEMRA_SERVE_B1FAST=0 MEMRA_SERVE_GS=0`) and pay the
measured solo cost (−8.33% q9 decode-only at c=1; the flag table below). It is also why
lane/spec-gate had to test its demotion handoff at a pinned batch shape
(`MEMRA_SPEC_DEMOTE_AT`) rather than by triggering it with load — under load, no comparison
isolates the property under test.

The contract is over **tokens**, not over the FP program that produces them, and since
2026-08-05 (`MEMRA_SERVE_B1FAST`, serve-path phase 2) a solo tick deliberately runs a
*different* program from a batched one: at `b_n==1` the tick uses the m=1 fused trunk
(`decode_layers_eager` — the same code `run-gen` runs), while `b_n>=2` uses the batched
body. Those two carry the long-accepted decode-config FP-composition gap
(`decode-batch-gate`'s jurisdiction), so the guarantee is exactly as stated — token
streams match — and is not a claim of bit-identical logits between a solo and a batched
tick. The direction is deliberate: a c=1 request now computes what the CLI computes
(strict bit-identity to `decode_step_h`, gated), which is why `serve-st-gate`'s
CLI-vs-server greedy token-stream check is a *stronger* assertion than before. Verified at
the stream level on q9 NVFP4-MTP: greedy 150 ids and seeded-sampled identical to the
`run-gen` oracle and across both arms (`research/servepath-p2-20260805/`). It is also a fixed defect, not a freebie: the batched
cuBLASLt prefill router and shared-expert-gate GEMMs were m-dependent, so under
cross-request prefill batching a MoE request's own expert selection changed with its
co-arrivals (the supported Qwen3.6-35B had the same defect as the onboards whose serve
gate exposed it — Ornith-35B 6/16, KAT 7/16, both 16/16 after the fix). Default
`MEMRA_ROUTER_PREFILL_EXACT` routes prefill through decode's m-invariant router/gate
kernels; a bit-identical batched twin recovers most of the prefill cost
(`MEMRA_ROUTER_BATCH`, FLAGS §3). Receipts: `research/concat-prime-exact-20260802/`,
`research/fast-router-20260802/`.

**Admission is VRAM-aware** (2026-08-02): once the first admitted session reveals the
model's per-session VRAM cost, further admissions require free ≥ 2x that cost — otherwise
the request *waits* in the same never-rejected FIFO as the session-count cap instead of
failing with a cache-alloc OOM (the c=16 8192-ctx failure mode under resident-if-fits,
caught by the serve gate as instant HTTP 400s — `research/fast-router-20260802/RESULTS.md`).
The first session always admits; an OOM with no active sessions is real capacity and
still errors loudly, with the CUDA error quoted.

### 64-client robustness (lane/admit-oom, 2026-08-06) — gated, not assumed

At `MEMRA_MAX_SESSIONS=64` with spec ON on a 24GB card, the 2026-08-02 cost model
under-charged the live burst and **every one of 64 streams died** with a quoted
`step error: DriverError(CUDA_ERROR_OUT_OF_MEMORY)` (0/64 well-formed, x3 runs; the worker
itself survived — it was never a hang or a panic). Two independent errors, both fixed:

- **The parked-session delta understated the live cost 1.49x**, and a roughly constant
  ~1.3 GiB draft-graph capture-arena transient is not proportional to session count at all,
  so no per-session headroom multiple could cover it. Admission now charges a flat
  `SPEC_SHRINK_RESERVE` (1.5 GiB) on **spec-capable models only** — the plain path is
  untolled and passed c=64 unaided.
- **Retires returned KV to the pinned async pool, invisible to driver `free`**, so the gate
  read a full card while gigabytes sat cached. The gate now reads `free + pool_cached`
  (deferrals 36 → 5, 59 sessions active sustained).
- **Step-OOM parks instead of killing**: a spec step that OOMs despite admission rebuilds
  its request and re-queues at the FRONT (`MEMRA_STEP_OOM_RETRIES`, default 3) — bounded,
  and only for a session that has emitted **nothing** and only on a quoted CUDA OOM, so a
  streamed prefix is never replayed to a client. Parking costs a re-prime: pure latency,
  never a correctness change.

Result: **64/64 well-formed, x3, peak 23.1 of 24.5 GB.** The c=8 no-regression control is
behaviorally identical (+0.49% agg tok/s, zero defer/park events). This is now a *gated*
property, not a claim: `tools/serve-stress-gate.sh` runs in `tools/local-ci.sh` and as the
`sstress` fast-gate arm, and it has teeth — `--teeth` forces the reserve to 16 MB and the
verdict inverts (11/64), so a gate observed only passing proves nothing.
Receipts: `research/admit-oom-20260806/`, `research/serving-density-20260806/VERDICT.md`.

### Config recommendation: send `max_tokens`

Admission sizes each session's KV ladder from the request's own bound. A request that
**omits `max_tokens`** falls back to the context ceiling, so at `MEMRA_CTX=32768` it
reserves ladder slack it will never use: measured **6.3% of a 96GB card at c=16 and 12.6%
at c=32** stranded on the 9B — more than sealed-prefix duplication costs at the same
shape. Right-sized requests (explicit `max_tokens`) strand ~0%. Set an explicit
`max_tokens` in serve configs and client defaults, and keep `MEMRA_CTX` at the workload
rather than the maximum. Receipt: `research/serving-density-20260806/VERDICT.md` (Q1).

## The exact-16 decode chunk tier

The batched tick decodes sessions in per-model chunks. Default width is **16 on models
where every matmul has a bit-exact 16-batch kernel class** (`decode_batch_exact16_ok`:
the b16 batched-mmvq family — Q8_0 qualifies only through its `_rp` mirror twin), **8
otherwise**; `MEMRA_DECODE_BATCH_CAP` stays the explicit measurement door. Qualifying
steps scope out every m>=16 GEMM/MMQ arm, so chunk-16 output is bit-identical to
isolated decode (gate2 bit-checked at steps 32 and 160). B=32 has no exact kernel class
— chunk policy stays <=16. On the H100 fleet model (9B Q8_0, mirror on by default) the
tier engages automatically on the next deploy; the H100 numbers above are chunk-8-era
and the chunk-16 fleet effect is pending on-box re-validation.

**Capacity envelope (24GB):** the mirror costs ~model-size VRAM, so c=32 sessions at the
default `MEMRA_CTX=8192` exceed VRAM (captured `CUDA_ERROR_OUT_OF_MEMORY` in the
pre-admission-wait receipts; ~27 sessions fit — since the VRAM-aware admission wait the
overflow queues instead of erroring). Set `MEMRA_CTX` to the workload — 2048 clears the
same cell (machine-specific config per the flags doctrine).

## Pipeline-parallel (PP-2) serving

For a model that fits only across two cards. Receipts:
[`research/pp2-batch-20260806/`](../research/pp2-batch-20260806/) (batched decode),
[`research/pp2-spec-20260806/`](../research/pp2-spec-20260806/) (the spec verdict),
[`research/pp2-hardening-20260806/`](../research/pp2-hardening-20260806/) (the fail-closed
guard). Rig for all three: 2x RTX PRO 6000 Blackwell Server Edition 96 GB, sm_120a, CUDA
13.2, SPOT box — **rented**, not owned. Flag reference: [FLAGS.md](FLAGS.md) `MEMRA_PP_*`.

The serving config, minimally:

```bash
MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 \
MEMRA_SERVE_SPEC=0 \
MEMRA_MODELS="big=/path/to/model.gguf" memra-server
```

`MEMRA_SERVE_SPEC=0` is **load-bearing, not tidiness** (see below). The server logs
`[pp] cross-device transport: stage0=dev0 stage1=dev1` when the split is live — a config that
silently did not split is the failure mode that banner exists to rule out.

**Exactness: the split adds zero deviation.** `decode-batch-gate --mode pp` records a
reference with the door OFF over the same loaded weights, replays the same token sequence
through the split, and compares every f32 logit of every row of every step bit by bit.
**0 differing bits** on all seven configs — `dev01`, `dev10` (reversed placement),
`singledev` (seam only, one card), `split5` (uneven cut), N=4 (`devices 0,0,1,1`), q27 (64
layers), and `wide` (B=12/16 under the `MEMRA_DECODE_BATCH_CAP=16` door). The B=1 fast path
is its own gate arm (arm 4) against the eager split, since it carries the accepted m=1 fusion
FP gap vs the batched body by design: **3,973,120 f32 logits bit-identical, 0 differing
bits**, across the same six configs.

**Cost: the boundary transfer does not bite at m>1.** q9, 64 steps, 512-token prompts,
greedy, N=5 rep-major interleaved in one lock hold on one binary (medians; cross-run
comparison would be clock-drift invalid):

| arm | B=1 | B=4 | B=8 |
|---|---|---|---|
| door shut, single device | 208.4 | 489.3 | 654.0 |
| split dev01 (**the serving config**) | **204.7** | 487.0 | 646.9 |
| ratio | 0.982x | **0.995x** | **0.989x** |

So batched PP-2 costs **0.5–1.5%** at B=4/8/16, and of that, transport is 0.986–0.997x of the
seam — almost all of the small loss is the seam, not PCIe. Both placement orders agree within
0.3%. Aggregate scaling survives the split: B=8 reaches 3.65x B=1's aggregate.

The B=1 row has history worth keeping: opening the pp door originally dropped every solo
session off the m=1 fusion chain (the `b1_fast` guard included `pp_cuts().is_none()`), a
permanent **−14.9%** tax on exactly the request shape an interactive 2-card box serves. Fixed
by giving the split its own B=1 path — each stage runs its layer range through
`decode_layers_eager`. `MEMRA_SERVE_B1FAST=0` is the rollback control and still measures the
old 177 (0.851x).

**Why `MEMRA_SERVE_SPEC=0`.** Speculative serving over PP-2 is *correct* — the verify trunk
takes its own stage split and the bit-identity battery is 7/7 ALL GREEN — and it is still
**not shippable for concurrent serving**:

- On the reversed placement it provokes a deterministic `CUDA_ERROR_ILLEGAL_ADDRESS` that is
  **sticky for the CUDA context**: once it fires, every later `new_session` inherits it. At
  c=4 that is **100% of requests lost** (0/48, wall 0.008 s), reproducible 3/3.
- On the other placement it is ~20x slow.
- The same placement with spec OFF is 96/96 clean and the fastest arm measured.

So PP-2 serves the **plain** path today. An artifact carrying an embedded MTP head self-specs
by default, which is why the flag must be explicit: without it every request funnels into the
verify trunk. `serve-smoke.sh` over the split (PP-2 dev01, spec off) returns **0 failed
checks** across `/models`, non-stream chat, SSE streaming, `/v1/completions`, greedy
determinism, 3 concurrent chats, and long generation — identical to the door-shut control,
i.e. the split adds nothing observable to the OpenAI surface.

**What refuses, deliberately.** The four decode paths that have no stage split
(`decode_step_batch`'s unsplit body, `decode_step_dc`, the graph capture wrapping dc, and
`decode_step_t*` spec verify) **fail closed** under an open pp door with a sharded
cross-device placement, behind one shared guard (`pp::refuse_unsplit_if_remote`). They were
not wrong, they were a silent perf cliff with a green battery: an unsplit trunk peer-reads
every remote stage's weights every step, measured **7.4 vs 208.9 tok/s at B=1 (28x)** and
**47.4 vs 657.0 at B=8 (13.9x)**. Exactness was never affected (peer reads return identical
bytes), which is exactly why a refusal rather than a warning was the right call.
`MEMRA_PP_ALLOW_UNSPLIT_BATCH=1` re-admits them as a measurement door only;
`MEMRA_PP_SHARD=0` is the non-measurement escape (weights all home — full speed, forfeits the
capacity PP-2 exists for).

## OpenAI tools surface (serve-tools lane, 2026-08-02)

`/v1/chat/completions` accepts `tools`, `tool_choice` (`"auto"`|`"none"`; `"required"` and
named-function forms 400 — the grammar engine isn't wired to tool selection yet),
assistant-history `tool_calls`,
`role:"tool"` result turns, and `reasoning_effort`/`reasoning`. The path is **template +
parsing only — zero engine changes**:

- Tool schemas render into the model chat template's own `<tools>` branch (the qwen3.5/3.6
  ChatML convention: schemas JSON in the system region, `<tool_call>`/`<function=…>` call
  format, `<tool_response>` result turns), byte-per-byte per the committed template dumps
  (`research/onboard-ornith-20260801/templates/`). Models whose template has no tools
  branch (hy3 dialect, gemma4, bare ChatML) reject `tools` with a 400 at admission.
- Emitted `<tool_call>` blocks are parsed from the generated stream into OpenAI-shape
  `tool_calls` (streaming deltas + non-stream `message.tool_calls`, deterministic ids,
  `finish_reason:"tool_calls"`); argument values coerce per the declared JSON-schema types.
  **Malformed policy:** a block that does not parse is surfaced verbatim as content — never
  an error, never dropped bytes; unterminated blocks flush raw at end of generation.
- **`reasoning_effort` — one surface, per-arch native thinking control** (owner directive
  2026-08-07: every supported model is a thinking model). The reasoning-capable-model
  convention: `low|medium|high` = thinking ON at that budget, `none|minimal` = thinking
  OFF, **absent = the model's own default** (never overridden — no silent behavior change
  for existing deployments). `reasoning: {enabled, effort}` (OpenRouter form) maps the same
  way; `{enabled: false}` is the explicit off, `{enabled: true}` thinking on at the
  template default budget. Unknown values 400. Per-model mapping (goldens rendered from
  each REAL shipped template: `research/step-sku-20260807/render-thinking-goldens.py`):

  | model class | native mechanism | absent (default) | none/minimal | low | medium | high |
  |---|---|---|---|---|---|---|
  | Qwen3.5/3.6, Ornith, AgentWorld, KAT (qwen ChatML class) | `enable_thinking` switch | thinking **ON** (open `<think>\n`, the template default) | closed `<think>\n\n</think>\n\n` | open `<think>` | open `<think>` | open `<think>` |
  | Gemma-4 family (12B/26B/31B/E4B) | `enable_thinking`, template default **false** | thinking **OFF** (closed `<\|channel>thought\n<channel\|>`) | closed channel | `<\|think\|>` system token + open turn | same | same |
  | Hy3 | template's own `reasoning_effort:` `no_think\|low\|high` | `no_think` (its jinja default) | `no_think` | `low`, open `<think:opensource>` | `low` (clamp — no medium level) | `high`, open think |
  | Step-3.7-Flash (`step35`) | `Reasoning: {level}` string in the system turn; `<think>` tail **unconditional** | no `Reasoning:` line (template default) | `Reasoning: low` (clamp — no off level) | `Reasoning: low` | `Reasoning: medium` | `Reasoning: high` |

  Level strings reach only templates that consume one (spawn-time `effort_levels` probe,
  keyed on the jinja's own `reasoning_effort is defined` input test — true for step35 and
  Hy3); binary-switch templates are driven by the on/off half alone, so prompts on models
  that never read a level cannot be perturbed by it. Serve-smoke receipts:
  `research/step-sku-20260807/raw/effort-smoke-*.log` (step35),
  `research/step-sku-20260807/raw/think-smoke-*.log` (qwen + gemma4 arms).
- **Isolation:** non-tools traffic bypasses the tools renderer AND the emission parser
  entirely (legacy render path, byte-identical streams); tools traffic is generation-
  identical for the identical rendered prompt (raw-completions bijection gate). `usage`
  now carries worker-truth `prompt_tokens` (rendered tools block included) +
  `completion_tokens` + `total_tokens` on stream and non-stream shapes, with the
  prompt-caching split (`prompt_tokens_details.cached_tokens`) — see "Prompt caching"
  below. Tools requests cache like any other: the prefix cache keys on the rendered
  prompt's token ids, so a repeated tools block is a cacheable prefix (no special-casing).

Receipts: `research/serve-tools-20260802/` (round-trip transcripts N=3 greedy on
Qwen3.6-35B + AgentWorld, streaming schema checker, malformed-policy transcript,
tok-check usage crosscheck, cross-binary c1 refs + c1-vs-c16) and
`research/integrate-cache-20260802/` (tools x cache intersection gate).

## OpenAI compatibility contract (serve-compat lane, 2026-08-03)

The five gap-scan listing-blockers (`research/gap-scan-20260802/REPORT.md`), fixed and
gated by the official `openai` Python SDK against a live server
(`research/serve-compat-20260802/`):

- **Envelope:** every OpenAI-shape completion and stream chunk carries `id`
  (`chatcmpl-…`/`cmpl-…`), `created`, and `system_fingerprint` (`memra-<git sha>`, baked
  at build); the id echoes as the `x-request-id` response header. The first stream delta
  carries `role:"assistant"`. Error bodies are the OpenAI object —
  `{"error": {"message","type","param","code"}}` — and mid-stream worker errors arrive as
  a final `data:` error chunk + `[DONE]`, never a named SSE event. SSE keep-alive comments
  flow every 5s (long-prompt prefill streams nothing before first token; OpenRouter cancels
  silent streams).

  **Precondition — which surface you are talking to.** Everything in this section describes the
  **OpenAI-shape** surface, and the stream terminator + the mid-stream error shape are gated on
  `chat || openai_compat()` (main.rs:1966, 2007). `openai_compat()` is true when
  `MEMRA_COMPAT=openai`, or when `MEMRA_COMPAT` is unset **and `MEMRA_API_KEY` is set** — the pi
  setup. On a **native-default** server (no `MEMRA_COMPAT`, no `MEMRA_API_KEY`) a streaming
  `/v1/completions` does the opposite of the sentence above: it emits a named `event: error` and a
  named `event: done`, with **no `data: [DONE]`**. That is deliberate, not a bug — native clients
  are memra's own tools, which do parse named events, and the validation harnesses rely on it.
  `/v1/chat/completions` is always OpenAI-shape (`chat` is true regardless). The shipped unit sets
  `MEMRA_COMPAT=openai` (`deploy/systemd/memra-server.service:92`), so a deployed server matches
  this section — but if you are testing a bare `memra-server` and your SDK reads a silent hang,
  this is why.
- **Reasoning separation:** on think-open prompts, `<think>` text routes to
  `message.reasoning` / `delta.reasoning` (+ `reasoning_details`, the OpenRouter
  dialect); `content` is post-think only. `include_reasoning:false` (or
  `reasoning: {exclude: true}`) drops the separated text. Non-think models keep
  byte-identical no-parser streams. **Gemma-4 dialect** (lane/gemma4-serve-gaps,
  2026-08-07): `<|channel>thought\n…\n<channel|>` blocks route to `reasoning` the same
  way — tags, the channel label and the bracketing newlines are syntax — and the splitter
  runs on *every* gemma4 chat request (channels can open mid-stream even under the
  closed-channel default). Turn-end control tokens (`<turn|>`, `<end_of_turn>`,
  `<|im_end|>`) stop generation (`eog_ids()` union) and never reach the client as text.
- **`max_tokens` omitted** ⇒ context-bounded budget (session ctx − prompt, capped at the
  model's trained context) — the OpenAI default-when-omitted semantics, not a silent
  128-token truncation. Explicit `max_tokens`/`max_completion_tokens` honored exactly.
- **`temperature` omitted ⇒ 1.0; `seed` omitted ⇒ fresh per request** (dogfood F4,
  2026-08-04). Both were `#[serde(default)]`, which is `0.0` for `f32` and `0` for `u64` —
  and both of those zeros are *meaningful values*, not "unset": temperature 0 is greedy
  argmax and seed 0 is a valid fixed stream. An omitting client (the OpenAI SDK's
  documented leave-it-out path, and this repo's own agentic driver) therefore got
  deterministic decoding pinned twice over: same context in, same token out, identical
  tool-call cycles forever. Now `temperature` omitted is OpenAI's documented 1.0 and an
  omitted `seed` draws fresh entropy per request. **Explicit values are honored exactly,
  including `temperature: 0` (greedy) and `seed: 0`** — every determinism gate in `tools/`
  and `research/` sends both explicitly, so all of them keep their behavior. Supply a
  `seed` whenever you want reproducibility; omit it to get variation.
  Corollary worth knowing: an omitted-`temperature` request is *pure* temperature-1.0
  sampling (`top_p` 1.0, `top_k`/`min_p` disabled, penalties off), which is exactly the
  regime that keeps the in-graph sampled draft chain — so the OpenAI default lands on the
  fast sampled-spec path, not a slow fallback.
- **Disconnect abort:** a hung-up client's session retires at the next tick (all serve
  paths: batched, graph, spec, legacy) and is billed to the abort point (the `[abort]`
  log line records prompt/cached/generated); queued requests from dead clients never
  reach the GPU.
- **Parameter breadth + honesty:** `frequency_penalty`/`presence_penalty`/
  `repetition_penalty` plumb to the sampler (whole-history window; greedy+penalized
  keeps the host-sampled path). `response_format` `json_object`/`json_schema` is REAL
  constrained decoding (see the section below). Semantic params we can't honor 400 with
  the param named (`logit_bias`, `logprobs`/`top_logprobs`, `n != 1`, `best_of != 1`,
  unknown `response_format` types); cosmetic fields (`user`, `stream_options`) are
  accepted and ignored. Streams exclude stop-sequence text exactly like non-stream
  responses (holdback buffer).

## Gateway listing surface

OpenRouter's current Provider Monitor schema is version **2.4**, but it is not the old
flat/catalog shape: new integrations declare typed `input_modalities` and
`output_modalities`, with pricing and capacity nested on the modality they belong to.
The older flat provider document remains supported only for existing integrations.

memra keeps three views separate because the current provider schema rejects unknown
fields (`additionalProperties: false`):

- **`GET /models`** keeps the historical OpenAI-style body byte-for-byte:
  `{"object":"list","data":[{"id":"<alias>","object":"model"}]}`. Existing pill/Hermes
  consumers stay on this default.
- **`GET /models?schema=openrouter`** is the OpenRouter Provider Monitor 2.4 document
  for a new provider integration. Use this full URL for the OpenRouter application.
- **`GET /v1/models`** keeps the existing catalog-style enrichment
  (`context_length`, `architecture`, `pricing`, `top_provider`) for current clients.
  It is not the strict Provider Monitor document.

The Provider Monitor view derives what the process knows:

- `id` and `name` are the exact `MEMRA_MODELS` alias.
- Text `max_context_length` and `tokenizer` come from the loaded model.
- Streaming and supported generation parameters come from the real HTTP surface;
  `tools` and `reasoning` appear only when the loaded template exposes them.

Everything else is operator-declared in a TOML file named by
`MEMRA_MODEL_METADATA`. The file is optional for local serving. If configured, it is
parsed before the GPU worker starts; unknown fields, invalid price strings, invalid
quantization names, zero limits, or aliases absent from `MEMRA_MODELS` are fatal.

```toml
# /etc/memra/models.toml
[models."provider/model-id"]
description = "Qwen3.6 27B served by memra."
quantization = "nvfp4"
max_prompt_length = 245760
max_output_length = 16384
is_ready = true
# Set the real deployment location before submitting the application:
# datacenters = [{ country_code = "US", region = "actual-region" }]

[models."provider/model-id".pricing]
# Per-token USD strings, not per-million-token numbers.
prompt = "0.000000234"        # $0.234 / 1M input tokens
cached_prompt = "0.0000000585" # 25% cache-read price
completion = "0.000001872"    # $1.872 / 1M output tokens

[models."provider/model-id".capacity]
# Optional honest declarations; omit values that are not measured.
prompt_tpm = 1000000
completion_tpm = 500000
request_rpm = 1000
concurrency = 16
```

```bash
MEMRA_MODELS="provider/model-id=/path/to/model.gguf" \
MEMRA_MODEL_METADATA=/etc/memra/models.toml \
memra-server

curl 'http://127.0.0.1:8080/models?schema=openrouter'
```

Supported metadata fields are `hugging_face_id`, `created`, `quantization`,
`description`, `max_prompt_length`, `max_output_length`, `is_ready`, `is_free`,
`discount_to_user`, `openrouter_slug`, `datacenters`, `zdr`, and `hipaa`.
`pricing` accepts `prompt`, `cached_prompt`, `cache_write`, `completion`,
`internal_reasoning`, and `request`; `capacity` accepts `prompt_tpm`,
`cached_prompt_tpm`, `completion_tpm`, `request_rpm`, and `concurrency`.
Prices and capacities are omitted when undeclared. memra never turns an absent price
into `"0"`; use an explicit zero only for a genuinely free SKU.

The remaining gateway controls are battery-gated (`research/serve-tail-20260804/`):

- **Rate-limit headers:** `X-RateLimit-Limit` / `-Remaining` / `-Reset` (emitted
  lowercase on the wire, as HTTP/2 requires; capitalized here by convention — a client
  parsing headers into a case-sensitive dict must key on `x-ratelimit-*`) on both
  completion routes with concurrency-slot semantics — a per-lane atomic gauge whose
  RAII slot rides the SSE stream to completion, so `Remaining` is truthful for the
  whole life of a stream. Sheds carry `429 + Retry-After`; `MEMRA_RL_RESET_S` is the
  no-signal fallback for `Reset` (with traffic, Reset = mean tokens/request x p50 step
  latency).
- **Graceful drain:** SIGTERM flips `/health` to `status:"draining"` (still **200** — see
  Health below) and `/readyz` to **503**, new completion requests get `503 + Retry-After`,
  in-flight requests — streams included — run to `[DONE]` within the `MEMRA_DRAIN_S`
  deadline (default 30s), then the process exits 0. Live receipt: a 1024-token stream
  completed mid-drain.

## Health, readiness, and fault handling (serve-hardening lane, 2026-08-06)

Receipts: `research/serve-hardening-20260806/`. Example unit:
`deploy/systemd/memra-server.service`.

**The full route table**, since the sections above only introduce routes as they become
relevant (bind address `MEMRA_ADDR`, default `127.0.0.1:8080`):

| route | notes |
|---|---|
| `GET /health`, `GET /livez` | the same handler — inference liveness (below) |
| `GET /readyz` | routability (below) |
| `GET /v1/models` | the existing catalog-style enriched listing |
| `GET /models` | the byte-compatible OpenAI-style listing used by existing clients and smoke gates |
| `GET /models?schema=openrouter` | strict OpenRouter Provider Monitor schema 2.4; operator metadata comes from `MEMRA_MODEL_METADATA` |
| `POST /v1/completions` | raw-prompt completions. **Streaming shape depends on `MEMRA_COMPAT`** — see the compatibility precondition above |
| `POST /v1/chat/completions` | always OpenAI-shape |
| `GET /metrics` | counters + the cache-hit metering surface (below) + the `spec` acceptance block |
| `GET /yield/metrics` | the dark-lane yield view |

**`/health` == `/livez` — inference liveness, not process liveness.** The GPU worker is
ONE `std::thread` owning the CUDA context. `/health` used to answer `{"status":"ok"}` off
the axum task, so a worker panic or a wedged card left a permanently green health check in
front of a box answering nothing. It now derives from a heartbeat the scheduler loop stamps
every iteration, plus a phase:

| worker phase | `/health` | why |
|---|---|---|
| `loading` | 503 | weights are not resident; the process answers nothing yet. On a FIRST load the port is not bound yet (bind follows the load), so a probe sees connection-refused — the same verdict for k8s and `serve-fleet.sh`. This state is reached over HTTP during a **respawn**, which is the case that matters |
| `idle` | 200 at any beat age | the worker blocks in `rx.recv()` — an idle server legitimately stamps nothing for hours, and a naive age check would call every quiet server dead |
| `busy` | 200 while the beat advances, 503 past `MEMRA_HEALTH_STALL_S` (120s) | work in flight must make progress; the bound covers a max-context prefill tick (see FLAGS for the derivation) |
| `dead` / fault latched | 503 immediately | worker panic or fatal Xid — a latch, not a timeout, so the flip is instant |

The response body is `{status, models, worker:{phase, beat_age_ms, tick_max_ms,
stall_threshold_ms, generation, xid_warnings}}`, plus a top-level `detail` on a red (which is
where a quoted panic payload lands). `status` is `ok` / `draining` / `unhealthy` on
`/health`-`/livez` and `ready` / `not_ready` on `/readyz`. So a red is self-explaining and
`tick_max_ms` — the longest scheduler iteration this process actually observed — is the live
receipt for revisiting the threshold.

Every red probe 503 follows the same retry-header contract as request-path overloads.
Worker-related `/health` and `/readyz` failures use the worker supervisor's 2-second respawn
backoff (`Retry-After: 2` + `retry-after-ms: 2000`). A draining `/readyz` uses
`MEMRA_DRAIN_S`, clamped to the SDK-honored 1..=60 second window, with the matching millisecond
twin. Retryable probe responses never carry the contradictory `x-should-retry: false`.

**`/readyz` — should traffic be routed here?** Ready = model loaded AND worker alive AND
not draining. Unready is NOT a restart request: draining and loading are healthy states
that simply must not be routed to, which is exactly why liveness and readiness are
separate endpoints (k8s deprecated `/healthz` at v1.16 for this split). Queue pressure
deliberately does not flip readiness — the interactive lane queues FIFO and never sheds,
so a deep queue is work in progress; capacity backpressure belongs on the request path.
`tools/serve-proxy.py` probes `/readyz` for rotation; `tools/serve-fleet.sh` probes
`/health` for its restart decision. vLLM has no readiness endpoint (503 only on
`EngineDeadError`) and TGI a single `/health`.

**Worker panic → supervised.** The worker thread runs inside `catch_unwind`: a panic marks
health dead with the quoted panic payload, then ONE respawn is attempted after a **`2 x attempt`
second** backoff — 2 s at the default max of 1 (`MEMRA_WORKER_RESPAWN`; the sleep exists so a
panic from a transient device condition gives the driver time to settle instead of re-hitting it
immediately) — and failing that the process exits **70** so the supervisor restarts it whole.
**Two distinct paths reach exit 70**, and an operator reading `systemctl status` should be able to
tell them apart: the respawn budget running out (`STATUS=worker unrecoverable; exiting`), and a
respawn whose **weight reload itself failed** (`STATUS=respawn load failed; exiting`) — the second
is not a panic, and it exits rather than looping because a load failure will not fix itself.
Exit 70 is sysexits' `EX_SOFTWARE`, chosen so it reads distinctly from the startup FATAL paths,
which exit 1 ("the engine died" vs "bad config"). One attempt, deliberately — CUDA errors are sticky per process, so a
respawn loop against a poisoned context produces a box that looks alive and serves nothing.
Proved on a real CUDA worker, not only in tests (`MEMRA_PANIC_AFTER` fault injection,
`research/serve-hardening-20260806/logs/worker-death.txt`): panic → 503 on all three routes
with the quoted payload in `detail` within ~200 ms → weights reloaded → `generation` 0 → 1 →
the respawned worker served a real completion; with `MEMRA_WORKER_RESPAWN=0` the process
exited 70 and the port went refused. A request that arrived during the dead window was
**served by the respawn** — the supervisor owns the command channel across restarts, so
queued work survives a worker death.

**GPU faults (`MEMRA_GPU_WATCH`).** A watcher thread tails Xid lines (`/dev/kmsg`, falling
back to `journalctl -k -f`) and latches unhealthy on the fatal classes
(48/64/79/94/95/119/120), counting the rest as warnings. It also probes `nvidia-smi` for
uncorrectable ECC and row-remap failures every `MEMRA_GPU_WATCH_S` seconds (default 60 — the
audit's published detection commitment is "checks every 60 s", so treat it as a stated fact about
the instrumentation rather than a free knob). The design constraint: Blackwell's worst wedge
(Xid 119/120, GSP RPC timeout) emits nothing to the process **and hangs the query tools**,
so the probe runs as a killed-on-deadline child and its own timeout
(`MEMRA_GPU_PROBE_TIMEOUT_S`) is the alarm. Health reads only atomics, so a hung
`nvidia-smi` can never block a health answer. A GPU fault survives a worker respawn: a new
thread on a wedged card is not recovery.

**The supervision contract (`deploy/systemd/memra-server.service`) has three couplings you can
break silently.** The unit is an example to copy, but these are not stylistic choices — each is
sized against a server-side default, and changing one side alone produces a unit that looks
correct and misbehaves only during a failure:

| directive | value | the coupling |
|---|---|---|
| `WatchdogSec` | 180 | MUST exceed `MEMRA_HEALTH_STALL_S` (default 120). The heartbeat that feeds `/health` also feeds systemd, so a watchdog under the legitimate-stall bound restarts a *healthy* server mid-prefill. Raise both together if you raise `MEMRA_MAX_SESSIONS` or the context |
| `TimeoutStopSec` | 60 | MUST exceed `MEMRA_DRAIN_S` (default 30), or systemd SIGKILLs a drain that is finishing streams correctly. The server also sends `EXTEND_TIMEOUT_USEC`; the static floor covers a build that does not |
| `TimeoutStartSec` | 600 | MUST exceed the slowest cold load (~120 s measured for a 27B NVFP4 from page cache; cold NVMe on a large bank is slower). Startup silence is a load, not a hang |
| `StartLimitIntervalSec` / `StartLimitBurst` | 3600 / 4 | systemd's defaults (10 s / 5) are sized for millisecond daemons and **cannot trip at all** here — 5 starts do not fit in 10 s when each start takes ~120 s, so a crash loop restarts forever instead of failing the unit for a human. 4 starts per hour ≈ "if it cannot survive four full loads, page someone" |
| `RestartSec` / `RestartSteps` / `RestartMaxDelaySec` | 10 / 4 / 160 | a card that just threw an Xid needs the driver to settle; a tight loop makes recovery less likely. The ramp needs systemd ≥ 254 — on older systemd delete the last two lines and keep the flat 10 s |
| `OOMPolicy` | `kill` | the default `stop` reaps only the offending process, and the kernel OOM killer can take out ONE thread — classically the worker — leaving a process that accepts connections and can never serve them, which is the exact invisible death this lane removes. **Host memory only**: CUDA OOM is the 503 above, never a process kill |

Two more worth knowing before you deploy. `Type=notify` + `NotifyAccess=main` means `READY=1`
fires after the models load **and** the socket binds — `systemctl start` returning is a real
readiness signal, which is why `TimeoutStartSec` must be generous. And Xid visibility can be
silently absent: `kernel.dmesg_restrict=1` makes `/dev/kmsg` root-only, so an unprivileged unit
sees Xids only through `journalctl`; grant `AmbientCapabilities=CAP_SYSLOG` +
`CapabilityBoundingSet=CAP_SYSLOG` or accept the fallback to the probe-hang and ECC/remap
detectors, which need no kernel log. The watcher logs which source it got, so this is never a
silent downgrade. The unit is deliberately **not** `ProtectSystem=strict` — model paths,
`/dev/nvidia*`, and the CUDA cache need real filesystem access, and a wrong sandbox fails at
load time looking like a model bug.

**Error taxonomy.** Every engine failure used to be `400 invalid_request_error` — which no
OpenAI SDK retries, and which a router cannot distinguish from a malformed request. The
class now comes from the producer:

| condition | status | `type` | `code` | retry headers |
|---|---|---|---|---|
| malformed field, bad template, bad `response_format` | 400 | `invalid_request_error` | — | `x-should-retry: false` |
| prompt ≥ context cap | 400 | `invalid_request_error` | `context_length_exceeded` | `x-should-retry: false` |
| unknown model id | 400 | `invalid_request_error` | `model_not_found` | `x-should-retry: false` |
| dark-lane QoS shed (`x-lane` judge/harvest over budget) | 429 | `rate_limit_error` | `rate_limit_exceeded` | `Retry-After: 2` + `retry-after-ms: 2000` |
| out of VRAM / step-OOM past its park budget / worker restarting | 503 | `server_error` | `overloaded` | `Retry-After: 5` + `retry-after-ms: 5000` |
| step, prefill, graph or constraint fault | 500 | `server_error` | `engine_error` | none (not time-bounded) |
| new request arriving during a drain | 503 | `server_error` | `draining` | `Retry-After: MEMRA_DRAIN_S` (≤60) + matching `retry-after-ms` |
| unknown `x-lane` value | 400 | `invalid_request_error` | `invalid_lane` | `x-should-retry: false` |
| batch-class api key requesting `x-lane: interactive` | 403 | `authentication_error` | — | `x-should-retry: false` |
| bad / disabled api key | 401 / 403 | `authentication_error` | — | `x-should-retry: false` |
| worker channel dropped (`cmd_tx.send` fails) | 503 | `server_error` | `overloaded` | `Retry-After: 2` + `retry-after-ms: 2000` |

Unknown model is a deliberate **400, not 404**: OpenRouter's uptime math counts 404s
against the provider and excludes 400s, and the `code` is what clients branch on either
way. `Retry-After` is always integer seconds ≤ 60 (RFC 9110 delay-seconds; litellm honors
only `0 < v ≤ 60`, openai-python abandons retry past 120s) with a matching
`retry-after-ms`, which openai-python reads first. A **mid-stream** failure — after the 200
is committed and no status code is left to change — emits the same error object as a
`data:` chunk and closes the connection.

The channel-drop path uses the same 2-second constant as the supervisor's first respawn delay, so
the HTTP hint cannot drift from the recovery ladder. The control-plane probe 503s described above
also pass through the shared contract builder; there are no bare 503 producers left in
`crates/memra-server/src`.

## Safetensors checkpoint serving (serve-st + fp8-ship lanes, 2026-08-04)

`MEMRA_MODELS` accepts safetensors checkpoint directories (`config.json` +
`model.safetensors[.index.json]`) and repack dirs alongside GGUF paths — validated at
parse time (a bogus dir fails naming the missing file). Chat templates come from the
checkpoint's own tokenizer config (`from_hf_dir`); template-less dirs 400 with a pointer
to `/v1/completions`. Official Qwen FP8 block-128 checkpoints load bit-exact (GPU
dequant, load wall 843.9 → 291.6 s = **2.89x faster load**) and **spec decode runs out of
the box on the checkpoint's embedded MTP head** — **128.06 tok/s** from the checkpoint's
own `mtp.safetensors` (**2.61x** the same-run plain 48.99), 136.75 with an own-trim
drafter, on rig **`rig2x5090-serve`** (rented 2x RTX 5090; there is no official-FP8 cell
on any RTX PRO 6000 board — do not merge the two). The win is **load time, not decode
throughput**: the e4m3-resident arm is flat by construction (weights dequantize onto the
Q8_0 arm), and spec **triples TTFT** on this arm (0.170 → 0.466 s).
Receipts: `research/fp8ship-20260804/official/`.

The ST-spec exactness scare (#68) was root-caused to a serve-side bug that was never
ST-specific: the per-session persistent draft graph replayed with dangling pool
addresses (capture transients not retained + the fa-partials pool freeing grown-past
buffers the capture baked) — reproducible on GGUF session bursts at n>=600 too. Fixed
via capture-retain keepers on `DraftGraphCtx` + retire-on-grow for the fa partials pool;
the quarantine is lifted and dir checkpoints are spec-eligible by default
(`MEMRA_SERVE_SPEC=0` is the rollback door). Gate: `tools/serve-st-gate.sh` — item 3 pins
the CLI ST-dir branch and the server to identical greedy token streams on a 64-token
window, and item 4 pins the DEFAULT (spec-on) server against the **tokenwise serve
oracle** (`MEMRA_SERVE_SPEC=0 MEMRA_SERVE_BATCH=0` — same worker, plain decode) at a
400-token window, prefix-tolerant for burst overshoot. Note what item 4 is *not*: the
comparator is deliberately the tokenwise **serve** arm, not the run-gen CLI, because both
the batched-plain path and the CLI carry their own accepted near-tie FP classes at long
windows (see [first-token cross-config
drift](#first-token-cross-config-drift-batched-prime--stated-honestly) below). Do not
restate this gate as "token-identical to the CLI oracle".

## Constrained decoding (`response_format`) — lanes constrained + constrained-full, 2026-08-03

`/v1/chat/completions` honors `response_format` `{"type":"json_object"}` and
`{"type":"json_schema","json_schema":{...,"schema":{...}}}` as REAL constrained decoding:
the schema compiles to an [llguidance](https://github.com/guidance-ai/llguidance) grammar,
and each step's packed token bitset uploads to a stable per-session device buffer (~31KB
H2D) where `mask_logits_f32` bans disallowed tokens on device — BEFORE the same
device-sample / lean-logits / CUDA-graph / speculative paths unconstrained sessions ride.
No path is lost to being constrained.

- **Cost:** plain constrained-greedy = **99.4% of unconstrained** (123.7 vs 124.4 tok/s,
  q9 N=3 same-session, local RTX 5090, Qwen3.5-9B NVFP4, 256-token greedy); per-step
  grammar compute 0.006–0.007 ms. **That 99.4% is the plain lane only — the speculative
  lane pays far more: 153.4 vs 194.4 = 79%.** Never quote 99.4% for the spec path. The
  remaining constrained-vs-unconstrained gap is draft acceptance under a tight grammar,
  not mask overhead.
- **Draft-side masking (lane/draft-mask, 2026-08-04):** the drafter is masked too. A
  constrained spec session clones the session's grammar matcher once per spec round
  (0.002 ms), advances the clone with each proposed token, and bans the illegal ids in the
  draft head's own logits — in-graph on the captured draft chain, permuted through `d2t`
  for trimmed draft heads. Proposals are legal by construction, so the verify-side
  truncation backstop (which stays, as the correctness backstop) stops firing:
  `gram_cuts` went 3/12, 3/15, 1/10, 28/30, 18/25 -> **0/N on every cell measured**.
  Bounded tight schema: acceptance 0.561 -> 0.651, 216.6 -> 227.5 tok/s (+5.0%, N=3 warm).
  Cells whose drafter already proposed legal tokens (json_object, loose prose) move inside
  noise; unconstrained traffic is inert. Rollback seam `MEMRA_DRAFT_MASK=0`. Receipts
  `research/draft-mask-20260804/`.
- **Exactness:** device-mask greedy is byte-identical to the host -inf oracle
  (`MEMRA_CONSTRAIN_HOST=1`), spec-constrained is byte-identical to plain-constrained,
  graphed is byte-identical to eager, draft-masking ON is byte-identical to OFF (greedy and
  seeded-sampled, 7 cells), and unconstrained requests are byte-identical to the pre-lane
  binary (the isolation contract). Kernel-check pins `mask_logits_col` bit-identity.
  One measured exception, documented because it is NOT a masking property: an unbounded
  schema that lets the model degenerate into arbitrary whitespace against a token cap has a
  draft-chain-SHAPE-dependent tail (verify batch shape T changes FP summation order, which
  flips argmax at the near-ties in that tail). The pre-lane binary shows the same
  divergence across `MEMRA_SPEC_K=3/2/1` on that cell with no draft-mask code present;
  with shape held fixed the arms are byte-identical. Bound the schema and it goes away.
- **Think interaction:** constrained requests force the template's no-think switch (a
  grammar masking from token 0 can never close an open `<think>` tail); a think-tail
  template without an `enable_thinking` switch is a loud 400.
- Unknown `response_format` types remain loud 400s. `/v1/completions` (non-chat) carries
  no `response_format`.

Receipts: `research/constrained-20260803/` (v1) + `research/constrained-full-20260803/`
(full battery: every path, cross-path identity, three-way perf).

## Prompt caching (cross-request prefix cache) — 2026-08-02

Two caching tiers serve prompt tokens without recomputing them:

1. **Continuation pool** (pre-existing, `MEMRA_KV_REUSE`): a retired session parks its whole
   (prompt + generation) state; a new prompt that EXACTLY EXTENDS it resumes. Single-use,
   exact-extension only — a new session that merely shares a system prompt always missed.
2. **Cross-request prefix cache** (`MEMRA_PREFIX_CACHE_MB`, default 256MB, 0 = off): compact
   device snapshots of primed state at token boundaries, keyed by the exact token-id prefix,
   per model, LRU under the byte budget. Entries are REUSABLE — a hit deep-copies the entry
   into the new session's cache, so one marketplace system prompt serves any number of
   sessions. Learning sequence for a shared-prefix pattern: request 1 seeds its full prompt,
   request 2 split-primes at the longest-common-prefix and inserts the boundary entry,
   request 3+ hit. Hybrid models are safe by construction: GDN conv/ssm state cannot be
   truncated to a shorter prefix, so the state is snapshotted AT the boundary while a fresh
   session primes — never rolled back.

**Exactness contract:** an entry stores the KV/recurrent bytes from WHATEVER prime config ran
(single, chunked, or concat batch-prime); decode from those bytes is deterministic, so a
cached hit is bit-identical to the run that computed the prefix — gated 16/16 partial-prefix
+ 16/16 full-prefix cached-vs-fresh greedy identity across depths
(`research/prompt-cache-20260802/gate-exact.jsonl`). Comparing a cached-hit stream against a
DIFFERENT prime config's fresh stream inherits the batched-prime near-tie first-token law
("First-token cross-config drift" below) — same documented class, reported not gated.

**Policy:** spec sessions bypass the prefix cache entirely (SpecSession owns trunk + draft
caches; a trunk-only prefix restore would leave draft state unprimed — the spec tier keeps
its own continuation pool). Legacy round-robin mode (`MEMRA_SERVE_BATCH=0`) also bypasses.
Sessions always win over the cache: a failed session-cache allocation evicts every entry and
retries before erroring.

**Per-tenant isolation (`cache_salt`) — PC-ISO, 2026-08-02:** every cross-request reuse
tier (prefix cache, continuation pool, spec pool) keys on (model, cache namespace), not
model alone. The namespace comes from the optional `cache_salt` string field on
`/v1/completions` and `/v1/chat/completions` (the vLLM `cache_salt` design, OpenAI-
compatible extension): requests only share cached prefixes with requests carrying the
SAME salt, in either direction, so `usage.prompt_tokens_details.cached_tokens` can only
ever reflect the caller's own namespace's history — the CacheProbe/PROMPTPEEK cross-tenant
hit-oracle mitigation (`research/cache-tools-20260802/REPORT.md` §1.4/§4). No salt = the
default `""` namespace: single-tenant deployments behave exactly as before (no new env
knob — the namespace is a request field, not a flag). The LRU byte budget stays GLOBAL
across namespaces (VRAM is one resource; only visibility is namespaced). A gateway
multiplexing many end-users through one API key — the marketplace listing shape — MUST set
a per-end-user/session salt. Gates: `research/pc-iso-20260802/` (same-salt hit, cross-salt
miss both directions, default-namespace blindness; the integrate-cache intersection gate
re-run unmodified as the no-salt regression).

**Accounting:** every response shape carries OpenAI-schema usage with the worker-truth split —
`usage.prompt_tokens`, `completion_tokens`, `total_tokens`, and
`prompt_tokens_details.cached_tokens` (tokens resumed from ANY cache tier: continuation pool,
spec resume, or prefix cache — the field name providers report cached reads under: OpenAI,
OpenRouter, and Grok chat all use `prompt_tokens_details.cached_tokens`). Cached
prefill costs ~0 to serve and bills at 25% of input on the OpenRouter hy3 endpoints — the
margin lever (`research/or-provider-20260802/REPORT.md`).

## Cache-hit metering (lane/cache-metering, 2026-08-07)

The aggregate receipt surface for the caching economics — the first-listed-week hit-rate
number is one query away. `GET /metrics` carries (cumulative since process start,
worker-truth, published every 32nd tick AND on every request retire so a post-workload
scrape is never stale):

| field | meaning |
|---|---|
| `prompt_tokens_in` / `cached_tokens_in` | every prompt token admitted / the subset served from any cache tier |
| `computed_tokens_in` | `prompt - cached` — the denominator of the revenue multiplier |
| `cache_hit_token_ratio` | `cached / prompt`, token-weighted — THE hit-rate number |
| `prefix_cache_hits/misses/inserts/evictions` | prefix-cache probe outcomes + churn |
| `prefix_cache_hit_tokens` | token-weighted hit mass (sum of served entry lengths) |
| `prefix_cache_entries/bytes` | resident state |
| `lcp_histogram` | `{edges, counts}`: one sample per prefix-cache probe — served entry length on a hit, best LCP on a miss. Lower-edge buckets `[0,1,16,32,64,128,256,512,1024,2048,4096]`, last unbounded; `[64,512)` (buckets 4..=6) is the tick-seg segmentation window |
| `tenants` | per-tenant `{prompt_tokens_in, cached_tokens_in, cache_hit_token_ratio}` rows — absent until the first admit |

`tenants` composes with PC-ISO tenancy: rows key on the TENANT half of the namespace
(keyring deployments get one row per tenant across its end-user salts, `t:<tenant>`;
no-keyring deployments key on the raw `cache_salt`, `""` = the default namespace). Rows are
bounded (256): overflow traffic aggregates under `"(other)"`, so totals stay exact while a
salt-spraying client cannot grow the map. Spec-tier and non-batched requests never probe the
prefix cache and are absent from the histogram by construction (their cached tokens still
count in `cached_tokens_in` via the continuation/spec pools).

**The economics query** (`tools/cache_economics.py <metrics-url-or-json>`): turns a scrape
into the earning-model row — `revenue_multiplier = billed_prompt_tokens /
computed_prompt_tokens` at a chosen cached-token billing factor (`--cache-billing-factor`,
1.0 = cached bills full price, 0.25 = the OR cached-input tier), plus per-tenant multipliers
and the tick-seg window share. JSON row on stdout (ledger-appendable), summary on stderr.

**Fleet receipt accumulator** (`tools/fleet-meter.sh`): the pre-listing hit-rate receipt for
controlled replay traffic. A one-shot scrape of `http://127.0.0.1:8002/metrics` appends only
the UTC timestamp, prompt/cached/computed counters, hit ratio, LCP histogram, tenants, and a
`restart` marker to `research/fleet-meter/rig5090-fleet.jsonl`. An unchanged scrape is
idempotently skipped. A failed scrape logs `skip` and exits successfully; it never starts,
stops, or otherwise mutates the owner-critical server.

**Fleet replay driver** (`tools/fleet-replay.py`): run only in dev-idle windows against the
existing port-8002 deployment. Its low defaults are five minutes at 3 requests/minute,
89.5:1 prompt:completion, 12 carried synthetic sessions, four tenant-scoped `cache_salt`
values, and eight shared 1k-4k-token system-prompt/tool-schema templates; exponential
inter-arrival times and 2-4-turn session bursts exercise both prefix sharing and continuation.
Set `MEMRA_API_KEY` to the local deployment key and run
`tools/fleet-replay.py --duration 300`. Any meter interval driven by this tool is labeled
**replay-calibrated**: it is a controlled synthetic workload and must never be described as
organic traffic.

```bash
tools/fleet-meter.sh --once                         # cron/timer-safe snapshot
tools/fleet-meter.sh --loop --interval-minutes 30  # foreground accumulator
python3 tools/fleet-report.py                       # all UTC days
python3 tools/fleet-report.py --days 7              # rolling weekly view
```

The example `deploy/systemd/memra-fleet-meter.{service,timer}` runs the one-shot form every
30 minutes. Copy the units, override their `/opt/memra` path and service account for the
host, then enable the timer; do not point the meter at a public endpoint.

The report diffs cumulative counters and histograms. A counter regression (or an explicit
`restart=true`) starts a new segment whose current values count from zero, so restarts never
produce negative traffic. The first snapshot intentionally counts the server's existing
cumulative receipt. Later intervals are attributed to the UTC day of their ending snapshot,
which bounds day-edge uncertainty to the snapshot cadence. Each daily row shows fleet prompt
tokens, cached/computed splits, hit-token ratio and day-over-day change, the revenue
multiplier band at cached-token billing factors 0.25 and 1.0, tick-seg `[64,512)` probe
share, and detected restart count. Revenue and tick-seg math comes directly from
`tools/cache_economics.py`; the report does not carry a second formula.

**Exactness gate** (`tools/cache-meter-gate.py`, serve-smoke arm 7b): N requests sharing a
K-token `prompt_ids` prefix must meter exactly — seed/LCP-split requests `cached_tokens: 0`,
steady-state hits `cached_tokens == K`, a same-prefix request under a different `cache_salt`
cold (PC-ISO), `/metrics` totals closed-form, histogram bucket-exact, economics row
crosschecked. 26/26 on the 5090; disabling the cache inverts 16/26 (teeth). Overhead A/B
(pre-lane binary vs instrumented, both resident, interleaved x5, N=100/arm, prefix-hit
steady state): p50 −0.03%, p95 −0.19% — no measurable serve overhead (<0.5% p95 bar).
Receipts: `research/cache-meter-20260807/`.

## Spec-decode acceptance telemetry (lane/accept-telemetry, 2026-08-05)

Always-on per-draft-position acceptance counters, the llama.cpp #26389 / vLLM spec-decode
counter schema. WHY: the 2026-08-05 dogfood head-to-head found short-context sampled
acceptance at 0.55 vs 0.73 full-draft — a posthoc dig that this surface turns into a live
gauge (drafter health on a new checkpoint is readable in minutes, and the K-policy work
gets a per-position decay curve for free).

**`GET /metrics` — the `spec` block**, per model, cumulative since the model loaded (models
load once per server process, so counters reset on restart, never mid-run). Absent until the
first spec burst — spec-off deployments see the exact pre-lane payload:

```json
"spec": {
  "q9": {
    "rounds": 118, "drafted": 354, "accepted": 213,
    "acceptance_rate": 0.602, "tokens_per_round": 2.805,
    "pos_drafted":  [118, 118, 118],
    "pos_accepted": [96, 71, 46],
    "accept_rate_per_pos": [0.814, 0.602, 0.390]
  }
}
```

`accept_rate_per_pos[j]` = P(draft position j accepted | a round offered position j) — healthy
spec decode decays monotonically from position 0 (acceptance is a prefix walk: position j can
only be accepted if 0..j-1 were). Arrays are trimmed to the deepest position ever drafted
(up to 8 tracked positions; totals count deeper drafts too). Normalization matches
`MEMRA_SPEC_STATS`: a p-min-cut chain token is counted in neither drafted nor accepted. The
opt-in round-stream arm (`MEMRA_SPEC_STREAM=1`) keeps its accept counts on device, so under
it per-position arrays cover the standard-path rounds only; totals stay complete.

**`usage.spec` — per-request summary.** Spec-decode requests carry their OWN
rounds/drafted/accepted + `acceptance_rate` in the response usage object (this request only —
pool-resumed sessions do not leak prior requests' counts). Additive and OpenAI-safe: official
SDKs ignore unknown usage fields, no existing field changes, and non-spec requests carry no
`spec` key at all.

**Cost:** host-side u64 adds at the round accounting the engine loop already does — zero
GPU syncs, zero per-token allocation, no hot-path lock (the worker merges per-burst deltas
into its own map; the metrics mutex is only taken on the existing 32nd-tick publish, plus a
force-publish when a spec session retires so one-shot requests are visible immediately).
Validation capture: `research/accept-telemetry-20260805/`.

## API keys — multi-key tenant auth (lane/api-keys, 2026-08-05)

Bearer auth that maps key → tenant, so cache isolation, QoS lane class, rate-limit
headers, and metering all key off a real tenant identity. Launch-shaped: a file-backed
keyring + a CLI, no web UI.

**Configuration.** `MEMRA_API_KEYS=/path/keys.toml` — TOML `[[keys]]` entries carrying
`sha256` (of the plaintext key — the plaintext is never stored), `tenant`
(`[A-Za-z0-9_-]+`), `lane` (`interactive` default | `batch`), `enabled`, and optional
`rate_limit`. An inline env form `tenant:sha256hex[:lane],...` exists for file-less
deploys. A malformed ring is a startup FATAL (never partially applied); the file
hot-reloads on mtime change (≤2s poll — chosen over SIGHUP: no signal thread, cannot be
missed), and a broken rewrite keeps the previous ring and logs loudly — auth never fails
open because of a typo.

**Lifecycle CLI.**
```
memra-server --gen-key acme [--lane batch] [--rate-limit 4] [--keys /path/keys.toml]
memra-server --revoke-key mk-acme-1a2b3c4d5e6f [--keys /path/keys.toml]
```
`--gen-key` prints the plaintext key (`mk-<tenant>-<48 hex>`) exactly ONCE on stdout and
appends the hash entry; `--revoke-key` disables by unambiguous prefix (or full key) — a
running server picks the revocation up on the next poll. `--keys` defaults to
`MEMRA_API_KEYS`.

**Request law.** `Authorization: Bearer <key>` on every `/v1` completion route:
- keyring match → that key's tenant context; **disabled key → 403** (actionable,
  distinct from unknown), **unknown key / missing header → 401**;
- `MEMRA_API_KEY` (the single static key — the daily driver and every serve script)
  keeps working unchanged as tenant `default`, with or without a keyring configured;
- neither configured → open (dev behavior), tenant `default`.

**What the tenant identity drives:**
- **Cache isolation:** with a keyring configured, the PC-ISO namespace is
  `t:<tenant>␟<cache_salt>` — one tenant's keys share cached prefixes, different tenants
  never do, and the `␟` (US, `\x1f`) separator is excluded from tenant ids so a
  client-controlled `cache_salt` cannot forge another tenant's namespace. `cache_salt`
  still sub-scopes WITHIN a tenant (a gateway multiplexing end-users through one key
  keeps setting per-user salts). No keyring → the raw-salt namespace, byte-identical to
  PC-ISO behavior.
- **QoS lane class:** `interactive`-class keys behave exactly like pre-lane traffic
  (default lane interactive, any `x-lane` honored). `batch`-class keys default to the
  harvest lane and are refused `x-lane: interactive` with a 403 — a bulk key cannot
  claim the protected class, by omission or by header.
- **Rate limits:** per-key `rate_limit` is a concurrency-slot override; the effective
  cap is **min(override, global lane cap)** — the global cap stays authoritative, an
  override can only narrow. A request that arrives while its tenant already holds every
  configured slot is rejected before worker admission with `429 rate_limit_exceeded`;
  two simultaneous arrivals cannot both pass the cap. The `X-RateLimit-*` trio reports
  the binding cap, with `Remaining` counting the tighter of the tenant and lane gauges.
  Multiple keys under one tenant intentionally share that tenant's gauge; issue distinct
  tenants when recipients need independent caps.
- **Metering seam:** every admitted request logs one flat
  `[meter] admit id=<x-request-id> tenant=<t> lane=<l> model=<m>` line — the public-repo
  half; the private fork's metering layer joins these against the worker-truth usage
  lines by request id for per-tenant billing.

Gate: `tools/apikeys-gate.sh` (unit laws + live two-tenant isolation proof via
cache-hit behavior; receipts `research/apikeys-20260805/`).

## Session affinity — resuming a REWRITTEN conversation (lane/session-affinity, 2026-08-05)

Both reuse tiers above require the new prompt to EXTEND what is cached (token prefix, or
text prefix). Real agent clients do not extend — they REWRITE. The owner's client strips
`<think>` blocks out of prior assistant turns before re-sending them, so turn N's prompt is
not a prefix-extension of anything, both probes miss, the parked multi-GB session is
discarded, and every turn re-primes the whole growing conversation.

Affinity answers a different question: not "does this prompt extend that session's bytes?"
but "is this the SAME CONVERSATION?" — and then resumes it at a retained boundary.

**Two identity tiers (nomination only):**

- **Explicit** — the client names its conversation. Accepted from `session_id` or `user` in
  the request body of `/v1/completions` and `/v1/chat/completions`, or the `x-session-id`
  header. Body beats header (the body is the caller's own statement of identity; a header can
  be injected by an intermediary); `session_id` beats `user`. An explicit id on one side only
  never matches: a named conversation and an anonymous one are not the same conversation.
- **Implicit** — nothing named, so identity is STRUCTURAL: the conversation is split at its
  control tokens (the chat template's own role markers) and each segment contributes a hash of
  its first and last few tokens. A rewritten segment BODY does not perturb its hash, so the
  chain's leading run survives a think-strip; three shared segments are required before an
  implicit fingerprint may name a conversation (a bare system prompt is shared by every fresh
  conversation and must not cross-link them).

**Identity nominates, BYTES decide.** A nominated session is resumed only if the new prompt
reproduces its committed tokens EXACTLY up to the boundary its last turn checkpointed. A
fingerprint collision therefore costs one wasted comparison, never a wrong resume. If the
rewrite reached BELOW the boundary, affinity declines and the request re-primes in full —
correctness first. Declines are logged with their offsets (`history diverged at N of
checkpoint M`), because a silent decline is indistinguishable from a broken mechanism.

**The boundary.** Each turn checkpoints the state at its PROMPT END — before the first
generated token. That is the only boundary worth keeping: a history-rewriting client mutates
what the session GENERATED, never the prompt it was given. Full-attention KV is truncatable
by length, so the checkpoint copies only the GDN conv/ssm recurrent state; the draft scratch
needs no copy (the next turn's fill rewrites it).

**Scope.** Affinity is stored per (model, cache namespace), so it adds no cross-tenant reach
beyond what the reuse tiers already have: a `cache_salt` is an affinity boundary too.
Constrained (grammar) requests never resume. Resumed sessions respect the same evict-first +
right-size ladder as new ones, and are tested against the room the request actually needs, so
a right-sized session stays affinity-eligible.

`MEMRA_AFFINITY=0` turns the mechanism off (rollback seam / exactness A/B arm; the winner is
the default and needs no flag). Receipts, byte-identity gate, and TTFT curves:
`research/session-affinity-20260805/`.

## Multi-tenant QoS — the x-lane SLO gate (lane/qos-p95, 2026-08-02)

Requests may tag a service class via the `x-lane` header: `interactive` (protected;
also the default when the header is absent — naked traffic is byte-identical),
`judge` (prefill-shaped), or `harvest` (decode-shaped bulk). The gate is engine-side
admission control: interactive always admits (waits FIFO past `MEMRA_MAX_SESSIONS`,
never rejected); judge/harvest admit only while the measured interactive decode-step
p99 stays under their fraction of `MEMRA_SLO_P99_MS` (50ms default) and shed with an
immediate `429 + Retry-After` otherwise — dark work is never queued inside the engine.
Inside the tick, interactive decode rows batch first and dark-lane prefill runs after
decode within measured SLO headroom only. Per-lane counters + the engine-truth step
p50/p99 export at `GET /yield/metrics`.

Measured at fleet scale (8 replicas, Qwen3.5-9B-Q8_0 on rented H100s, c=96 harvest + c=4
interactive, 4 conditions interleaved, N=3 passes with full teardown/bring-up per cell,
`research/qos-p95-20260802/`): the lane-blind proxy FIFO alone inflates contended
interactive p95 to 7.15s (~4x alone); with lanes on and the proxy cap at 16 (so engine
admission owns the queue — the gate cannot fix a queue it never sees), p95 drops to
3.69s (~2x alone) at -11% bulk throughput vs the cap-16 ceiling. `MEMRA_SLO_P99_MS`
is the dial: 25ms makes contended interactive statistically equal to alone
(p50 1.637s / p95 2.158s) with bulk paying -67%. Lane knobs in [FLAGS.md §1](FLAGS.md).

**Attribution, required whenever the 7.15 → 3.69s figure is quoted:** raising the proxy
cap from 8 to 16 *by itself* moves p95 **7.15 → 4.335s** (that control cell is in the same
RESULTS.md); the lane gate accounts for **4.34 → 3.69s**. Roughly half the headline
improvement is the queue, not the engine gate — which is the point of the sentence above,
not a caveat to it. Quoting 7.15 → 3.69s as "what lanes do" is refutable from our own log.

## Streaming cadence + admission latency — the felt-TTFT arc (lanes sse-cadence 2026-08-05, admission-latency 2026-08-06)

Two fixes, one arc: solo first text went **0.41 s → 0.12 s** and contended first text
**1.60 s → 0.15 s** (27B NVFP4+MTP, K=3, local 5090, N=5 medians in one lock hold), and
neither number scales with `MEMRA_SPEC_BURST` anymore.

- **Round-cadence SSE** (lane/sse-cadence): spec-burst sessions used to emit ONE
  `Event::Token` per burst — at B128 that meant 2 chunks per response and 1.16 s to first
  text. The worker now flushes text at every spec-round commit through an `on_commit` seam
  in the engine's spec loop (same detokenize-tail + `utf8_delta` cursor, same
  EOS-text-never-streamed rule), so first text is ~0.12 s and inter-chunk gap p50 ~27 ms at
  ANY burst size for a solo stream (B32 fix-off was 0.41 s / 299 ms). Content is
  byte-identical either way — only chunk boundaries move. Throughput parity measured c=1
  and c=8. Rollback: `MEMRA_SSE_PER_BURST=1`. Receipts:
  `research/sse-cadence-20260805/VERDICT.md`.
- **Admission yield + cold-first ordering** (lane/admission-latency): a request arriving
  mid-burst used to wait the whole in-flight burst out (contended first text 0.54 s at B32 /
  1.60 s at B128, i.e. burst size set round-robin admission latency). Two pieces, one flag:
  a pending admit (`PENDING_ADMITS` gauge, polled by the round hook above) ends the
  in-flight burst at the next round boundary, and sessions that have emitted nothing yet
  burst before mid-generation peers. Contended first text is now **0.123 s (B32) / 0.152 s
  (B128)** — the solo class at any burst. Content byte-identical on/off, solo AND contended.
  The cost lives at c=8 saturation only: −3.4% agg tok/s for 3.8x better p50
  (newcomer-first vs lockstep-fair; p95 tail pays); c=1 parity. Rollback:
  `MEMRA_ADMIT_YIELD=0` (both pieces). Receipts: `research/admission-20260806/VERDICT.md`.

Burst default stays **B32** by the strict flip criterion, but the two old flip-blockers are
gone: B128 buys +8.4% (c=1) / +8.5% (c=8) and now trails B32's contended first text by one
29 ms round-cadence quantum instead of a 3x cliff — a live owner call, per the
`MEMRA_SPEC_BURST` row in FLAGS.md.

## Dead-darklane background jobs — valleys carry owner work (lane/darklane-training, 2026-08-07)

The standing lab thesis: idle serve capacity carries owner research/training jobs, yielding
instantly to paying traffic. This section is the ENGINE mechanics only — which jobs run,
what a valley is worth, and every scheduling-policy/economics question live in the product
repo; the seam between the two is exactly `MEMRA_BG_JOB` + the checkpoint protocol below.

**Valley detection** invents no sensor. The scheduler already flips its health phase to
IDLE precisely when `active` and `queue` are both empty, and the phase stamp refreshes the
heartbeat — so `phase == IDLE` + heartbeat age IS the idle duration, at zero new hot-path
cost. The `PENDING_ADMITS` gauge closes the HTTP→worker handoff gap (a submitted request
the worker hasn't popped is traffic, not idleness). Exposed two ways: **`/metrics
serve_idle_seconds`** (always published; 0.0 the instant there is any work) and the
in-process `ValleySignal` hook (`darklane.rs`) the runner polls. Receipt — signal accrues
while idle, reads 0.0 sampled mid-generation, re-accrues from a fresh epoch after:
`research/darktrain-20260807/raw/valley-signal.log`.

**The lane class sits BELOW every serving lane.** Harvest is still a *request* class the
engine admits, schedules, and sheds; a background job is not a request at all — it runs
only while the engine has NOTHING (no interactive, no judge, no harvest, no queue, no
pending admits) and yields on the first sign of any of them. The hysteresis is asymmetric
on purpose: yield fires on the busy EDGE with no debounce (paying traffic never waits for
a threshold), resume waits a full `MEMRA_VALLEY_S` (default 2 s) of quiet, because a
between-requests gap in a live conversation is not a valley.

**Yield mechanism v1 — simplest honest first.** The job (`MEMRA_BG_JOB`, arbitrary
command) is a child process in its own process group; yield is SIGSTOP to the group,
resume SIGCONT. The bound is the poll interval (`MEMRA_BG_POLL_MS`, 25 ms) plus signal
delivery — **measured 19.4 ms median / 23.3 ms max** request-fired-to-job-stopped (N=5,
one per rep, i.e. one poll interval; target <500 ms). Serve-impact stress (N=5 interleaved
reps, fresh boot per rep per arm, c=8×16 streaming bursts vs an 8-spinner CPU job, 5090):
burst p95 delta **+0.77%**, TTFT p95 **+1.11%**, agg tok/s **−0.54%** — under the 2% bar
(`research/darktrain-20260807/raw/bgstress-n5.log`). Two operator truths: a SIGSTOPped
process KEEPS its memory (VRAM included — the budget is carved out for the life of the
job, not per valley), and the runner cleans up on drain (CONT→TERM→KILL past grace) while
PDEATHSIG covers the SIGKILL path, so no orphan ever stays frozen.

**GPU memory discipline** is fits-or-refused at launch: `MEMRA_BG_VRAM_MB` (default 0 =
CPU-only) is granted only while `min free across visible GPUs >= budget +
MEMRA_MOE_RESIDENT_HEADROOM_GB` — min, not sum, because on a PP-2 pair both cards carry
serve shards. Unreadable `nvidia-smi` = refusal (fail closed); a refused job retries next
valley (headroom moves as sessions retire). v1 enforces fit at launch; staying inside the
budget at runtime is the job's contract, and the VRAM-aware admission gate defends serving
against a job that lies the same way it defends against everything else.

**Checkpoint/resume — the training-class seam** (`MEMRA_BG_YIELD_MODE=checkpoint`), for
jobs whose stopped working set must not squat on VRAM. The "checkpoint callback" is
process-level: SIGUSR1 to the group means *checkpoint now and exit 75* (EX_TEMPFAIL); the
runner relaunches the same command next valley and the job resumes from its own file.
Exit 0 = complete, never relaunched; any other exit = failed, loud, never relaunched. A
job that outlives `MEMRA_BG_CKPT_GRACE_MS` (5 s) after SIGUSR1 is SIGKILLed — the yield
bound holds even against a wedged job, and semantics are at-least-once (a training step
may repeat, never be lost; checkpoint writes must be atomic — write-tmp-then-rename).
Write single-command jobs as `MEMRA_BG_JOB="exec python3 train.py ..."`: the command runs
under `sh -c`, and without `exec` the shell parent dies of the unhandled SIGUSR1 before
the job's exit 75 can propagate. The live-server receipt caught exactly this
(`raw/ckpt-serve.log`: "job exited None during preemption") — the runner's
during-preemption branch classifies ANY exit after SIGUSR1 as checkpointed-and-relaunch,
so the cycle still resumed from step 129 correctly, but `exec` is what makes the
protocol exit visible.
Toy proof: `tools/bg-ckpt-counter.py` (counter checkpoints on SIGUSR1, exits 75, resumes
from the file; the unit test `checkpoint_mode_preempts_and_resumes_counter` pins the whole
cycle GPU-free). An in-process trainer API can replace this seam later without touching
the valley/scheduler half.

Observability: `/metrics` gains a `bg` block only when `MEMRA_BG_JOB` is set (state,
launches/yields/resumes/preempts, ckpt_kills, last yield-signal micros, job pid, budget)
— unset deployments see the pre-lane payload byte-identical.

## Knobs

Serving flags (batch cap, device sampling, lean logits, prime batching, spec burst) are
cataloged in [FLAGS.md §7](FLAGS.md) under "Serving (memra-server)"; fleet topology knobs
(`GPUS`, `REPLICAS_PER_GPU`, `CAP`, ports, health cadence) are env-overridable at the top of
`tools/serve-fleet.sh`. The exactness contract holds under batching: `decode-batch-gate` runs
inside `tools/validate-h100.sh` **twice** — `--mode config --batch 8` (the default-env battery,
fused tier live in the reference) and `--mode strict --batch 4` under the equalized composition
(`MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1`). Each invocation runs gate1 (B=1 vs `decode_step_h`),
gate2 (per-seq isolation — batchmates must not change your stream), and gate3, whose three
sub-checks are (a) device-argmax == host-argmax of the same row, (b) sampled draws at B=N ==
the same metas at B=1, and (c) `gate3c`, lean-vs-full logits identity. **`gate3c` is a
sub-check of gate3, not a fourth gate** — gate3 prints one PASS/FAIL line covering all three,
so a green line is the only signal that (c) ran; the sub-check names surface in the output only
when one fails. The stage-split modes (`--mode pp`, `--mode ppspec`) SKIP gate1/2/3 by design —
they are single-device jurisdiction — and neither PP mode is wired into `validate-h100.sh`; PP
exactness has its own invocations (see [TESTING.md](TESTING.md)).

## First-token cross-config drift (batched prime) — stated honestly

Serving primes prompts BATCHED (`prime_cache`, prefill GEMMs) while the historical oracle
stream is tokenwise (`decode_step`, m=1). These are different numeric configs by design —
same law as forward-vs-decode and the decode-batch gate's config mode — so on near-tie
prompts the FIRST generated token of a request can differ from the tokenwise oracle
stream, and everything after it follows the new prefix. Measured on the six-model
2026-08-02 sweep (`research/prime-gate-coverage-20260802/`, 144 prompts): **10/144 first
tokens flip (~7%)**, every flip at a tokenwise top1-top2 margin <= 0.70, batched prime
bit-deterministic, no content leakage across chunk boundaries, and forward_last sides
with the batched prime in 8/10 flips — the tokenwise config is usually the outlier, so
this is config roulette on a near-tie, not a wrong path. On the gemma prefill lanes the
config can even move per PROCESS (cuBLASLt heuristic algo selection; one observed
instance in the 144-row double pass, bit-deterministic within a process). Dense Q8_0
models (9B judge, Ornith-9B — the fleet class) flipped 0/48. Consequences can be visible (the Qwen3.6-35B
pp512 probe greedy-emits `"\n"` + EOS at 2 tokens where the tokenwise stream writes 128):
within contract, but real. `MEMRA_PRIME_TOKENWISE=1` pins the oracle stream at prefill
cost; the run-gen `batched-prime` gate line + the `prime-gate` battery bound the class
(structured divergence fails hard, near-tie flips are reported).

### Chunked prefill is split-stable — since the grain-free fix (found 2026-08-05; mechanism corrected and FIXED same day)

The class below is **history**: since lane/chunkinv-flip, chunked prefill is bit-identical
across `MEMRA_PRIME_CHUNK` values by default (see "FIXED BY DEFAULT" further down). What
follows is the finding and root cause, kept because the mechanism correction is the
evidence for the fix. A sharper statement of the same class, found while building
serve-smoke check 10: **changing only the prefill chunk split changed greedy output.**
Arms were the same four recorded prompts with a per-turn `cache_salt` (so nothing
resumes — every request primes cold), `MEMRA_AFFINITY=0`, varying only `MEMRA_PRIME_CHUNK`:

| prompt tokens | 2048 vs 64 | 2048 vs 32 |
|---|---|---|
| 48 | identical | identical |
| 97 | identical | **differs @ char 45** |
| 149 | **differs @ char 172** | **differs @ char 52** |
| 195 | identical | identical |

No reuse required, and 149 tokens is far too short for a long-window explanation. Every
resume tier inherits this by construction: a resume primes `[rewind boundary .. end]` as
its own chunk sequence rather than one full prime.

**Mechanism — corrected 2026-08-05 by `lane/chunk-invariance`.** This section originally
said "a different split changes the reduction order in the prefill GEMMs." **That is
measurably wrong.** The prefill GEMM is m-INVARIANT: feeding the same activation rows at
m=32 and at m=33..80 leaves rows `[0,32)` BIT-IDENTICAL for both the quantized `wq` and the
`output` head, so growing a batch does not move an existing row's value. And the divergence
is not a distributed last-bit band — it is a **step at the first chunk boundary**: per-row
maxdiff is exactly `0.000e0` for every row before the boundary and O(1) (6.9) immediately
after it, with `first_div_pos` equal to the chunk size exactly in every arm.

The real cause is a numeric-**class** edge, in `full_attn_prime_fa_dispatch`
(`hybrid_forward.rs`), selected by `base_len == 0` — *"is this the first chunk?"*:

- chunk 0 → `fa_prefill` over this batch's **f32** K/V;
- every later chunk → `fa_prefill_view_ws` over the **q8_0/q5_1 quantized KV cache**.

So `MEMRA_PRIME_CHUNK` decides at which token position the prefill stops reading f32 K/V
and starts reading dequantized cache. Rows before that position are computed identically in
both configs (hence the bit-identity); rows after it carry q8_0/q5_1 quantization error, and
a near-tie argmax flips. Eliminated by measurement, not assumption: `MEMRA_PRIME_DEQW=0`
(the other quantized-cache FA kernel) diverges identically, and `MEMRA_GDN_CHUNKED=0`
(sequential GDN scan, no WY segmentation at all) still diverges — the GDN state carry is
**not** the cause.

**FIXED BY DEFAULT — grain-free (lane/chunkinv-flip, 2026-08-05).** The `base_len == 0`
f32 special case is gone: chunk 0 quantizes its K/V into the cache first and attends through
`fa_prefill_view_ws` exactly like every later chunk (quantize-then-attend). One numeric class
for every row means the chunk size cannot decide where a precision edge falls, so **chunked
prefill is byte-identical across `MEMRA_PRIME_CHUNK` values with no door and no grain knob**
(chunkinv gate, naked env, both pinned prompts EXACT at chunks 2048/64/32).
`MEMRA_PRIME_CHUNK` is again a pure memory/transient knob. Rollback seam:
`MEMRA_PRIME_F32CHUNK0=1` restores the legacy f32 first-chunk arithmetic (and is the gate
canary's injection). The interim `MEMRA_PRIME_INVARIANT`/`MEMRA_PRIME_GRAIN` pin-the-boundary
door was superseded by this fix and removed at v0.71 per the flags doctrine (the research
record keeps its history). History + root-cause receipts:
`research/chunk-invariance-20260805/VERDICT.md`; flip receipts:
`research/chunkinv-flip-20260805/`.

What this changes and what it does not:

1. Gates MAY now assert byte-equality between two prefills of the same prompt at different
   chunk boundaries — `tools/chunk-invariance-gate.sh` asserts exactly that as its default
   (`--expect-invariant`, no env). serve-smoke check 10's scoping note is retired with it.
2. The exactness CLASS of short (single-chunk) prompts changed at the flip: chunk 0 now
   reads quantized KV — the same arithmetic long prompts always had past the first boundary.
   Near-tie argmax flips vs the old f32-first-chunk output are the documented contract
   change (quantified teacher-forced in `research/chunkinv-flip-20260805/`), not a bug.

**Scope: this is a per-architecture property.** The fix above is a property of the
`full_attn_prime_fa_dispatch` path, and the gate runs on the shipped arches. A different
attention family can re-enter the class through its own door, and one did — twice, both
closed and both gated: the `step35` bring-up arch (Step-3.7-Flash) was **chunk-DEPENDENT
past its 512-token SWA window** via kernel *selection* (a chunk whose `t_kv` exceeded the
window took the f32 windowed floor while a chunk that fit took FA, so the FA rows formed a
prefix `P = c*floor(win/c)` and the verdict depended only on `P` — pinned by a
pre-registered 4/4 falsification battery incl. a one-token c=513-vs-512 verdict flip;
receipts [`research/step37-p2-20260806/`](../research/step37-p2-20260806/), commit `66a81371`).
**FIXED 2026-08-07 in two stages, both gated:**
(1) *within one `prime_cache` call* — the SWA arm keys on the request's `seq_end`, not the
chunk's `t_kv`, making `P` identically 0 at every chunk size; gate `chunkinv35` (+
`chunkinv35c` canary via `MEMRA_STEP35_SWA_TKV`), default measured +0.009%
([`research/step35-chunkfix-20260807/`](../research/step35-chunkfix-20260807/));
(2) *across calls* — serve splits a prompt over SEVERAL `prime_cache` calls (per-tick budgets,
dark lanes SLO-capped = load-dependent; plus the prefix-cache LCP split), so `prime_cache` now
carries `queued_after` and `seq_end = cache.pos + t + queued_after` is request-level whatever
the tick segmentation; gate `tickinv35` (+ `tickinv35c` canary via `MEMRA_PRIME_CALLLOCAL`,
whose `sp<L>` split arms also pin the off-grid-resume hole — vLLM #51113's second law)
([`research/tick-seg-20260807/`](../research/tick-seg-20260807/)).
The SECOND door opened when the SWA prefill moved from the f32 floor to the windowed hd128
FA stamp (lane/pp-prefill 2026-08-07, `MEMRA_STEP35_SWA_FA` seam): the FA kernel's
online-softmax tiles group keys relative to the **view start**, and the SWA view offset is
a chunk boundary — so an unaligned offset regrouped the same absolute keys into different
BK=32 tiles at different chunk sizes. Closed by aligning the view offset down to the tile
size (the ≤31 extra leading keys are fully masked for every query — a bitwise no-op in both
kernels, measured on the floor arm). `chunkinv35` caught the second door on its first
battery, and its canary (`MEMRA_STEP35_SWA_TKV=1`, restoring BOTH pre-fix halves) is
verified red-capable (`research/pp-prefill-20260807`, batteries 1-3).

The behavior is **gated in both directions**: fast-gate ids `chunkinv` / `chunkinvc`
(routed from the `hybrid_forward.rs` map row): the default arm asserts byte-identity naked;
the canary arm injects `MEMRA_PRIME_F32CHUNK0=1` and must break, proving the gate detects
the mechanism. Reproducers + raw rows:
`research/session-affinity-20260805/chunk-order-probe.py` and `chunk-order.jsonl` (12 rows =
3 chunk sizes x 4 prompts, each with its text; under two minutes on the 9B), plus the
engine-level root-cause arm `concat-prime-probe chunkinv` and
`research/chunk-invariance-20260805/`.
