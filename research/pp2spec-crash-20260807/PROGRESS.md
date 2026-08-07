# pp2spec-crash — task #87: the sticky ILLEGAL_ADDRESS under spec+PP-2 concurrency

**Lane**: `lane/pp2spec-crash` off `006aca75`. **Rig**: darklanes-g7e12, 2x RTX PRO 6000
Blackwell Server 96GB, sm_120a, CUDA 13.2, driver 595.71.05 (SPOT box, shared, flock
GPU-lock discipline). Model: q9 = Qwen3.5-9B-NVFP4-MTP GGUF on box NVMe.

**Inherited finding** (lane/pp2-spec, merged 5882b753): spec over `MEMRA_PP_DEVICES=1,0`
loses 100% of requests at c=4 (0/48, 3/3 reps), the fault is STICKY (every later
new_session in the process inherits `CUDA_ERROR_ILLEGAL_ADDRESS`), spec-OFF same placement
is 96/96 clean, and `MEMRA_SPEC_NOGRAPH=1` fails identically (draft graph exonerated).
Quarantined at parse time via `DraftVerdict::RefuseSpecOverPp2`.

## Diagnostic door

The quarantine walls off the debugger too, so commit 3f56c8ce adds
`MEMRA_PP2SPEC_UNQUARANTINE=1` — skips both refusal sites (preflight + load-path verdict),
WARNs loudly at boot, dies with the quarantine when #87 is fixed. Since the finding lane,
the spec-gate (#89) also merged: at c>=4 it routes new arrivals to batched decode, which
would HIDE the two-live-spec-sessions trigger — every repro arm therefore pins
`MEMRA_SPEC_GATE=0`.

## Round 1 — repro + first localization (raw/round1/, box receipts ~/receipts/pp2crash)

Three phases, one script (`raw/round1/run-pp2crash-repro2.sh`), tree @ 3f56c8ce:

| phase | env delta | c=2 | c=4 | verdict |
|---|---|---|---|---|
| A bare | (none) | 1/8 ok | 0/16 ok, wall 0.070s | **REPRODUCES** |
| L launch-blocking | `CUDA_LAUNCH_BLOCKING=1` | 12/12 ok | 16/16 ok | **CLEAN** |
| B memcheck | compute-sanitizer memcheck + `MEMRA_SPEC_NOGRAPH=1` | 8/8 ok | 8/8 ok | **CLEAN, 0 findings** |

Phase A quotes (raw/round1/A-server.log):

    [worker] spec pool evicted (2) after alloc failure; retrying (evict-first learned)
    [worker] spec session alloc failed (DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, "an illegal
             memory access was encountered")); tokenwise path

then the sticky repeat for every later request — same signature as the finding lane, 3/3
faithful. Client-side (A-points.jsonl): c=2 errors are `step error: ILLEGAL_ADDRESS` and
`cache alloc failed: ILLEGAL_ADDRESS`; c=4 is all `cache alloc failed` in 70 ms = context
dead on arrival.

**The kernel-level fault is captured this time** (raw/round1/xid-at-fault.log, journalctl
at 06:07:02 = the exact A-phase c=2 fault minute):

    NVRM: Xid (PCI:0000:32:00): 31, pid=112468, name=memra-gpu-worke, channel 0x00000002,
    intr 00000000. MMU Fault: ENGINE GRAPHICS GPC0 GPCCLIENT_T1_1 faulted @ 0x484_c6b3c000.
    Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_READ

PCI 32:00 = **GPU index 0 = stage 1** (placement dev10: stage0=dev1, stage1=dev0). A
VIRT_READ FAULT_PDE = a kernel READ through an unmapped page-directory entry — a stale or
never-mapped device pointer, not an out-of-bounds offset within a live allocation
(memcheck's specialty, consistent with B finding nothing).

## What round 1 establishes (quoted, not inferred)

1. **Timing-dependent, not a logic bug in the math**: `CUDA_LAUNCH_BLOCKING=1` (every
   launch synchronous) is 28/28 clean; the sanitizer's serialization also masks it. The
   race window needs genuinely-async cross-stream execution. This KILLS the "padded
   side-tensor corrupts KV slot ownership" shape (SGLang #33253) as primary — that class
   is timing-independent and would fault under L too.
2. **The faulting access is a READ on the stage-1 device (dev0) through an unmapped PDE.**
   Consistent with a use-after-free (stream-ordered free retired the mapping while another
   stream's kernel still read it) or a pointer to another device's memory dereferenced
   without peer mapping. The TRT-LLM #16170 relay-starvation analogue (stale/reused
   boundary pointer while the owner thread blocks) remains live; so does a WAR/RAW hazard
   on a shared buffer between two spec sessions' interleaved bursts (SGLang #33587 shape).
3. **Two live spec sessions on the split ARE the trigger** (re-confirmed): c=2 lost 7/8 in
   phase A while c=1 and every spec-OFF arm in the record are clean.
4. Sanitizer note: the 36 `Program hit CUDA_ERROR_NOT_FOUND ... cuModuleGetFunction`
   records in B-server-sanitizer.log are the engine's known fallible symbol probe
   (`Engine::func` tries 5 modules in sequence, lib.rs:1123-1131) — benign, present in
   every clean run.

## Round 2 — the faulting kernel is NAMED (raw/round2/)

Coredump-on-exception preserves async timing until the trap. Two data points:

- **Phase C** (coredump env, c=4 only, NO c=2 warmup phase): 16/16 CLEAN — the trigger
  wants the c=2+warmup arrival pattern; raw c=4 on a fresh server didn't fire this time.
- **Phase D** (coredump env, exact phase-A sequence): c=2 1/8, c=4 0/16 — the A signature,
  and the driver caught the trap. `cuda-gdb` on `core-r1-125335.nvcudmp`
  (raw/round2/D-cudagdb-core-r1-125335.log), quoted:

      CUDA Exception: Warp MMU Fault
      The exception was triggered at PC 0x7c986b7ad960  embed_gather_u32
      #0  0x00007c986b7ad9f0 in embed_gather_u32<<<(16,1,1),(256,1,1)>>> ()
      [device 0, sm 0, warp 5, lane 0]

  Registers R10/R11 = 0x484_c6b3c500 — the SAME PAGE as round 1's Xid 31
  (`faulted @ 0x484_c6b3c000`). A reproducible fault VA across independent runs is a
  pointer-valued read of a stale/unmapped mapping, not a random-offset overrun.

`embed_gather_u32` at grid (16,1,1) = the T=1 single-token gather (n_embd 4096 / 256).
In the spec serving path that shape is the DRAFT CHAIN's first node.

## Round 3 — both draft arms fault at the SAME VA (raw/round3/)

Two arms, exact-A sequence, coredump env, fresh server each (`run-pp2crash-round3.sh`):

| arm | draft path | c=2 / c=4 | faulting kernel | fault VA (R10:R11) |
|---|---|---|---|---|
| nograph (`MEMRA_SPEC_NOGRAPH=1`) | eager chain | 1/8, 0/16 | `embed_gather_u32_t` (grid 16,1,1) | 0x484_c6b3c500 |
| graph2 (default) | captured graph replay | 1/8, 0/16 | `embed_gather_u32` (grid 16,1,1) | 0x484_c6b3c500 |

Same fault VA in three independent server processes, two different kernels, both being
the draft chain's embed gather (eager arm: `mtp_head_forward_dev` op A
`embed_gather_device_t(g, &[e_tok], ..)` spec.rs:735; graph arm: `mtp_head_forward_cap`
`embed_gather_device` spec.rs:1281). Both kernels read exactly two device buffers: the
resident embed TABLE (`model.embd_gpu`, a per-model `OnceLock` uploaded once) and the
tiny token-id buffer. The token-id buffers differ per arm (fresh `htod_u32_v` vs the
session's persistent `g_tok`) — but the fault VA is IDENTICAL across arms and processes,
so the common operand — **the resident embed table pointer** — is the stale mapping.
(An identical VA also kills "garbage token index": a wild index would fault at
table_base + garbage*row_bytes, different every time.)

Working hypothesis, to be settled by operand readout (round 4): the embed table is
uploaded through the PRIMARY engine's stream (dev0-homed under placement 1,0 — note the
primary is stage 1 here) via `embd_gpu.get_or_init(|| e.upload_u8(..))`, while a second
live spec session's traffic runs concurrently; the table allocation lands in the async
pool and something stream-ordered frees or remaps that VA range. Candidate mechanisms to
discriminate: (a) the upload races a pool free from another session's dropped transients
(alloc reuse across streams without dependency — pool has REUSE_ALLOW_OPPORTUNISTIC=0,
INTERNAL_DEPENDENCIES=1, but those govern ONE device's pool and same-pool reuse only);
(b) the table OnceLock is initialized by a different Engine (stage engine vs primary) in
one session and dereferenced under the other stage's context.

## The SENTINEL-TOKEN arithmetic (the round-3 VA decoded)

The q9 artifact's `token_embd.weight` is NVFP4 (`ggml_dtype 40`), dims [4096, 248320]
(read off the GGUF header on-box). NVFP4 block = 64 elems -> 36 bytes, so
`row_bytes = 4096/64*36 = 2304`. The device argmax kernels (`argmax_partial_f32` /
`argmax_final_f32`, kernels.cu:122/161) initialize `best_i = 0x7fffffff` and only replace
it when a comparison WINS — if every logit is NaN (`v > best_v` false, `v == best_v`
false), the sentinel survives and `token_out[0] = 0x7FFFFFFF`. And:

    0x4_C6B3_CE00 + 0x7FFFFFFF * 2304 = 0x484_C6B3_C500  (the fault VA, exactly)

i.e. the fault VA IS `embd_base + 0x7fffffff * row_bytes` for a table based at
0x4C6B3CE00 — a plausible dev0 async-pool VA, and the SAME across processes because the
pool's VA reservation is deterministic at equal load points. Both draft arms read the
argmax output as the next chain token (graph: in-graph argmax -> `g_tok` -> next replay's
embed; eager: `argmax_token_device` -> `tok_d` -> `embed_gather_device_t`), so ONE
all-NaN draft-head logits row poisons the chain's next embed lookup identically in both
arms. This also explains stickiness (the MMU fault kills the context) and why
launch-blocking/sanitizer mask it (the NaN producer is a cross-stream race that
serialization removes).

What makes the logits NaN is now the question — h_seed (the verify's hidden handed to the
draft chain, produced INSIDE the stage-split verify and published via `publish_to`) is
the leading candidate operand; a WAR/UAF on spec scratch between two concurrent sessions
is the leading mechanism class.

## Round 4 — aborted (box contention)

The first FULL-coredump attempt timed out on the GPU flock (a co-tenant lane held a
long window: `probe-pp2-valley.sh`). Re-dispatched as round 5 with a 4h flock wait.

## Round 5 — operand readout blocked; EVT escape hatch is itself broken under PP-2 (raw/round5/)

- **G** (full coredump): reproduced (same `embed_gather_u32`, grid 13073, dev0), but
  cuda-gdb cannot read param memory even from the full dump
  (`CUDBG_ERROR_INVALID_MEMORY_ACCESS` on every `@parameter` read — the param-space
  readout appears broken for this driver/dump combination, it misattributes the param
  window to `fa_decode_vec_q_rows_v4_512_tb`), and global probes of the predicted table
  base come back `Cannot access memory`. Operand-level confirmation via the dump is a
  dead end on this toolchain; the VA arithmetic (exact to the byte, three processes)
  stands as the evidence.
- **H** (`MEMRA_EVT=1` x2): INSTANT hard failure, 0/8 at c=2 INCLUDING the solo warmup,
  wall 0.03s, quoted: `step error: DriverError(CUDA_ERROR_INVALID_CONTEXT, ...)`. No
  illegal address anywhere. The cross-stream event-tracking escape hatch is INCOMPATIBLE
  with the ppN multi-context runtime (cudarc records events in one context and waits on
  them in another) — the probe is INCONCLUSIVE for the race question, and the flag is a
  landmine for any future PP-2 debugging (documented here so nobody burns a day on it).

### The mechanism, restated precisely

The MMU fault needs NO stale mapping: `0x7FFFFFFF * 2304 = ~4.6 TB` past the table base
— a VA that was never mapped. The whole crash is: **draft-head logits row reads as
all-NaN under 2+ concurrent spec sessions over the PP-2 split; the device argmax's
sentinel (0x7FFFFFFF) survives every NaN comparison; both draft arms feed that token id
straight into the next chain step's embed gather; the gather dereferences
table+sentinel*row_bytes; MMU fault; context dies.** c=1 never trips because the NaN
producer needs a second session's interleaved GPU work.

Localization to the DRAFT chain (not the verify) is pinned by the faulting kernels:
sentinel-fed `embed_gather_u32{,_t}` at T=1/grid-16 exist only in the draft chain; a
verify-side sentinel would fault the T=K+1 gather (`embed_gather_device_td`) or panic
the d2t map indexing first.

## Round 6 — the traps fired, and they change the game (raw/round6/)

Four sentinel traps landed (32d86e21: greedy graph, sampled graph, eager chain, verify
accept walk — each refuses `idx >= vocab` with quoted NaN counts instead of feeding
0x7FFFFFFF into `embed_row()`). Trap build, exact-A x3 reps:

| rep | c=2 | c=4 | recovery (c=1 x4 after) |
|---|---|---|---|
| T1 | 7/8 | **16/16** | 4/4 |
| T2 | 7/8 | **16/16** | 4/4 |
| T3 | 7/8 | **16/16** | 4/4 |

The trap line, quoted (identical in all three reps):

    step error: draft(graph) argmax sentinel 0x7fffffff >= d_vocab 248320 at round 0
    j=0 pos=296: round-seed NaN 4096/4096 — refusing to dereference the embed row (#87 trap)

What this establishes:

1. **The NaN theory is CONFIRMED at the operand level**: the draft-graph seed buffer
   read back 4096/4096 NaN — the ENTIRE [n_embd] hidden is NaN, round 0, j=0.
2. **Stickiness is architecturally gone with the trap in place**: one request errors
   (7/8 at c=2), everything else — including the previously-100%-fatal c=4 phase —
   serves. 48/48 requests after the trap vs 0/48 in the finding lane. The MMU fault WAS
   the whole process-killing mechanism; refusing the dereference contains the blast to
   the one poisoned burst.
3. `round 0, j=0, pos=296`: the poison arrives with the FIRST chain step of a burst —
   the seed was NaN BEFORE any drafting. The seed at round 0 comes from the pre-loop
   init: `decode_step_h(last_token)` (the INIT FEED, spec.rs:3750) or a continuation's
   carried `last_h`/`prompt_h` tail — all products of the STAGE-SPLIT trunk walk
   (`decode_step_h_ppn` / `prime_cache`) handed across streams.
4. 7/8+16/16 serving at a c=4 that used to be 0/48 ALSO means: with the fault contained,
   spec+PP-2 concurrency fundamentally works — the remaining bug is "one burst's
   entry hidden is NaN under concurrent session traffic", a much smaller animal.

Refined trap (746a14ea) re-running: g_seed (self-fed = head OUTPUT at j) vs h_seed_buf
(round INPUT, untouched since round-start copy) — discriminates poisoned-arrival from
head-compute-NaN. j=0 with g_seed=NaN could still be either (the replay had already run).
