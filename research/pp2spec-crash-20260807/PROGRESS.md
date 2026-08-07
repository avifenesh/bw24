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

## Round 4 — running: FULL coredump for operand readout

Lightweight dumps omit param memory. The full dump (`run-pp2crash-round4.sh`) reads the
faulting kernel's params (embd*, token_d*), token_d[0], and probes whether the fault VA
falls inside the live table mapping.
