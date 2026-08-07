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

## Round 2 — next: name the kernel

`CUDA_LAUNCH_BLOCKING` and the sanitizer both perturb timing too much; coredump-on-
exception does not (async until the trap, then the driver dumps the faulting kernel).
Running: `run-pp2crash-coredump.sh` — c=4 bare repro with
`CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1` + lightweight dump + `cuda-gdb` batch readout.
