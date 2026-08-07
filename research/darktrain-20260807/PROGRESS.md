# lane/darklane-training — dead-darklane background jobs, engine machinery (2026-08-07)

The FIRST lane of the darklane-training program. Owner thesis: idle serve capacity carries
owner research/training jobs, yielding instantly to paying traffic. This lane ships the
engine-side skeleton with receipts: valley detection, a yield-first background job runner,
the VRAM budget contract, and the checkpoint/resume seam. Policy/economics = product repo.

Rig: local 5090 Laptop (24 GB), q9 Qwen3.5-9B NVFP4 + owntrim draft, all GPU windows under
`flock /tmp/gpu5090.lock`. Box (2x PRO 6000, PP-2) reserved for the final pair receipt.

## Deliverable 1 — valley detection (COMMITTED efff4b26)

No new sensor: worker truth already encodes idleness — the scheduler flips PHASE_IDLE
exactly when `active`+`queue` are empty and the phase stamp refreshes the heartbeat, so
`phase==IDLE` + beat age IS the idle duration at zero hot-path cost. `PENDING_ADMITS`
closes the HTTP→worker handoff gap. Surface: `/metrics serve_idle_seconds` (always
published) + the `ValleySignal` hook (`crates/memra-server/src/darklane.rs`); threshold
`MEMRA_VALLEY_S` (2.0 s, read-once). Asymmetric: `busy_now()` has no debounce,
`in_valley()` waits the full threshold.

Receipt `raw/valley-signal.log` (probe-valley-signal.sh): accrues 2.593→4.113 s idle;
0.0 sampled mid-flight inside a 2000-token generation; re-accrues fresh (2.02 s) after.
Harness trap caught by the probe's own first run: a 256-token budget completed in 1.2 s
(spec, 258 tok) BEFORE the 2 s mid-flight sample — budget resized to the measured rate.
Unit: `valley_signal_reads_worker_truth`. Suite 99/99 at the commit.

## Deliverable 2 — background job runner with yield

`MEMRA_BG_JOB` (sh -c, own process group, PDEATHSIG=KILL) + supervisor thread
(`memra-bg-runner`, poll `MEMRA_BG_POLL_MS`=25 ms). State machine: waiting → running →
(yielded ⇄ running) → done/failed, with preempted/refused_vram as re-entry states. Yield =
SIGSTOP to the group on the busy edge; resume = SIGCONT after a full valley. Exit protocol:
0 = complete (terminal), 75 = checkpointed (relaunch next valley), else failed (terminal,
loud). Drain path CONTs+TERMs (KILL past 2 s) so no SIGSTOPped orphan survives; PDEATHSIG
covers the SIGKILL path. `/metrics` gains a `bg` block ONLY when configured (pre-lane
payload byte-identical otherwise).

Receipts (see raw/ + RESULTS below): yield latency direct measurement + N=5 interleaved
serve-impact stress vs no-job baseline, c=8x16 streaming bursts, fresh boot per rep/arm.

## Deliverable 3 — GPU memory discipline

Fits-or-refused at launch: `MEMRA_BG_VRAM_MB` (0 = CPU-only) granted only while
`min free across visible GPUs >= budget + MEMRA_MOE_RESIDENT_HEADROOM_GB` (min not sum —
PP-2 pairs shard serve across both cards). Fail-closed on unreadable nvidia-smi. Refusal
is re-tried next valley (headroom moves as sessions retire). v1 enforces fit at launch;
runtime containment is the job's contract (documented in FLAGS.md) — no cgroup-style VRAM
enforcement exists to delegate to. Unit: `vram_budget_refused_when_it_does_not_fit_and_
fail_closed` (refuse, fail-closed, and fits-launches arms).

## Deliverable 4 — checkpoint/resume seam (training-class jobs)

`MEMRA_BG_YIELD_MODE=checkpoint`: SIGUSR1 = "checkpoint and exit 75"; relaunch next valley
resumes from the job's own file; `MEMRA_BG_CKPT_GRACE_MS` (5 s) then SIGKILL (dirty
preempt — at-least-once, never lost). Toy proof `tools/bg-ckpt-counter.py` (atomic
write-tmp-rename checkpoints; standalone run: preempted at step 14 → exit 75 → resumed at
14 → complete at 30, exit 0). Unit `checkpoint_mode_preempts_and_resumes_counter` pins the
full preempt→relaunch→resume-past-checkpoint cycle GPU-free.

## Deliverable 5 — docs

docs/SERVING.md "Dead-darklane background jobs" (engine mechanics; policy seam named),
docs/FLAGS.md rows: MEMRA_VALLEY_S, MEMRA_BG_JOB, MEMRA_BG_POLL_MS, MEMRA_BG_YIELD_MODE,
MEMRA_BG_CKPT_GRACE_MS, MEMRA_BG_VRAM_MB.

## RESULTS (filled from raw/ as runs land)

- valley-signal: PASS 3/3 checks (raw/valley-signal.log).
- bg-stress N=1 shakeout: yield 3.0 ms; p95 delta +3.17% on n=2 bursts/arm — noise-level
  N, superseded by the N=5 run (raw/bgstress-n5.log).
- bg-stress N=5 (raw/bgstress-n5.log, 20 burst points in bgstress-points.jsonl, yields in
  bgstress-yield.jsonl; single lock hold, interleaved base/bg per rep, one thermal regime):

  | metric        | base median (n=10) | bg median (n=10) | delta   |
  |---------------|--------------------|------------------|---------|
  | lat_p50_s     | 1.444              | 1.449            | +0.30%  |
  | lat_p95_s     | 2.478              | 2.497            | +0.77%  |
  | ttft_p50_s    | 0.146              | 0.135            | −7.05%  |
  | ttft_p95_s    | 0.514              | 0.520            | +1.11%  |
  | agg_tok_s     | 413.057            | 410.821          | −0.54%  |

  Yield wall (request fired → job /proc state 'T'): 19.4 ms median, 23.3 ms max, 2.7 ms
  min (N=5 — the spread IS the 25 ms poll interval; target <500 ms). Liveness every rep:
  launches=1 yields=3 resumes=2. STOP-bar verdict: +0.77% p95 ≤ 2% — WITHIN BAR; the
  mechanism ships (opt-in by construction: unset MEMRA_BG_JOB = no runner, no /metrics
  block, pre-lane payload byte-identical). (N=1 shakeout's +3.17% was n=2-bursts noise;
  the N=5 interleave is the citable number.)
- ckpt-serve (raw/ckpt-serve.log, REAL server, 5090): valley launch → one chat request
  preempts (SIGUSR1, preempts=1, ckpt_kills=0, yield signal 6 µs) → checkpoint holds 129 →
  next valley relaunches "resume from checkpoint" → counter continues 129→293. Trap found:
  under `sh -c` the shell parent dies of the unhandled SIGUSR1 before exit-75 propagates
  ("job exited None during preemption") — the during-preemption branch already treats that
  as checkpointed, and SERVING.md now says to `exec` single-command jobs.
- PP-2 box receipt (raw/pp2-valley.log + server logs; 2x PRO 6000, q27 NVFP4 over
  MEMRA_PP_DEVICES=0,1 + MEMRA_SERVE_SPEC=0, under flock /tmp/memra-gpu.lock behind the
  tick-seg hold): 7/7 ok — valley idle=5.725 s, job launched; ONE chat request flipped the
  job to /proc 'T' in **37.2 ms** while the PP-2 request completed normally (64 tok,
  1.05 s); next valley resumed (yields=1 resumes=1); the unfittable-budget arm
  (147252 MB vs min-free) landed `refused_vram` with the loud
  `[darklane] REFUSED ... min free Some(82514)MB` line — the min-across-GPUs term read the
  pair with q27 shards resident on both cards. Single cycle, labeled as such.

## Gates

- serve-smoke: 16/16 ok (raw/serve-smoke.log).
- bg-stress N=5: 0 failed, STOP-bar WITHIN (raw/bgstress-n5.log).
- kernel-check: ALL GREEN (raw/kernel-check-tail.log). First attempt OOM'd against a
  concurrent 14.6 GB gemma-gate tenant (quoted CUDA_ERROR_OUT_OF_MEMORY, compute-apps
  recorded in the log) — rerun clean after it exited. Zero engine/kernel files touched
  by this lane (`git diff --name-only e54dd2e6..HEAD`: crates/memra-server only).
- workspace tests: all crates green (memra-server 103/103; full `cargo test --workspace`).

## Follow-up lane should build

- Compose the runner with a REAL GPU training job (the checkpoint seam is proven on a toy;
  the first real consumer will find the VRAM-budget contract's edges — esp. allocator
  behavior of a framework that grabs the whole budget at import).
- In-process job API (the module doc's named seam) if/when process-level SIGSTOP cost or
  checkpoint latency becomes the binding constraint — measure first.
- Multi-job queue (v1 is single-job by design — one MEMRA_BG_JOB), priority within the
  background class, and the product-side policy loop consuming serve_idle_seconds.
- cgroup v2 / MPS-based runtime VRAM containment if the honest-job contract proves
  insufficient in practice.
