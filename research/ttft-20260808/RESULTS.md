# TTFT anatomy and kill - 2026-08-08

## Verdict

The TTFT performance targets pass on the integrated PP-2 train:

- 228-token short turn: **0.589 s p50**, N=8 (target <0.8 s).
- 4107-token rendered 4k turn: **5.992 s p50**, N=5 (target <=7.5 s).

The lane is nevertheless **STOPPED, not releasable**. The final `serve-smoke` battery ended
with `2 failed`: the session-affinity replay was deterministic, but only one arm exercised a
rewind. Per the owner stop bar, no further GPU experiments, merge, tag, or push followed.

## Rig and protocol

The requested box2 address was unavailable: SSH timed out and the EC2 inventory observed during
the run showed `darklanes-sprint-pair2` terminated; the final recheck no longer listed it.
Measurements therefore used the designated fallback box1, the same `g7e.12xlarge` PP-2 class,
at `18.195.123.14`.

- GPUs: 2x NVIDIA RTX PRO 6000 Blackwell Server Edition, PP devices `0,1`.
- Model: Step-3.7-Flash IQ4_XS shards plus the external Q8_0 MTP draft.
- Serving measurements: spec off, sequential requests, unique cache namespaces, one excluded
  warmup, then N=8 short and N=5 4k.
- Every accepted run began with 0 MiB GPU memory used. Thermal start/end values and binary/model
  hashes are in each `summary.log`.
- Prompt file SHA-256:
  `23c1d8384a16c7c0bcb7736b412d43e64c0b4d8e238703864e928565f824ae11`.

## Phase anatomy

All values below are p50 milliseconds. `first SSE` is the handoff from first decoded token to the
first serialized SSE `data:` frame; keepalive comments are excluded.

### Before the solo-prefill fix (`c5e26522`)

| shape | N | client TTFT | parse | admission | queue | tokenize | prime wait | prime | decode wait | first SSE | server total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short, 228 tok | 8 | 588.116 | 0.013 | 0.011 | 0.033 | 0.662 | 5.500 | 581.471 | 0.135 | 0.023 | 587.929 |
| 4k, 4107 tok | 5 | 7117.664 | 0.044 | 0.014 | 0.037 | 8.882 | 1.178 | 7107.127 | 0.135 | 0.027 | 7117.486 |

The suspected fixed serve-layer cost is not present on the current train. Excluding prime,
server work is 6.458 ms on the short request and 10.359 ms on 4k. Queue wait is 0.033-0.037 ms,
and first-SSE handoff is 0.023-0.027 ms. Prime compute owns 98.9% of short server TTFT and
99.9% of 4k server TTFT.

### Causal controls

| arm | short TTFT | short prime | 4k TTFT | 4k prime | conclusion |
|---|---:|---:|---:|---:|---|
| grouped default, tick 1024 | 588.116 | 581.471 | 7117.664 | 7107.127 | integrated baseline |
| `MEMRA_MOE_GROUPED=0` | 934.811 | 928.069 | 10855.249 | 10844.910 | serve is using Lever C; rollback costs 0.347 s short / 3.738 s 4k |
| `MEMRA_PREFILL_TICK=8192` | 589.722 | 582.774 | 5995.418 | 5985.078 | remaining 4k loss is outer 1024-token call segmentation |

Lever C is live in serving. The remaining divergence from the standalone grouped-prefill rate was
four scheduler-level `prime_cache` calls, each choosing geometry from its 1024-token slice. One
outer call restores the engine's eight-microbatch geometry and reaches 686 tok/s over the rendered
4107-token prompt.

## Fix

Commit `c09afe4c` widens only a naked, sole, fresh interactive request to one bounded outer
prefill call of at most 8192 tokens. It does not widen when another unfinished session or queued
request exists. An explicit `MEMRA_PREFILL_TICK` remains authoritative, and concurrent sessions
retain the existing 1024-token fairness cap. A remainder below `PRIME_MIN_T` is merged rather
than stranded on tokenwise prefill.

### After (`c09afe4c`)

| shape | N | client TTFT | parse | admission | queue | tokenize | prime wait | prime | decode wait | first SSE | server total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short, 228 tok | 8 | 589.238 | 0.015 | 0.012 | 0.045 | 0.694 | 5.743 | 582.295 | 0.136 | 0.029 | 589.053 |
| 4k, 4107 tok | 5 | 5991.509 | 0.051 | 0.019 | 0.044 | 8.769 | 1.078 | 5981.119 | 0.141 | 0.039 | 5991.215 |

| shape | before p50 | after p50 | delta | target | result |
|---|---:|---:|---:|---:|---|
| short | 0.588 s | 0.589 s | +0.001 s | <0.8 s | PASS |
| 4k | 7.118 s | 5.992 s | -1.126 s (-15.8%) | <=7.5 s | PASS |

## Gates

| gate | verdict |
|---|---|
| local `memra-server` unit suite | 122 passed, 0 failed |
| model-backed `kernel-check` | ALL GREEN |
| PP-2 `run-gen` | prefill/decode and batched-prime/tokenwise argmax 6776 MATCH |
| PP-2 `run-spec` K=1..8 | 8/8 self-consistency PASS; pinned acceptance pattern retained |
| `serve-smoke` | **FAIL: 2 failed**; affinity replay did not exercise the required rewind |
| overall | **STOP** |

## Commits

- `9ea793cb feat(serve): trace per-request TTFT phases`
- `9a264c76 perf(serve): merge grouped Step prefill`
- `6b3544a9 fix(serve): trace completion routes only`
- `6c3cf7a7 test(serve): add TTFT anatomy harness`
- `c5e26522 fix(serve): exclude SSE keepalives from TTFT`
- `0fef5ff5 test(serve): parameterize TTFT control arms`
- `c09afe4c perf(serve): widen solo fresh prefill`

## Receipts

- `raw/baseline-fixed/`: before N=8/N=5 clients, joined rows, full 15-line phase trace.
- `raw/grouped-off/`: Lever C rollback control and full phase trace.
- `raw/prefill-tick8192/`: geometry control and full phase trace.
- `raw/solo-prefill-after/`: committed default-after N=8/N=5 receipt and full phase trace.
- `raw/exactness/`: kernel-check, run-gen, and run-spec raw logs plus combined summary.
- `raw/serve-smoke/`: final red smoke stdout and retained final server log.
- `raw/box2-unavailable.log`: final timeout and account-inventory recheck for the requested rig.
