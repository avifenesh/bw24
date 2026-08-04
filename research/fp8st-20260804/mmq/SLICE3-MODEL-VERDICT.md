# Slice 3 — model-level exactness verdict, per-block FP8 MMQ prefill kernel

Lane `lane/fp8-mmq`, 2026-08-04. Rig: RTX 5090 Laptop (sm_120a), nvcc 13.1, all runs under
`flock /tmp/gpu5090.lock`. Kernel: `crates/memra-engine/cu/mmq_fp8_blk.cu`, flag `MEMRA_FP8_MMQ=1`
(default OFF).

Checkpoint: `/data/ai-ml/hf-models/qwen3-1.7b-blk128fp8-synth` — the genuine block-128 FP8
safetensors dir ARM B' built from the local Qwen3-1.7B BF16 dir
(`research/fp8st-20260803/armb/make_blk128_fp8_ckpt.py`): 196 2-D Linear weights as `F8_E4M3`
codes + BF16 `weight_scale_inv` of shape `[ceil(out/128), ceil(in/128)]`, per-block `s = amax/448`.
Dynamic range varies block to block — the property ARM A's global fold destroys. 2.7 GB, dense arch.

Harness: `crates/memra-engine/src/bin/fp8_mmq_stream.rs` (new). `run-gen` could not be used: it
takes the HybridModel path only and panics `not a hybrid arch` on a dense checkpoint — the same wall
ARM B' hit (`research/fp8st-20260803/armb/rungen-gpu.log`). The new bin drives
`Model::forward_last` per step, i.e. re-prefills the growing sequence, so with the flag on **every**
projection GEMM of **every** step routes through the new tile (m = T >= 16 clears
`GEMM_M_THRESHOLD`). A decode-cache stream would run m=1 MMVQ and never touch the kernel.

## Which branch of the exactness bar applies: (b)

The free-running greedy 128-token streams DIFFER.

| arm | stream digest | vs floor |
|---|---|---|
| floor (no FP8 flags, Q8_0 requant) | `0x85b3dfe4e7f01ec9` | reference |
| ARM B' (`MEMRA_FP8_BLK_GPU=1`) | `0x85b3dfe4e7f01ec9` | **IDENTICAL** |
| MMQ (`MEMRA_FP8_MMQ=1`) | `0xe0b75a1beb37e253` | differs, first flip step 1 |
| ARM A (`MEMRA_FP8_FOLD=1 MEMRA_ST_E4M3=1`) | `0xe1ef37a71d360c9f` | differs |

So branch (b) of the bar is the operative one, exactly as the brief anticipated: per-block-FP8
arithmetic is not Q8_0-requant arithmetic. The real gate is therefore the kernel-check bit-identity
against a host reference of the SAME arithmetic (slice 1, `ALL GREEN`, log
`fp8-mmq-check-synth.log`), and the model level supplies the 6-draw sampled loop + quality sanity
below rather than stream identity.

ARM B' matching the floor bit-for-bit is not a coincidence and is load-bearing here: it is the
control proving the tape and the harness are arm-stable, so a difference in the MMQ column is the
kernel's arithmetic and not harness nondeterminism.

## Teacher-forced drift — the measurement that is actually comparable

Free-running streams that flip once at a near-tie are incomparable afterwards: every later position
sees a different prefix, so a raw stream diff cannot separate "the arithmetic drifted badly" from
"one 0.26-logit tie flipped and rerouted the rest". `MEMRA_FP8_MMQ_TF=<log>` appends the id from the
floor's tape instead of the arm's own argmax, so both arms see **bit-identical inputs at every
position** and each disagreement is attributable to that position's arithmetic alone.

Step-0 logits (n = 151936, `rms(ref)` = 5.7945), `logit-drift.log` / `drift-table.log`:

| arm | max_abs | rms(diff) | rms(diff)/rms(ref) | differing f32 | top-1 | top-10 |
|---|---|---|---|---|---|---|
| ARM B' | 0.0000e0 | 0.0000e0 | 0.000e0 | 0/151936 | same | 10/10 |
| **MMQ** | 6.5074e-1 | 1.6531e-1 | **2.853e-2** | 151936/151936 | same | 10/10 |
| ARM A | 1.0573e0 | 2.3956e-1 | 4.134e-2 | 151936/151936 | **flipped** | 9/10 |

Teacher-forced argmax disagreements over the 128-step tape:

| arm | disagreements |
|---|---|
| floor (self-reproduction control) | 0/128 |
| ARM B' | 0/128 |
| **MMQ** | **3/128** (steps 1, 23, 67) |
| ARM A | 7/128 |

MMQ drift is 1.45x smaller than ARM A's on rms and 1.63x smaller on max_abs, with 3 disagreements
vs 7, and MMQ keeps the floor's top-1 and top-10 at step 0 where ARM A already loses top-1. That
ordering is the expected one from the mechanism: MMQ re-quantizes **nothing** on the weight side
(the checkpoint's e4m3 bytes ARE the MMA A operand and each block keeps its own f32 scale), so its
entire residual is the activation path (per-32 e4m3, `d = amax/448`) — the same activation format
the existing f8f4/w4a8 kernels use, not a new one. ARM A additionally destroys per-block weight
dynamic range.

## 6-draw sampled loop

`sixdraw.sh`, 6 distinct prompts (chat opener, arithmetic, code, prose, numbered list,
repetition-prone tail), each >= 16 tokens so m clears the threshold from step 0, 32 steps each,
floor tape then MMQ teacher-forced on it. Raw logs in `sixdraw/`, summary `sixdraw.out`,
disagreement lines `sixdraw-disagreements.log`.

| draw | disagreements |
|---|---|
| 1 | 0/32 |
| 2 | 0/32 |
| 3 | 0/32 |
| 4 | 1/32 (step 22) |
| 5 | 0/32 |
| 6 | 0/32 |

**1/192 positions** across the six draws.

## Every divergence is a near-tie, not a mis-ranking

The broken-kernel signature would be the arm picking tokens the floor ranked far down. Instead all
four disagreements (3 on the 128-tape + 1 in the 6-draw loop) land where the floor's own top-1/top-2
margin is in the bottom decile of that tape's margin distribution (`flip-margins.log`):

floor tape margins: n=128, min 0.0235, p10 0.2314, median 3.6871, max 12.1490.

| flip | floor margin | percentile of the tape's margins |
|---|---|---|
| step 1 (128-tape) | 0.263948 | 10.9th |
| step 23 (128-tape) | 0.023468 | 0.8th |
| step 67 (128-tape) | 0.171801 | 7.0th |
| draw 4 step 22 | 0.098108 | 2.3th |

Largest flip margin 0.2639 = 0.072x the tape median, and within 1.60 sigma of the measured step-0
logit `rms(diff)` of 1.6531e-1. Every flip is exactly what a drift of that magnitude must produce at
a tie that tight; none of them is a wide-margin flip, which is the class that cannot come from
FP composition. Same law the batched-prime gate already uses (`forward.rs`, gate #46).

## Quality sanity

Mean token NLL over a frozen real-text window, one prefill, `nll-window.txt` (1024 tokens): real
held-out GSM8K test prose from the local parquet cache. Deliberately NOT the floor's own greedy
output — an NLL on that would reward whichever arm reproduces the floor, i.e. the thing under test.
Every arm reads byte-identical input. Logs `nll-*.log`.

| arm | mean_nll | ppl | delta vs floor |
|---|---|---|---|
| floor | 1.280530 | 3.598545 | — |
| ARM B' | 1.280530 | 3.598545 | 0.000000 |
| **MMQ** | **1.281883** | **3.603419** | **+0.001353 (+0.11%)** |
| ARM A | 1.265222 | 3.543880 | -0.015308 |

MMQ costs +0.0014 nats — 0.11% perplexity, indistinguishable from the floor at this window size.
ARM A scores *below* the floor on this single window; that is a single-window artifact and is
labeled as such, not read as ARM A being better. It does mean this instrument, at N=1 window, has
no power to separate arms at the ~0.015-nat scale, so it is reported as a sanity check (MMQ did not
break the model) and NOT as a quality ranking. The teacher-forced drift table is the discriminating
measurement.

## Verdict

**PASS on branch (b).** The mandatory gate — kernel bit-identity vs a host reference of the same
arithmetic — is green from slice 1 (9 shapes including ragged n and k tails, 0 differing bits,
with a committed negative control that fails at O(1) rms_rel). At the model level: MMQ's greedy
stream is not bit-identical to the Q8_0 floor, as the bar allows and the mechanism requires; the
divergence is 3/128 and 1/192 positions, all in the bottom decile of margins and all within
1.6 sigma of the measured logit drift; step-0 top-1 and top-10 match the floor exactly; drift is
strictly smaller than the already-rejected ARM A on every measure; and NLL moves +0.11%.

Honest note, per the bar: per-block-FP8 arithmetic != Q8_0-requant arithmetic. This arm is not a
bit-exact drop-in for the floor stream and must not be described as one. Its exactness claim is
scoped to what was proven: the kernel computes its own arithmetic exactly, and that arithmetic's
model-level residual is confined to the activation quantizer shared with the existing f8f4/w4a8
path.

Scope limit: the 27B ST checkpoint battery is not in this file — the only block-128-grid checkpoint
on this rig is the 1.7B one above (`nvidia-qwen36-27b-nvfp4` ships per-tensor `weight_scale`, not a
grid; the 2026-08-03 scan found 0 local dirs with a grid). The 27B-class run needs the vast box's
`/root/models`.

## Files

| file | what |
|---|---|
| `stream-battery.sh` | the 6-run stream + NLL battery (floor / ARM B' / MMQ / ARM A, free + teacher-forced) |
| `stream-battery.out` | its rc lines |
| `stream-{floor,armbprime,fp8mmq,arma}.log` | free-running 128-step greedy, per arm |
| `stream-tf-{floor,armbprime,mmq,arma}.log` | teacher-forced on the floor tape, per arm |
| `logits-{floor,armbprime,mmq,arma}.bin` | raw step-0 logit vectors (LE f32, 151936 each) |
| `logit-drift.log`, `drift-table.log` | the drift statistics computed from those |
| `nll-window.txt`, `nll-*.log` | the frozen quality window + per-arm NLL |
| `sixdraw.sh`, `sixdraw.out`, `sixdraw/`, `sixdraw-disagreements.log` | the 6-draw loop |
| `flip-margins.log` | flip margins vs the tape's margin distribution |
| `fp8-mmq-check-synth.log`, `negative-control.log` | slice-1 kernel gate + its negative control |
