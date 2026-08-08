# REPRO — the 27B Q8_0 `board-2048` prefill-vs-decode argmax MISMATCH

Lane `lane/q8-argmax`, 2026-08-06. Task #77: the standing pre-existing exactness red carried
into the v0.71.0 release battery.

## What actually fails (and what does NOT)

The failing gate is **`run-gen`'s intra-run prefill-vs-tokenwise-decode argmax check**
(`crates/memra-engine/src/bin/run_gen.rs:880-896`) — *not* a cross-rig golden comparison and
*not* a fast-gate probe row. One binary, one rig, one process: it runs `forward_last`
(batched prefill) and a `decode_step` loop over the same prompt, then hard-asserts that the
two last-position argmaxes agree.

```
prefill argmax=332  decode argmax=485  logit maxdiff=4.659e-1  MISMATCH
[gate] prefill: l[332]=12.8352 l[485]=12.8045 | decode: l[332]=12.6235 l[485]=12.7234
thread 'main' panicked at crates/memra-engine/src/bin/run_gen.rs:896:5:
decode-step diverges from prefill — cache threading bug
```

Both sides' own view of both ids is printed, so the arithmetic is fully visible:

| side | l[332] | l[485] | its own top-2 margin |
|---|---|---|---|
| prefill (batched `forward_last`) | 12.8352 | 12.8045 | **0.0307** toward 332 |
| decode (tokenwise `decode_step`)  | 12.6235 | 12.7234 | **0.0999** toward 485 |

The two configs disagree by `maxdiff = 0.466` somewhere in the 248k-wide vocab, and the top-2
gap at the decision position is **0.031** — 15x smaller than the config spread. That is the
signature of a near-tie coin flip, not of a wrong number.

`maxdiff = 0.466` is **ordinary for this gate**: the same artifact reports MATCH at
maxdiff 0.34 / 0.41 / 0.47 / 0.52 / 0.65 / 0.88 on the other prompts of the same sweep
(`pod/inherited-v071/` prompt table below), and the repo-wide MATCH population runs to
2.4e0. maxdiff is NOT the discriminator; the top-2 margin is.

## Invocation (exact)

```
LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat \
MEMRA_NGEN=32 MEMRA_PROMPT_FILE=research/e2e/prompts/board-2048.txt \
  ./target/release/run-gen /root/models/Qwen3.6-27B-Q8_0.gguf
```

Rig: RunPod **RTX PRO 6000 Blackwell Workstation, 188 SM**, driver 570.211.01, CUDA 13.1
compat. Community pod — exactness rows only, no perf claimed anywhere in this lane.

## Inherited receipts (v0.71 prep lane, 2026-08-06) — reproduced here, not trusted blind

Byte-identical `prefill argmax=332 decode argmax=485 maxdiff=4.659e-1` across FOUR binaries
(v0.71 candidate, v0.70.0, v0.69.0, an 08-04 control) — pre-existing, not a release
regression. Raw: `pod/inherited-v071/`.

The v0.71 triage also already **refuted the k27 `fa_split_keys` rung** on the short default
prompt: `MEMRA_FA_SPLIT=8`, `=16`, `=1` and an older tree all print the *same* line
(`271` vs `1178`, maxdiff 4.234e-1) to every printed digit. This lane re-runs the split arms
on the *board-2048* prompt (the one that actually fails the release battery) because the
triage's split arms used run-gen's default 55-token prompt, a different decision position.

## Prompt-sweep context (same binary, same Q8_0 artifact, 188-SM pod)

| prompt | verdict | maxdiff |
|---|---|---|
| **board-2048** | **MISMATCH** (332 vs 485, margin 0.031) | 4.659e-1 |
| p1-code-short | MATCH | 3.761e-1 |
| p2-code-medium | MATCH | 4.085e-1 |
| p3-agentic-long / -v2 / -v3 | MATCH | 8.800e-1 / 6.500e-1 / 5.168e-1 |
| p4-16k / p5-32k | MATCH | 4.819e-1 / 4.665e-1 |
| pp512 / pp2048 | MATCH | 5.902e-1 / 3.439e-1 |

One prompt in eleven. The failing one is the only one whose last-position top-2 sit inside
the config spread.

## The 2026-07-31 precedent — SAME token pair, DIFFERENT rig, DIFFERENT model

`research/tune-data/h100board-vllm-20260731-realtext-logs/q9-memra.log` (H100 lane, five
consecutive runs) on the **same board-2048 prompt text** against a **9B Q8_0**:

```
prefill argmax=485  decode argmax=332  logit maxdiff=6.748e-1  MISMATCH
[gate] prefill: l[485]=14.2880 l[332]=14.2842 | decode: l[485]=14.1897 l[332]=14.2719
```

Prefill margin **0.0038**. Same `{332, 485}` pair, opposite orientation, on sm_90a with a
different model size and a different toolchain. The near-tie is a property of **this prompt
text at its final position**, not of the 188-SM pod, not of the 27B artifact, and not of any
kernel this lane could fix.
