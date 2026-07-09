# rebaseline2 — 2026-07-09 full re-baseline (post sampled-spec)

Protocol: N=2 per arm per cell, medians. Pairs interleaved per model within one hour-regime.
Idle-gated (<1000 MHz SM) before every block; GPU state logged per run. Raw logs:
`research/tune-data/rebaseline-logs/`.

Cells: plain tg128 @d512 / @d6257 (`decode-bench` vs `llama-bench -fa 1 -ctk q8_0 -ctv q5_1`,
`GGML_CUDA_GRAPH_OPT=1`); spec p1/p2 GREEDY temp-0 (`run-spec` NGEN=256 vs `llama-spec-round.sh`
serve protocol); spec p3 SAMPLED (p3-agentic-long-v3, temp 0.7, seed 42, chat-templated;
llama via /v1/chat/completions temperature 0.7 max_tokens 256).

## Table (tok/s, median of 2; ratio = bw24/llama)

### A — Qwen3.5-9B NVFP4 GGUF

| cell | bw24 | llama | ratio |
|------|------|-------|-------|
| plain d512 | 135.7 | 126.7 | 1.07x |
| plain d6257 | 127.2 | 120.0 | 1.06x |
| spec p1 greedy | 238.9 | 123.6 | 1.93x |
| spec p2 greedy | 210.2 | 122.6 | 1.71x |
| spec p3 sampled | 212.0 | 119.2 | 1.78x |

bw24 spec config: K=3 pmin=0.3 trim=`frspec-9b-32768.gguf` (9B-native trim FOUND on disk;
initial pass ran UNTRIMMED by mistake — untrimmed numbers p1 199.8 / p2 176.7 / p3 177.7 kept in
`A-9b-gguf-spec-p{1,2,3}*.log` for reference, table rows are the TRIMMED re-run,
`A-9b-gguf-spec-TRIMMED.log`). llama = serve-best raw decode (no 9B MTP draft exists), same-hour
rounds. Acceptance: p1 67.7%, p2 68.1%, p3 81.0%. p3 gate: PASS (seeded rerun identical) x2.

### B — Qwen3.5-9B NVFP4 ST (modelopt) — llama fields the 9B GGUF (A's file)

| cell | bw24 | llama | ratio |
|------|------|-------|-------|
| plain d512 | 132.0 | 126.5 | 1.04x |
| plain d6257 | 124.0 | 120.4 | 1.03x |
| spec p1 greedy | 193.8 | 123.6 | 1.57x |
| spec p2 greedy | 193.0 | 122.6 | 1.57x |
| spec p3 sampled | 158.7 | 119.2 | 1.33x |

bw24 spec config: K=2 pmin=0.3 trim=`frspec-9bst-modelopt-32768.gguf`.
Acceptance: p1 68.1%, p2 74.0%, p3 66.7%. p3 gate: PASS (seeded rerun identical) x2.
NOTE: B bw24 spec cells ran ~1h before the llama serve rounds (thermal states comparable,
1845-2137 MHz / 62-70 C both arms) — hour-regime pairing is looser here than for A/C/D.

### C — Qwen3.6-27B NVFP4 GGUF

| cell | bw24 | llama | ratio |
|------|------|-------|-------|
| plain d512 | 48.4 | 44.9 | 1.08x |
| plain d6257 | 45.9 | 43.0 | 1.07x |
| spec p1 greedy | 104.1 | 88.7 | 1.17x |
| spec p2 greedy | 88.8 | 93.8 | 0.95x |
| spec p3 sampled | 103.5 | 93.4 | 1.11x |

bw24 spec config: K=3 pmin=0.4, MTP draft `mtp-Qwen3.6-27B-Q4_K_M.gguf`, trim
`mtp-Qwen3.6-27B-Q4_K_M-frspec-code75-32768.gguf`. llama: MTP serve (draft-mtp n-max 3
p-min 0.1). Acceptance: p1 77.1%, p2 68.3%, p3 91.1%. p3 gate: PASS (seeded rerun identical) x2.
llama p3 sampled spread 89.89/96.88 (7.5%) — sampled llama arm is regime-noisy.

### D — Qwen3.6-27B NVFP4 ST (NVIDIA official, BW24_NV_W4=1) — llama fields the 27B GGUF (C's file)

| cell | bw24 | llama | ratio |
|------|------|-------|-------|
| plain d512 | 48.9 | 44.5 | 1.10x |
| plain d6257 | 46.6 | 42.9 | 1.09x |
| spec p1 greedy | 98.7 | 86.4 | 1.14x |
| spec p2 greedy | 89.8 | 91.7 | 0.98x |
| spec p3 sampled | 83.8 | 99.3 | 0.84x |

bw24 spec config: NV_W4=1 K=3 HPOST=1, pmin p1=0.4 / p2=0.3 / p3=0.3 (brief gave 0.4/0.3
per-content; p3 pmin=0.3 chosen and noted), trim `frspec-corpus-32768.gguf`.
Acceptance: p1 70.7%, p2 68.2%, p3 72.2%. p3 gate: PASS (seeded rerun identical) x2.
**TEXT-AUDIT FLAG (p3): bw24 sampled continuation degenerates into a "Wait, the trace says:"
repetition loop from ~token 60 even at temp 0.7 seed 42** (see `D-27b-st-spec.log` sampled-text
block). Cell reported per protocol, flagged, not excluded.

### E — Qwen3.6-35B-A3B MoE IQ4_XS

| cell | bw24 | llama | ratio |
|------|------|-------|-------|
| plain d512 | 178.2 | 167.8 | 1.06x |
| plain d6257 | 163.7 | 160.9 | 1.02x |
| spec p1 greedy | 280.4 | 251.9 | 1.11x |
| spec p2 greedy | 226.0 | 221.4 | 1.02x |
| spec p3 sampled | 255.7 | 248.9 | 1.03x |

bw24 spec config: K=3 pmin=0.4 PMIN0=1 trim=`mtp-Qwen3.6-27B-Q4_K_M-frspec32768.gguf` (generic
vocab trim, transfers per 2026-07-08 row). llama: self-MTP (embedded NextN, NO -md; passing the
model as -md OOMs — captured "unable to allocate CUDA0 buffer"), --spec-type draft-mtp
--spec-draft-n-max 2 --spec-draft-p-min 0.1; MTP engagement verified via draft_n/draft_n_accepted
in timings (p2: 202/153). llama p1 uses ignore_eos=true (raw p1 prompt EOSes at 1 token,
known from 2026-07-08 row). Acceptance bw24: p1 79.0%, p2 70.9%, p3 88.9%. p3 gate: PASS
(seeded rerun identical) x2.

## Text-audit verdicts (p3 sampled, BW24_PRINT_TEXT=1)

- A (9B GGUF): coherent crash analysis; one mojibake token ("line cé203"); NO loop.
- B (9B ST): coherent; NO loop.
- C (27B GGUF): coherent structured thinking-process; NO loop.
- D (27B ST): **REPETITION LOOP** ("Wait, the trace says:" x10+ from ~token 60). FLAGGED.
- E (35B): coherent structured thinking-process; NO loop.

## Anomalies

1. Model A initial spec pass ran untrimmed (trim `frspec-9b-32768.gguf` exists on disk but no
   explicit 9B-GGUF FRSPEC_TRIM path appeared in the latest 9B GGUF JSONL rows); re-run trimmed,
   both sets logged. Trim is worth +19-20% on every prompt (199.8→238.9 p1 etc).
2. First 27B llama-spec attempt (hand-rolled HTTP client in a background heredoc) hung with the
   server healthy; killed, replaced with the proven `llama-spec-round.sh` + curl-file-payload
   route per owner instruction. No numbers were taken from the hung path.
3. llama sampled p3 arm shows 7-13% run-to-run spread (89.9-102.1 across C/D pairings, same
   server config) — sampled chat requests on llama are noisier than its greedy /completion path.
4. B's hour-regime pairing is looser (see B section note).
5. 35B llama self-MTP with `-md <same file>` OOMs (double model load); running WITHOUT -md
   engages the embedded NextN head (draft_n present in timings) — this is the working self-MTP
   config on 24 GB.
6. llama 35B p1 needed `ignore_eos: true` (raw prompt EOSes at 1 token); bw24 p1 does not EOS
   under run-spec, so the p1 pair compares 256-token generations on both sides but llama's is
   EOS-suppressed continuation.
7. D spec p3 pmin: brief specified per-content pmin only for p1 (0.4) / p2 (0.3); p3 sampled ran
   at pmin=0.3 (choice documented, not swept).
8. C plain d512 llama run-spread 44.27/45.48 (2.7%) — largest plain-cell spread of the session;
   all other plain cells <=1%.
