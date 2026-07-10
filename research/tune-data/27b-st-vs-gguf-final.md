# 27B ST vs GGUF — sampled-p3 robustness matrix (2026-07-10)

Scope note: this lane started as a full same-hour cell1-5 comparison; the owner cut scope
mid-run to focus on the decision-critical piece — is the ST checkpoint's long-context
sampled degeneration systematic or a single-draw artifact? Cells 1-4 (plain decode, pp-only,
greedy spec p1/p2) were **not run to completion** this session (partial cell-1 logs from the
aborted full plan are kept, unfinished, in `27b-final-logs/cell1-*.log` — do not cite them as
results). Everything below is the sampled-p3 matrix only. Both arms are bw24 (no llama).

## Rig / protocol

- RTX 5090 laptop (24GB), idle-gated before every block (`clocks.sm` 172-180 MHz, `free -g`
  available 36-38GB throughout).
- bw24 @ `2a47133` (`decode-bench` / `run-gen` / `run-spec` release binaries).
- Sequencing: all 6 configs per arm run back-to-back (GGUF block first, then ST block) rather
  than interleaved — loop verdict is a content property (not thermal-sensitive); tok/s here is
  secondary to the loop question.
- Sampled mode: `BW24_SPEC_TEMP=0.7`, `BW24_SEED∈{42,7,1234}`, `BW24_CHAT=1`, `BW24_NGEN=256`,
  `BW24_PRINT_TEXT=1`, N=1 per seed (seeded sampled runs are deterministic — `run-spec` itself
  reruns `generate_spec` a second time internally and asserts token-identity, so N=1 per seed is
  exact, not a single noisy draw).
- Prompts: `research/e2e/prompts/p3-agentic-long-v3.txt` (agentic, 5420 tok) and
  `/data/projects/hqmtp/eval/prompts/pe4-16k.txt` (16k eval prompt, 17224 tok after chat-template).
- Loop verdict: mechanical, not judged — every sampled-text block was extracted and checked for
  (a) any non-blank line repeated ≥4x verbatim, or (b) any sentence-level phrase (split on
  `.`/`!`/`?`/newline, ≥15 chars) repeated ≥6x, via `sort | uniq -c` plus a Python
  `collections.Counter` pass (line-only `uniq -c` under-counts the ST arm, which emits
  long un-wrapped paragraphs with few newlines — phrase-splitting was needed to catch loops
  that live inside one physical line).
- Configs used (per ARMS spec in the task): GGUF = `BW24_SPEC_K=3 BW24_SPEC_PMIN=0.4
  BW24_MTP_DRAFT=mtp-Qwen3.6-27B-Q4_K_M.gguf BW24_FRSPEC_TRIM=...frspec-code75-32768.gguf`.
  ST = `BW24_NV_W4=1 BW24_MMQ_F8F4=1 BW24_SPEC_K=3 BW24_SPEC_HPOST=1 BW24_SPEC_PMIN=0.4
  BW24_FRSPEC_TRIM=...frspec-corpus-32768.gguf` (PMIN=0.4 used for `pe4-16k` too — it isn't one
  of the named p1/p2/p3 prompts, so it was grouped with the p1/p3 PMIN value as the closer
  analog to agentic-long content; this is a judgment call, flagged here).
- Every run's self-consistency gate line read exactly `PASS (seeded rerun identical)` — 12/12.

## Loop matrix (arm x prompt x seed)

| arm | prompt | seed | verdict | gen tok/s | spec tok/s (ratio) | prime (TTFT) | acceptance |
|---|---|---|---|---|---|---|---|
| GGUF | p3-agentic-long-v3 | 42 | coherent | 45.93 | 100.67 (2.19x) | 4.048s | 185/203 = 91.1% |
| GGUF | p3-agentic-long-v3 | 7 | coherent | 45.91 | 97.32 (2.12x) | 4.064s | 181/201 = 90.0% |
| GGUF | p3-agentic-long-v3 | 1234 | coherent | 45.74 | 97.01 (2.12x) | 4.084s | 179/208 = 86.1% |
| GGUF | pe4-16k | 42 | coherent | 41.51 | 65.06 (1.57x) | 14.030s | 158/221 = 71.5% |
| GGUF | pe4-16k | 7 | coherent | 41.49 | 63.28 (1.53x) | 14.021s | 152/232 = 65.5% |
| GGUF | pe4-16k | 1234 | coherent | 41.37 | 62.31 (1.51x) | 14.092s | 152/232 = 65.5% |
| ST | p3-agentic-long-v3 | **42** | **LOOP** | 46.63 | 84.44 (1.81x) | 3.841s | 166/211 = 78.7% |
| ST | p3-agentic-long-v3 | 7 | coherent | 46.61 | 79.84 (1.71x) | 3.846s | 161/213 = 75.6% |
| ST | p3-agentic-long-v3 | 1234 | coherent | 46.65 | 77.80 (1.67x) | 3.841s | 156/214 = 72.9% |
| ST | pe4-16k | 42 | coherent | 42.16 | 68.56 (1.63x) | 13.257s | 158/199 = 79.4% |
| ST | pe4-16k | 7 | coherent | 42.19 | 67.00 (1.59x) | 13.289s | 161/217 = 74.2% |
| ST | pe4-16k | **1234** | **LOOP** | 42.18 | 74.29 (1.76x) | 13.296s | 170/199 = 85.4% |

GGUF: **6/6 coherent.** ST: **4/6 coherent, 2/6 LOOP** (one on each prompt, different seeds:
p3/seed42 and pe4-16k/seed1234). No OOM on the ST arm at 16k context — both arms handled the
17224-token prompt without incident.

## Loop-case text quotes (mechanical evidence)

**ST / p3-agentic-long-v3 / seed 42** — line `- Wait, let's look at the actual text carefully.`
repeats 10x verbatim:
```
1.  **Analyze User Input:**
   - The user provides a large block of text that appears to be a mix of Python code snippets, documentation, and error traces.
   - The core of the prompt seems to be a Python traceback/error related to a `selectors` module célection.
   - The error seems to be a Python traceback/error related to a `selectors` module.
   - The user asks to diagnose the crash.
   - The user asks to diagnose the crash.
   - Wait, let's look at the actual text carefully.
   - Wait, let's look at the actual text carefully.
   [... x10 total, then run ends mid-clause]
```

**ST / pe4-16k / seed 1234** — phrase `Explain each with a concrete failure scenario.` repeats
11x inside one paragraph (no line breaks — this is why phrase-splitting was necessary, not just
`sort | uniq -c` on lines):
```
Here's a thinking process:

1.  **Analyze User Input:**
   - The user provided a large chunk of Python code that appears to be a mix of standard library code (likely `argparse` or a custom parser implementation) mixed with some custom/modified code. Wait, looking closely, it's actually a mix of standard library code and some custom/modified code. [repeated once more]
   - Wait, I need to identify the three most likely sources of the three most likely sources of runtime bugs in this code.
   - Wait, the prompt says: "Review the code above. Identify the three most likely sources of the three most likely sources of the three most likely sources of runtime bugs in the provided code. Explain each with a concrete failure scenario. Explain each with a concrete failure scenario. [... x11 total]
```

Both loop cases show the same signature: an early meta-commentary/self-correction phrase
("Wait, ...") that starts repeating and then locks into a short cycle, cutting off the actual
task response before it starts. GGUF never exhibits this in any of the 6 draws.

## Verdict (report only — decision is the owner's)

- **Not fully systematic, but not a single-draw fluke either.** The ST checkpoint loops on
  2 of 6 sampled draws (33%), spread across both prompts and different seeds — this is a
  recurring failure mode of the ST checkpoint under `temp=0.7` long-context sampling, not
  something tied to one specific prompt or one specific seed.
- **GGUF shows zero loops across the identical 6-draw matrix** (same prompts, same seeds, same
  temp). This is the sharpest signal in the data: same sampling protocol, same content, GGUF
  arm never degenerates, ST arm degenerates 1/3 of the time.
- Acceptance and tok/s are not the discriminator here — the LOOP seed on `pe4-16k` (seed 1234,
  85.4% acceptance) actually has the **highest** acceptance of the three `pe4-16k` seeds; the
  loop is not signaled by low acceptance in this data. Self-consistency gates (argmax/seeded
  rerun) do not catch this either — a looping generation reproduces identically on rerun (it's
  deterministic given seed), so the gate says PASS while the content is degenerate.
- Raw logs (full run output, all 12 configs): `research/tune-data/27b-final-logs/p3matrix-*.log`.
  JSONL (one row per arm/prompt/seed, tag `27b-p3-matrix-2026-07-10`):
  `research/tune-data/27b-p3-matrix-2026-07-10.jsonl`.

## Anomaly (infra, not a data-quality flag on the numbers above)

The first ST attempt (p3/seed42) was killed by a 5-minute shell timeout — **below the
documented ~8-minute cold-load floor for this arm** (my error: I under-shot the ops note).
After the kill, `nvidia-smi` reported a stuck `2805 MHz` / `100% util` / `72W` state with
**no owning process** (`fuser`/`lsof`/`pmon` on `/dev/nvidia0` showed only `llama-server`
(BGE, untouched) and desktop compositor — confirmed via `journalctl`/`dmesg`, no Xid/driver
crash logged) for ~10 minutes straight, unchanging to the decimal. A real `decode-bench` probe
run during this window returned the same clean number as before the incident (48.1 vs 48.2
tok/s), and the telemetry reset to idle (180 MHz / 0%) immediately after that probe completed —
so it reads as a stale nvidia-smi counter left over from the aborted process, not an actual GPU
hang or contention; it did not corrupt any measurement above. All 12 final runs after that point
completed cleanly with the GPU idle-gated (172-180 MHz) beforehand. `llama-server` (BGE, :8181)
was never touched.
