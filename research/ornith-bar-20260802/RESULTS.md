# ornith-bar: Ornith-9B best-vs-best + 5090 KQRP sweep — results (2026-08-01/02)

Lane `lane/ornith-bar` (from `restructure/public-split`, 595f5c43). Rig: RTX 5090 Laptop
24.5GiB, driver 595.71.05, platform_profile `performance`, `gpu-full-power on`. Every GPU run
under `flock /tmp/gpu5090.lock` (one co-lane shares the rig; the co-resident
`llama-server --embedding` is allowlisted and untouched). llama.cpp arm: the local fork build
`bb090d1f1` (libs 0.0.9839). All memra arms from this worktree's release build (sm_120a,
naked defaults unless stated). Model shas: pinned in `research/onboard-ornith-20260801/` and
`research/ornith-drafters-20260801/manifests/` (byte-identical paths, see `MANIFEST.log`).

Session: one window, 17:22–17:42Z, rig otherwise idle (busy-proc gate before every arm),
temps 59–78 °C across the 9B cell, 64–73 °C across the 35B batteries (per-row `temp_c` in the
jsonls). All medians N=3 (9B cell, 35B re-cell) or N=5 (KQRP), interleaved same-session.

## 1. Ornith-9B Q8_0 — BEST-vs-BEST cell (the deployment decider)

Convention = the board's spec-row rule: each engine at ITS best config on the same gguf.

- **memra best** = the ADOPTED own-gen trimmed drafter (`research/ornith-drafters-20260801/`),
  `MEMRA_MTP_DRAFT=draft-ornith9b-owntrim-nvfp4head-q4blk.gguf MEMRA_SPEC_K=3`, otherwise
  naked. run-spec interleaves plain + spec in-process; self-consistency gate per run.
- **llama best** = best PLAIN (llama-completion, `-ngl 999 -fa on -ctk q8_0 -ctv q5_1`,
  greedy, --ignore-eos, 256 new tokens). Its draftless speculative doors were given a fair
  best-effort and are **structurally broken on this arch** — lookup emits mid-run
  `inconsistent sequence positions` (M-RoPE) even at draft-max 3, its outputs diverge from
  plain greedy (degenerate repetition at dm≥8, sampler ggml_abort at dm=16/p2), lookahead
  fails the M-RoPE `X < Y` position rule outright. Full screen receipts:
  `llama-spec-doors-screen.md` + `o9b-llama-look*ep0.log` + `o9b-lookup-sweep-*.log`.
  llama has no Ornith draft artifact (fork EAGLE3/MTP paths need one).

Protocol: interleaved llama→memra pairs, rep loop outside the class loop, N=3, same session.
e2e = prime/prompt-eval wall + 256/decode (decode-rate term; llama's sampler+unaccounted time
excluded — the generous-to-llama variant). Raw: `o9b-cell.jsonl` (163 rows),
`o9b-{llama-plain,memra-spec}-*-rep{1..3}.log`, console `o9b-cell-console.log`.

| class | metric | memra best (spec K=3) | llama best (plain) | ratio | 1.1x bar |
|---|---|---|---|---|---|
| p1-code-short (27 tok) | decode tok/s | **188.1** (plain 87.0, acc 61.1%) | 84.2 | **2.23x** | |
| | prefill tok/s | 794 | 811 | 0.98x | |
| | e2e s (256 tok) | **1.395** | 3.084 | **2.21x** | **PASS** |
| p2-code-medium (1845 tok) | decode tok/s | **150.4** (plain 85.0, acc 47.0%) | 83.9 | **1.79x** | |
| | prefill tok/s | 5302 | 4923 | 1.08x | |
| | e2e s | **2.050** | 3.429 | **1.67x** | **PASS** |
| p3-agentic-long (6257 tok) | decode tok/s | **144.8** (plain 84.9, acc 47.8%) | 82.6 | **1.75x** | |
| | prefill tok/s | 5146 | 4909 | 1.05x | |
| | e2e s | **2.984** | 4.376 | **1.47x** | **PASS** |

Rep spreads are tight (memra spec decode reps within ±0.6%, llama within ±1.3% — see
summary-output.txt). memra self-consistency: 9/9 PASS (spec ≡ plain tokens every run).

**VERDICT: Ornith-1.0-9B Q8_0 CLEARS the deployment bar on every prompt class —
best-vs-best e2e 2.21x / 1.67x / 1.47x ≥ 1.1x. DEPLOY-grade under the board convention.**
(The plain-vs-plain 0.97x parity cell from `research/ornith-serve-20260801/` stands; the bar
is best-vs-best, and llama has no working speculative path for this model. Serve-level
context: o9b held greedy batch-isolation 16/16 in the serve lane — no batch-prime caveat.)

## 2. Ornith-35B Q4_K_M — 5090 KQRP sweep (Hopper-default, previously unswept here)

No local q27-class artifact exists (the two candidate 27B Q4_K_M ggufs are an MTP-head-only
2.0GB file and the NVFP4-trunk daily — header dumps in `q27-local-check.log`), so per
mission the sweep is Ornith-35B alone. Board shape (pp512.txt, NGEN=128), interleaved x5,
`MEMRA_KQRP` off (5090 default) vs `=1`. Raw: `kqrp-sweep.jsonl`, `kqrp-{off,on}-rep{1..5}.log`.

| metric | KQRP off (default) | KQRP on | on/off |
|---|---|---|---|
| decode tok/s (N=5 med) | **139.16** [135.9–139.4] | 134.54 [134.0–135.0] | **0.967x** |
| prefill tok/s (N=5 med) | 496.1 | 491.2 | 0.990x |
| run-gen argmax | MATCH 5/5 | MATCH 5/5 | |
| mirrors engaged | 0 | 191 tensors | |
| generated tokens sha | 61c48f62f98e5dbc | 61c48f62f98e5dbc | bit-identical, 10/10 runs |

**KQRP is a -3.3% decode regression on the 5090** (ranges cleanly separated), prefill flat.
Correctness verified on this rig: argmax MATCH per arm and bit-identical output across all
ten runs — the mirrors compute the same numbers, they're just slower on this GDDR7 memory
system for the 35B's mirror-eligible trunk slice (191 tensors ≈ 1.12 GiB of a 19.7 GiB
model). The Hopper-only default is CORRECT for this card; no default change, no serving flag.
The layout-v2 q27-class check stays owed to a box that has the q27 gguf (/opt/dlami/nvme).

## 3. Ornith-35B plain vs llama — same-session restatement (idle regime)

KQRP did not move the 35B (it regressed), so the mission's conditional re-cell at KQRP=on is
moot; this is the same-session plain restatement in the idle regime (the serve-lane cell was
hot/co-loaded). llama-bench (`-ngl 999 -fa 1 -ctk q8_0 -ctv q5_1 -p 512 -n 128 -r 3`) vs
run-gen naked, interleaved x3. Raw: `o35b-plain-recell.jsonl`, `o35b-recell-*-rep*.log`.

| metric | memra (N=3 med) | llama (N=3 med) | ratio |
|---|---|---|---|
| decode tg128 | 139.1 (argmax MATCH 3/3) | 193.2 | **0.72x** |
| prefill pp512 | 495.7 | 3952.9 | **0.125x** |
| e2e wall (512+128) | 1.953 s | 0.792 s | **0.41x** |

Confirms the serve-lane cell (0.72x / 0.15x / 0.42x) — regime-independent.

## 4. The priced remaining gap for Ornith-35B (bar: e2e ≥ 1.1x, board shape)

Budget arithmetic: llama wall 0.792 s → memra must reach ≤ 0.720 s. With the ADOPTED K=2
drafter (spec-vs-plain 1.38x/1.09x/1.05x by class), memra's best decode today is
192/152/146 tok/s by class:

- **p2/p3-class: no prefill speed can clear the bar today** — the decode term alone
  (128/152 = 0.844 s, 128/146 = 0.876 s) exceeds the whole 0.720 s budget.
- p1-class: needs pp512 ≥ ~9560 tok/s (19x today's 496) at today's decode — also out of reach.

So the levers, priced:

1. **KQRP: refuted here** (-3.3% decode). Costs nothing to keep off (it is off). Zero gain.
2. **Q4_K trunk f16 prefill mirrors (round-49 class): NOT the 35B lever.** Trunk q4_K/q6_K
   is 1.12 GiB = 5.7% of weight mass (experts 18.16 GiB = 92%; header dump
   `o35b-tensor-dump.txt`); Amdahl caps the prefill gain under ~+10%
   while the f16 mirror costs ~3.9 GiB VRAM this card does not have beside a 19.7 GiB model
   (the SLRU expert cache would shrink and decode would pay). Priced and correctly not built.
3. **Decode: expert residency, worth +39%.** llama-bench holds the whole model resident on
   this 24.5 GiB card (with q8_0/q5_1 KV) and decodes 193; memra runs the 18.2 GiB expert
   bank through the SLRU spill cache and decodes 139 (0.72x). The trunk is exonerated by
   this sweep (KQRP moved nothing but itself), and the same-arch resident control beats
   llama 1.13x same-session (serve-lane receipts) — the gap attribution is the expert fetch
   path. A residency-budget lane (fit experts + trunk + KV resident, as llama proves
   possible) is the decode path to ~parity; then the drafter gives 265/210/203 by class.
4. **Prefill: the MoE Q4_K expert prefill lane, worth 4.3–11.6x — the binding lever.**
   At decode parity + drafter, the bar needs pp512 ≥ 2134 (p1) / 4563 (p2) / 5753 (p3)
   vs today's 496. The same-arch ctrl (IQ3_S/IQ4_XS experts) does ~2400 on this rig and
   8428 on H100 with the grouped expert-f16 dequant lane (Hopper default; round 47/49
   receipts) — the Q4_K-expert variant of that lane on sm_120a is the unbuilt piece.
   p1-class clears at ctrl-parity; p2/p3-class need llama-class-plus (~4.6–5.8k) prefill.

**VERDICT: Ornith-1.0-35B stays HOLD (pre-deployment).** Order of attack for a future lane:
expert residency (decode 0.72x → ~1x) first — without it no prefill number clears p2/p3 —
then the Q4_K expert prefill lane to ≥2.1k (p1) / ≥5.8k (all classes).

## Files

- `run-9b-cell.sh` (+`screen` phase), `run-kqrp-sweep.sh`, `summarize.py`, jsonls + per-run
  logs as cited above; `summary-output.txt` = frozen summarize.py output;
  `llama-spec-doors-screen.md`; `q27-local-check.log`; `MANIFEST.log`;
  smoke logs `smoke-{llama-plain,memra-spec}-p1.log`.
