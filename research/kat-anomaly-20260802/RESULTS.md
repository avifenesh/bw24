# kat-anomaly: why KAT-Coder IQ4_XS decoded at 0.55x — and the fix (2026-08-02)

Lane `lane/kat-anomaly` (from `restructure/public-split`, 5f15f838 — post resident-if-fits +
fast-router). Rig: RTX 5090 Laptop, 24463 MiB, platform_profile `performance`. Every GPU run
under `flock /tmp/gpu5090.lock` (one co-lane shares the rig; the co-resident
`llama-server --embedding` (332 MiB) is allowlisted and inside every peak figure). llama.cpp
arm: local fork build `bb090d1f1` (same binary as the residency-cap lane). Model:
`/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf`
(sha-verified at onboarding). Control:
`/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`.

## 1. Residency check (mission item 1): NOT the mechanism

Merged-tree decision line, naked KAT (`smoke-kat-naked.log`, `decision-lines.log`, every rep):

```
[moe] resident-experts decision: experts 17.11GB + trunk 1.68GB vs free 23.93GB (expert budget 20.24GB) -> RESIDENT
```

And the anomaly-era receipts show KAT was ALREADY RESIDENT under the old budget math —
captured verbatim from `research/ornith-serve-20260801/{server,board}-kat-*-rep1.log`:

```
[moe] resident-experts decision: per-layer 427MB x 40 layers = 17.1GB vs budget 19.1GB -> RESIDENT
```

(KAT's uniform IQ4_XS bank projects to 17.1GB, inside even the old 19.1GB budget — unlike
Ornith-35B's 20.9GB misprojection.) The cheapest hypothesis is refuted by its own receipts:
the 104-106 tok/s cells were measured with a resident expert bank. Residency-cap changed
nothing for KAT (17.11GB exact vs 20.24GB budget, RESIDENT before and after).

## 2. The mechanism: IQ4_XS trunk matvecs on the Stage-A f32 oracle path

Static attribution (tensor-mix dumps from the GGUF headers, `kat-tensor-mix.txt` /
`ctrl-q35-tensor-mix.txt`, script `gguf_tensor_dump.py`):

| | KAT-Coder IQ4_XS | ctrl Qwen3.6-35B UD-IQ4_XS |
|---|---|---|
| experts | uniform IQ4_XS, 17.113 GB | IQ3_S gate/up + IQ4_XS/Q6_K/Q4_K down, 15.601 GB |
| trunk 2-D matmuls | **IQ4_XS** attn_qkv x30, attn_gate x30, ssm_out x16, attn_q x5, shexp x20-layers (~0.52 GB) + Q6_K/Q5_K/Q8_0 rest | **Q8_0** everything (~2.06 GB) |
| active bytes/decode tick | ~1.95 GB (fewer than ctrl!) | ~2.55 GB |

KAT touches FEWER bytes per token yet decoded at 0.55x — a kernel-path problem, not
bandwidth. Geometry is identical to the control (same qwen35moe stack, field-by-field —
onboarding receipts), so the only delta is the quant mix. The dispatch:

- `Engine::mmvq_supports` (lib.rs) admits `Q8_0|Q4_K|Q5_K|Q6_K|NVFP4|Q4_0` — no IQ4_XS.
- `uses_q8_1_fast` admitted IQ4_XS only under the opt-in `MEMRA_IQ_FAST` (flags audit verdict:
  "UNCLEAR — no concluding JSONL row found").
- Without it, every non-expert IQ4_XS matmul fell through `matmul`/`matmul_pre` to the
  **Stage-A generic `qmatvec_f32`** — the f32 dequant-in-kernel correctness-oracle path
  (the same class as `MEMRA_FAST=0`), at m=1 decode AND at prefill m (no `mmq_supports`/
  `gemm_supports` arm either).
- The kernel it should ride (`qmatvec_iq4_XS_dp4a`) already exists (the expert paths use the
  same body); `MEMRA_IQ_FAST=1` was a complete, never-concluded door.
- Expert-bank IQ4_XS was never the problem: the ctrl's IQ4_XS down_exps decode full-speed
  through the fused/grouped expert dispatch (its own qtype tables).

## 3. The op-class experiment = the fix, interleaved x5 same-session

Board shape (pp512.txt, NGEN=128), arms interleaved kat-naked -> kat-iqfast -> ctrl ->
llama-kat per rep (rep loop outside), N=5 medians, run-gen argmax gate per memra run,
busy-proc gate + per-run 1s peak-VRAM sampler. Raw: `kat-sweep.jsonl`, `*-rep{1..5}.{log,vram}`,
console `sweep-console.log`, `token-hashes.log`, `decision-lines.log`. llama rows re-parsed
from raw logs (`note:"reparsed-from-raw-log"` — llama-bench json is stderr-mixed in-log).
Thermal: single 40-min window, all arms share it (temps in jsonl rows).

| arm | decode tok/s (N=5 med) | prefill tok/s (N=5 med) | argmax | peak MiB |
|---|---|---|---|---|
| kat-naked (Stage-A trunk) | 106.69 [106.51-106.94] | 228.10 [227.5-228.7] | MATCH 5/5 | 19118 |
| kat-iqfast (`MEMRA_IQ_FAST=1`) | **193.41** [189.72-193.93] | **697.00** [695.8-699.8] | MATCH 5/5 | 19118 |
| ctrl-q35 (`MEMRA_PRIME_TOKENWISE=1`) | 189.07 [186.68-191.28] | 2315.10 | MATCH 5/5 | 17994 |
| llama-kat (fork bb090d1f1) | 190.28 [188.95-194.86] | 4113.86 [3476-4222] | — | — |

- **Anomaly killed: decode 106.69 -> 193.41 (+81.3%)** — one dispatch admission, zero new
  kernels. KAT now decodes ABOVE the same-session control (1.023x), exactly where the
  active-bytes math says it belongs.
- **vs llama decode: 0.55x -> 1.016x** (same-session, ranges overlap — parity-class).
- **Prefill: 228 -> 697 (+206%)**, vs llama 0.169x (was 0.05x) — prefill is now the whole
  remaining gap, same shape as the Ornith-35B #44 verdict.
- Determinism: token sha per arm 5/5 identical (naked `5246b77accbdf82a`, iqfast
  `9102ffd0b8241a65`). The arms differ from each other — expected: oracle->dp4a is a numerics
  class change (the same class every other dtype made when Stage-B became default); the
  argmax gate (prefill==decode) holds in every run, spec gates below arbitrate the rest.
- ctrl runs `MEMRA_PRIME_TOKENWISE=1` per the residency-cap branch finding (pp512 near-tie
  first-token flip on naked batched-prime; decode gen-only rate is prime-mode-independent);
  its sha `106567788698cf2f` is the post-flip bit-identity guard anchor.

## 4. What shipped: IQ4_XS trunk dp4a admission is the default

Winners are defaults (flags doctrine). `MEMRA_IQ_FAST` flips from undocumented opt-in to
DEFAULT ON with `MEMRA_IQ_FAST=0` as the rollback seam (lib.rs `iq_fast_enabled()`;
`uses_q8_1_fast` + the `matmul` IQ4_XS arm; docs/FLAGS.md updated). Dispatch-parity by
construction: IQ4_XS has no mmvq/batched kernels, so every m=1..15 rides the same
`qmatvec_iq4_XS_dp4a` per-column program (m>=16 falls to the same dp4a grid — no GEMM arm
exists), i.e. verify == decode kernel class at every tier, same law as the other dtypes.
Supported models are dispatch-unchanged by construction: no other local artifact carries
IQ4_XS NON-expert 2-D matmuls (ctrl trunk Q8_0, Ornith-35B Q4_K, Ornith-9B Q8_0, gemma
Q4_0/Q6_K, NV-27B NVFP4) — guarded by the ctrl bit-identity run below.

Post-flip verification (naked = new default), same session:

| check | result |
|---|---|
| kernel-check | **ALL GREEN** (283 OK, 0 fail — `kernel-check-post.log`) |
| KAT naked x3 (`post-default-rep{1..3}.log`) | decode 187.83/187.68/191.24 (median 187.83), argmax MATCH 3/3, token sha `9102ffd0b8241a65` 3/3 = **bit-identical to the sweep's iqfast arm** across the rebuild |
| ctrl bit-identity guard (`post-ctrl-guard.log`) | sha `106567788698cf2f` = pre-flip ctrl exactly (decode 184.14, argmax MATCH) — **dispatch-unchanged for supported models, proven not assumed** |
| rollback seam `MEMRA_IQ_FAST=0` (`post-rollback-seam.log`) | 105.18 tok/s + sha `5246b77accbdf82a` = the old naked stream exactly |
| run-spec K=1..8 self-consistency + drafter battery | `gates/`, §5 |

## 5. Drafter re-verdict (#42) + deployment bar

Full battery on the post-flip binary (`gates-katcoder.sh`, same protocol as
`research/ornith-drafters-20260801/` RECIPE §5; parser `summarize-gates.py`; raw `gates/`):

- **run-spec K=1..8 self-consistency: PASS 8/8** (spec ≡ plain, acceptance > 0 every K;
  per-K acceptance K1 92.4% … K8 39.1%) — the fix holds the spec exactness contract.
- Acceptance table (greedy, ngen 256, deterministic per (prompt,K); tok/s cells single-run):

| K | p1-code-short | p2-code-medium | p3-agentic-long |
|---|---|---|---|
| 2 | 84.7% / 1.51x | 55.3% / 0.96x | 58.0% / 0.96x |
| 3 | 73.3% / 1.21x | 51.0% / 0.95x | 45.6% / 0.85x |
| 4 | 61.5% / 1.10x | 41.0% / 0.84x | 35.8% / 0.78x |

- e2e spec vs plain @K=2, interleaved in-process x3 (the verdict numbers):

| class | reps | median ratio | median acc |
|---|---|---|---|
| p1-code-short | 1.25x, 1.25x, 1.25x | **1.25x** | 84.7% |
| p2-code-medium | 0.97x, 0.96x, 0.96x | 0.96x | 55.3% |
| p3-agentic-long | 0.96x, 0.95x, 0.95x | 0.95x | 58.0% |

- K=1 probes (`gates/probe-k1-*.log`): p2 1.00x (74.1%), p3 0.98x (70.7%) — no K rescues
  p2/p3.

**Verdict:** #42's stated mechanism ("slow plain decode makes spec rounds cost more than
they save") is REFUTED — the decode is fixed — but the global NO-ADOPT survives on new
grounds: at 193 tok/s plain, the drafter's fixed per-round cost needs >~60% acceptance to
pay, and p2/p3 sit at 55-58%. Code-short FLIPS to a clear win (1.25x e2e, 84.7% acceptance
— still the batch's best drafter). Serving guidance: `MEMRA_MTP_DRAFT=...draft-katcoder-...
+ MEMRA_SPEC_K=2` is net-positive for code-short-dominant serving only; no global default.
(Old-lane comparison: acceptance moved 82.5/61.7/55.4 -> 84.7/55.3/58.0 with the numerics
class change; ratios moved 1.09/0.91/0.85 -> 1.25/0.96/0.95 with the faster denominator.)

**Deployment bar (>=1.1x e2e vs llama, board shape 512+128, same-session cell medians):**

| leg | memra | llama | ratio |
|---|---|---|---|
| decode plain | 193.41 | 190.28 | **1.016x** (was 0.55x) |
| decode w/ drafter K=2 (code-short) | ~242 (193.41 x 1.25) | 190.28 | **~1.27x** |
| prefill pp512 | 697.0 | 4113.9 | 0.169x (was 0.05x) |
| e2e proxy plain | 1.396 s | 0.797 s | 0.57x (was 0.23x) |
| e2e proxy spec (code-short) | 1.264 s | 0.797 s | 0.63x |

KAT stays **onboarded, pre-deployment**: the decode anomaly is closed (and inverted), the
bar-binding gap is now prefill alone — same shape as the Ornith-35B #44 verdict, owned by
the IQ4_XS-trunk MMQ port priced in §6.

## 6. What remains (priced, not built here)

- **IQ4_XS trunk prefill MMQ port**: pp512 697 vs llama 4114 (0.169x). The trunk now rides
  dp4a grid(out_f, m) — no weight reuse across tokens. `cu/mmq_iq_experts.cu` already
  implements the IQ3_S/IQ4_XS int8-MMA MMQ tile loaders for expert-segmented GEMMs; a dense
  trunk arm = a launcher + `mmq_supports` admission + the exactness battery. Its own lane
  (order days, benefits every IQ4_XS-trunk artifact). Ceiling ~ctrl-class 2315 (the rest of
  the ctrl's gap to llama is the MoE expert prefill, the unowned #44 flag-3 lane).
- **IQ3_S trunk**: no local artifact carries one; `qmatvec_iq3_s_dp4a` does not exist. Do not
  admit IQ3_S without writing the kernel (B3 comment, lib.rs).

## Files

`run-kat-sweep.sh`, `kat-sweep.jsonl` (N stated per row), `sweep-console.log`,
`{kat-naked,kat-iqfast,ctrl-q35,llama-kat}-rep{1..5}.{log,vram}`, `token-hashes.log`,
`decision-lines.log`, `smoke-kat-{naked,iqfast}.log`, `gguf_tensor_dump.py`,
`{kat,ctrl-q35}-tensor-mix.txt`, `kernel-check-post.log`, `run-postflip.sh`,
`post-default-rep{1..3}.log`, `post-ctrl-guard.log`, `post-rollback-seam.log`,
`gates-katcoder.sh`, `summarize-gates.py`, `gates/` (gate-k1-8, acc-k{2..4}-*,
e2e-k2-*-rep{1..3}, probe-k1-{p2,p3}).
