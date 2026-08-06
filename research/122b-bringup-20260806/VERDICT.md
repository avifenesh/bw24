# 122B bring-up — VERDICT (lane/122b-bringup, 2026-08-06)

Rig: RunPod PRO 6000 WK 96GB COMMUNITY pod (80.15.7.37). **Community-pod caveat on every
number: relative evidence, single runs unless stated — not board material.** Build: train
HEAD 4cbf5e39 + the one fix below (`/root/bw24-122b`). Model:
Qwen3.5-122B-A10B UD-IQ4_XS (Unsloth), merged 60,229,510,432 bytes,
sha256 `9c9701c1...af04c3f92` (full hashes in logs/sha-122b*.log), staged at
`/dev/shm/122b/` (RAM-backed — disk has 55G free < the 60GB artifact).

## Headline

**The card's premium SKU boots, gates green, and serves c=8 — after one real engine bug
found and fixed at the new head geometry.** The 122B is the first gqa=16 model memra has
ever loaded (32 Q heads / 2 KV heads; everything prior was gqa<=8), and it walked straight
into a latent FA v4 buffer overflow.

## THE BUG — FA v4 decode family overflows fa_v4_smem at gqa>8

- Symptom: decode logits ALL-NaN on any prompt whose t_kv crosses the vec floor (96);
  prefill logits healthy; deterministic (x2 identical); bisect 400..3200-char prompts all
  FAIL, 5/28-token prompts MATCH.
- Isolation (arm battery, logs/arm-*.log): v4 + v4-deep MISMATCH+NaN; **v3, v2, smem/reg,
  scalar all MATCH**; MEMRA_FAST=0 oracle still NaN (matvec-class-independent) ⇒ the v4
  decode family is the source.
- Root cause (cu/flash_attn.cu:7048): `fa_v4_smem` sizes its per-warp Q arrays
  `q_ints[8][64]` / `q_d[8][8]` for gqa<=8. The block launches `(32, gqa, 1)` and
  `fa_v4_stage_q` writes `q_ints[wy]` for wy 0..gqa-1 — at gqa=16, warps 8..15 write past
  the Q arrays **into the K tile** (`k_ints`/`k_d`), corrupting scores → NaN. The hd512
  lane already carries the equivalent capacity guard at dispatch (lib.rs "gqa <= 16 =
  fa_v4_smem_512's q-array capacity"); hd256 v4 never got one.
- Fix shipped (22fd5f6a): load-time guard in hybrid.rs — `gqa > 8` stores
  `FA_V4_MAX_DEFAULT = 0`, flipping every v4 dispatch site (eager/rows/dc/rows_dc/
  windowed/seqs) to the v3 lane together (decode/verify parity preserved).
  `MEMRA_FA_V4_MAX` stays the diagnostic seam.
- Verified: previously-failing 110-tok prompt MATCH + x2-identical; the original 4k-prompt
  repro MATCH (verify-122b.out).
- **Follow-up (fix brief):** a real v4 gqa16 extension = size the Q arrays [16][64]/[16][8]
  (+6KB smem, occupancy re-check) or split the GQA group across 2 CTAs; gate on
  kernel-check bit pins + the 3-model battery + a measured v3-vs-v4 delta at this shape
  before defaulting. Until then the 122B rides v3 (also: v4-deep, seqs-batched FA and the
  PDL v4 windowed arms are all correctly excluded by the same key).

## Gate battery (all under the guard, defaults otherwise)

| Gate | Verdict | Receipt |
|---|---|---|
| boot + single-stream | PASS — resident (experts 53.75GB + trunk 6.46GB), coherent text | logs/boot-rungen.log |
| kernel-check (model-backed) | **ALL GREEN** (356 OK incl. MoE router/prefetch cells) | logs/gate-kernel-check.log |
| run-gen argmax x2 | MATCH, x2 token-identical (110-tok + 4k prompts) | logs/fix-default-r{1,2}.log, fix-4k.log |
| run-spec K=1..3 | **UNAVAILABLE** — artifact ships no MTP head (`nextn=0`, quoted error in logs/gate-runspec-k1.log) | see drafter status |
| chunkinv | PASS | logs/gate-chunkinv.log |
| serve boot + smoke | PASS 0 failed (spec arm SKIP, no draft) | logs/gate-serve-smoke.log |
| capacity c=8 @ 8k ctx | **8/8 clean**, 0 OOM lines, VRAM peak 62,010 MiB | cap-122b.out, logs/cap-rows.jsonl, logs/cap-vram.csv |

## VRAM census vs the assessment

Boot plateau 58.9GB; c=8 serving peak 62.0GB on the 96GB card — consistent with the
assessment's 60.2GB weights arithmetic (+KV+overhead). ~34GB headroom at c=8/8k.

## Perf snapshots (single runs, community pod — context only)

- decode ~108-124 tok/s single-stream (v3 lane; 10B active on a 188-SM card)
- prefill pp125 1358 tok/s, pp2136 4945 tok/s
- c=8: TTFB ~2.8s (~1.1k-tok prompts, 8-way concurrent), 256 tok completions in ~9.6s wall

## Drafter status

The UD-IQ4_XS GGUF carries **no NextN/MTP tensors** (`nextn_predict_layers=0`; header
tensor scan confirms blk.0..47 only). The HF source config says `mtp_num_hidden_layers: 1`
— the head exists upstream but Unsloth's GGUF drops it. Spec serving therefore needs the
own-gen trimmed-drafter recipe (the q27/q9 pattern: donor-block extraction + own-gen
trim + acceptance gates) as a follow-up lane, or a GGUF re-export that keeps the NextN
tensors.

## Deployment-gap list (before this SKU can serve for real)

1. **Drafter** — no MTP head in-artifact; own-gen drafter lane or re-export (above).
2. **FA v4 gqa16** — perf follow-up; v3 lane is the correctness-green path today.
3. **Artifact durability** — 60GB lives in /dev/shm only; pod disk (55G free) cannot hold
   it. Durable home (or deploy-time re-download vs the pinned sha) required.
4. **Quant exception** — prod=8bit rule vs this SKU only fitting at 4-bit: standing owner
   call (assessment §5.1); bring-up was explicitly allowed ahead of it.
5. **128k admission** — this receipt is 8k-ctx c=8; the assessment's ~21×128k envelope
   needs its own capacity ladder before listing.
6. **Real-rig battery** — all of this is community-pod; the gate battery re-runs on the
   deployment card before any tag/serve default.
