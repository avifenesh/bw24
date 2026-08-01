# Hy3 layer103.5 serving artifact — inventory (2026-08-01)

Purpose: de-risk the Aug-2 8xH100 Hy3 PP-2 serving spike (research/product-vision-20260801/
ASSESSMENT.md) by putting Hy3 on Hopper (sm_90a) for the first time, today, on the Mumbai
H100 board box.

## What the artifact is

`/data/ai-ml/hf-models/hy3-layer103p5-bw24-runtime` (local rig) is a **runtime view**, not a
weight store. As of 2026-07-30 the 73.1 GiB expert payload was reclaimed from local disk and
is published on HF Hub (see `RESTORE.md` + `relocation-receipt.json` in that dir). Contents:

| file | bytes | role |
|---|---|---|
| `manifest.json` | 9.8M | `bw24-expert-overlay-v2` manifest: 29,700 expert tensors, per-tensor `file/qtype/ne/bytes/expert_stride`, `pruned_experts` per layer, tier plan (sha-pinned) |
| `experts` (symlink, absent until restored) | — | -> expert payload: 237 `.bin` fragment shards, 78,490,288,128 bytes total |
| `frspec-code-wiki-16384.gguf` / `-32768.gguf` | 66k/131k | frspec draft tables for run-spec |
| `relocation-receipt.json` | 761 | overlay relocation receipt (hash chain) |
| `RESTORE.md` | 759 | restore instructions |

- Format: `bw24-expert-overlay-v2` (pre-rename spelling — see gap G1 in gaps.md).
- Tier mix (`tier_summary`): IQ3_S, IQ4_XS, Q2_K, Q3_K, Q4_K, Q8_0 — mixed per-expert
  layouts, i.e. the `HostExps::layouts == Some(..)` metadata-aware path, NOT the
  uniform fast path.
- `pruned_experts`: 79 layers, 64-96 pruned ids per layer; ids keep original router
  positions and are masked before top-k (`active_experts()`); weights are absent.
- `source_dir` (baked absolute): `/data/ai-ml/hf-models/hy3-layer103p5-sparse-source` —
  non-expert fallback, an HF-checkpoint-shaped dir: 20 real non-expert shards (23.30 GiB)
  from pinned `tencent/Hy3` @ `716aa7241bd6d95896be4ebfc761162a9c4d49ef`, 79 expert-only
  shard names symlinked to a 10-byte empty safetensors placeholder, + config/tokenizer/index.
- Logical model: 103,489,802,752 bytes (release.json `candidate.logical_model_bytes`);
  payload+source staged ≈ 96.4 GiB.

## Hash chain (pinned)

| item | sha256 |
|---|---|
| runtime `manifest.json` | `b8bdd684a0112312f3714024b97b9c18c8a3e7e474cbd7111f6f6021be6a644c` |
| published `manifest.json` | `08f206aed555752982585a59a7b5096b9cc6e71faf1f84ad5c6dd60476b7509a` |
| source `config.json` | `663036ceca3d8a178cd772739566c262caffdecebaed6c1d76b464d729bb2951` |
| source `model.safetensors.index.json` | `9594f1a9419e62ca7afca51bb644f38ef19039374f7812449381ccf42f0ef79b` |
| plan (canonical) | `3a953d9904bfc5a8792e6136b498c68f2a656cefbcd6390bea3cb9fb4fe4419c` |

Release receipt chain: local `~/hf-release-hy3-layer103.5/` (release.json: paired evals
5W/108T/2L vs baseline, score 76 vs 73).

## Serving entrypoint

`run-gen <runtime-dir> --prompt "..."` and `run-spec <runtime-dir>` — a DIRECTORY argument
with `manifest.json` opens `memra_gguf::source::Hy3RepackSource` (crates/memra-gguf/src/
source.rs), which resolves overlay tensors from `experts/` fragments and everything else
through the `source_dir` safetensors fallback; tokenizer comes from `source_dir`.
MoE experts ride the SLRU spill/cache stack (`MEMRA_MOE_*`, `MEMRA_SPILL_*` — docs/FLAGS.md).
The 5090 launcher `tools/run_hy3_local_5090.sh` is the 24GB CPU-companion hybrid profile;
on 80GB H100 the plain SLRU defaults apply (no CPU expert companion).

## Staging rule receipts

- CLAUDE.md rule: record the staged manifest hash; do not report persistent-EBS 4 KiB fault
  throughput as memra spill speed. Staged (NVMe) runtime manifest sha256 verified equal to
  the pinned runtime hash above (see transfer-receipt.md). All weights staged on Mumbai
  local NVMe (`/opt/dlami/nvme`), NOT the 94%-full root EBS.
