# 122B bring-up — PROGRESS (lane/122b-bringup, restaffed 2026-08-06)

Pod: RunPod PRO 6000 WK 96GB COMMUNITY (ssh -p 48084 root@80.15.7.37, WARM).
Community-pod caveat applies to every number here: RELATIVE evidence, not board material.

## Artifact — DOWNLOADED, MERGED, VERIFIED (prior staffing, salvaged)

- Source: Unsloth Qwen3.5-122B-A10B UD-IQ4_XS, 3 splits (dl-122b.log 05:50-05:56Z)
- Merged: `/dev/shm/122b/Qwen3.5-122B-A10B-UD-IQ4_XS.gguf` — 60,229,510,432 bytes
  (60.2 GB — exactly the assessment's HF blob receipt)
- sha256 (merged): `9c9701c1673f80cc164bf66e3a82b957dc18a7223676339a9bbd76175c4d8f92`
- sha256 (splits): see logs/sha-122b.log
- NOTE: lives in /dev/shm (RAM-backed) — survives the pod, not a reboot. Disk / has 55G
  free (< the 60GB artifact); durable copy is a re-download, receipts are the hashes above.

## Boot — PASS (prior staffing, salvaged; pod tree @7ac05f54-vintage +122b enablement)

- Loads as `qwen35moe`, 48 trunk layers, "optional MTP skipped" (drafter probe pending)
- Resident decision: experts 53.75GB + trunk 6.46GB -> RESIDENT (92.15GB expert budget)
- Single-stream: "capital of France" 64 tok @ 124.18 tok/s, argmax MATCH, coherent text
- VRAM plateau 58.9GB (boot-vram.csv) vs assessment arithmetic 60.2GB weights-class: consistent

## THE BUG — decode-path all-NaN logits above a prompt-length threshold

Receipts (logs/, salvaged from /root/receipts-122b):
- argmax-run1/run2 (4k-class prompt): prefill argmax healthy (l[271]=17.31), decode logits
  ALL NaN -> argmax 0, run_gen gate panics. x2 IDENTICAL (deterministic).
- bisect: ~110-tok prompt FAIL, 800/1600/3200-char prompts FAIL — every decode NaN;
  5-tok boot and 28-tok probe MATCH and generate.
- Hypothesis (prior agent, to be verified this staffing): FA v4 decode kernel NaNs at the
  122B attention shape (hd256, nkv=2) once t_kv crosses the vec-lane floor
  (MEMRA_FA_VEC_MIN default 96). 35B-A3B shares nkv=2/hd256 and is green — so the delta
  (n_head/gqa group? IQ4_XS K/V? 12-full-attn-layer hybrid?) is the isolation target.

## Plan (this staffing)

1. [x] Salvage receipts + this file, commit (this slice)
2. [ ] Rsync worktree @4cbf5e39 (train HEAD) -> pod /root/bw24-122b, rebuild
3. [ ] Reproduce NaN on the 4cbf build; pin the threshold (t_kv ~96 crossover?)
4. [ ] Arm isolation via env doors: MEMRA_FAST=0 oracle, MEMRA_FA_V4=0, MEMRA_FA_V3=0,
       MEMRA_FA_V2=0, MEMRA_FA_VEC_MIN=big (scalar floor), MEMRA_FA_SPLIT
5. [ ] If an arm is clean: workaround config receipt + FA v4 fix brief; full gate battery
       under the workaround (kernel-check MoE cells, argmax x2, run-spec probe, serve boot
       + smoke, chunkinv, c=8 8k-ctx capacity no-OOM)
6. [ ] MTP head probe (artifact tensors) — drafter status for the deployment-gap list
