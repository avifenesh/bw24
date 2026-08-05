# q27 deep dive PHASE 2 (MTP/spec tier) — pod state 2026-08-05 (WARM, DO NOT TERMINATE)

Pod: RunPod PRO 6000 WK 96GB COMMUNITY, ssh -p 48084 root@80.15.7.37.
BOARD CAVEAT unchanged (510W cap, mem droop) — RELATIVE deltas only.
LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat MANDATORY for all CUDA binaries.

## Trees
- /root/bw24     — the phase-1 rsync tree (ed815eee vintage + dev seams). PRE-H3.
- /root/bw24-h7  — train HEAD 7ac05f54 (git-archive from local, clean), built release
                   12:47Z (/root/build-h7.log). THE tree for any further phase-2 work.
- /root/bw24-ctrl, /root/pilot, /root/hf-models, /root/llama.cpp — untouched.

## Models (/root/models)
- Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf (nv), Qwen3.6-27B-Q8_0.gguf (q8)
- draft-owntrim-nvfp4head-q4blk.gguf -> /root/mb/drafts/... (trim drafter, the daily config)
- draft-full-untrimmed.gguf (donor-extracted FULL-vocab head, built by the killed lane 10:58Z;
  MEASURED: loses e2e everywhere vs trim, see RESULTS — keep for reference, do not serve)

## Receipts
- /root/receipts-p2/            — ALL phase-2 receipts (killed-lane salvage + h7-* rerun).
  scripts/ has every driver. logs/h7-* = the 7ac05f54 rows (gates, sk-*, bu-*, pmin/pk/pv,
  ident). nsys/ = serve-session profiles (nv K5, q8 K4) + kernel-sum CSVs.
- /root/bw24/research/q27-deepdive-p2-20260805/RESULTS.jsonl — 15 rows, the lane deliverable.
  Rsynced to local research/q27-deepdive-p2-20260805/ (pod-receipts-preH3/ + RESULTS.jsonl).

## Headlines (details in RESULTS.jsonl)
- Round anatomy: verify owns 75-92% of the round; commit glue 2-3% (no 5090 T=1 glue tax at
  188 SM). Sampled serve path syncs in commit-host (68-69%) instead of verify-wait — same
  round, different sync point.
- spec-econ verify curve: flat to T=8, x3 CLIFF at T=9 both artifacts => q8 K=8 serving
  collapses (64 tok/s). Cap K<=7.
- K policy: q8 K=4 (B32) / K=6 (B128); nv K=5 at any burst. Cross-tree agreement pre/post H3
  to 0.2% (H3 does not touch the spec path).
- Burst 32->128: +7.5% nv / +4% q8 c=1, +7.6% c=8, p50 improves, byte-identical. B64 flat (nv).
- p-min 0.3 (+PMIN0): +6.1% nv / +2.3% q8 further; acceptance +15-20pp; 0.2/0.4 both worse;
  exactness 9 PASS + byte-identical. Stack: nv 161.8->184.5 (+14%), q8 111.6->133.9 (+20%).
- Short-ctx acceptance dip REPRODUCES (0.51-0.59 fresh vs 0.63-0.66 @6k) on every arm incl.
  untrimmed head => depth property, not trim. Untrimmed draft buys +1-5pp acceptance short-ctx
  but LOSES e2e on all 16 cells (-5..-9%) => keep trim on 96GB too.

## Open / next
- Default flips (K, burst, pmin) need prod-class silicon re-mint + the 5090 gate battery
  (flags doctrine: winners become defaults — after the target-rig battery, not from a
  community board).
- ncu still HARD-BLOCKED here (ERR_NVGPUCTRPERM) — occupancy questions go to prod silicon.
- memra-server is NOT left running (all measurement servers were torn down; GPU idle).
