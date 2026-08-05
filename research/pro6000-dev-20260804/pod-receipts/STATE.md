# pro6000-dev pod state — 2026-08-05 (left WARM per owner instruction)

Pod: RunPod PRO 6000 WK 96GB COMMUNITY cloud, ssh -p 48084 root@80.15.7.37, $1.69/hr.
BOARD CAVEAT: different physical board than the 20260804 prod pod — driver 570.211.01
(needs LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat for all CUDA binaries), power cap 510W
(prod: 600W), runs 89C under load, mem clock droops to 13365/14001 MHz. Numbers from this
board are ~9% below the prod board at identical code (control receipt: ctrl-serve/).

## Resident
- /root/bw24            — train HEAD 623ce27e + 2 MEASUREMENT SEAMS (MEMRA_NV_MR, MEMRA_NV_DUAL
                          + MSWEEP_M1_RP in mvq_msweep) — diff in receipts-dev/dev-seams.patch.
                          Built release (all bins) against cuda-13.1 + compat.
- /root/bw24-ctrl       — prod-session commit 2299ee0f, run-gen built (board-drift control).
- /root/models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf   sha256 d8d71c7e... (byte-verified)
- /root/models/draft-owntrim-nvfp4head-q4blk.gguf  -> /root/mb/... sha256 b445fbb1...
- /root/stage/          — both tree tarballs (623ce27e + 2299ee0f) from HF Avifenesh/pro6000-stage
- /root/receipts-dev/   — this session: spin-probe, gates, msweep/, msweep-m1/, e2e/,
                          exact-spec/, ctrl-serve/, dev-seams.patch (rsynced to local
                          research/pro6000-dev-20260804/)
- HF token at /root/.cache/huggingface/token (delete if pod is handed off)

## Session verdicts (receipts in pro6000wk-runpod-community.jsonl locally)
- Gates ALL GREEN (6th GB202 die). Board health PASS w/ caveats.
- Batched (verify) NVFP4 variant table: AUTO CONFIRMED on 188 SMs, 63 cells.
- m=1 NVFP4 mr2->mr1: +0.95% d512 / +0.98% dlong e2e, exact (argmax+spec PASS).
  Auto-rule proposal (SM-count keyed) pending 170-SM 5090 measurement — NOT merged.
- Serve spec optimum K=5 on nv (162.7 vs 154.0 at K=3, N=5); bare optimum stays K=3-4.

## To resume
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
cd /root/bw24    # run-gen/run-spec/memra-server/mvq-msweep all built
