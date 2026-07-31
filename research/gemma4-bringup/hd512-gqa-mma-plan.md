# hd512 globals decode kernel — gqa-packed mma port (campaign plan, 2026-07-30)

**CORRECTION (2026-07-31, supersedes the geometry below):** the 12B globals are MQA —
nkv=1, gqa=16 (not nkv=2/gqa=8), and the tb512 launch sees t_kv ~520 (a bounded view,
not the full 1736): grid (1,17,1) = 17 blocks on 82 SMs, SP512 default 32. The split
axis is COMBINE-BOUND (sp512 sweep receipts, rig5090.jsonl 2026-07-31) — the port's win
must come from in-kernel efficiency (mma + in-kernel stream-k fixup replacing the
separate combine), not more splits. Projected win ~+1% on the g12-plain-d1736 cell
(0.13ms of an 11.4ms step). PRIORITY NOTE: the H100 fronts (gemma prefill MMQ 0.23-0.32x,
q9 w8a8 0.73x) are 10-50x larger prizes — do those first (2026-07-31 goal order).

The g12-plain-d1736 0.973x mechanism (pinned, rig5090.jsonl 2026-07-30 rows): the hd512
GLOBALS decode lane costs 30.8us/layer (tb512 24.5 + combine_dc 6.3) vs llama's ~14.7
deflated — 8 global layers = 0.13ms/step, half the cell's gap. Windowed lane is parity.

## Measured facts (do not re-derive)

- gemma-4-12b globals: head_dim=512, n_head_kv=2, gqa=8 (16 q heads), t_kv full (1736
  in the cell). Windowed layers: hd256, nkv=8, window 512, fp8 wkv — PARITY vs llama.
- Current arm: `fa_decode_vec_q_rows_v4_512_tb` (flash_attn.cu:9016), dispatched at
  lib.rs:8059 (`tb512`), grid=(nkv=2, n_splits=t_kv/sp, t=1), sp from fa_split_keys
  (nkv<=4 ladder: 8 below d3072) -> 434 blocks x (32,gqa?) threads. Score phase: ONE
  key per lane, lane<nt_r (sp=8 -> 8/32 lanes active). Staging: K+V tile dequant to
  smem, used once (t=1). ~155GB/s effective; ncu (w_sp sibling): 0.39 waves, occupancy
  9.85%, DRAM 14%.
- llama (nsys graph-node, deflate 1.34x): `flash_attn_ext_f16<512,512,1,8>` ONE launch
  per layer per step (8/step total), 13.9us raw / ~10.4 deflated + fixup_uniform 5.8/4.3.
  ncols2=8 = the 8 GQA q-heads ride as mma columns; f16 KV; stream-k over KV chunks.
- Seam probes ALL flat-or-worse on the cell (interleaved+cooldown, receipts in
  g12-d1736-hd512-probe-20260730.log): FA_V512=1 flat, FA_TB512=0 -2%, FA_SPLIT=16/32
  flat, FA_SPW ladder flat-topped at default 32.
- Launch cadence: ours ~0.32ms/step eager (630 launches) vs llama graph ~0.19; graph
  door CLOSED on 12B (-5.6%, 2026-07-23), PDL waves A/B already default-on.

## The port (v1)

New kernel `fa_decode_hd512_gqa_mma` (flash_attn.cu), new dispatch arm in the hd512
wrapper (lib.rs tb512 site), opt-in door `BW24_FA_GQMMA=1` until battery + interleaved
receipts, then default flip for hd512 decode+verify TOGETHER (parity law: both flavors
resolve through the same wrapper; combine layout partO/partM/partL unchanged).

- grid (n_head_kv, n_splits) with sp=64..128 keys/split (tune; stream-k-style fixed
  partition, combine handles split reduction exactly as today).
- block: 4 warps (128 thr).
- Stage per chunk of 16 keys: dequant K to bf16/f16 smem tile [16 x 512]; Q for the 8
  gqa heads staged once per block (bf16, [8 x 512]).
- Score: mma.sync m16n8k16 (f16 in, fp32 acc): A=K tile rows, B=Q cols -> S[16 x 8].
  32 k-steps over 512 dims. ALL warps cooperate (split 512 dims across warps, reduce).
- Online softmax per head over the split's keys (m,l partials per (head,split) — same
  partM/partL contract).
- PV: P[16x8] x V[16x512] -> acc[8x512] via mma (A=P^T) or scalar v1 (each thread owns
  dims); V dequant to smem like K.
- NEW NUMERIC CONFIG (f16 mma scores vs dp4a int8): battery arbitrates — run-gen argmax
  (12B/31B), K=1..8 self-consistency, stream gates; decode==verify flip together.

## Gate battery for the flip
kernel-check, run-gen argmax 12B+31B, run-spec K=1..8, BW24_GRAPH_GATE stream gate,
g12-plain-d1736 interleaved x5 vs llama (target: 0.973 -> >=1.00), g12-spec-d1736,
g31 depth cells (hd512 shared), serve-smoke.

## Status
- [ ] kernel v1 compiles
- [ ] parity harness vs tb512 output (same partials contract, rel tol)
- [ ] cell A/B interleaved x5
- [ ] battery + default flip decision
