# Q4_K f16 prefill mirrors — q27 (round 49, 2026-08-01, H100 GPU 3)

Q4_K joins the f16-mirror carve-out (Q8_0 2026-07-26, Q4_0 campaign-A, Q6_K round 47).
The q27 trunk bulk (294 Q4_K tensors, 40.4GB @2B/w) rode mul_mat_q_q45k int8-MMA for
prefill; the cuBLASLt f16 lane beats that class at large m. Dequant kernel
(memra_q4kf16_dequant_kernel, f16_prefill.cu): 144B superblock — fp16 d + fp16 dmin +
12B 6-bit packed scales/mins (get_scale_min_k4) + 128B nibbles, value = d*sc*q - dmin*mn;
unpack verified against qmatvec.cu deq_q4_k/q4k_scale_min. Admission in build_q8_f16
(in_f%256==0, row_bytes == in_f/256*144).

BUDGET PRIORITY (the design decision): Q4_K admits as a SECOND pass over the trunk walk
so the shared MEMRA_PP_F16_BUDGET_MB (default 32768) keeps FULL Q6_K coverage as its
floor — Q6_K mirrors replace a ~10x dequant-GEMM (no MMQ arm exists), Q4_K mirrors
upgrade a working int8-MMA arm; a joint walk would evict late-layer Q6_K mirrors for the
weaker lever. Layer-order prefix within the Q4_K class: default budget admits 183/294
tensors (22420 MB) after Q6_K's ~9.5GB.

## Gates (all on GPU 3, this binary = arc4 @ v059+q4k)

- kernel-check (q27 model arg): rc=0, fails=0 (kernel-check.log, 271 lines). NEW gates:
  - GEMM blk.3.attn_q.weight [Q4_K] f16 T=16/64/128/512: rel 4.1-4.4e-3 OK (band 1e-2)
  - GEMM output.weight / blk.0.attn_v.weight [Q6_K] f16: rel 3.5-4.3e-3 OK — the round-47
    Q6_K mirror had NO battery entry; added with Q4_K ("gates outside the battery rot").
- run-gen argmax q27 board-2048: MATCH, maxdiff 6.513e-1 (default budget);
  MATCH, maxdiff 5.124e-1 (MEMRA_PP_F16_BUDGET_MB=43008). NO round-45-style flip —
  the Q4_K f16 class HOLDS on the q27 hybrid; admission stands.
- run-spec K=1..8 (HPOST=1 PMIN=0.3, board-2048): all 8 self-consistency PASS.
- q35 (UD-IQ4_XS) run-gen argmax: MATCH, maxdiff 8.402e-1 — no-regression confirmation;
  q35 carries ZERO Q4_K tensors (gguf type dump: IQ3_S/IQ4_XS/Q8_0/Q6_K only).
- q9-Q8_0 carries zero Q4_K tensors; gemma q4_0 artifacts carry zero Q4_K (type dumps) —
  blast radius of the carve-out on the current fleet is q27 only.
- validate-h100.sh --quick (q27, GPU 3, post-A/B): ALL GATES GREEN — policy tests,
  kernel-check, decode-batch config+strict, decode-dc, graph-decode capture/replay
  bit-identity, graph-session (validate-h100-quick.log). The captured prime graph bakes
  f16 GEMM pointers, so the graph gates were the untested risk surface for the new
  mirrors in the prime path — clear.

## Prefill A/B (base = memra-int v059: Q6_K f16 + KQRP; new = +Q4_K f16)

Interleaved x3 per class per bin, same-session, N=3 medians (ab.sh, parse_ab.py,
ab-summary.txt, raw per-run logs in ab-logs/). MEMRA_SPEC_K=3 HPOST=1 PMIN=0.3 NGEN=256.

| class      | metric        | base   | new (default budget) | delta |
|------------|---------------|--------|----------------------|-------|
| board2048  | prefill tok/s | 3205.0 | 4934.9               | +54%  |
| agentic500 | prefill tok/s | 2780.7 | 4402.8               | +58%  |
| board2048  | plain decode  | 88.46  | 89.55                | flat  |
| agentic500 | plain decode  | 90.21  | 89.75                | flat  |
| board2048  | spec K=3      | 109.91 | 101.39 (acc 66.1->57.3) | -7.7% |
| agentic500 | spec K=3      | 144.15 | 146.43 (acc 84->86)  | +1.5% |

Decode flat = the m>=16 arm holds; decode never touches the mirror.

## Budget probe (x2 interleaved, board-2048)

| budget MB | mirrors        | prime s | prefill tok/s | spec K=3 | acc  |
|-----------|----------------|---------|---------------|----------|------|
| 32768     | 183 (22420 MB) | 0.416   | 4923          | 99.6     | 57.3 |
| 43008     | 265 (32660 MB) | 0.311-0.313 | 6564      | 109.2-109.7 | 64.8 |

+105% prefill vs base at 43008, AND the board-2048 spec acceptance recovers to ~base
(66.1 -> 64.8): the acceptance dip at default budget is a partial-coverage artifact —
the layer-prefix mirror mixes f16-prime and int8-prime numerics mid-trunk. Single-prompt
evidence (round-45/48 roulette law applies), but both axes prefer fuller coverage.
VRAM peak: 66037 MiB (default budget runs) / 76277 MiB (43008 runs) on the 81559 MiB
H100. Default stays 32768 (a hopper-wide default bump would move VRAM on every model on
a box — g31's own note recommends 57344 explicitly); serving configs set the env per
model/box (flags doctrine: VRAM budgets are machine-specific config).

## e2e math (256 gen tokens, board-2048 primes)

- plain: base 256/(0.639+2.894)=72.5 -> new 256/(0.415+2.859)=78.2 (+7.9%)
- spec:  base 256/(0.639+2.329)=86.3 -> new@default 87.1 (+0.9%) -> new@43008 ~96.5 (+11.8%)
- agentic spec: 127.7 -> 135.3 (+5.9%, default budget)

## Raised-budget agentic spot-check (SINGLE RUN, labeled as such)

probe-bud43008-agentic-r1.log: prime 0.105s = 6038 tok/s (base 2781 = +117%; default-
budget new 4403 = +37%), plain 90.18, spec K=3 146.27, acceptance 84.7% (base 84.0) —
no acceptance regression on the agentic class at fuller coverage.

## Round 49b increment: Q5_K f16 mirrors (48 ssm_out tensors, 176B superblock)

Q5_K (the last mul_mat_q_q45k prefill class in q27 — GDN ssm_out [6144,5120] x48) rides a
THIRD budget pass strictly after all Q4_K, so the default-budget composition — and every
banked 49 gate/A/B above — stays byte-identical (verified: 49b binary at default budget
builds the same 183 Q4_K mirrors / 22420 MB, zero q5k, argmax maxdiff bit-same 6.513e-1).
Unpack per qmatvec.cu deq_q5_k: same get_scale_min_k4 scales, qh bit g of qh[l].

- Gates (49b binary): kernel-check rc=0 fails=0 — NEW battery case blk.0.ssm_out.weight
  Q5_K GEMM (rel 3.2e-7) + Q5_K f16 mirror (rel 2.4-6.0e-3, band 1e-2)
  (kernel-check-q5k.log); run-gen argmax MATCH default (6.513e-1) AND q5k-active config
  (4.231e-1, MEMRA_KQRP=0 MEMRA_PP_F16_BUDGET_MB=50688); run-spec K=1..8 8/8 PASS
  (runspec-k1-8-q27-49b.log).
- Marginal value probe (x2 interleaved, KQRP off to free VRAM; probe-q5k-bud*.log):
  budget 47616 = full q4k (288 tensors, 35.6GB) + 27/48 q5k -> prime 0.266-0.267s;
  budget 50688 = + full q5k (2880 MB) -> prime 0.254-0.255s = 8063 tok/s (+152% vs
  base 3205). Q5_K tail marginal ~9.5ms/GB.
- ON THIS BOX the Q5_K tier is dark in the serving config: full q4k+q6k = 45GB budget
  already exceeds what fits beside the KQRP decode mirrors (round 48, +15% decode), and
  Q5_K only admits after the whole Q4_K class. It pays on bigger-VRAM boxes or KQRP-off
  diagnostic runs; the class is now implemented + battery-gated either way.

## Next wall (nsys, ONE whole-run capture at 43008 — unwindowed, direction only)

nsys-kern-sum-bud43008.txt (run-gen NGEN=8 board-2048; the .nsys-rep binary, 237MB,
stays box-side — the text summary is the committed evidence). Whole-run totals, NOT
prime-windowed: the remaining non-f16 prefill-class kernels are the un-mirrored tail
riding mul_mat_q_q45k — q5_K <128,0,5> 192 calls / 151ms total and the q4_K budget tail
<128,0,4> 92 calls / 148ms total — while the Lt f16 GEMM (nvjet_hss 320x128) runs 960
calls at med 467us. Named next rung: Q5_K f16 dequant mirror (48 tensors, 3.0GB @2B/w,
176B superblock — same get_scale_min_k4 unpack + qh bit) AND the q4_K tail — both are
budget/VRAM-bound on this box (full coverage needs ~47GB f16 budget -> ~80.5GB peak;
the 81.5GB card says the last ~5GB of coverage is an OOM-margin owner call). A proper
prime-windowed attribution (NVTX or two-capture subtraction) is the follow-up.

## Files

- crates/memra-engine/cu/f16_prefill.cu   — memra_q4kf16_dequant_kernel + memra_q4_K_dequant_f16
- crates/memra-engine/src/f16_ffi.rs      — extern + build_q4k_f16_raw + Q4_K admission in build_q8_f16
- crates/memra-engine/src/hybrid.rs       — Q8RP walk: Q4_K second pass (budget priority to Q6_K)
- crates/memra-engine/src/bin/kernel_check.rs — f16-mirror gate coverage for Q4_K + Q6_K

Thermal regime: all comparisons interleaved same-session on a dedicated GPU 3; no
cross-day denominators. Raw logs: ab-logs/ (12 runs), probe-bud*-r*.log (4 runs),
kernel-check.log, rungen-argmax-*.log, runspec-k1-8-q27.log, vram-samples.log (5s
cadence across the A/B + probe window), build log = build-arc4.log.
