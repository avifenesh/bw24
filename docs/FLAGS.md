# Environment flags — the audited catalog

Doctrine (CLAUDE.md): **winners are defaults** — naked commands run the tuned path. A flag exists only when it is (a) runtime parameter, (b) machine-specific config, (c) documented rollback seam, (d) diagnostic, or (e) explicitly-blocked experimental door. When an experiment concludes negative or flat, the flag and dispatch arm are deleted — the `research/tune-data/*.jsonl` row is the record, not dead code. The 2026-07-08 flag audit enforced this repo-wide; removed-flag ledger at bottom.

Provenance dates refer to `research/tune-data/rig5090.jsonl` / `g7e-rtx6000.jsonl` rows and
HANDOVER.md sections from that date.

---

## 1. Runtime parameters

### Generation (run-gen / run-spec / run-eagle)

| flag | default | what it does |
|---|---|---|
| `BW24_PROMPT` | numeric ids `101..612` | prompt TEXT (tokenized with the model's tokenizer) |
| `BW24_PROMPT_FILE` | — | prompt text from a file (wins over `BW24_PROMPT`) |
| `BW24_NGEN` | bin-specific (64–256) | number of tokens to generate |
| `BW24_CHAT` | off | `1` wraps the prompt in the model's chat template (run-gen) |
| `BW24_PRINT_TEXT` | off | `1` decodes gold tokens between markers (agent-loop harness, run-spec, 2026-07-08) |
| `BW24_GEN_ONLY` | off | run-spec: skip prime-inclusive timing, report gen-only |
| `BW24_SEED` | 0 | sampler seed (run-gen sampling path) |
| `BW24_TEMP` | greedy | temperature; enables the sampling chain |
| `BW24_TOP_K` / `BW24_TOP_P` / `BW24_MIN_P` | off | sampling filters (run-gen) |
| `BW24_PENALTY_REPEAT` / `_FREQ` / `_PRESENT` / `_LAST_N` | off | repetition penalties (run-gen) |
| `BW24_STOP` | EOS only | comma-separated stop strings (run-gen) |

### Speculative decode (run-spec / serve)

| flag | default | what it does |
|---|---|---|
| `BW24_SPEC_K` | 3 | draft depth K. Per-(model,content): 27B K=3, 9B K=2–3, 35B K=2 (2026-07-05/06 sweeps) |
| `BW24_SPEC_PMIN` | 0.0 | p-min confidence gate — stop the draft chain when head confidence drops below it (0.15–0.3 typical) |
| `BW24_SPEC_PMIN0` | off | `1` lets p-min gate slot 0 too (zero-draft rounds). Pays below ~75% base acceptance, hurts above ~90% (2026-07-08, 35B +13–23 tok/s) |
| `BW24_SPEC_HPOST` | off | `1` feeds the MTP head POST-output_norm hidden — acceptance lever on the 27B (>100 tok/s crossing, 2026-07-06). Per-model choice |
| `BW24_FRSPEC_TRIM` | off | `<frspec.gguf>`: self-trimmed draft lm_head — gathers top-frequency rows from the model's own output.weight via the file's d2t ranking (generic ranking transfers across same-vocab heads; specialized ones do not — 2026-07-07) |
| `BW24_MTP_DRAFT` | off | `<draft.gguf>`: replace the MTP head with a standalone draft GGUF (exactness unaffected — verify arbitrates) |

### Benchmark harness (run-gen / session-bench)

| flag | default | what it does |
|---|---|---|
| `BW24_PP_ONLY` | off | prefill-only timing mode |
| `BW24_PP_REPS` | 1 | prefill repetitions (N-median protocol) |
| `BW24_PP_WARMUP` | 1 | prefill warmup runs |
| `BW24_NMEASURE` | 32 | decode tokens measured after counter reset |
| `BW24_TURN_TOKENS` | 512 | session-bench: suffix length per turn |
| `BW24_HIST_FILE` | required | session-bench: conversation history text file |

### Server (bw24-server)

| flag | default | what it does |
|---|---|---|
| `BW24_MODELS` | — | model list `alias=path,...` |
| `BW24_ADDR` | bind default | listen address |
| `BW24_API_KEY` | none | bearer key; setting it also defaults `BW24_COMPAT=openai` |
| `BW24_COMPAT` | native (`openai` when API_KEY set) | response shape: `openai` = OpenAI completions SSE |
| `BW24_CTX` | 8192 | session-cache context floor (KV @8192 ≈ 119MB/session on the 9B) |
| `BW24_KV_REUSE` | on | `0` disables the KV prefix-reuse pool (session-gate validated; 42.6x turn-start at 40k) |
| `BW24_SERVE_SPEC` | on | `0` disables the spec-decode serve path (greedy + MTP-head requests) |
| `BW24_SPEC_BURST` | 32 | tokens per spec burst — round-robin latency vs per-burst fixed cost (throughput-neutral, 2026-07-06 A/B null) |

---

## 2. Machine-specific config

| flag | default | what it does |
|---|---|---|
| `BW24_PP_FP8` | off | `1` enables the cuBLASLt FP8-activation prefill GEMM on FP8-native (safetensors) checkpoints. +78–129% pp on g7e (2026-07-08); local NV-27B pp1855 1341→1412 (+5.3%, 2026-07-09). Opt-in: costs resident-copy VRAM (`BW24_PP_FP8_BUDGET_MB`); `BW24_ST_E4M3` is the no-duplicate route |
| `BW24_PP_FP8_BUDGET_MB` | 1536 | VRAM budget for resident FP8 weight copies (rig-dependent; 2560 = +16.6% pp1845 on 24GB, 2026-07-08). Irrelevant under `BW24_ST_E4M3` (one copy, no stash) |
| `BW24_ST_E4M3` | off | `1` = F8-E4M3-origin safetensors projections keep RAW e4m3 as the ONE resident copy (no Q8_0 re-encode): decode dequants e4m3 in-kernel (`qmatvec_e4m3_mmvq` + batched twins), prefill rides the FP8 GEMM on the same bytes. NEW NUMERIC CONFIG (checkpoint-native weight precision — the Q8_0 hop was lossy); gate the full battery in-config. Frees ~GBs on 24GB rigs → full FP8 prefill coverage (lane e4m3dec 2026-07-08) |
| `BW24_KV_K` | `q8_0` | K-cache format arm. `fp8` FLIP-BLOCKED (2026-07-09 A/B, tag kvk-fp8-ab-9bst): e2e FLAT (+0.3-0.9% ms/tok — the −7-14% micro didn't survive; micro≠e2e) AND 9B ST spec self-consistency FAIL (acceptance 74%→20.5%, drift accumulates across K reads). q8_0 stays |
| `BW24_KV_V` | `q5_1` | V-cache format arm: `q4_0` FLAT + quality-taxed (argmax MISMATCH in-config), `fp8` borderline (−2pt) — measured 2026-07-08; q5_1 stays |
| `BW24_FA_SMEM_TKV` | 1024 | t_kv crossover to the smem-broadcast FA decode twin (`0` = never). Swept 2026-07-05: flat 512–2048 on real prompts; micro says lower may pay on the 35B (untested e2e arm: 96) |
| `BW24_FA_SPLIT` | ctx-adaptive | force fixed FA split-keys. EXACTNESS: split count changes combine FP order — run-spec must re-gate (32 broke 9B self-consistency, 2026-07-03/07) |
| `BW24_PRIME_CHUNK` | 4096 | chunked-prime chunk size in tokens (`0` = monolithic). Long-ctx OOM/transient control |
| `BW24_MOE_VRAM_FRAC` | 0.85 | SLRU expert-cache fraction of free VRAM (sweep 2026-07-06: 0.40=25.0 → 0.85=28.5 tok/s). Lower it on rigs co-running other GPU work |
| `BW24_MOE_SLOTS` | auto | force an exact SLRU slot count (spill experiments used 64/512) |
| `BW24_MOE_RESIDENT` | on | `0` forces the SLRU path even when experts fit VRAM (fits-VRAM resident = 169.55 vs 28.5 tok/s on the local 35B) |
| `BW24_MOE_RESIDENT_GB` | 80% of free | resident-experts budget override (g7e M3 partial-resident tier) |
| `BW24_MOE_PINNED` | pinned when MOE_CACHE on | force pinned host expert slabs |
| `BW24_SPILL_DISK` | off | set = enable the NVMe disk tier for MoE experts (M3-class models that exceed host RAM) |
| `BW24_SPILL_PINNED_FRAC` | 0.60 | fraction of MemAvailable the spill tier may pin |
| `BW24_ST_PINNED` | off | `1` pins the safetensors expert store — ONLY for fits-in-RAM checkpoints (pinning 26GB evicted the page cache: 30x regression, 2026-07-07) |
| `BW24_ST_REPACK_DISK` | on | `0` forces in-RAM gather instead of the `.bw24-repack` disk cache (safetensors stream-repack loader) |
| `BW24_KQ_NVFP4` | 0 | load-time k-quant→NVFP4 re-encode: `1` = Q4_K, `2` = +Q5_K. +3.9% 9B plain but ~2x quant error and an acceptance tax — bpw equality ≠ quality-class equality (2026-07-08). Speed-mode opt-in only |
| `BW24_MMQ_F8F4` | off | `1` = W4A8-FP8 prefill MMQ (e4m3 fold + f8f6f4 MMA, 381-TF class). pp +3.9-6.3% ALL models, TTFT -4-5.6%; e2e spec MODEL-SIGNED via the prefill-KV acceptance law (27B ST +7.2%, 9B -3.5/-6.1%) — per-model serve adoption (NV-27B ST config), never a global default (2026-07-10 flip battery). Expert-tile twin measured NEGATIVE (-6.4% best variant, not pipe-bound) and was deleted — tag moe-f8f4-negative |
| `BW24_NV_W4` | off | `1` re-quants F8 attention weights → NVFP4 on the NVIDIA-official 27B (+20% plain, acceptance held, 2026-07-07). Opt-in until the text battery proves the class |

### Build-time (build.rs / nvcc)

| flag | default | what it does |
|---|---|---|
| `BW24_NVCC` | `nvcc` on PATH | nvcc binary override |
| `BW24_CUDA_ARCH` | 120a | target arch (the sm_89 lane built with `89`; lane closed) |
| `BW24_CUTLASS` | off | set = compile the CUTLASS sm120 NVFP4 GEMM (`cutlass_smoke`, `BW24_FP4_CUTLASS` door) |
| `BW24_CUTLASS_ROOT` | flashinfer venv tree | CUTLASS header tree location |
| `BW24_MMQ_X_Q45K` | 64 | k-quant MMQ X-tile (compile-time sweep seam) |
| `BW24_MMQ_X_W4A8` / `BW24_MMQ_Y_W4A8` | 128 / 128 | W4A8 MMQ tile sizes. Swept 2026-07-06: defaults optimal on the 5090 (X=32 −28%, Y=64 −2%); kept as autotuner levers for other silicon |
| `BW24_GEMM_K1_LAUNCH` | shipped (128,128,8) | `"BM,BN,NWARP"` kernel1 launch-tile override — MUST match the `-D K1_BM/K1_BN/NWARP` the swept fatbin was built with (tools/sweep) |

`BW24_*_FATBIN*` (ENGINE/FLASH/GEMM/QMATVEC/ROUTER/HYBRID) are **internal build plumbing**
(build.rs → rustc-env fatbin paths), not user flags; `BW24_KV_KFMT`/`BW24_KV_VFMT` are the
`-D` defines build.rs derives from `BW24_KV_K`/`BW24_KV_V` per flash fatbin. `BW24_MMVQ_ROWS`
is a compile-time `#define` in qmatvec.cu, not an env var.

---

## 3. Rollback seams (default ON — `=0`/set reverts to reference path)

These exist because correctness discipline needs a same-binary oracle. Each is a measured winner; the seam is the documented way back.

| flag | revert semantics | provenance |
|---|---|---|
| `BW24_FAST=0` | Stage-A f32-dequant matvec class — THE correctness oracle | default-on 2026-07-08 (env-law retirement) |
| `BW24_MMVQ=0` | dp4a matvec class (m=1 AND batched verify switch together — dispatch-parity law) | default-on 2026-07-08; parity fix 2026-07-07 |
| `BW24_MOE_CACHE=0` | stage-every-token expert dispatch (no SLRU) | default-on 2026-07-08 |
| `BW24_MOE_PREFETCH=1` | pipeline the next routed expert's cache misses on the copy stream | experimental; target-rig gate required |
| `BW24_MOE_PAGE_PREFETCH=1` | issue rolling `MADV_WILLNEED` for mmap-backed GGUF/repack ranges | experimental; cold-cache G7e + RTX 5090 A/B required |
| `BW24_MOE_PAGE_PREFETCH_WINDOW` | future experts kept in the rolling page-prefetch window (default `1`; `0` disables) | only read when `BW24_MOE_PAGE_PREFETCH=1`; tune to storage latency/page-cache budget |
| `BW24_NO_FA_VEC` (set) | scalar `fa_decode_f32` bit-reference (eager + rows + graph in lockstep) | vec default-on 2026-06-28 |
| `BW24_FA_V2=0` | per-key online-softmax FA twins (v2 = tile-batched, own numeric config) | default-on 2026-07-08, the depth-slope fix |
| `BW24_FA_ROWS_OFF=1` | per-row verify FA loop instead of the fused rows kernel | rows landed 2026-07-03 (+13.8% 9B p2) |
| `BW24_SPEC_LEAN=0` | zeroed verify buffers + rows dispatch at t=1 | default-on 2026-07-08 (+1.5–2.4% 35B) |
| `BW24_SPEC_M2=0` | per-m grid.y=m verify dispatch at t=2 (no small-m batched twin) | default-on 2026-07-09 (flattened the 35B verify K-curve; K=4 within 1–3% of K=3) |
| `BW24_SPEC_FUSED_T=0` | per-tensor decode-exact verify trunk calls (no t=2-4 launch fusion) | default-on 2026-07-09; bit-identical per (tensor,token,row) by construction (kernel-check Q8-FUSED2-B/FUSED3-B gates) |
| `BW24_FA_V3=0` | FA v2 decode twins (v3 = dp4a-K hybrid, own numeric config — int8 Q quantization) | default-on 2026-07-09 after full battery green (kernel-check + run-gen MATCH 35B/9B + run-spec PASS + graph gate bit-identical); requires default q8_0/q5_1 KV + hd%128==0 (host-gated, auto-falls back to v2) |
| `BW24_SPEC_REPLAY=1` | legacy rollback+replay partial accept (also the j==0 fallback) | replay-free default 2026-07-03 (+10–32%) |
| `BW24_SPEC_NOREFRESH=1` | chain-approximate draft-KV entries (no true-hidden refresh) | refresh default 2026-07-03 (+4–6% acc) |
| `BW24_SPEC_NOGRAPH=1` | eager draft chain (no CUDA-graph draft) | graph draft 2026-07-03 |
| `BW24_PRIME_TOKENWISE=1` | tokenwise decode-step prime (escape; <16-tok prompts take it anyway) | batched prime 2026-07-03 (23x TTFT at 6k) |
| `BW24_PRIME_APPEND_LOOP=1` | per-row KV append instead of the batched `_rows` kernel | measured equal 2026-07-03 |
| `BW24_PRIME_DEQW=0` | inline-dequant prefill FA (no bf16 dequant-once workspace) | deqw default 2026-07-05 (32k prime 1.60x) |
| `BW24_PRIME_DEQW_DB=0` | single-buffer workspace staging (no cp.async double-buffer) | 2026-07-05 |
| `BW24_GDN_CHUNKED=0` | sequential GDN prefill scan | chunked default 2026-07-04 (+4.6% pp512 9B); `BW24_GDN_CHUNK` (default 32) = chunk size |
| `BW24_B8=0` | m=5..8 verify back to per-m grid.y=m dispatch | b8 tier 2026-07-05 (K=4 cliff fix, +30%) |
| `BW24_NO_BATCHED` (set) | per-m grid.y=m path for ALL m=2..8 — the batched-verify A/B reference | 2026-07-03 |
| `BW24_Q5K_ISSUE=0` | reference q5_K MMVQ bodies (`2` = force-il A/B probe) | il default 2026-07-08 (+1.8% 9B plain) |
| `BW24_Q8_DUAL=0` | two separate q8_0 matvec launches (no gate+up fusion) | fused 2026-07-03 |
| `BW24_NO_FUSE_NORMQ` (set) | unfused rms_norm + quantize (decode norm-fusion off) | fused 2026-07-03 |
| `BW24_NO_GEMM` (set) | force the dp4a fallback for prefill matmuls — the int8-GEMM bit-reference | 2026-06-28 dispatch |
| `BW24_NOFA` (set) | naive SDPA instead of the hand-written prefill FlashAttention (also the auto-fallback for unstamped head_dims) | FA prefill 2026-06 |
| `BW24_KS=0` | drop the 2026-07-06 rpsc smem-scale-prestage entries from variant AUTO | rpsc landed 2026-07-06 (g7e +2.6% K=4; 5090 neutral, harmless) |
| `BW24_EVT=1` | restore cudarc cross-stream event tracking (elision default: −7ms/tok host on 35B) | elision default 2026-07-05 |
| `BW24_RP=0` | GGUF-layout NVFP4 (no A6 split-plane repack); also the W4A4 door key, see §5 | rp default 2026-07-05 |
| `BW24_ST_DIRECT=0` | safetensors NVFP4 → GGUF-layout conversion instead of direct split-plane load | 2026-07-07 |
| `BW24_MMQ_W4A8=0` | int8 GEMM prefill everywhere (no W4A8 MMQ tile) | W4A8 default 2026-07-05 (1.54–1.9x prime) |
| `BW24_MOE_Q8=0` | Stage-A f32-dequant expert kernels — restores BYTE-identity for the MOE_GATE oracle | dp4a experts 2026-07-06 (+22%) |
| `BW24_MOE_Q8_KQ=0` | exclude k-quant arms from the q8 expert dot set | 2026-07-06 (+9 tok/s 35B) |
| `BW24_MOE_DEC=0` | `_em` per-token re-decode expert dot (no decode-once) | dec default 2026-07-05 (3.34x 35B prefill) |
| `BW24_MOE_GDEC=0` | sequential per-expert launch chain (stage-2 grouped decode off) | 2026-07-04 |
| `BW24_MOE_DEV=0` | host routing (no zero-DtoH device dispatch); `BW24_FUSED_ROUTER=0` implies it | stage-3 2026-07-05 |
| `BW24_FUSED_ROUTER=0` | host softmax+sort routing | fused router 2026-07-05 (g7e stage-3 arc) |
| `BW24_MOE_PAIRS=0` | per-expert loop for real prefill (no pair-batched launches) | 2026-07-06 |
| `BW24_MOE_PREWARM=0` | organic SLRU residency (no one-shot layer prewarm) | 2026-07-05 |
| `BW24_MOE_MMA=0` | dp4a expert prefill (no int8-MMA expert MMQ; t floor 16 keeps verify on dp4a) | 1.5x pp 2026-07-05; spec-safety floor 2026-07-06 |
| `BW24_MOE_DEVQ8_GU` / `BW24_MOE_DEVQ8_DOWN` | force dev-q8 kernel variants (auto = measured winners w8h2 / GU=v, +2.5% 35B) | down8 merge 2026-07-08 |
| `BW24_MOE_DEVQ8_WPB` (default 4) | warps/block for the `_r` twins (probe knob) | 2026-07-06 |

## 4. Diagnostics & test config

| flag | what it does |
|---|---|
| `BW24_MOE_STATS=1` | per-layer expert-cache hit/miss/staged-bytes prints (forces the stats-visible dispatch path) |
| `BW24_MOE_TRACE=<path>` | append (layer, step, expert ids) per decode step — routing-locality analysis (`research/scripts/moe_trace_analyze.py`, 2026-07-07 M3 measurement) |
| `BW24_SPEC_STATS=1` | per-slot accept histogram + draft-length histogram |
| `BW24_DEBUG_SPEC=1` | per-round spec decode trace |
| `BW24_MOE_CSR` | `0` = rollback the CSR expert-dedup gate_up on spec verify (default ON 2026-07-10: owner-scan dedup of the 38-40% duplicated expert weight-stream+decode, +1-2% spec e2e all K); `2` = run BOTH paths + byte-compare (debug) |
| `BW24_MOE_OVERLAP` | `1` = log cross-token expert-activation overlap at spec verify (unique/pairs ratio, diagnostic) |
| `BW24_FA_V4` | `0` = rollback to the v3 FA lane. DEFAULT ON 2026-07-10: key-per-lane score phase (zero shuffles/key), kernel -7.5..-15% at depth, e2e p2 +5.7% (35B), p3 +0.7-3.9% (all models), p1 -0.8% (noise); full battery green on all three models incl sampled seeded identity + kernel-check; acceptance deltas trajectory-class (±). `noB3`/`stage` = phase probes (WRONG OUTPUT, bench only) |
| `BW24_SPEC_DEVACC` | `1` = device-side accept walk + seed gather + KV/recur rollback (round-stream stages a/b; token-identical, perf-neutral machinery — the stage-c consumer measured negative) |
| `BW24_SPEC_STREAM` | `1` = pre-issued M-round zero-readback bursts (round-stream stage c, `BW24_SPEC_STREAM_M` rounds/burst default 4). MEASURED NEGATIVE 2026-07-10 (35B serve p2 -16%, p1 -4%; 27B p2 wash): always-K draft + fixed-width verify waste > the ~1.5-2ms/round trip savings at real acceptance rates. Kept as the experimental seam; token-identity holds by construction |
| `BW24_ROUTER_KERNEL` | `0` = rollback to the cuBLAS router (per-column gemv at verify t, gemvx at t=1). Default ON since 2026-07-10, extended to t=1 the same day: decode + verify route through the SAME kernel (parity by construction — the cuBLASLt n-dependence class is structurally gone). Verify +2-4% spec e2e; t=1 e2e wash (kept for parity) |
| `BW24_PROFILE_GEN` | `1` = cudaProfiler{Start,Stop} brackets run-gen's timed generate_with (prime included); `2` = capture starts at the DECODE LOOP (prime excluded) — pair with `nsys -c cudaProfilerApi`. Window-cutting whole-run captures misattributes the argmax-gate loop into decode shares (2026-07-10) |
| `BW24_PROFILE_SPEC` | `1` = cudaProfiler{Start,Stop} brackets generate_spec (prime included); `2` = capture starts at the ROUND LOOP (prime excluded) — pair with `nsys -c cudaProfilerApi`. Built 2026-07-10 after phase-isolation-by-subtraction proved unworkable on MoE (primes are not fungible: the first cold-stages the expert cache) |
| `BW24_LAYER_PROBE=1` | sync+print after every forward stage — bisects an in-graph ILLEGAL_ADDRESS (M3 bring-up tool) |
| `BW24_GDN_DIFF=1` | dual-run oracle: chunked GDN prefill checked against the sequential scan per call |
| `BW24_MOE_GATE=1` | byte-identity oracle: grouped-vs-sequential MoE FFN compare (pair contract; known benign q8-quantize diff class documented at the gate site) |
| `BW24_MOE_GDEC_GATE=1` | byte-identity oracle: grouped-decode vs sequential-axpy expert accumulation |
| `BW24_MMVQ_BV=<variant>` | force ONE NVFP4 batched/mmvq variant everywhere (base/pf/r2/r2w8/pfr2/ca/car2/rp*) — the per-variant measurement seam (auto = wave-aware winners; forced-BV concluded auto-optimal 2026-07-06) |
| `BW24_KQ_BV=<variant>` | same, narrowed to k-quant dispatch only (interleaved k-quant A/B without touching NVFP4) |
| `BW24_FA_FLOOR=1` | prefill-FA floor kernel variant (fa_hd128_check gate bin) |
| `BW24_MSCALE_PROFILE` / `BW24_MSCALE_NOEAGER` | verify-mscale probe bin controls |
| `BW24_TEST_MODEL`, `BW24_LLAMA_TOKENIZE`, `BW24_ST_TEST_DIR` | test-suite input paths (tokenizer parity, safetensors header tests) |

## 5. Experimental doors (opt-in, documented block)

| flag | door | block |
|---|---|---|
| `BW24_MMQ=1` (+`BW24_RP=0`) | native W4A4 FP4 MMQ prefill — 1.03–1.06x llama, 1.4–1.76x our default (2026-07-08) | EXACTNESS-BLOCKED: e2m1 activation grid forks argmax/text on long real prompts (p3 reject reproduced 2026-07-07 + 2026-07-08; agent-loop 1/8 self-consistency FAIL). Speed-mode candidate, never default |
| `BW24_FP4=1` | hand-rolled W4A4 GEMM in `matmul` (decode/mid-m band) | same accuracy class (maxdiff ~1.0 vs W4A8 0.159); explicit speed/accuracy tradeoff only |
| `BW24_FP4_CUTLASS` (+build `BW24_CUTLASS`) | CUTLASS sm120 NVFP4 GEMM for m>=128 prefill (`BW24_FP4_CUTLASS_OTF=1` = per-call repack, no resident VRAM doubling) | same W4A4 exactness block + resident repack ~doubles NVFP4 weight VRAM (OOMs 27B/24GB) |
| `BW24_MOE_GROUPED=1` | expert-grouped MoE prefill prototype (spill-regime 3.4–3.9x vs sequential-stage at cap64, 2026-07-04) | superseded by pairs/dev/MMA on the daily path (barely moves it, 2026-07-06); kept as the `BW24_MOE_GATE` oracle pair + seed of the specced fused-MoE-MMQ prefill arc |
| `BW24_MOE_Q8_NVFP4=1` | NVFP4 expert dp4a dot (M3) | BLOCKED: broke the M3 decode-vs-verify gate (3.4e1); M3 stays f32 until macro-handling parity is proven. Also irrelevant while M3 is PCIe-bound |
| `BW24_MOE_MMA_T=<n>` | MMA t-floor override (bisect seam; <16 puts spec verify on MMA) | verify must stay dp4a (dispatch-parity law) — measurement only |
| `BW24_IQ_FAST` | opt-in IQ4_XS fast matvec (non-expert path) | UNCLEAR — no concluding JSONL row found; left untouched by the audit |
| `BW24_EAGLE` / `BW24_EAGLE_ALIGN=0` | EAGLE draft lane (run-eagle bin; ALIGN=0 = un-shifted MTP-style pairing A/B) | experimental lane, not on the daily path |

---

## 6. Research platform (MTP-heal, dual-shape)

bw24's second shape is a research platform. The first protocol (MTP-heal) measures MTP draft-head
acceptance at FULL PRECISION as the ceiling, then on the NVFP4 daily GGUF — the delta is the quant
hit on drafting. See HANDOVER "BW24 DUAL-SHAPE".

| flag | what it does |
|---|---|
| `BW24_FULL_PREC=1` | FULL-PRECISION LOADER MODE (default OFF). Bypasses the standing loader law (large BF16/F8 → Q8_0/NVFP4 re-encode, the "Float-poison" tripwire — the very re-encodes this mode must NOT do). Everything loads Float; large 2D bf16 matmul weights stay **bf16-resident** (`GpuTensor::FloatBf16`) with dequant-on-use, so the 9B (~18GB bf16) + f32 activations fit 24GB instead of blowing to ~38GB as an all-f32 materialization. Compute rides the Stage-A f32 oracle path end to end (`BW24_FAST=0`-class). **SLOW IS FINE** — this mode is for the exactness ceiling, not speed. Also forces the EAGER spec draft (CUDA-graph capture can't enclose cuBLASLt f32 GEMV or the dequant alloc) and disables `BW24_FRSPEC_TRIM` (the ceiling wants the model's natural full head; the trim gather is Quant-only anyway). The Float-poison tripwire warnings are correct here and are suppressed. |

Per-slot acceptance for this protocol comes from the existing **`BW24_SPEC_STATS=1`** (§4) — its
`per_slot=[...]` line is the deep-K decay profile; no separate flag was needed.

**Acceptance battery harness** (`tools/`, no GPU-free CI — runs on the rig):
- `acceptance_battery.sh <model> <out.jsonl>` — fixed prompt set (p1/p2/p3) + the 8-turn agent loop, N≥3, one JSONL row per (prompt,K,run). `FULL_PREC=1` for the bf16 ceiling arm, plain for the NVFP4 arm.
- `agent_loop_acceptance.sh <model> <out.jsonl> <arm>` — the 8-turn accumulative agent-loop protocol (recreated in-repo; feeds each turn's output forward).
- `acceptance_parse.py` — one run-spec invocation (merged stdout+stderr, `BW24_SPEC_K=<k>` + `BW24_SPEC_STATS=1`) → one JSONL row.
- `acceptance_delta.py <bf16.jsonl> <nvfp4.jsonl>` — the deliverable delta table: per-(prompt,K) median acceptance, ceiling vs quant, and the hit.

```
# bf16 ceiling:
FULL_PREC=1 tools/acceptance_battery.sh /data/ai-ml/hf-models/qwen35-9b-hf out-bf16.jsonl
# NVFP4 hit:
tools/acceptance_battery.sh /data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf out-nvfp4.jsonl
# delta:
tools/acceptance_delta.py out-bf16.jsonl out-nvfp4.jsonl --json summary.json
```

---

## Removed in the 2026-07-08 flag audit (concluded flags — JSONL rows are the record)

| flag | verdict | record |
|---|---|---|
| `BW24_FA_PPOOL` | micro −5–12% but e2e +0.5–0.8% at d6257, under bar | rig5090 2026-07-08 (fadepth) |
| `BW24_FA_CMB_WIDE` | wide combine ~−1µs — the 98-split serial FP chain bounds it, not SM count | rig5090 2026-07-08 (fadepth) |
| `BW24_FA_SK` | shared-K verify fold: occupancy collapse, p3 spec 108→51.9 tok/s | g7e 2026-07-05 |
| `BW24_Q4V` | issue-reduced q4_K matvec: DRAM hides it, ~+0.08% decode = neutral class | rig5090 2026-07-08 (q4issue) |
| `BW24_Q6_ISSUE` (5 variants) | all variants identical at locked clock — the premise refuted itself (DVFS artifact) | rig5090 2026-07-08 (q6issue) |
| `BW24_GDN_FUSE` | fused GDN prep+scan: NEUTRAL on eager | rig5090 2026-07-08 (merge row) |
| `BW24_MMVQ_MR` (+mr4, q4_K/q6_K mr2 kernels) | override never used: q4_K mr2 +0.7% (noise), q6_K flat, mr4 crashes | 2026-07-05/07 notes; HANDOVER |
| `BW24_GEMM_M` | m=4 MMA verify NEGATIVE (96.6 vs 99.1; grid starves + FP-order acceptance dip) | rig5090 2026-07-06 |
| `BW24_MMQ_STREAMK` (+sk/fixup kernels) | 1.11x per-GEMM but k-split f32 reorder flips model argmax (FP-order lesson #3) | rig5090 2026-07-03 |
| `BW24_SPEC_ADAPT` | adaptive-K: honest loss to static per-class optima (EMA lag) | rig5090 2026-07-07 |
| `BW24_SPEC_KVLOCAL` | legacy round-local draft scratch: −35 accept pts, incompatible with sessions | rig5090 2026-07-03 sweep; HANDOVER |
| `BW24_SPEC_HSAME` (+pseudo-seed passes/graph) | legacy same-row pairing: −16 accept pts vs predecessor pairing | rig5090 2026-07-04 |
| `BW24_MOE_GHOST` + `BW24_MOE_FAST_ADMIT` | second-miss ghost filter: net loss in BOTH regimes (spill +3.4% with it off; 96GB bypassed it) — first-miss admit is the only policy, FAST_ADMIT became a no-op | rig5090 2026-07-06 / g7e 2026-07-04 |
| `BW24_MOE_ORDER` | descending-m_e expert order won (1.34x first-forward); the `=id` restore arm was dead | rig5090 2026-07-04 |
| `BW24_FA_VEC` | vestigial: dispatch reads `BW24_NO_FA_VEC` since the 2026-06-28 default-flip. Writes removed; the kernel-check scalar-vs-vec gate now toggles `BW24_NO_FA_VEC` (the old toggle had gone vacuous) | audit 2026-07-08 |
| `BW24_GEMM` | zero read sites (gate went unconditional when MMQ Phase 0 shipped); writes removed from scripts/docs | audit 2026-07-08 |
| `BW24_NO_EVT` | legacy no-op alias of the EVT elision default | audit 2026-07-08 |

Bench bins deleted with their flags: `q4v_bench`, `q6iss_bench`, `fa_ppool_bench` (superseded by
`fa_v2_bench`, which keeps the FA depth/split/smem sweep).
