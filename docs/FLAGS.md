# Environment flags — the audited catalog

Doctrine (CLAUDE.md): **winners are defaults** — naked commands run the tuned path. A flag exists only when it is (a) runtime parameter, (b) machine-specific config, (c) documented rollback seam, (d) diagnostic, or (e) explicitly-blocked experimental door. When an experiment concludes negative or flat, the flag and dispatch arm are deleted — the `research/tune-data/*.jsonl` row is the record, not dead code. The 2026-07-08 flag audit enforced this repo-wide; removed-flag ledger at bottom.

Provenance dates refer to `research/tune-data/rig5090.jsonl` / `g7e-rtx6000.jsonl` rows and
HANDOVER.md sections from that date.

---

## 1. Runtime parameters

### Generation (run-gen / run-spec / run-eagle)

| flag | default | what it does |
|---|---|---|
| `MEMRA_PROMPT` | numeric ids `101..612` | prompt TEXT (tokenized with the model's tokenizer) |
| `MEMRA_PROMPT_FILE` | — | prompt text from a file (wins over `MEMRA_PROMPT`) |
| `MEMRA_NGEN` | bin-specific (64–256) | number of tokens to generate |
| `MEMRA_CHAT` | off | `1` wraps the prompt in the model's chat template (run-gen) |
| `MEMRA_PRINT_TEXT` | off | `1` decodes gold tokens between markers (agent-loop harness, run-spec, 2026-07-08) |
| `MEMRA_GEN_ONLY` | off | run-spec: skip prime-inclusive timing, report gen-only |
| `MEMRA_SEED` | 0 | sampler seed (run-gen sampling path) |
| `MEMRA_TEMP` | greedy | temperature; enables the sampling chain |
| `MEMRA_TOP_K` / `MEMRA_TOP_P` / `MEMRA_MIN_P` | off | sampling filters (run-gen) |
| `MEMRA_PENALTY_REPEAT` / `_FREQ` / `_PRESENT` / `_LAST_N` | off | repetition penalties (run-gen) |
| `MEMRA_STOP` | EOS only | comma-separated stop strings (run-gen) |

### Speculative decode (run-spec / serve)

| flag | default | what it does |
|---|---|---|
| `MEMRA_SPEC_K` | 3 | draft depth K. Per-(model,content): 27B K=3, 9B K=2–3, 35B K=2 (2026-07-05/06 sweeps). `run_hy3_local_5090.sh` additionally accepts `all` to omit the override and run the mandatory K=1..8 battery in one model load |
| `MEMRA_SPEC_HOST_EMBD` | off | `1` keeps the token-embedding table in host memory for the Hy3 spill profile, reducing HBM pressure at the cost of a host-row transfer |
| `MEMRA_SPEC_PMIN` | 0.0 | p-min confidence gate — stop the draft chain when head confidence drops below it (0.15–0.3 typical) |
| `MEMRA_SPEC_PMIN0` | off | `1` lets p-min gate slot 0 too (zero-draft rounds). Pays below ~75% base acceptance, hurts above ~90% (2026-07-08, 35B +13–23 tok/s) |
| `MEMRA_SPEC_HPOST` | off | `1` feeds the MTP head POST-output_norm hidden — acceptance lever on the 27B (>100 tok/s crossing, 2026-07-06). Per-model choice |
| `MEMRA_FRSPEC_TRIM` | off | `<frspec.gguf>`: self-trimmed draft lm_head — gathers top-frequency rows from the model's own output.weight via the file's d2t ranking (generic ranking transfers across same-vocab heads; specialized ones do not — 2026-07-07) |
| `MEMRA_MTP_DRAFT` | off | `<draft.gguf>`: replace the MTP head with a standalone draft GGUF (exactness unaffected — verify arbitrates). The standard artifact is the regime draft (docs/DRAFT-REGIME.md): own-gen ranks + NVFP4 head + Q4_K_M block, one file, no other flags |
| `MEMRA_GEMMA_DRAFT_RANKS` | off | `<ranks.txt>`: gemma FR-Spec drafter head trim — corpus-ranked vocab ids, head gathered to those rows (150→18 MB at 32k). ADOPTED for 26B serving: +2.8% short/+5.8% depth spec at IDENTICAL acceptance (2026-07-12; ranks file: `research/gemma4-bringup/gemma4-frspec-ranks-32768.txt`). Historical negative verdicts were measured through two since-fixed verify bugs |
| `MEMRA_GEMMA_TRIM_ADAPT` | 512 | spare head slots for the serve-time adaptive trim (needs `..._DRAFT_RANKS`; `0` = static trim). Coverage escapes — tokens the trim can't propose — are the entire trim cost (oracle 2026-07-19); the loop learns them from prompt ids + verify argmaxes and persists to `<ranks>.learned`, so a distribution's escapes pay their first-miss round once per lifetime, not per request. Flipped the 31B chat cell from −17% to +2.5% and made trim ≥ untrimmed on EVERY measured cell (2026-07-19) |
| `MEMRA_SPEC_CAPMAX` | 7 | gemma adaptive-K cap ceiling. >7 opens the b16 verify tier (t=9..16) — correct since 2026-07-12 (three host bugs fixed) but measured perf-negative (depth K8-10 283-293 vs K6 301-306); the door is a measurement, not a crash |

### Benchmark harness (run-gen / session-bench)

| flag | default | what it does |
|---|---|---|
| `MEMRA_PP_ONLY` | off | prefill-only timing mode |
| `MEMRA_PP_REPS` | 1 | prefill repetitions (N-median protocol) |
| `MEMRA_PP_WARMUP` | 1 | prefill warmup runs |
| `MEMRA_NMEASURE` | 32 | decode tokens measured after counter reset |
| `MEMRA_FORCE_TOKENS_FILE` | off | run-gen diagnostic: teacher-force numeric token ids from a file so placement A/Bs traverse an identical decode route |
| `MEMRA_TURN_TOKENS` | 512 | session-bench: suffix length per turn |
| `MEMRA_HIST_FILE` | required | session-bench: conversation history text file |

### Server (memra-server)

| flag | default | what it does |
|---|---|---|
| `MEMRA_MODELS` | — | server model list `alias=path[+draft.gguf],...` — `+draft` attaches that model's regime draft per model (docs/DRAFT-REGIME.md); paths accept `hf:` specs |
| `MEMRA_ADDR` | bind default | listen address |
| `MEMRA_API_KEY` | none | bearer key; setting it also defaults `MEMRA_COMPAT=openai` |
| `MEMRA_COMPAT` | native (`openai` when API_KEY set) | response shape: `openai` = OpenAI completions SSE |
| `MEMRA_CTX` | 8192 | session-cache context floor (KV @8192 ≈ 119MB/session on the 9B). Machine envelope: on the 24GB 5090 under the q8rp-mirror config, c=32 ctx-8192 sessions OOM (captured `CUDA_ERROR_OUT_OF_MEMORY`; ~27 admit) — set it to the workload, 2048 clears the same cell at 502 tok/s (`research/batched-tick-inc3-20260801/`) |
| `MEMRA_KV_REUSE` | on | `0` disables the KV prefix-reuse pool (session-gate validated; 42.6x turn-start at 40k) |
| `MEMRA_REUSE_POOL` | 2 | parked-cache pool cap per model (VRAM budget: ~119MB/entry at ctx 8192 on the 9B). At the default, sequential multi-turn workloads of >2 sessions hit the park-evicts-next-entry cascade (0/N resumes; pinned 2026-07-31) — raise to the expected concurrent-session count when VRAM allows |
| `MEMRA_SERVE_SPEC` | on | `0` disables the spec-decode serve path (greedy + MTP-head requests) |
| `MEMRA_SPEC_BURST` | 32 | tokens per spec burst — round-robin latency vs per-burst fixed cost (throughput-neutral, 2026-07-06 A/B null) |

---

## 2. Machine-specific config

| flag | default | what it does |
|---|---|---|
| `MEMRA_PP_FP8` | off | `1` enables the cuBLASLt FP8-activation prefill GEMM on FP8-native (safetensors) checkpoints. +78–129% pp on g7e (2026-07-08); local NV-27B pp1855 1341→1412 (+5.3%, 2026-07-09). Opt-in: costs resident-copy VRAM (`MEMRA_PP_FP8_BUDGET_MB`); `MEMRA_ST_E4M3` is the no-duplicate route |
| `MEMRA_PP_FP8_BUDGET_MB` | 1536 | VRAM budget for resident FP8 weight copies (rig-dependent; 2560 = +16.6% pp1845 on 24GB, 2026-07-08). Irrelevant under `MEMRA_ST_E4M3` (one copy, no stash) |
| `MEMRA_ST_E4M3` | off | `1` = F8-E4M3-origin safetensors projections keep RAW e4m3 as the ONE resident copy (no Q8_0 re-encode): decode dequants e4m3 in-kernel (`qmatvec_e4m3_mmvq` + batched twins), prefill rides the FP8 GEMM on the same bytes. NEW NUMERIC CONFIG (checkpoint-native weight precision — the Q8_0 hop was lossy); gate the full battery in-config. Frees ~GBs on 24GB rigs → full FP8 prefill coverage (lane e4m3dec 2026-07-08) |
| `MEMRA_KV_K` | `q8_0` | K-cache format arm. `fp8` FLIP-BLOCKED (2026-07-09 A/B, tag kvk-fp8-ab-9bst): e2e FLAT (+0.3-0.9% ms/tok — the −7-14% micro didn't survive; micro≠e2e) AND 9B ST spec self-consistency FAIL (acceptance 74%→20.5%, drift accumulates across K reads). q8_0 stays |
| `MEMRA_KV_V` | `q5_1` | V-cache format arm: `q4_0` FLAT + quality-taxed (argmax MISMATCH in-config), `fp8` borderline (−2pt) — measured 2026-07-08; q5_1 stays |
| `MEMRA_FA_SMEM_TKV` | 1024 | t_kv crossover to the smem-broadcast FA decode twin (`0` = never). Swept 2026-07-05: flat 512–2048 on real prompts; micro says lower may pay on the 35B (untested e2e arm: 96). fp8 layer classes (gemma GKV/WKV) are excluded — the smem V-stage is q5_1-hardcoded |
| `MEMRA_FA_SPW` | 32 | gemma windowed-attention split width. Re-swept 2026-07-12 under the raw-e4m3 sV occupancy ceiling (t=1 decode is grid-limited): 32 wins plain (+4 at 1.7k vs 48); SPEC serving sets `64` (verify fills the grid via rows, bigger splits amortize staging — depth K6 301 vs 249 at 32). MUST be uniform across decode/verify (a t-keyed probe broke combine-order parity: stream 9/128). The default is tuned for the 82-SM 5090 Laptop; the optimum tracks SM count (a 188-SM RTX PRO 6000 measured +4% at `24`, 2026-07-20) — on bigger GPUs, sweep {16,24,32,48,64} |
| `MEMRA_FA_SP512` | 16 | gemma global (hd512) split override — occupancy for the nkv=2 dpl16 grid (+5.7% depth spec when landed; re-swept under e4m3 2026-07-12: 16 stands) |
| `MEMRA_FA512_MIN` | 512 | hd512 vec-lane crossover floor (scalar's more-blocks latency hiding wins below it) |
| `MEMRA_FA_VEC_MIN` | 96 (per-model) | FA vec-lane t_kv floor override (gemma model init lowers it — v4 wins at every depth there) |
| `MEMRA_FA_SPLIT` | ctx-adaptive | force fixed FA split-keys. EXACTNESS: split count changes combine FP order — run-spec must re-gate (32 broke 9B self-consistency, 2026-07-03/07) |
| `MEMRA_PRIME_CHUNK` | 4096 | chunked-prime chunk size in tokens (`0` = monolithic). Long-ctx OOM/transient control |
| `MEMRA_MOE_VRAM_FRAC` | 0.85 | SLRU expert-cache fraction of free VRAM (sweep 2026-07-06: 0.40=25.0 → 0.85=28.5 tok/s). The measured Hy3 5090 launcher sets 0.90 for both soft and hard fractions; lower it on rigs co-running other GPU work |
| `MEMRA_MOE_HARD_VRAM_FRAC` | 0.80 | machine-specific hard ceiling for the SLRU allocation (`0.10..=0.95`). Raise only after a local OOM-gated sweep. Hy3 on the 24GB RTX 5090 validated `0.85` at 21.21GiB peak framebuffer use: 2,006 slots, exact argmax/tokens, and 0.53 -> 1.03 tok/s on the warm repeat together with worker depth 16 |
| `MEMRA_MOE_SLOTS` | auto | force an exact SLRU slot count (spill experiments used 64/512) |
| `MEMRA_MOE_RESIDENT` | on | `0` forces the SLRU path even when experts fit VRAM (fits-VRAM resident = 169.55 vs 28.5 tok/s on the local 35B) |
| `MEMRA_MOE_RESIDENT_GB` | free − trunk − headroom | resident-experts budget override, absolute GB (g7e M3 partial-resident tier). Default is RESIDENT-IF-FITS (2026-08-02): exact expert-bank bytes summed from the GGUF header (per-layer x n_layer misprojects UD-quants both ways: Ornith-35B +7%, Qwen3.6-35B −2%) vs free VRAM minus the file's non-expert bytes minus the headroom reserve. The old 0.80 x free default reserved 20% of the card and spilled the Ornith-35B 19.5GB bank that fits: −33% decode / −54% prefill (research/residency-cap-20260802/) |
| `MEMRA_MOE_RESIDENT_HEADROOM_GB` | 2.0 | machine-specific headroom the resident decision reserves beside weights (CUDA ctx + KV + workspace; measured ~1.7GB at board shape on the 24GB 5090, serve c=8 peaked 23220/24463 MiB). Raise it on rigs co-running other GPU work or serving many long-ctx sessions |
| `MEMRA_MOE_PINNED` | pinned when MOE_CACHE on | force pinned host expert slabs |
| `MEMRA_MOE_SIZE_AWARE` | off | `1` partitions SLRU slots by expert projection size so mixed-size Hy3 tiers cannot strand capacity |
| `MEMRA_MOE_LFU` | off | `1` enables frequency-aware GPU SLRU eviction; this is separate from the fixed LRU policy in the optional CPU projection cache |
| `MEMRA_MOE_LFU_DECAY` | off | finite `(0,1]` epoch decay for GPU-cache frequency scores; `1.0` retains counts while still enabling forward-epoch accounting |
| `MEMRA_MOE_LFU_MTP_WEIGHT` | 1.0 | finite `0.25..=64` multiplier applied to GPU-cache frequency updates from MTP/verify epochs |
| `MEMRA_SPILL_DISK` | off | set = enable the NVMe disk tier for MoE experts (M3-class models that exceed host RAM) |
| `MEMRA_SPILL_PINNED_FRAC` | 0.60 | fraction of MemAvailable the spill tier may pin |
| `MEMRA_SPILL_IO` | `mmap` | expert storage backend: `mmap` (default and fallback), `pread` (blocking positioned-read oracle), `worker` (bounded CPU positioned-read prefetch), or `direct` (the same worker pipeline with O_DIRECT for 4 KiB-aligned expert extents). Invalid values, unaligned direct extents, and backend/read failures degrade to mmap and increment fallback/error counters |
| `MEMRA_SPILL_PREAD_DEPTH` | 2 | pinned-buffer count and worker-thread count for `worker` (`1..=64`; also sizes the blocking `pread` pool). Hy3 on the local RTX 5090 measured depth 16 as the knee: it matched depth 32 at 1.18 tok/s after a reverse-order repeat, used 107MB rather than 214MB of pinned RAM, and removed depth-8 ring saturation |
| `MEMRA_SPILL_WORKER_EXPERT_WINDOW` | `(pread_depth-1)/3`, minimum 1 | complete experts kept in the rolling positioned-read window; each expert occupies three projection buffers |
| `MEMRA_CPU_EXPERT_LIB` | off | path to a trusted build of memra's optional Hy3-only native CPU-expert companion (ABI v2; no external inference runtime). `dlopen` executes constructors before ABI validation. Complete frozen-cache hits stay on CUDA; experts missing any projection run wholly on CPU |
| `MEMRA_CPU_EXPERT_THREADS` | 8 | OpenMP compute threads for one-token CPU expert work (`1..=256`); the two-pass means for 8 and 12 threads differed by 0.7%, but the winner reversed by about 8% between individual reverse-order observations (100 timed calls each), so the lower-contention value remains the local default |
| `MEMRA_CPU_EXPERT_IO_THREADS` | compute thread count | positioned-read workers used by the CPU expert backend (`1..=256`) |
| `MEMRA_CPU_EXPERT_PIPELINE` | off | `1` overlaps companion reads with compute (cached experts compute immediately, missing experts in read-completion order, output byte-identical; the Hy3 launcher opts in). Requires `OMP_WAIT_POLICY=ACTIVE`: with passive waits the ready-batch region entries pay futex+C-state wakes and decode compute inflated 3.0→8.4 s per 32 tokens; with ACTIVE the spinners starve the caller between calls and e2e is flat vs serial (2026-07-22 bisect chain, N=3 4.55 vs N=5 control 4.50). Stays a door until the single-region structural fix removes the wait-policy dependence |
| `MEMRA_CPU_EXPERT_IO_CPUSET` | inherit | optional cpu list (e.g. `8-15`) pinning the async read pool off the compute cores |
| `MEMRA_CPU_EXPERT_EXECUTOR_CPUSETS` | inherit | `"0-7;8-15"` pins each CPU-expert executor to its own core group and sizes its OMP team to that group (`MEMRA_CPU_EXPERT_EXECUTOR_THREADS="8;8"` overrides). Asymmetric partitioning for hybrid P/E silicon: one team per group, so a slow group never straggles a fast group's barrier. MEASURED +11% at m=3 mixed prompts (5.74/5.73/5.77 vs 5.30/4.94/5.27) and far more reproducible than the P-only baseline. Knees: 8 E-cores in a second group is the optimum (a 14-core E team falls back to baseline) and executors should be FEWER than in-flight calls, or the weakest group becomes the critical path. Requires GOMP_CPU_AFFINITY unset (a global affinity overrides per-thread pinning) and a process taskset covering the groups. Helps only when several expert calls are in flight — single-stream issues one at a time |
| `MEMRA_CPU_EXPERT_CACHE_SHM` | off | `1` backs the expert cache with a named tmpfs segment and persists the index across restarts: a clean reopen starts warm (validated 100% hit / zero re-reads in the cross-process smoke). Exclusive flock; a second concurrent process silently gets a private cache. Crash → dirty header → cold-but-correct start; cache keys pin device/inode/size/ctime so a changed source tree can never serve stale bytes. Costs THP on this rig (`shmem_enabled=never`) unless the sysctl is flipped to `advise` |
| `MEMRA_CPU_EXPERT_CACHE_SHM_NAME` | `/memra-expert-cache-v1` | shm segment name for the persistent cache |
| `MEMRA_CPU_EXPERT_FREEZE_PROFILE` | off | path to a freeze-profile sidecar. A warmup run saves the frozen residency set (versioned header binds slot geometry); a later run restages it directly and SKIPS the profiling warmup. Measured pair (2026-07-22): 5285/5285 blocks restaged, startup wall 7:37.6→6:02.0 (−21%), identical 32 tokens, argmax MATCH, decode band unchanged. Mismatched/stale profiles fall back to a normal warmup; the post-freeze argmax gate still validates the assignment either way |
| `MEMRA_CPU_EXPERT_CACHE_GB` | 16 | requested normal-RAM cache ceiling for CPU-routed expert projection bytes; the measured Hy3 5090 launcher requests 20 GiB and still applies the live-stack headroom gate below |
| `MEMRA_CPU_EXPERT_RESERVE_GB` | 4 | live-stack RAM headroom retained by capping `MEMRA_CPU_EXPERT_CACHE_GB` to `MemAvailable - reserve` when the CPU companion initializes and prints the effective budget |
| CPU expert cache eviction | LRU | fixed winners-only policy; experimental LFU and layer-aware least-stale arms did not beat the measured LRU path and were removed |
| `MEMRA_CPU_EXPERT_IO` | `buffered` | CPU expert reads: `buffered` or aligned `direct` positioned I/O |
| `MEMRA_CPU_EXPERT_MIRROR_MAP` | off | tab-separated inode/alternate-path map for splitting large direct reads across byte-identical expert mirrors on two NVMe devices |
| `MEMRA_CPU_EXPERT_FREEZE_CACHE` | off | `1` freezes HBM expert residency so CPU/GPU backend assignment cannot drift after warmup; required for exact repeatable Hy3 hybrid serving |
| `MEMRA_CPU_EXPERT_FREEZE_WARMUP_TOKENS` | off | discarded tokens used to profile/fill HBM before freezing; the Hy3 5090 launcher uses 128 after one same-session, identical-token N=32 pair measured 4.52 tok/s versus 4.46 at 8 tokens (single pair, not a median), and `MEMRA_CPU_EXPERT_FREEZE_WARMUP_SPEC_K` selects speculative warmup depth in run-spec |
| `MEMRA_CPU_EXPERT_FREEZE_PROFILE_ADMIT` | off | `1` lets CPU-routed warmup misses vote for HBM admission before the residency set freezes; current-token execution remains on CPU |
| `MEMRA_CPU_AFFINITY` | `0-7` in `run_hy3_local_5090.sh` | CPU list used by the Hy3 launcher for `taskset` and `GOMP_CPU_AFFINITY`; wrapper-only machine topology setting |
| `MEMRA_CPU_EXPERT_BATCHED_PRIME` | off | rollback seam: `1` restores batched prompt replay after Hy3 CPU/GPU expert residency freezes; the local spill path otherwise replays tokenwise to avoid transiently rereading the missing expert bank |
| `MEMRA_ST_PINNED` | off | `1` pins the safetensors expert store — ONLY for fits-in-RAM checkpoints (pinning 26GB evicted the page cache: 30x regression, 2026-07-07) |
| `MEMRA_ST_REPACK_DISK` | on | `0` forces in-RAM gather instead of the `.memra-repack` disk cache (safetensors stream-repack loader) |
| `MEMRA_KQ_NVFP4` | 0 | load-time k-quant→NVFP4 re-encode: `1` = Q4_K, `2` = +Q5_K. +3.9% 9B plain but ~2x quant error and an acceptance tax — bpw equality ≠ quality-class equality (2026-07-08). Speed-mode opt-in only |
| `MEMRA_MMQ_F8F4` | off | `1` = W4A8-FP8 prefill MMQ (e4m3 fold + f8f6f4 MMA, 381-TF class). pp +3.9-6.3% ALL models, TTFT -4-5.6%; e2e spec MODEL-SIGNED via the prefill-KV acceptance law (27B ST +7.2%, 9B -3.5/-6.1%) — per-model serve adoption (NV-27B ST config), never a global default (2026-07-10 flip battery). Expert-tile twin measured NEGATIVE (-6.4% best variant, not pipe-bound) and was deleted — tag moe-f8f4-negative |
| `MEMRA_NV_W4` | off | `1` re-quants F8 attention weights → NVFP4 on the NVIDIA-official 27B (+20% plain, acceptance held, 2026-07-07). Opt-in until the text battery proves the class |

### Build-time (build.rs / nvcc)

| flag | default | what it does |
|---|---|---|
| `MEMRA_NVCC` | `nvcc` on PATH | nvcc binary override |
| `MEMRA_CUDA_ARCH` | auto-detected | build target: `120a` (Blackwell consumer), `90a` (Hopper), `100a` (B200), `89` (portable eval). Unset = probe the GPU via `nvidia-smi compute_cap` (12.x→120a, 9.0→90a, 10.0→100a, 8.9→89); GPU-less/unknown falls back to `120a` so the CI compile gate is unchanged. Explicit value always wins. `89` builds the secondary correctness-first eval lane: native-FP4 prefill compiled off, tiled GDN on its generic fallback, prompt attention through f32 SDPA |
| `MEMRA_ARCH_CHECK` | on | `=0` skips the Engine::new device-vs-binary compute-capability guard (fatbins are single-arch SASS; the guard fails early with a rebuild hint on mismatch) |
| `MEMRA_CUTLASS` | off | set = compile the CUTLASS sm120 NVFP4 GEMM (`cutlass_smoke`, `MEMRA_FP4_CUTLASS` door) |
| `MEMRA_CUTLASS_ROOT` | flashinfer venv tree | CUTLASS header tree location |
| `MEMRA_MMQ_X_Q45K` | 64 | k-quant MMQ X-tile (compile-time sweep seam) |
| `MEMRA_MMQ_X_W4A8` / `MEMRA_MMQ_Y_W4A8` | 128 / 128 | W4A8 MMQ tile sizes. Swept 2026-07-06: defaults optimal on the 5090 (X=32 −28%, Y=64 −2%); kept as autotuner levers for other silicon |
| `MEMRA_GEMM_K1_LAUNCH` | shipped (128,128,8) | `"BM,BN,NWARP"` kernel1 launch-tile override — MUST match the `-D K1_BM/K1_BN/NWARP` the swept fatbin was built with (tools/sweep) |

`MEMRA_*_FATBIN*` (ENGINE/FLASH/GEMM/QMATVEC/ROUTER/HYBRID) are **internal build plumbing**
(build.rs → rustc-env fatbin paths), not user flags; `MEMRA_KV_KFMT`/`MEMRA_KV_VFMT` are the
`-D` defines build.rs derives from `MEMRA_KV_K`/`MEMRA_KV_V` per flash fatbin. `MEMRA_MMVQ_ROWS`
is a compile-time `#define` in qmatvec.cu, not an env var.

---

## 3. Rollback seams (default ON — `=0`/set reverts to reference path)

These exist because correctness discipline needs a same-binary oracle. Each is a measured winner; the seam is the documented way back.

| flag | revert semantics | provenance |
|---|---|---|
| `MEMRA_FAST=0` | Stage-A f32-dequant matvec class — THE correctness oracle | default-on 2026-07-08 (env-law retirement) |
| `MEMRA_PDL=0` | plain `cuLaunchKernel` for ALL PDL-attributed kernels (master seam: the six glue kernels + every later wave) | default-on 2026-07-13; E4B +1.0-1.2%, 26B/31B/qwen flat; SASS-audited entry syncs |
| `MEMRA_PDL_MMVQ=0` | revert PDL wave-A alone: the four decode-hot mmvq matvec kernels (`q4_0_mmvq_rp`, `fused2_mr1_rp`, `fused3_mr1_rp`, `q6_K_mmvq`) | default-on 2026-07-23; 12B +0.19%, E4B A/B positive, 31B/26B flat |
| `MEMRA_PDL_WB=0` | revert PDL wave-B alone: dense-glue quartet (rms_norm_f32, add_rms_norm_f32, add_scale_rms_norm_q8_1, quantize_q8_1) + flash trio (append_kv dc, the two q8 rows-combines) | default-on 2026-07-23; B1a 12B +0.61% |
| `MEMRA_WPF=0` | no wo-plane L2 prefetch across the E4B fa window | default-on 2026-07-13; +0.65% E4B; 26B/31B probed flat/negative and NOT wired |
| `MEMRA_F2B=0` | separate per-tensor b-tier verify launches (no segmented-grid qkv/gate+up fusion) | default-on 2026-07-13; 31B depth +5%, short +3.9%, 26B +0.9%; bit-exact (VERIFY-GATE 0.000e0) |
| `MEMRA_MMVQ=0` | dp4a matvec class (m=1 AND batched verify switch together — dispatch-parity law) | default-on 2026-07-08; parity fix 2026-07-07 |
| `MEMRA_MOE_CACHE=0` | stage-every-token expert dispatch (no SLRU) | default-on 2026-07-08 |
| `MEMRA_MOE_PREFETCH=1` | pipeline the next routed expert's cache misses on the copy stream | experimental; target-rig gate required |
| `MEMRA_MOE_PAGE_PREFETCH=1` | issue rolling `MADV_WILLNEED` for mmap-backed GGUF/repack ranges | experimental; cold-cache G7e + RTX 5090 A/B required |
| `MEMRA_MOE_PAGE_PREFETCH_WINDOW` | future experts kept in the rolling page-prefetch window (default `1`; `0` disables) | only read when `MEMRA_MOE_PAGE_PREFETCH=1`; tune to storage latency/page-cache budget |
| `MEMRA_MOE_MMAP_ADVICE` | whole-map expert advice: `random` (default) or `normal` | `normal` restores ordinary Linux readahead; invalid values warn and fall back to `random` |
| `MEMRA_NO_FA_VEC` (set) | scalar `fa_decode_f32` bit-reference (eager + rows + graph in lockstep) | vec default-on 2026-06-28 |
| `MEMRA_FA_V2=0` | per-key online-softmax FA twins (v2 = tile-batched, own numeric config) | default-on 2026-07-08, the depth-slope fix |
| `MEMRA_FA_ROWS_OFF=1` | per-row verify FA loop instead of the fused rows kernel | rows landed 2026-07-03 (+13.8% 9B p2) |
| `MEMRA_SPEC_LEAN=0` | zeroed verify buffers + rows dispatch at t=1 | default-on 2026-07-08 (+1.5–2.4% 35B) |
| `MEMRA_SPEC_M2=0` | per-m grid.y=m verify dispatch at t=2 (no small-m batched twin) | default-on 2026-07-09 (flattened the 35B verify K-curve; K=4 within 1–3% of K=3) |
| `MEMRA_SPEC_FUSED_T=0` | per-tensor decode-exact verify trunk calls (no t=2-4 launch fusion) | default-on 2026-07-09; bit-identical per (tensor,token,row) by construction (kernel-check Q8-FUSED2-B/FUSED3-B gates) |
| `MEMRA_FA_V3=0` | FA v2 decode twins (v3 = dp4a-K hybrid, own numeric config — int8 Q quantization) | default-on 2026-07-09 after full battery green (kernel-check + run-gen MATCH 35B/9B + run-spec PASS + graph gate bit-identical); requires default q8_0/q5_1 KV + hd%128==0 (host-gated, auto-falls back to v2) |
| `MEMRA_SPEC_REPLAY=1` | legacy rollback+replay partial accept (also the j==0 fallback) | replay-free default 2026-07-03 (+10–32%) |
| `MEMRA_SPEC_NOREFRESH=1` | chain-approximate draft-KV entries (no true-hidden refresh) | refresh default 2026-07-03 (+4–6% acc) |
| `MEMRA_SPEC_NOGRAPH=1` | eager draft chain (no CUDA-graph draft) | graph draft 2026-07-03 |
| `MEMRA_PRIME_TOKENWISE=1` | force tokenwise decode-step prime in every phase (<16-tok prompts take it anyway). Frozen Hy3 CPU/GPU expert serving selects tokenwise automatically after its profiling warmup. Also the bisect knob for a pinned OPEN correctness item: on this tree the naked q35 pp512 greedy stream flips to early-EOS on a near-tie FIRST token — the flip lives in the batched-prime path's last-position logits, `=1` restores the tokenwise-oracle stream, and run-gen's argmax gate does not cover the batched-prime seed position (`research/residency-cap-20260802/RESULTS.md` §4, probed pre-patch — not a residency effect) | batched prime 2026-07-03 (23x TTFT at 6k) |
| `MEMRA_PRIME_APPEND_LOOP=1` | per-row KV append instead of the batched `_rows` kernel | measured equal 2026-07-03 |
| `MEMRA_PRIME_DEQW=0` | inline-dequant prefill FA (no bf16 dequant-once workspace) | deqw default 2026-07-05 (32k prime 1.60x) |
| `MEMRA_GEMMA_GKV=0` | gemma GLOBAL (hd512) layers back to q8_0/q5_1 KV (default = e4m3 via the kf8vf8 module) | fp8-globals default-on 2026-07-11 — the depth-plain lever (dequant-latency-bound, ncu) |
| `MEMRA_GEMMA_WKV` | gemma WINDOWED (hd256) layers: e4m3 vs q8_0/q5_1 KV. DEFAULT IS SERVING-KEYED (2026-07-12): `MEMRA_DRAFT` set (spec serving) → q8/q5_1, else → e4m3. Explicit `=0`/`=1` always wins | fp8-windows win depth-plain (+3%, 2026-07-12 A/B) but GUT the MTP drafter's acceptance (31B short .758→1.000 on q8/q5 = spec 88→122.7, ABOVE llama-mtp 112; 26B depth .57→.89) — the drafter's single swa attention is fragile to e4m3 KV noise. Globals (`GKV`) cost no acceptance |
| `MEMRA_GEMMA_ROWS_W=0` | per-token loop instead of the parity-law rows/rows_w twins (gemma decode+verify) | rows twins = the parity-law foundation, 2026-07-10/11 |
| `MEMRA_GEMMA_DRAFT_DC=0` | gemma drafter attention back to the host-len kvmod arm (default = device-len fa_decode_dc/rows_w arms riding the main layers' len_d) | burst-arc step (a/b), 2026-07-12 — required by the draft graphs + burst |
| `MEMRA_E4B_GRAPH` | E4B whole-token graph-exec-update serving (`graph_update.rs`: ONE captured step, per-token fa geometry retuned to the live eager split counts). DEFAULT IS BUDGET-KEYED (2026-07-13): ON at `max_new >= 256` under the window, else eager-dc. `=1` forces, `=0` kills | steady-state replay beats eager but the ~30ms capture crosses over ~200 tokens (128tok −1.3%, 400tok +0.9%, valid-window interleaved); stream 400/400 IDENTICAL vs dc-eager |
| `MEMRA_FA_SPW2=0` | disable the warp-1 staging helper on the windowed v4 kernel (gqa==1) | +0.4-0.7 marginal keep 2026-07-11 |
| `MEMRA_FA_V4=0` | pre-v4 FA decode lanes (v4 = key-per-lane score phase; format-aware e4m3 arms 2026-07-12) | default-on 2026-07-10 after the 3-model battery |
| `MEMRA_FA_I2=0` | single-key walk in the gemma global rows kernel (i2 = 2-key interleave) | i2 default-on 2026-07-11 (depth +3.5%; i4 was negative — register pressure) |
| `MEMRA_FA_TB512=0` | per-row (grid.z) globals rows kernel instead of the t-batched shared-tile twin | tb512 default-on 2026-07-14: one block per (kv_head, split) stages its tile once and loops the verify rows — kills the x t DRAM re-read of the full-ctx globals (31B depth spec +1.4%, plain flat; fixed absolute split grid = new numeric config shared by every hd512 caller; stream identical, acceptance unshifted, spec 256/256 x3 models). The in-kernel dp4a port alone (z-form) probed FLAT — the lane was DRAM-re-read-bound, not unpack-bound |
| `MEMRA_QWEN_DC=0` | qwen greedy serving falls back to the host-logits eager loop (full-vocab dtoh + host argmax per token) instead of the dc-eager loop | dc-eager default-on 2026-07-15: device argmax + device counters = 4B/token host traffic (the B=2-4 multi-model serving contract), flat single-stream, streams identical. The graph routes were measured and lost: per-bucket captures -24% (ladder-rung recapture storms), exec-update replay -4.5% (SSM copy-back ~50MB/token vs eager's pointer swap); eager's ~470 launches pipeline behind the single readback — jsonl 2026-07-15 |
| `MEMRA_Q4RP=0` | GGUF-layout Q4_0 decode (default = split-plane repacked mirrors, built at load) | q4rp default-on 2026-07-10 (+1.9-6.3% all gemma cells; the 18B-stride cure) |
| `MEMRA_Q8RP=0/1` | Q8_0 split-plane decode mirrors, built at load (default = ON on the Hopper lane only — VRAM cost == the mirrored trunk; `=1` opts in elsewhere when model + mirror + KV fit). The mirror is also the exact-16 serve tier's admission ticket on Q8_0 trunks — only the `_rp` twins carry the bit-exact b16 kernel class (see `MEMRA_DECODE_BATCH_CAP`, §7) | q8rp hopper-default 2026-07-26 (H100 ncu: the 34B-stride GGUF layout held Max BW at 41-46% from sector overfetch; mirrors bit-identical). 5090 serve receipts 2026-08-01: mirror alone +5.9% at chunk 8, exact-16 chunk +18.8% at c=16 same-mirror (`research/batched-tick-inc3-20260801/`) |
| `MEMRA_KQRP=0/1` | K-quant (q4_K/q6_K) split-plane decode mirrors, built at load (default = ON on the Hopper lane only — VRAM cost == the mirrored trunk; a 24GB card cannot hold model + mirror + KV) | kqrp hopper-default 2026-08-01 (q27 ncu: q4_K mmvq 65% excessive sectors at DRAM 41-54%, q6_K 78% at 40-47% — the 144B/210B superblock-stride twin of the Q8_0 2026-07-26 fix; bit-identical KQRP kernel-check gates) |
| `MEMRA_SPEC_ADAPT=0` | fixed-K gemma draft rounds (default = adaptive k_next = accepted+1, clamp [floor, cap]) | gemma adaptive default-on 2026-07-10 (+10% depth spec). NOTE: the qwen adaptive-K arm of the same name was retired 2026-07-07 — this is the gemma revival, different policy |
| `MEMRA_SPEC_ADAPT=1` (qwen path) | opt-in port of the gemma accepted-run law to the qwen MTP spec (spec.rs): next round drafts accepted+1, clamp [floor(pos), min(K, CAPMAX)]; same floor/cap/pos-key envs as gemma. Fixed-K stays the qwen default | ported + REFUTED 2026-08-01 (lane/qwen-adaptive-k, H100 x3 interleaved): q27 tuned K=3 flat-to-negative (short +0.8% noise / board −1.9% / agentic −0.5%; PMIN=0 −2.1%), q35 K=2 −6.4%; wins only vs untuned deep K (K=6 floor4 +1.5% over fixed K=6, still −3.7% under fixed K=3) — the EMA-arm verdict class again, different law. Exactness PASS K=1..8 both models. Do not merge as a default; kill the arm unless a config appears that wants it |
| `MEMRA_SPEC_ADAPT_FLOOR=<n>` | pin the adaptive-K floor at every position (default is per-model AND position-keyed: n_embd≥3500 → 4, ≥2500 → 2, else 1; past pos 1024 high floors (≥4) relax to 1, mild floors persist; clamps to K) | floor default 2026-07-25/26: floor-1 collapses to shallow drafts after a miss; expensive-verify models win big raising it (31B chat K5 floor4 124.3 vs 103.8; 12B chat K4 235.0 vs 200.6; floor 5+ falls off). At depth acceptance drops (0.69-0.74) and forced-deep turns net-negative (31B d1736 floor4 99-101 vs floor1 103.8-104.2) — hence the pos key |
| `MEMRA_SPEC_FLOOR_CTX=<n>` | position boundary for the adaptive-K floor (default 1024): full floor below it; past it floors ≥4 relax to 1, mild floors persist | measured: 31B d1736 wants floor1 (104.2 vs 97-101 under floor2/4, two batteries); 26B d1736 wants its floor2 (304-305 vs ~297); explicit ADAPT_FLOOR bypasses the key |
| `MEMRA_SPEC_PMIN_INROUND=<p>` | pin the in-round draft-confidence cut at every position/round (0 disables). DEFAULT is self-keyed: cut at 0.7, active only at pos ≥ floor_ctx AND in rounds following a miss — one small dtoh sync per draft step, chain stops early, verify at shrunk width | 2026-07-28: always-on wins low-accept depth (26B +1.4-3.2%, 31B +2%) but taxes chat and 0.95-accept cells (−0.9 to −6%); the self-key idles where rounds full-accept — battery par-to-plus on all six cells, acceptance +1.3-2.8pts on 26B/12B depth |
| `MEMRA_PRIME_DEQW_DB=0` | single-buffer workspace staging (no cp.async double-buffer) | 2026-07-05 |
| `MEMRA_GDN_CHUNKED=0` | sequential GDN prefill scan | chunked default 2026-07-04 (+4.6% pp512 9B); `MEMRA_GDN_CHUNK` (default 32) = chunk size |
| `MEMRA_B8=0` | m=5..8 verify back to per-m grid.y=m dispatch | b8 tier 2026-07-05 (K=4 cliff fix, +30%) |
| `MEMRA_NO_BATCHED` (set) | per-m grid.y=m path for ALL m=2..8 — the batched-verify A/B reference | 2026-07-03 |
| `MEMRA_Q5K_ISSUE=0` | reference q5_K MMVQ bodies (`2` = force-il A/B probe) | il default 2026-07-08 (+1.8% 9B plain) |
| `MEMRA_Q8_DUAL=0` | two separate q8_0 matvec launches (no gate+up fusion) | fused 2026-07-03 |
| `MEMRA_NO_FUSE_NORMQ` (set) | unfused rms_norm + quantize (decode norm-fusion off) | fused 2026-07-03 |
| `MEMRA_NO_GEMM` (set) | force the dp4a fallback for prefill matmuls — the int8-GEMM bit-reference | 2026-06-28 dispatch |
| `MEMRA_NOFA` (set) | naive SDPA instead of the hand-written prefill FlashAttention (also the auto-fallback for unstamped head_dims) | FA prefill 2026-06 |
| `MEMRA_KS=0` | drop the 2026-07-06 rpsc smem-scale-prestage entries from variant AUTO | rpsc landed 2026-07-06 (g7e +2.6% K=4; 5090 neutral, harmless) |
| `MEMRA_EVT=1` | restore cudarc cross-stream event tracking (elision default: −7ms/tok host on 35B) | elision default 2026-07-05 |
| `MEMRA_RP=0` | GGUF-layout NVFP4 (no A6 split-plane repack); also the W4A4 door key, see §5 | rp default 2026-07-05 |
| `MEMRA_ST_DIRECT=0` | safetensors NVFP4 → GGUF-layout conversion instead of direct split-plane load | 2026-07-07 |
| `MEMRA_MMQ_W4A8=0` | int8 GEMM prefill everywhere (no W4A8 MMQ tile) | W4A8 default 2026-07-05 (1.54–1.9x prime) |
| `MEMRA_PP_Q8MMQ=0` | hand-rolled `qmatvec_gemm_q8_0` tiling GEMM for Q8_0 prefill (no int8-MMA MMQ) | default 2026-07-09 (35B pp 2456→3069) |
| `MEMRA_PP_Q4MMQ=0` | hand-rolled `qmatvec_gemm_q4_0[_rp]` tiling GEMM for Q4_0 prefill (no int8-MMA MMQ) | default 2026-07-22 (gemma-12B lane) |
| `MEMRA_FA512_STAGE=f32` | f32-staged hd512 FA prefill kernel (no bf16 pre-convert; bit-identical either way) | bf16 default 2026-07-22 (12B pp +23%) |
| `MEMRA_QKVNORM_W=0` | block-tree `rms_norm_qkv_f32` at prefill depth (no warp-per-row float4 twin) | default 2026-07-22 (12B lane) |
| `MEMRA_FAW_STAGE=f32` | f32-staged SWA windowed FA prefill (no bf16 pre-convert; bit-identical either way) | bf16 default 2026-07-22 (12B lane) |
| `MEMRA_FA512_SP=0` | z=2 hd512 FA (GEMM0 recomputed per O-half; no split-K single-pass) | sp default 2026-07-22 (kernel-diff lever) |
| `MEMRA_FA_F16PV` | f16-P/V prefill class. Default is serving-keyed (the wkv acceptance-law pattern): plain serving ON (stamp v4: 12B 1.045x, 31B 0.979x), SPEC serving (MEMRA_DRAFT set) OFF — f16 prime numerics shift the sub-argmax logit space the drafter feeds on (26B d1736 acceptance 0.883→0.405, -40% e2e, every argmax/stream gate green; f32 prime restores 0.846 and +3.5% on the 31B chat cell). Explicit 0/1 wins | 2026-07-23 default; spec flip 2026-07-26 |
| `MEMRA_FA512_W4=1` | 4-warp sp16 without head-pairing (GEMM0 split-K 4-way + cp.async) — measured net-negative on the 150W laptop (clock tax), superseded by the hp arm | opt-in probe arm 2026-07-23 |
| `MEMRA_FA512_HP=0` | hd512 head-pair rollback to the 2-warp sp16 kernel | hp DEFAULT 2026-07-23 (laptop kernel 10.2->3.0ms) |
| `MEMRA_FAW_HP=0` | SWA head-pair rollback to the per-head p1 stamp | hp DEFAULT 2026-07-23 (SWA kernel 870->677us) |
| `MEMRA_MMQ_SK` | stream-k MMQ prefill GEMM. Default: plain serving ON everywhere (12B pp512 +3.3%, pp1736 +1.0%); SPEC serving is per-model — big dense (n_embd≥3500) forces tiling, MoE/small keep sk. The sk autotune's per-process kernel coin made 12B spec cells BIMODAL (d1736 205 @ 0.756 / 260 @ 0.943 identical invocations; tiling ×6 stable 263-269 @ 0.953, chat +3%; 31B neutral) while the 26B drafter measures BETTER under sk's fold order (depth 305 @ 0.826 vs 293 @ 0.750). Explicit 0/1 wins | sk default 2026-07-23; per-model spec key 2026-07-27 |
| `MEMRA_MOE_Q8=0` | Stage-A f32-dequant expert kernels — restores BYTE-identity for the MOE_GATE oracle | dp4a experts 2026-07-06 (+22%) |
| `MEMRA_MOE_Q8_KQ=0` | exclude k-quant arms from the q8 expert dot set | 2026-07-06 (+9 tok/s 35B) |
| `MEMRA_MOE_DEC=0` | `_em` per-token re-decode expert dot (no decode-once) | dec default 2026-07-05 (3.34x 35B prefill) |
| `MEMRA_MOE_GDEC=0` | sequential per-expert launch chain (stage-2 grouped decode off) | 2026-07-04 |
| `MEMRA_MOE_DEV=0` | host routing (no zero-DtoH device dispatch); `MEMRA_FUSED_ROUTER=0` implies it | stage-3 2026-07-05 |
| `MEMRA_FUSED_ROUTER=0` | host softmax+sort routing | fused router 2026-07-05 (g7e stage-3 arc) |
| `MEMRA_MOE_PAIRS=0` | per-expert loop for real prefill (no pair-batched launches) | 2026-07-06 |
| `MEMRA_MOE_PREWARM=0` | organic SLRU residency (no one-shot layer prewarm) | 2026-07-05 |
| `MEMRA_MOE_MMA=0` | dp4a expert prefill (no int8-MMA expert MMQ; t floor 16 keeps verify on dp4a) | 1.5x pp 2026-07-05; spec-safety floor 2026-07-06 |
| `MEMRA_MOE_FUSE_ACTQ=0` | two-pass act epilogue in the MMA prefill arms (separate silu/gelu-mul + D4 quantize launches, f32 act buffer materialized) — fused path is byte-identical (kernel-check gated) | fused epilogue 2026-08-01 (2-2.5x epilogue kernel, g26 pp +0.9%) |
| `MEMRA_MOE_DEVQ8_GU` / `MEMRA_MOE_DEVQ8_DOWN` | force dev-q8 kernel variants (auto = measured winners w8h2 / GU=v, +2.5% 35B) | down8 merge 2026-07-08 |
| `MEMRA_ROUTER_V2=0` | lone-warp router GEMV (no 8-warp smem-reduce twin; NEW FP order — battery-arbitrated) | w8 default qwen 2026-07-31; gemma4 re-arbitrated onto it 2026-08-01 (+13% g26 decode) |
| `MEMRA_SHEXP_DOT=0` | cuBLASLt m=1,n=1 GEMM + separate sigmoid for the MoE shared-expert gate (default = `sigmoid_dot_rows_f32`, one fused 8-warp block-reduce dot + sigmoid launch, wired into EVERY decode-class arm so dispatch choice cannot change bits; since 2026-08-02 prefill rides it too — see `MEMRA_ROUTER_PREFILL_EXACT`). Numeric-config class, same as `MEMRA_ROUTER_V2` | default-on 2026-07-31 (round 45): q35 decode d1736 160.0 -> 181.9 (+13.7%, N=5 interleaved); kernel-check `sigmoid_dot` maxdiff=0; decode-batch STRICT PASS |
| `MEMRA_ROUTER_BATCH=0` | per-(expert,token) router GEMV at every t (no 8x8 register-tiled batch twin at t>=8). **Perf-only rollback seam, NOT a numeric config**: the batch twin is BIT-IDENTICAL per row to `router_gemv_f32_w8` (kernel-check m=1..2048 sweep on real router weights), so this flag can never change output bits. (Killed same-lane arms, JSONL as record: 8x16 tile slower at every t; `sigmoid_dot_rows` twin slower at every t) | batch twin default 2026-08-02 (`lane/fast-router`): router launch 3.5x at m=2048, recovers most of the prefill cost of `MEMRA_ROUTER_PREFILL_EXACT` — receipts `research/fast-router-20260802/` |
| `MEMRA_MOE_DEVQ8_WPB` (default 4) | warps/block for the `_r` twins (probe knob) | 2026-07-06 |
| `MEMRA_MOE_GU_IL=1` | interleave gate/up expert rows at resident upload (locality probe; measured neutral) | 2026-07-11 |

## 4. Diagnostics & test config

| flag | what it does |
|---|---|
| `MEMRA_MOE_STATS=1` | per-layer expert-cache hit/miss/staged-bytes prints (forces the stats-visible dispatch path) |
| `MEMRA_SPILL_STATS=1` | print cumulative positioned-read snapshots when server requests finish: reads, logical bytes, errors, short reads, mmap fallbacks, buffer waits, and worker ring-full events. Snapshots are totals, not per-request deltas; do not sum them |
| `MEMRA_MOE_TRACE=<path>` | append (layer, step, expert ids) per decode step — routing-locality analysis (`research/scripts/moe_trace_analyze.py`, 2026-07-07 M3 measurement) |
| `MEMRA_MOE_WEIGHT_TRACE=<path>` | append the PREFILL router's per-(layer, token) selected experts + weights — the routing-divergence oracle behind the m-invariance razor (`concat_prime_probe`, `research/concat-prime-exact-20260802/`). Forces the stats-visible dispatch path; diagnosis only |
| `MEMRA_DUMP_HN=<path>` | append pre-head hidden states per decode step (head-MIPS feasibility probe, offline bound analysis; both decode arms share the door so they stay observably identical). Diagnostic only |
| `MEMRA_F16G_DEBUG=1` | grouped f16 expert lane stage bisect: full NaN/Inf scans of w/act/y plus post-permute/post-scatter bad-value counts — localizes a corrupt stage. Adds D2H syncs; diagnosis only |
| `MEMRA_FA_V4_MAX=1` | force the v4 FA lane at every t_kv (bypass the crossover) — correctness-forcing knob for the fp8 lane matrix (2026-07-12 closure battery) |
| `MEMRA_DRAFT_GRAPH_CHECK=1` | re-run the gemma draft chain eagerly after each graph replay and diff the drafted slots (non-destructive replay-vs-eager bisect) |
| `MEMRA_E4B_GRAPH_GATE=N` (gemma-gate) | E4B graph-door stream gate: `generate()` door OFF then ON on fresh caches, streams must be identical (the warmup-side-effect + exec-update oracle) |
| `MEMRA_GEMMA_GRAPH=1` | 12B/31B whole-token alloc-free graph door (slotted decode step, captured once, replayed per token). Bit-identical (96/96 stream gate) but −5.6% vs dc-eager on the 12B (campaign closed 2026-07-23: ladder 81.3→84.8 vs eager 89.8; residual is GPU-side graph-exec scheduling, not allocs/launch cost). Experimental door, default OFF |
| `MEMRA_GRAPH_GATE=N` (gemma-gate) | 12B/31B graph-door stream gate: eager vs graph token streams over N tokens must be identical + both throughputs printed |
| `MEMRA_GRAPH_DRAIN=N` | raise the graph door's drain window above the pinned 1 (EXPERIMENTS ONLY: relaunching the same graph exec before its prior launch completes is ILLEGAL — chunk=4 reproduced ILLEGAL_ADDRESS) |
| `MEMRA_GRAPH_CENSUS=1` | print the captured graph's node-type census (kernel/memset/memcpy/alloc/free counts) at instantiate |
| `MEMRA_VERIFY_GATE2=K` (gemma-gate) | CHAINED batched-verify oracle: prefix tokenwise, then two back-to-back `decode_step_t` calls — per-position argmax must match the tokenwise chain. `MEMRA_VERIFY_GATE2_DEV=1` runs the device-token verify arm (the spec round's exact path) |
| `MEMRA_ROUND_GRAPH_CHECK=1` | round-graph bisect: run the captured round body EAGERLY (no capture/replay) — splits body-semantics bugs from replay-mechanics bugs |
| `MEMRA_E4B_DCG_EAGER` | E4B door bisect arm: run the dcg step eagerly per token instead of capture/replay — `1` = exact live bucket, `2` = the capture's win bucket (separates bucket-path numerics from the replay mechanism) |
| `MEMRA_E4B_GRAPH_TIMING=1` | per-phase host timing of the E4B graph loop (dtoh-wait / fa_apply / launch sums) |
| `MEMRA_GRAPH_NODES_DUMP=1` | dump the captured E4B graph's kernel-node inventory (symbol + grid + count) + fa update-unit count at capture |
| `MEMRA_BURST_VCHECK=1` | run the gemma verify-STREAM on each eager round's batch before the eager verify and diff argmaxes + per-layer KV byte sums (the burst verify's in-place gate harness) |
| `MEMRA_GATE_SEED=<n>` | decode-batch-gate synthetic-prompt token-pattern offset (default 0 = the historic prompt). Cross-config drift is near-tie roulette on any single synthetic prompt, so calibration re-sweeps draw several seeds — gate1-config sweeps 6 and FAILs only on divergence before step 3 (round 45; the 2026-07-31 shexp-dot re-sweep) |
| `MEMRA_SPEC_STATS=1` | per-slot accept histogram + draft-length histogram |
| `MEMRA_DEBUG_SPEC=1` | per-round spec decode trace |
| `MEMRA_MOE_CSR` | `0` = rollback the CSR expert-dedup gate_up on spec verify (default ON 2026-07-10: owner-scan dedup of the 38-40% duplicated expert weight-stream+decode, +1-2% spec e2e all K); `2` = run BOTH paths + byte-compare (debug) |
| `MEMRA_MOE_OVERLAP` | `1` = log cross-token expert-activation overlap at spec verify (unique/pairs ratio, diagnostic) |
| `MEMRA_FA_V4` | `0` = rollback to the v3 FA lane. DEFAULT ON 2026-07-10: key-per-lane score phase (zero shuffles/key), kernel -7.5..-15% at depth, e2e p2 +5.7% (35B), p3 +0.7-3.9% (all models), p1 -0.8% (noise); full battery green on all three models incl sampled seeded identity + kernel-check; acceptance deltas trajectory-class (±). `noB3`/`stage` = phase probes (WRONG OUTPUT, bench only) |
| `MEMRA_SPEC_DEVACC` | `1` = device-side accept walk + seed gather + KV/recur rollback (round-stream stages a/b; token-identical, perf-neutral machinery — the stage-c consumer measured negative) |
| `MEMRA_SPEC_STREAM` | `1` = pre-issued M-round zero-readback bursts (round-stream stage c, `MEMRA_SPEC_STREAM_M` rounds/burst default 4). MEASURED NEGATIVE 2026-07-10 (35B serve p2 -16%, p1 -4%; 27B p2 wash): always-K draft + fixed-width verify waste > the ~1.5-2ms/round trip savings at real acceptance rates. Kept as the experimental seam; token-identity holds by construction |
| `MEMRA_ROUTER_KERNEL` | `0` = rollback to the cuBLAS router (per-column gemv at verify t, gemvx at t=1). Default ON since 2026-07-10, extended to t=1 the same day: decode + verify route through the SAME kernel (parity by construction — the cuBLASLt n-dependence class is structurally gone). Verify +2-4% spec e2e; t=1 e2e wash (kept for parity) |
| `MEMRA_ROUTER_PREFILL_EXACT` | `0` = rollback to the batched cuBLASLt GEMMs for the MoE **prefill** router (`ffn_gate_inp`) and the shared-expert gate dot (`ffn_gate_inp_shexp`). Default ON 2026-08-02 (`lane/concat-prime-exact`): both cuBLASLt forms are **m-DEPENDENT** — a row's output changes when other rows join the call (router rows [0,19) move 3.9e-3 between m=19 and m=65; shexp gate 1.07e-4 between m=74 and m=75), and under cross-request concat prime batching that made a served request's own expert selection a function of its CO-ARRIVALS (16% of (layer,token) pairs got different experts). The in-house `router_gemv` / `sigmoid_dot_rows` twins — the ones decode and spec verify already use — are m-INVARIANT, so prefill rides them at every t. Fixes serve greedy c=1-vs-c=16 byte identity: Ornith-35B 6/16 -> **16/16**, KAT 7/16 -> **16/16** (Ornith-9B + Qwen3.6-35B ctrl stay 16/16). Receipts: `research/concat-prime-exact-20260802/`. The `=0` arm is a numeric A/B seam, NOT a serving configuration — it forfeits the isolation guarantee |
| `MEMRA_PROFILE_GEN` | `1` = cudaProfiler{Start,Stop} brackets run-gen's timed generate_with (prime included); `2` = capture starts at the DECODE LOOP (prime excluded) — pair with `nsys -c cudaProfilerApi`. Window-cutting whole-run captures misattributes the argmax-gate loop into decode shares (2026-07-10) |
| `MEMRA_PROFILE_SPEC` | `1` = cudaProfiler{Start,Stop} brackets generate_spec (prime included); `2` = capture starts at the ROUND LOOP (prime excluded) — pair with `nsys -c cudaProfilerApi`. Built 2026-07-10 after phase-isolation-by-subtraction proved unworkable on MoE (primes are not fungible: the first cold-stages the expert cache) |
| `MEMRA_LAYER_PROBE=1` | sync+print after every forward stage — bisects an in-graph ILLEGAL_ADDRESS (M3 bring-up tool) |
| `MEMRA_GDN_DIFF=1` | dual-run oracle: chunked GDN prefill checked against the sequential scan per call |
| `MEMRA_MOE_GATE=1` | byte-identity oracle: grouped-vs-sequential MoE FFN compare (pair contract; known benign q8-quantize diff class documented at the gate site) |
| `MEMRA_MOE_GDEC_GATE=1` | byte-identity oracle: grouped-decode vs sequential-axpy expert accumulation |
| `MEMRA_MMVQ_BV=<variant>` | force ONE NVFP4 batched/mmvq variant everywhere (base/pf/r2/r2w8/pfr2/ca/car2/rp*) — the per-variant measurement seam (auto = wave-aware winners; forced-BV concluded auto-optimal 2026-07-06) |
| `MEMRA_KQ_BV=<variant>` | same, narrowed to k-quant dispatch only (interleaved k-quant A/B without touching NVFP4) |
| `MEMRA_FA_FLOOR=1` | prefill-FA floor kernel variant (fa_hd128_check gate bin) |
| `MEMRA_MSCALE_PROFILE` / `MEMRA_MSCALE_NOEAGER` | verify-mscale probe bin controls |
| `MEMRA_TEST_MODEL`, `MEMRA_LLAMA_TOKENIZE`, `MEMRA_ST_TEST_DIR` | test-suite input paths (tokenizer parity, safetensors header tests) |
| `MEMRA_KC_MODELS_DIR` | model dir for kernel-check's REAL-WEIGHT sections (the router weight-oracle bit-identity + m-invariance sweep added by `lane/fast-router`; battery scripts set it, unset skips with a note) |

## 5. Experimental doors (opt-in, documented block)

| flag | door | block |
|---|---|---|
| `MEMRA_LOCKSTEP_BATCH_ATTN=1` | batched m-band full-attention mixer for multi-stream lockstep decode: q/k/v + output projections run once at m (streams share weights), KV-bound work stays per stream. Bit-identical (same-prompt m=2 token-exact vs the per-stream path) | MEASURED FLAT 2026-07-25: m=2 4.72/4.72, m=3 5.24 vs 5.35. Full-attn is the minority layer type in the Hy3 hybrid, so the weight-read saving covers few layers and is cancelled by the norm->q8_1 fusion this path forgoes plus its gather/scatter copies. Primitive retained for a continuous-batching serve loop (batching across requests at higher m, no fused alternative to lose) |
| `MEMRA_MMQ=1` (+`MEMRA_RP=0`) | native W4A4 FP4 MMQ prefill — 1.03–1.06x llama, 1.4–1.76x our default (2026-07-08) | EXACTNESS-BLOCKED: e2m1 activation grid forks argmax/text on long real prompts (p3 reject reproduced 2026-07-07 + 2026-07-08; agent-loop 1/8 self-consistency FAIL). Speed-mode candidate, never default |
| `MEMRA_FP4=1` | hand-rolled W4A4 GEMM in `matmul` (decode/mid-m band) | same accuracy class (maxdiff ~1.0 vs W4A8 0.159); explicit speed/accuracy tradeoff only |
| `MEMRA_FP4_CUTLASS` (+build `MEMRA_CUTLASS`) | CUTLASS sm120 NVFP4 GEMM for m>=128 prefill (`MEMRA_FP4_CUTLASS_OTF=1` = per-call repack, no resident VRAM doubling) | same W4A4 exactness block + resident repack ~doubles NVFP4 weight VRAM (OOMs 27B/24GB) |
| `MEMRA_MOE_GROUPED=1` | expert-grouped MoE prefill prototype (spill-regime 3.4–3.9x vs sequential-stage at cap64, 2026-07-04) | superseded by pairs/dev/MMA on the daily path (barely moves it, 2026-07-06); kept as the `MEMRA_MOE_GATE` oracle pair + seed of the specced fused-MoE-MMQ prefill arc |
| `MEMRA_MOE_F16G=0\|1\|2` | grouped f16 expert prefill lane: dequant the ACTIVE experts once per (layer, projection) to f16, one grouped GEMM over the CSR groups. `1` = cublasGemmGroupedBatchedEx + per-projection sync (round 47; the grouped API runs on cublas-internal streams unordered with ours). `2` = single-kernel grouped m16n8k16 GEMM on the engine stream (round 49): zero syncs, f32 C with the act row-scale folded in. t floor 16 keeps decode/verify on dp4a. Dequant coverage: IQ4_XS/IQ3_S/Q4_0/Q3_K/Q4_K/Q6_K (round 49 — the round-47 IQ4_XS/Q4_0-only table admitted ~1/41 q35 layers, the real cause of that FLAT verdict) | PROMOTED — HOPPER DEFAULT mode 1 (2026-08-01, round 49): with full dequant coverage the q35 board prime measured 5490 (MMQ) / 8380 (mode 1, +53%) / 7990 (mode 2) x3 interleaved, argmax MATCH — the last board loss flipped (see §7). sm_120a keeps the MMQ default: 5090 2026-08-01 (receipts `research/expert-grouped-20260801/`) all gates green but q35 pp2048 e2e FLAT (the f16 workspace write/read traffic eats the kernel-rate win at 858 GB/s) and g26 negative (Q4_0 MMQ wins). f16-mirror numeric class, argmax/spec gated per model; `=0` kills anywhere. PER-MODEL NARROWING (2026-08-01, round 50): the gemma-MoE (gelu) dispatch site defaults OFF — the round-49 default REGRESSED g26 board-2048 prefill -8.3% interleaved x5 on-box (def median 10380 with wild 8.9k-11.7k spread vs off 11317 ±0.13%; receipts `research/f16g-permodel-20260801/`). The silu/qwen class keeps the default; explicit `=1`/`=2` still opens the gemma door (`moe_f16g_gemma_on`) |
| `MEMRA_F16G_SK=0\|32\|128` | mode-2 sk kernel form (round 51, lane/sk-bm128): the single-kernel grouped GEMM runs as a persistent problem-visitor over the REAL CSR tiles (grid-stride flat tile list from a smem prefix over the device offsets — kills the round-49 grid's ~92% early-exit churn under q35's ~17x skew) with TWO tile forms: 32x64x32 2-stage (round-49 geometry) and 128x64x64 3-stage cp.async (cutlass's tile shape on the same sm_80-portable mma.sync m16n8k16 — no wgmma/TMA). `0` = the round-49 grid-scan kernel (rollback seam); `32`/`128` force one visitor form (A/B arms); unset = hybrid split at `MEMRA_F16G_SK_CROSS`. sk128 smem 82944 B dynamic needs the >48KB opt-in — SetAttribute CHECKED with device-fit fallback to the 32x64 form. Every form is BYTE-IDENTICAL to the round-49 kernel (same ascending mma k-chain per element; kernel-check f16g-sk section gates maxdiff==0) | 2026-08-01 receipts `research/sk-bm128-20260801/`: kernel-check ALL GREEN both rigs, argmax MATCH, spec K=1..8 PASS. 5090: q35 board-2048 F16G=2 3403.7 -> 3568.4 (+4.8% interleaved x5, zero overlap; GEMM stage -9.4%). H100: old 7966.1 / new 8299.9 / cublas mode-1 8563.2 -> new sk = 96.9% of cublas (+4.2%, gap 4.7%->3.1%) — PARITY NOT REACHED, mode 1 keeps the Hopper default; residual = the 32x64 2-stage tail form (41.3ms = 31% of stage; cross fine-sweep x8/x16/x24 refuted) — next rung is a deeper tail form (BK=64 3-stage small-BM or register double-buffering) |
| `MEMRA_F16G_SK_CROSS=<m>` | visitor crossover: groups with m_e >= m ride the 128x64x64 form, smaller ones 32x64 (tune seam; ignored under `MEMRA_F16G_SK=0/32/128`) | per-arch swept 2026-08-01: 5090 default 64, H100 32 (`research/sk-bm128-20260801/`) |
| `MEMRA_MOE_Q8_NVFP4=0` | rollback: NVFP4 expert dp4a dot -> f32 arm | DEFAULT ON since 2026-07-17 — the old M3 decode-vs-verify break was the missing per-expert macro fold (fixed); ct-NVFP4 35B runs the q8 arm at IQ4_XS parity. M3/Hy3 never reach the q8 arms (sigmoid-router gates) |
| `MEMRA_KV_FP8` | qwen/non-gemma trunk KV in e4m3 (hd128 rides the register g-lane; draft scratch stays q8_0/q5_1). Default OFF everywhere. The 2026-07-12 9B '+0.7-4% w/ depth' did NOT reproduce on the 2026-07-28 build (12k A/B: fp8 −1%; d1736 flat; acceptance identical, all gates green) — adoption reverted by measurement. Remaining value = ~45% smaller KV for ctx-limited serving. Explicit 0/1 wins | per-model verdict 2026-07-12; adoption attempt + revert 2026-07-28 |
| `MEMRA_MOE_MMA_T=<n>` | MMA t-floor override (bisect seam; <16 puts spec verify on MMA) | verify must stay dp4a (dispatch-parity law) — measurement only |
| `MEMRA_IQ_FAST=0` | rollback: non-expert IQ4_XS matmuls back to the Stage-A f32 oracle path | DEFAULT ON since 2026-08-02 (`research/kat-anomaly-20260802/`) — the old opt-in default was the KAT-Coder IQ4_XS decode anomaly: its IQ4_XS trunk (attn_qkv/attn_gate/ssm_out/shexp, ~0.52GB/tick) rode the oracle kernel; dp4a admission moved decode 106.7 -> 193.4 (x5 interleaved, +81%) and pp512 228 -> 697. Expert-bank IQ4_XS has its own dispatch — supported models (Q8_0/Q4_K/NVFP4 trunks) are dispatch-unchanged |
| `MEMRA_PP_STAGES=2` / `MEMRA_PP_SPLIT=<layer>` | pipeline-parallel 2-stage seam (M1 of the multi-GPU build): decode_step runs layers [0,split) as stage 0 ON ITS OWN CUDA STREAM, hands the [n_embd] f32 residual through a persistent boundary slot (transport-selected: dtod same-device / cudaMemcpyPeerAsync cross-device, evented TX/RX), runs [split,n) as stage 1 on its stream. Split defaults n_layers/2. Generic eager + gemma4 arms only (batch/dc/graph/spec warn-once) | M1 increment 1 (2026-08-01): BIT-IDENTICAL via pp2-gate — q9 q8_0 splits 16/5/31 + g12 q4_0 splits 24/7 (`research/m1-pp2-20260801/`). Increment 2 (2026-08-01): per-stage streams + events + transport selection + stage-owned KV allocation, BIT-IDENTICAL again across every knob (`research/m1-inc2-20260801/`). Default OFF, zero behavior change unset |
| `MEMRA_PP_DEVICES=<d0>,<d1>` | pp2 stage->device placement (increment 2): stage s runs on device ds — remote stage gets its own Engine/context, KV for its layer range allocates on its HBM (`Cache::new_pp2`), boundary switches to cudaMemcpyPeerAsync on the publishing stream (M0: 2.8x NCCL at PP activation sizes; cuDeviceCanAccessPeer-guarded, peer-inaccessible = loud FAIL). Weights stay on the primary device (peer-read bring-up placement) | default unset = all stages on the primary device (today's behavior; `0,0` exercises the plumbing on one device — gated bit-identical). Cross-device gates on the 8x box: `MEMRA_PP_DEVICES=0,1 pp2-gate <model>` |
| `MEMRA_PP_OVERLAP=1` | pp2 double-buffered boundary slots (M2 seed): TX(t+1)'s write-after-read guard waits only the previous use of ITS slot, so a pipelined loop can issue token t+1 stage 0 while token t stage 1 runs. Scheduling structure only — every token's math is event-ordered | default OFF; single-device gated bit-identical (the eager decode API still syncs per step, so real overlap needs the M2 loop restructure) |
| `MEMRA_PP_STREAMS=0` | pp2 rollback seam to the increment-1 same-stream boundary (two plain dtod copies, no per-stage streams/events/devices) | rollback only; increment-2 choreography is the default when the door is open |
| `MEMRA_EAGLE` / `MEMRA_EAGLE_ALIGN=0` | EAGLE draft lane (run-eagle bin; ALIGN=0 = un-shifted MTP-style pairing A/B) | experimental lane, not on the daily path |
| `MEMRA_GEMMA_DRAFT_GRAPH=1` | gemma draft chain replays as ONE captured CUDA graph (keyed kr/rung/window-regime; pos slots + g_seed are eager-filled graph inputs) | steady-state perf ~0 (launch tax hidden at 96.7% busy) — exists as the BURST enabler; capture-retain allocator mode ships with it (2026-07-12) |
| `MEMRA_GEMMA_SPEC_BURST=<M>` (+`_DRAFT_GRAPH=1`) | pre-issue M full spec rounds with ONE host sync per burst (verify-stream + device accept/seed/rollback/ring on the round_stream generics) | EXACT (stream 128/128 short+depth, 26B+31B) but PERF-NEGATIVE single-stream — no draft/verify overlap materializes on one stream and fixed-K costs the adaptive policy (26B short 304 vs 379; 31B 74 vs 88). Default 0. Stage 2 = second-stream speculative draft(N+1) under verify(N) |
| `MEMRA_GEMMA_ROUND_GRAPH=1` | the WHOLE gemma spec round (draft chain + stream verify + device accept/seed/rollback/commit + `spec_adapt_k` device adaptive depth) captured once per regime, ONE graph launch per round | EXACT (26B 64/64 short + 128/128 depth, 31B 64/64) with BETTER round dynamics (tok/round 5.33 vs 4.74) but PERF-NEGATIVE at fixed shapes (−10/−11% interleaved, 2026-07-13) — the replay pays fixed k_cap drafts + fixed-width verify, not launch gaps (the enqueue-ahead law, 4th confirmation). Door kept for the exec-update-width phase 2; 31B pair pending a valid window |

---

## 6. Research platform (MTP-heal, dual-shape)

memra's second shape is a research platform. The first protocol (MTP-heal) measures MTP draft-head
acceptance at FULL PRECISION as the ceiling, then on the NVFP4 daily GGUF — the delta is the quant
hit on drafting. See HANDOVER "MEMRA DUAL-SHAPE".

| flag | what it does |
|---|---|
| `MEMRA_FULL_PREC=1` | FULL-PRECISION LOADER MODE (default OFF). Bypasses the standing loader law (large BF16/F8 → Q8_0/NVFP4 re-encode, the "Float-poison" tripwire — the very re-encodes this mode must NOT do). Everything loads Float; large 2D bf16 matmul weights stay **bf16-resident** (`GpuTensor::FloatBf16`) with dequant-on-use, so the 9B (~18GB bf16) + f32 activations fit 24GB instead of blowing to ~38GB as an all-f32 materialization. Compute rides the Stage-A f32 oracle path end to end (`MEMRA_FAST=0`-class). **SLOW IS FINE** — this mode is for the exactness ceiling, not speed. Also forces the EAGER spec draft (CUDA-graph capture can't enclose cuBLASLt f32 GEMV or the dequant alloc) and disables `MEMRA_FRSPEC_TRIM` (the ceiling wants the model's natural full head; the trim gather is Quant-only anyway). The Float-poison tripwire warnings are correct here and are suppressed. |

Per-slot acceptance for this protocol comes from the existing **`MEMRA_SPEC_STATS=1`** (§4) — its
`per_slot=[...]` line is the deep-K decay profile; no separate flag was needed.

**Acceptance battery harness** (`tools/`, no GPU-free CI — runs on the rig):
- `acceptance_battery.sh <model> <out.jsonl>` — fixed prompt set (p1/p2/p3) + the 8-turn agent loop, N≥3, one JSONL row per (prompt,K,run). `FULL_PREC=1` for the bf16 ceiling arm, plain for the NVFP4 arm.
- `agent_loop_acceptance.sh <model> <out.jsonl> <arm>` — the 8-turn accumulative agent-loop protocol (recreated in-repo; feeds each turn's output forward).
- `acceptance_parse.py` — one run-spec invocation (merged stdout+stderr, `MEMRA_SPEC_K=<k>` + `MEMRA_SPEC_STATS=1`) → one JSONL row.
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

## 7. The H100 / sm_90a lane (merged from `backend/sm90a-boot` rounds 26-36; unified-tree rounds 37-49)

On an H100 the arch auto-detects (`MEMRA_CUDA_ARCH=90a` forces it). Every promoted
config below followed the repo law:
harness proof -> engine seam -> full battery (kernel-check pins, greedy/stream identity,
decode/prime/graph gates, interleaved x5 A/B) -> default-ON with a `=0` revert. The
evidence ledger is [ARCHITECTURE-H100.md](../ARCHITECTURE-H100.md); the one-command
battery is `tools/validate-h100.sh <model.gguf> [--quick]`.

### Promoted defaults (Hopper; `=0` reverts each)

| flag | default | what it does |
|---|---|---|
| `MEMRA_GDN_MMA` | ON (90a) | GDN K4/K5 chunk pair on mma tensor cores (bf16 W/k mirrors); kernel-check pins both configs |
| `MEMRA_GDN_WGMMA` | ON (90a) | K4+K5 FUSED wgmma kernel (`gdn_k45_wgmma`) + K2 on wgmma; Y/Ssnap tensors never materialized; qb16/Pb16 mirrors; nested inside the mma config. Official prefill +2.56%, batched prime +3.8% |
| `MEMRA_FA3` | ON (90a) | FA3 full-attn prefill: TMA swizzled Q/K/V ring + wgmma, 4.8x the mma kernel (v11) |
| `MEMRA_L2_V2` | ON | l2-norm v2 with in-epilogue bf16 mirror folds (kb16 + qb16 — the standalone mirror passes are gone) |
| `MEMRA_GDN_DB` | ON | de-broadcast q/k at num_k heads (`hk` plumbed through the chunked chain) |
| `MEMRA_EMBED_DEV` | ON | device-side embed gather (host dequant + htod per token gone; +12.4% official lane) |
| `MEMRA_GEN_GRAPH` | ON at budget >= 256 | hybrid graph door in `generate_with`: capture/replay decode over the batched-prime cache. Official decode 190.3 -> 220.7 tok/s (+16%); bit-identical (graph-decode-gate, 256 steps x 16 fa buckets). `1` forces at any budget |
| `MEMRA_CONV_FUSE` | ON | carried-ring conv + SiLU + GDN repack in one pass (bit-identical class) |
| `MEMRA_GDN_VL` | ON | varlen batched cores: ONE launch per stage for all B sequences (bit-gateable vs per-seq) |
| `MEMRA_MOE_F16G` | ON (90a, mode 1, silu class) | grouped f16 expert prefill lane (round 49 promotion): q35 board prime 5490 -> 8380 (+53%) at 41/41 dequant coverage — the last board loss flipped. Round 50: the gemma-MoE (gelu) site defaults OFF per model (g26 board-2048 prefill -8.3% under the round-49 default; `=1`/`=2` still opens it). sm_120a keeps MMQ (measured FLAT/negative on the 5090). Modes + full history in §5; `=0` reverts |
| `MEMRA_PP_F16` | ON (90a), per-model | f16 prefill mirrors on the cuBLASLt lane (dequant-once at load, decode untouched — m>=16 arm only). Coverage classes: Q8_0 (2026-07-26, pp512 +80%), Q6_K (round 47, q27 pp2048 +64%), Q4_K + Q5_K (round 49, q27 pp2048 +54% at default budget). PER-MODEL argmax-gate arbitration (round 45): the qwen Q8_0 dense class defaults OFF (the f16-vs-int8 numeric gap flips a real-prompt top-tie deterministically — gate-violating defaults don't ship); gemma + the MoE hybrids hold MATCH and keep mirrors. `=1` forces (diagnostic), `=0` kills everywhere |
| `MEMRA_PP_F16_BUDGET_MB` | 32768 | VRAM cap for the f16 mirrors, layer-order prefix; admission passes ordered Q6_K -> Q4_K -> Q5_K so full Q6_K coverage (the ~10x dequant-GEMM replacement) is the floor. Machine-specific config: q27 serving at 43008 measured pp2048 6564 (+33% over default; +Q5_K full stack 8063). On 80GB the f16-mirror budget and the KQRP decode mirrors compete for the same ~15GB — per-box serving configs choose (round 49b) |

### Serving (memra-server)

| flag | default | what it does |
|---|---|---|
| `MEMRA_PRIME_BATCH` | 6 | max sequences per cross-request prime batch (B=8 measured within spread of 6) |
| `MEMRA_PRIME_BATCH_MAX_T` | 2048 | max per-seq tokens eligible for batch (raised 320 -> 2048 round 35: the old crossover was measured on pre-wgmma cores; batched now wins at every tested T) |
| `MEMRA_PRIME_BATCH_HOLD_MS` | 4 | batch-formation hold for staggered arrivals |
| `MEMRA_SERVE_GS` | ON | solo greedy interactive sessions ride GraphSession replay. Round 35: promotion happens AFTER chunked prefill (`graph_session_from_cache`) — the old token-wise re-prime was a 3x end-to-end loss on long prompts (6.4 -> 2.8s measured) |
| `MEMRA_GS_MIN` | 384 | min generation budget for GraphSession promotion (capture amortization) |
| `MEMRA_DECODE_BATCH_CAP` | auto | batched-decode chunk width door (engine assert + scheduler chunking, clamp 1..32). Unset = per-model policy: **16 on exact-16 tier models** (`decode_batch_exact16_ok` — every matmul has a bit-exact b16-class kernel; Q8_0 needs the q8rp mirror; the step scopes verify_exact so the m>=16 GEMM/MMQ arms never fire), else 8. Inc3 receipts (5090, 9B Q8_0+mirror, `research/batched-tick-inc3-20260801/`): gate2 bit-strength PASS at chunk 12/16 (s32+s160), 32-seq engine tick +12% vs chunk 8 same-mirror (chunksweep x5). Chunk widths above the exact tier remain a MEASUREMENT DOOR: m=16 GEMM/MMQ classes bit-shift logits at step 0 (maxdiff ~1.3-2.3e-1) and m>16 has no exact kernel class at all. HISTORY: the pre-inc2 H100 probe (`research/darklane-serving-20260801/`) measured cap 15/16 flat-or-worse on the then per-seq-serial-bound tick — that verdict was re-swept after inc2's z-batched attention made the tick weight-stream-bound (the stale-verdict law) |
| `MEMRA_SERVE_DEVSAMPLE` | ON | `0` = host-sample every batched-tick row from last_logits (the pre-2026-08-01 tick). Default: greedy-no-penalty rows sample via device argmax (bit-identical to host argmax), pure-temperature rows via seeded per-row gumbel draw (distribution-equal, batch-composition-independent — decode-batch-gate gate3); top-k/top-p/min-p/penalty rows keep the host path. Kills the measured 1.36 ms/row host temp-sample at 248k vocab (~45% of the B=8 9B tick) |
| `MEMRA_BATCH_PHASE` | off | `1` = sync-bounded per-phase accumulators inside decode_step_batch (12 phases, printed by decode-batch-bench). Shares rank the tick; the added syncs inflate the total — diagnosis only |
| `MEMRA_BATCH_FA` | ON | `0` = per-seq fa_decode loop in the batched tick (the pre-inc2 launch train). Default: one blockIdx.z=sequence `fa_decode_vec_q_seqs_v4` launch + one combine per layer via the per-step K/V pointer table — per-seq bit-identical (kernel-check seqs pins; decode-batch-gate strict s160). Auto-falls back per step for non-v4 rows (t_kv<96, fp8-KV, hd!=256) or an `fa_split_keys` rung straddle inside the batch (batched-tick inc2, 2026-08-01: B=8 engine +19.7%) |
| `MEMRA_BATCH_APPEND` | ON | `0` = per-seq KV-append loop in the batched tick. Default: one z-batched `append_quantize_kv_q8_0_q5_1_seqs` launch per layer through the same pointer table — cache bytes bit-identical (kernel-check pin). fp8-KV rides the per-seq g-module path |
| `MEMRA_SERVE_LEANLOGITS` | ON | `0` = full [B, n_vocab] logits D2H every batched tick (the pre-inc2 tick). Default: device-sampled rows skip the D2H; the last row parks on-device per cache (`Cache.last_logits_dev`) and is D2H'd once at retire for the KV-reuse pool (the audited last_logits consumer). decode-batch-gate gate3c pins lean==full bitwise. Serving: 9B temp-0.7 c=8/16/32 medians 654/657/659 tok/s vs 565/519/508 lean-off (H100, N=4, `research/batched-tick-inc2-20260801/`) |

### Tuning / diagnostic seams (sm_90a lane)

| flag | default | what it does |
|---|---|---|
| `MEMRA_CUDA_ARCH` | 120a | build arch: `90a` = the Hopper lane (wgmma/TMA kernels compile; portable-PTX correctness path rides along) |
| `MEMRA_FA_SPLIT` | ctx-adaptive ladder | fixed FA-decode split size; ladder re-swept on 132-SM H100 round 35 — holds (sp32 +0.35% = noise) |
| `MEMRA_PRIME_CHUNK` | 0 (monolithic) | prefill chunking size in tokens (cross-call state-carry battery leg) |
| `MEMRA_EVT` | off | `1` re-enables cudarc cross-stream event tracking (blocks graph capture-over-primed-cache: `graph_session_from_cache` refuses) |

### Refuted on this lane (mechanism in ARCHITECTURE-H100.md; no flag shipped)

Monolithic + piecewise prime CUDA graphs (Lt address-variant kernels); W8A8/fp8/CUTLASS/
Lt-autotune prefill GEMMs at m=512 AND m=2048; FA3 shapes v6-v9; 12 k45 scheduling forms;
K3 solve ILP-split AND tf32 inverse-product; v6 exchange-free k45; Q8_0-EXACT int8 GEMM
(triple-refuted: per-block rescale 5.4x naive, 17x pipelined — ptxas C7517 serializes
cross-bank GMMA register reads); FA3-TMA-ring k45 rebuild (refuted by composition).
The remaining single-seq prefill residual (~27% vs vLLM's INT8 GEMM class) is an
owner-gated accuracy decision (w8a8-class numerics change model outputs).

## Removed in the 2026-07-08 flag audit (concluded flags — JSONL rows are the record)

`MEMRA_SPEC_DSPARK` (lived hours, 2026-07-30): DSpark-class marginal-rate verify window
(arXiv 2607.05147) — S_{j+1}·T(j) > E[tok](j)·t_draft with profiled t_draft/t_verify EMAs.
Measured FLAT: never cuts at accept ≥ 0.8 (31B depth: identical rounds/accepts, −0.4%
noise), par on the 26B depth high mode (278 vs 281, −1.2%); the fixed 0.7 self-keyed cut
already covers the low-accept case. Arm deleted; note: the survival estimator used the
current-conditional-as-proxy — a learned estimator (the paper) could cut smarter.

| flag | verdict | record |
|---|---|---|
| `MEMRA_FA_PPOOL` | micro −5–12% but e2e +0.5–0.8% at d6257, under bar | rig5090 2026-07-08 (fadepth) |
| `MEMRA_FA_CMB_WIDE` | wide combine ~−1µs — the 98-split serial FP chain bounds it, not SM count | rig5090 2026-07-08 (fadepth) |
| `MEMRA_FA_SK` | shared-K verify fold: occupancy collapse, p3 spec 108→51.9 tok/s | g7e 2026-07-05 |
| `MEMRA_Q4V` | issue-reduced q4_K matvec: DRAM hides it, ~+0.08% decode = neutral class | rig5090 2026-07-08 (q4issue) |
| `MEMRA_Q6_ISSUE` (5 variants) | all variants identical at locked clock — the premise refuted itself (DVFS artifact) | rig5090 2026-07-08 (q6issue) |
| `MEMRA_GDN_FUSE` | fused GDN prep+scan: NEUTRAL on eager | rig5090 2026-07-08 (merge row) |
| `MEMRA_MMVQ_MR` (+mr4, q4_K/q6_K mr2 kernels) | override never used: q4_K mr2 +0.7% (noise), q6_K flat, mr4 crashes | 2026-07-05/07 notes; HANDOVER |
| `MEMRA_GEMM_M` | m=4 MMA verify NEGATIVE (96.6 vs 99.1; grid starves + FP-order acceptance dip) | rig5090 2026-07-06 |
| `MEMRA_MMQ_STREAMK` (+sk/fixup kernels) | 1.11x per-GEMM but k-split f32 reorder flips model argmax (FP-order lesson #3) | rig5090 2026-07-03 |
| `MEMRA_SPEC_ADAPT` (qwen EMA policy) | adaptive-K: honest loss to static per-class optima (EMA lag). The FLAG NAME lives again for gemma (§3) with a different policy (accepted+1 clamp), ported back to the qwen path 2026-08-01 as an opt-in (§3) | rig5090 2026-07-07 |
| `MEMRA_SPEC_KVLOCAL` | legacy round-local draft scratch: −35 accept pts, incompatible with sessions | rig5090 2026-07-03 sweep; HANDOVER |
| `MEMRA_SPEC_HSAME` (+pseudo-seed passes/graph) | legacy same-row pairing: −16 accept pts vs predecessor pairing | rig5090 2026-07-04 |
| `MEMRA_MOE_GHOST` + `MEMRA_MOE_FAST_ADMIT` | second-miss ghost filter: net loss in BOTH regimes (spill +3.4% with it off; 96GB bypassed it) — first-miss admit is the only policy, FAST_ADMIT became a no-op | rig5090 2026-07-06 / g7e 2026-07-04 |
| `MEMRA_MOE_ORDER` | descending-m_e expert order won (1.34x first-forward); the `=id` restore arm was dead | rig5090 2026-07-04 |
| `MEMRA_FA_VEC` | vestigial: dispatch reads `MEMRA_NO_FA_VEC` since the 2026-06-28 default-flip. Writes removed; the kernel-check scalar-vs-vec gate now toggles `MEMRA_NO_FA_VEC` (the old toggle had gone vacuous) | audit 2026-07-08 |
| `MEMRA_GEMM` | zero read sites (gate went unconditional when MMQ Phase 0 shipped); writes removed from scripts/docs | audit 2026-07-08 |
| `MEMRA_NO_EVT` | legacy no-op alias of the EVT elision default | audit 2026-07-08 |

Bench bins deleted with their flags: `q4v_bench`, `q6iss_bench`, `fa_ppool_bench` (superseded by
`fa_v2_bench`, which keeps the FA depth/split/smem sweep).
