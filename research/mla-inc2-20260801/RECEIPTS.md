# MLA bring-up increment 2 — receipts (2026-08-01, lane/mla-inc2)

Weights-free increment: everything a GLM-5.2 bring-up needs that does NOT need the model
bytes. Raw test logs in this directory (`cpu-tests.log`, `gpu-load-test.log`); artifact pin
in `ARTIFACT.md`. Base: `restructure/public-split` @ 4e395fa3 (increment 1 merged).

## Deliverables + gate verdicts

| # | Deliverable | Gate | Verdict |
|---|---|---|---|
| 1 | `glm-dsa` parse arm: `Arch::GlmDsa`, `MlaConfig` (+`DsaConfig`) on `ModelConfig` (config.rs) | unit tests vs pinned GLM-5.2 values: q_lora 2048, kv_lora 512, nope/rope 192/64, qk/v 256/256, latent 576, V-view 512, scale 1/16, router sigmoid+2.5+norm, shared 1, dense-lead 3, DSA 32/128/2048, 21-full/57-shared indexer layout, block_count 79 = 78+1 MTP | **GREEN** (`parse_glm52_pinned_metadata`, `parse_glm52_without_indexer_types_key`, `non_glm_arch_has_no_mla`) |
| 2 | Micro-GGUF fixture: `micro_gguf.rs` writer + `write_glm_dsa_micro` (2 trunk layers + MTP, hidden 64, kv-rank 16, §3.1 names/shapes incl. kv_b→(k_bᵀ,v_b) split + partial indexer + real-artifact nextn set), generated at TEST TIME (~100 KB, nothing committed) | parse + tensor-presence audit + value-level split-convention check through the reader | **GREEN** (`micro_fixture_parse_and_tensor_audit`, `micro_fixture_kv_b_split_convention`) |
| 3 | `MlaAttnLayer` loader arm behind the Arch gate: `Mixer::Mla`, `MlaGeom` latent-cache config, shape audits at load; every forward `match` routes Mla to a named `unimplemented` guard (increment 4), `mixer_in_q8_1_fast` returns false | workspace compiles clean (`cargo check --workspace --all-targets`); device load on the 5090 under flock: 2 trunk Mla layers + Mla MTP head, geometry + conversion-split shapes asserted on device tensors | **GREEN** (`gpu_load_glm_dsa_micro_fixture`, see `gpu-load-test.log`) |
| 4 | CPU-reference block forward at fixture scale: fixture tensors → attn_norm → q_a/q_a_norm/q_b → kv_a split + kv_a_norm → interleaved rope → mla.rs naive AND absorbed cores → wv_b decompress → wo | absorbed ≡ naive on the BLOCK OUTPUT, rel ≤ 1e-5, on every layer incl. MTP: prefill t=6, decode t_q=1/t_kv=9, chunked 3-over-11; plus bit-level w_uk/w_uv vs unsplit kv_b cross-check | **GREEN** (`cpu_block_forward_absorbed_equals_naive_{prefill,decode,chunked}`) |
| 5 | Artifact pin | repo+revision+per-file bytes+sha256, header-verified metadata, `hf download` commands | **DONE** — `ARTIFACT.md`: unsloth/GLM-5.2-GGUF @ abc55e72, UD-Q4_K_XL, 11 parts, 467.29 GB |

Zero-behavior-change evidence: full workspace compiles with no warnings introduced;
`cargo test -p memra-engine --lib` 32 passed / 0 failed (incl. the 4 increment-1 mla oracle
tests, unchanged); `memra-gguf --lib` 59 passed + the 2 PRE-EXISTING environmental minimax
failures (they hardcode `/data/ai-ml/hf-models/minimax-m3-nvfp4-reap50/*`, absent on this
machine — unrelated to this lane, fail identically on the base commit). memra-kv /
tokenizer / sampling suites green. No other arch touches any new code path: the Mla arm is
keyed on `cfg.mla`, which only `glm-dsa` populates.

## Ground-truth deltas found this increment (vs increment-1 assumptions)

1. The real unsloth GGUF ships **without** `attention.indexer.types` — llama.cpp's hardcoded
   GLM-5.2 default table (21 full / 57 shared for ctx ≥ 1M) is load-bearing, not just BC.
   Parse arm replicates it (`config::glm52_default_indexer_types`), tested.
2. The artifact carries indexer tensors on **all 79 layers** (incl. MTP), contradicting the
   increment-1 note that 5.2 ships them only on full layers. Recorded in ARTIFACT.md as an
   on-box audit item; harmless to memra (indexer tensors load only on FULL layers, inc-6).
3. `attention.head_count_kv = 1` in the artifact (MQA), `block_count = 79` (includes MTP) —
   both now pinned by header read, matching the fixture and the parse tests.
4. The artifact's nextn set is eh_proj+enorm+hnorm+shared_head_norm only (no
   shared_head_head / embed_tokens) — fixture updated to mirror exactly.

## Increment 3 (on-box, weights land 2026-08-02): the plan

Everything below is a *scaling exercise* over gates that already exist at fixture scale —
no new plumbing classes. See the lane hand-off message for the hour-by-hour version.

1. Pull + verify artifact per ARTIFACT.md (sha256 every part), gguf-dump audit vs the parse
   arm (incl. the indexer-tensors-on-all-layers question).
2. Parse + load the real 79-layer file through the (already-tested) loader arm; fix only
   quant-specific gaps (3D wk_b/wv_b row_bytes derivation for Q4/Q8 — the one known TODO,
   guarded by a load-time assert).
3. Layer-0 CPU reference vs llama.cpp `--dump-tensors`-class activations, maxdiff < 1e-3;
   then full-stack CPU forward argmax vs llama.cpp same-GGUF same-prompt (DESIGN §4 row 3).
