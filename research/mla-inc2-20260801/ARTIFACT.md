# GLM-5.2 GGUF artifact pin — for the 8xH100 box (arrives 2026-08-02 11:30Z)

Research date: 2026-08-01, all facts from the HF JSON APIs at the pinned revisions plus a
direct read of the GGUF headers (the 9.4 MB metadata shard + 512 KiB range-reads of the other
10 shard headers, ~14 MB total; sha256 of the fetched metadata shard verified against its LFS
oid). NO weights downloaded anywhere. Constraints honored: NOT a REAP variant (owner:
REAP quality-rejected), Q4 class fitting 8x80 GB with KV/activation room, reputable quantizer.

## Decision: `unsloth/GLM-5.2-GGUF` @ `abc55e72527792c6e77069c99b4cb7de16fa9f23`, quant `UD-Q4_K_XL`

- 11 parts, **467,289,111,904 bytes = 435.19 GiB** (~4.96 bits/param over 753.86B stored).
- Fit: 640 GB (596 GiB) VRAM − 435.2 GiB weights ≈ **161 GiB left** for latent-KV + activations
  (~20 GiB/GPU under EP/PP8). MLA latent cache is 87.8 KB/token f16 across 78 layers, so even a
  256K-token session is ~22 GiB — fits.
- Full 744B expert bank (`expert_count = 256`, no pruning), imatrix-calibrated
  (`unsloth_calibration_GLM-5.2.txt`, 1002 entries / 88 chunks), repo lastModified 2026-06-23,
  251k downloads.
- unsloth ships NO plain Q4_K_M — its Q4 arms are all UD-prefixed. `UD-Q4_K_M` exists at
  465,825,525,088 B (433.83 GiB, shares the identical first/metadata shard); the XL is +1.46 GB
  for a strictly better tensor mix — take the XL.
- bartowski has NO GLM-5.2 repo (only `bartowski/zai-org_GLM-5.1-GGUF`; both plausible 5.2
  names 401 on HF).

### File manifest (sha256 = HF LFS oid; verify each part after download)

| file (`UD-Q4_K_XL/`) | bytes | sha256 |
|---|---|---|
| GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf | 9,423,744 | 3256ac8c290273f0965ff39e93a8bcd07dc99bcd23e923bd4b7306ef39061038 |
| GLM-5.2-UD-Q4_K_XL-00002-of-00011.gguf | 49,433,942,336 | aaedfb89d314d6967a80005b93a9c460a494babc6c3e4f0138e21891e21572e1 |
| GLM-5.2-UD-Q4_K_XL-00003-of-00011.gguf | 48,566,415,136 | a2b45b63075b2e1bc8a73c9ce531ccea54c03001286a80f77454871aa93fdca8 |
| GLM-5.2-UD-Q4_K_XL-00004-of-00011.gguf | 48,566,415,136 | b5404d8d17b63e127aa34c1f98cef64d3722050d8ef1a0792dba816477f4c606 |
| GLM-5.2-UD-Q4_K_XL-00005-of-00011.gguf | 48,566,415,136 | 9ab79e1947115be35da815c1be2812a1451d3ec11f9f5d60dd3ba152e1ed5be2 |
| GLM-5.2-UD-Q4_K_XL-00006-of-00011.gguf | 48,566,415,136 | 43a2631ee392492f8857bae6c88660e0f1cac0fd6bc40d832538ac5421b3167b |
| GLM-5.2-UD-Q4_K_XL-00007-of-00011.gguf | 48,566,415,136 | 1efd96717a956a160a1717999c7dedbe601b5787ea6220d8185d232e95eff893 |
| GLM-5.2-UD-Q4_K_XL-00008-of-00011.gguf | 48,566,415,136 | 3460334e8148d12402c8f5adf684b132918504bbea4d3aecd74801121e8c8a99 |
| GLM-5.2-UD-Q4_K_XL-00009-of-00011.gguf | 48,566,415,136 | 7f6be8ce1c9dcb973ede026b7341657f8add8617f386f77cc165ff697cf9620d |
| GLM-5.2-UD-Q4_K_XL-00010-of-00011.gguf | 48,566,415,136 | 6a26bf391e6f1de947e63016d11ada565f7476a06cb90b444f6db334baa949f9 |
| GLM-5.2-UD-Q4_K_XL-00011-of-00011.gguf | 29,314,424,736 | 27032b927daa606872d887c56631c5278a788b39d219784b262e1df3d4cb851e |

Repo carries GGUFs + README + imatrix file only — **no config.json / tokenizer files**;
tokenizer (`gpt2`/`glm4` pre, 154,880 tokens) and the chat template are embedded in the GGUF
header. HF source endpoints:
`https://huggingface.co/api/models/unsloth/GLM-5.2-GGUF` (revision `sha`),
`.../tree/abc55e72527792c6e77069c99b4cb7de16fa9f23/UD-Q4_K_XL` (per-file `lfs.oid`/`size`).

### GGUF header facts (read from the artifact itself, 2026-08-01)

Confirmed against the parse arm this increment ships (`crates/memra-gguf/src/config.rs`):

| key | value | note |
|---|---|---|
| `general.architecture` | `glm-dsa` | |
| `glm-dsa.block_count` | **79** | INCLUDES the NextN layer (78 trunk + 1) — memra convention holds |
| `glm-dsa.nextn_predict_layers` | 1 | |
| `glm-dsa.attention.head_count` / `head_count_kv` | 64 / **1** | converter writes MQA head_count_kv |
| `glm-dsa.attention.key_length` / `value_length` | **576 / 512** | latent row / V prefix view |
| `glm-dsa.attention.key_length_mla` / `value_length_mla` | 256 / 256 | |
| `glm-dsa.attention.q_lora_rank` / `kv_lora_rank` | 2048 / 512 | |
| `glm-dsa.rope.dimension_count` / `rope.freq_base` | 64 / 8e6 | |
| `glm-dsa.expert_count`/`used`/`shared`/`leading_dense` | 256 / 8 / 1 / 3 | |
| `glm-dsa.attention.indexer.{head_count,key_length,top_k}` | 32 / 128 / 2048 | |
| `glm-dsa.attention.indexer.types` | **ABSENT** (69 KVs total) | see below |
| `general.file_type` | 15 (Q4_K_M base) | UD mix on top |
| split | `split.count = 11`, `split.tensors.count = 1809` | |

**indexer.types is absent** in this 2026-06 conversion → llama.cpp (and now memra's parse arm,
tested `parse_glm52_without_indexer_types_key`) falls back to the hardcoded GLM-5.2 default
table: 21 full / 57 shared for ctx ≥ 1M. The parse arm reproduces exactly that.

**Recorded discrepancy (do not silently reconcile):** increment-1 RECEIPTS §5 says GLM-5.2
"ships indexer tensors only on full layers", but THIS artifact carries per-layer indexer
tensors on **all 79 layers** (395 = 5x79: `indexer.attn_k`, `attn_q_b`, `k_norm.weight`,
`k_norm.bias`, `proj`) — including the MTP block, which runs dense MLA and never uses them.
On-box audit item for increment 3: gguf-dump the real file, confirm, and decide whether
shared-layer indexer tensors are copies/zeros (they are never dispatched either way — memra
only loads indexer tensors on FULL layers, increment 6).

**MTP/NextN: shipped.** blk.78 has 27 tensors (vs 23 for a trunk layer); the nextn set is
`eh_proj.weight` [12288,6144] Q8_0 + `enorm`/`hnorm`/`shared_head_norm` (F32). NO
`nextn.embed_tokens` / `nextn.shared_head_head` — the head reuses `output.weight` (memra's
MtpHead already falls back; the micro fixture mirrors this exact set).

## Download (run on the box, NVMe target)

```bash
# primary — pinned revision, only the chosen quant
hf download unsloth/GLM-5.2-GGUF \
  --revision abc55e72527792c6e77069c99b4cb7de16fa9f23 \
  --include "UD-Q4_K_XL/*.gguf" \
  --local-dir /opt/dlami/nvme/models/GLM-5.2-GGUF

# post-download gate: verify EVERY part against the manifest table above
cd /opt/dlami/nvme/models/GLM-5.2-GGUF/UD-Q4_K_XL && sha256sum *.gguf

# load by pointing at part 00001 (multi-part GGUFs auto-join)
```

Expect ~40-60 min at 1-2 GB/s HF throughput; start the pull first thing, work on
increment-3 scaffolding while it streams.

## Alternative arm (if the unsloth pull fails verification)

`DevQuasar/zai-org.GLM-5.2-GGUF` @ `823d10f1e11a5fff13a6b9e67f06b96476f7a605`, plain
`Q4_K_M`, 30 parts at repo root, total 454,552,364,896 B = 423.3 GiB, full 753.3B bank,
arch `glm-dsa`, ctx 1,048,576. Caveats: small community quantizer (3.2k downloads), appears
static (non-imatrix). Full per-file sha256s: tree endpoint at the pinned sha.

```bash
hf download DevQuasar/zai-org.GLM-5.2-GGUF \
  --revision 823d10f1e11a5fff13a6b9e67f06b96476f7a605 \
  --include "zai-org.GLM-5.2.Q4_K_M-*.gguf" \
  --local-dir /opt/dlami/nvme/models/zai-org.GLM-5.2-GGUF
```

Other repos surveyed and excluded: REAP variants (0xSero, pipenetwork — owner-rejected),
weight-modified full-bank repos (huihui-ai abliterated, phaseonx11 uncensored), sub-Q4 mixes
(sokann, antirez, easiest-ai-shawn).

## llama.cpp reference note

No stated minimum version in the repo README. The GGUFs exist since 2026-06-17, so `glm-dsa`
support predates that; the increment-1 pinned master commit
`ddd4ec1428a6201e18975ea52b07c71e0f9aef26` (2026-08-01) loads it — use that (or newer) as the
on-box comparison build for the increment-3 argmax gates.
