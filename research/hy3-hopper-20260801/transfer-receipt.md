# Hy3 -> Mumbai H100 transfer receipt (2026-08-01)

Box: darklanes-bench Mumbai H100 (`13.203.16.133`, H100 80GB HBM3, 16 vCPU, 249G RAM).
Disk at start: root EBS 290G/94% used (19G free) — NOT used for weights; local NVMe
`/opt/dlami/nvme` 3.5T with 3.2T free is the staging target (per the /data->NVMe staging
rule). **No deletions were needed or performed** on the box; the historical "EBS ~93% full"
constraint is moot because everything stages on NVMe (5% used after staging).

## Transfer route

The expert payload was NOT rsync'd from the local rig (it no longer exists there —
reclaimed 2026-07-30, published to HF). Both big pieces came down HF Hub -> Mumbai directly
(minutes, vs hours over the home uplink):

1. `Avifenesh/Hy3-REAP-Layer103p5-bw24` (full repo, 487 files) ->
   `/opt/dlami/nvme/models/hy3-layer103p5-bw24-restored` (73.1 GiB payload + receipts).
   Log: `logs/dl-overlay.log`.
2. `tencent/Hy3` @ `716aa7241bd6d95896be4ebfc761162a9c4d49ef`, the 20 non-expert shards +
   config/tokenizer/index (23.3 GiB) -> `/opt/dlami/nvme/models/hy3-sparse-source-dl`.
   Logs: `logs/dl-source.log`, `logs/dl-source-2.log` (shard 00006 missed by the first
   batch fetch; refetched and verified present).
3. rsync from local rig (small): runtime view dir (manifest.json 9.8M, frspec drafts,
   RESTORE.md, relocation-receipt.json) -> `/opt/dlami/nvme/models/hy3-layer103p5-bw24-runtime`;
   10-byte `.expert-only.empty.safetensors` placeholder + `sparse-source-receipt.json`.

Tooling: no pip on the box; bootstrapped `uv` and ran `uvx --from 'huggingface_hub[cli,hf_transfer]'
hf download` with `HF_HOME=/opt/dlami/nvme/hf`.

## Assembly (exact mirror of the local layout)

- `/opt/dlami/nvme/models/hy3-sparse-source/`: 107 entries = 20 real shard symlinks ->
  `hy3-sparse-source-dl/`, 79 expert-only shard names -> `.expert-only.empty.safetensors`,
  6 config/tokenizer files, placeholder, receipt.
- `/data/ai-ml/hf-models/hy3-layer103p5-sparse-source` -> the above (the manifest bakes this
  absolute `source_dir`; symlinking keeps `manifest.json` byte-identical instead of editing it).
- runtime dir `experts` -> `../hy3-layer103p5-bw24-restored/experts`.

## Verification (on-box, 2026-08-01)

| check | result |
|---|---|
| staged runtime `manifest.json` sha256 | `b8bdd684a0112312f3714024b97b9c18c8a3e7e474cbd7111f6f6021be6a644c` == pinned ✓ |
| staged published `manifest.json` sha256 | `08f206aed555752982585a59a7b5096b9cc6e71faf1f84ad5c6dd60476b7509a` == pinned ✓ |
| staged `config.json` sha256 | `663036ce…` == source fingerprint ✓ |
| staged `model.safetensors.index.json` sha256 | `9594f1a9…` == source fingerprint ✓ |
| expert `.bin` shard count | 237 == RESTORE.md ✓ |
| expert payload bytes | 78,490,288,128 == manifest `payload_bytes` ✓ |
| all 99 index shard names resolve | OK ✓ |

## Build on box

Lane tree rsync'd to `/opt/dlami/nvme/hy3-hopper/memra` (isolated — `~/memra` belongs to
another live lane, sk-bm128, and was NOT touched). `~/.cargo/bin/cargo build --release`
(kernel-check, run-gen, run-spec): `MEMRA_CUDA_ARCH auto-detected 90a (compute_cap 9.0)`,
finished in 3m57s. Log: `logs/build.log`.
