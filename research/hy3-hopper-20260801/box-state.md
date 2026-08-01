# Mumbai box state left for the Aug-2 spike (as of 2026-08-01)

Everything below lives on the Mumbai darklanes-bench H100 (`13.203.16.133`) local NVMe.
NVMe is instance-ephemeral: if the box stops, restage per transfer-receipt.md (the
downloads took ~6 min total from HF; the assemble script is
`/opt/dlami/nvme/hy3-stage/mumbai-assemble.sh`-equivalent, reproduced in this lane's
transfer-receipt.md).

| path | what |
|---|---|
| `/opt/dlami/nvme/models/hy3-layer103p5-bw24-runtime/` | serving runtime view (manifest sha `b8bdd684…` verified, `experts` symlink, frspec drafts) — pass THIS dir to run-gen/run-spec |
| `/opt/dlami/nvme/models/hy3-layer103p5-bw24-restored/` | published overlay repo (73.1 GiB experts payload, 237 shards verified) |
| `/opt/dlami/nvme/models/hy3-sparse-source/` | reconstructed non-expert fallback (20 real shards + 79 empty placeholders + tokenizer/config, fingerprints verified) |
| `/data/ai-ml/hf-models/hy3-layer103p5-sparse-source` | symlink -> above (the manifest's baked absolute source_dir) |
| `/opt/dlami/nvme/hy3-hopper/memra/` | lane/hy3-hopper tree + sm_90a release build (kernel-check, run-gen, run-spec) — includes the G1 fix |
| `/opt/dlami/nvme/hy3-stage/logs/` | all raw logs (mirrored into this receipts dir) |
| `~/memra` | NOT ours — the sk-bm128 lane's tree; untouched |

Shared-box rule honored throughout: every GPU-touching run wrapped in
`flock /tmp/gpu-h100.lock`. Root EBS untouched by weights (94% full; nothing deleted).

Quick re-run recipes:

    # greedy first-light
    flock /tmp/gpu-h100.lock env MEMRA_CHAT=1 MEMRA_NGEN=32 MEMRA_PROMPT="..." \
      /opt/dlami/nvme/hy3-hopper/memra/target/release/run-gen \
      /opt/dlami/nvme/models/hy3-layer103p5-bw24-runtime

    # spec probe (NEVER use the default single-token prompt for acceptance numbers)
    flock /tmp/gpu-h100.lock env MEMRA_SPEC_K=2 \
      /opt/dlami/nvme/hy3-hopper/memra/target/release/run-spec \
      /opt/dlami/nvme/models/hy3-layer103p5-bw24-runtime <tok ids...>
