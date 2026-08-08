# Final gate audit

Status: clean final receipt.

- Exact source commit: `8d8ba1eaad71c4d2c36426f20d24f7e4d28d14be`.
- GPU window: 2026-08-08 11:03:00Z-11:04:29Z.
- Both GPUs were at 0 MiB and the compute-process lists were empty at entry
  and exit.
- Single-card, PP-2 dev10, and PP-2 dev01 run-spec logs each contain eight
  K rows, eight self-consistency PASS rows, and the final PASS marker.
- PP-2 default: LOW=0/HIGH=1, 4/4 complete, zero `[spec-acc]` lines.
- Single-card default: LOW=2/HIGH=4, 4/4 complete, 13 `[spec-acc]` lines.
- #87 forced-spec quick gate: c=2 8/8, c=4 16/16, recovery c=1 4/4;
  85 `[spec-acc]` lines.
- Every policy/crash server reported at least 0.5 seconds of
  `serve_idle_seconds` before SIGTERM.
- No `CUDA_ERROR_ILLEGAL_ADDRESS`, `CUDA_ERROR_DEINITIALIZED`, #87 trap,
  argmax sentinel, pending-flush failure, Xid 31, page-table fault, panic,
  or abort line was found.
- `tools/serve-smoke.sh`: 0 failed. Its optional Gemma arm skipped because
  that artifact is absent on box2.

The q9 trunk and draft hashes match the measurement receipt. Binary hashes for
this exact build are recorded in `binary-sha256.txt`.
