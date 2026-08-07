# cx-503b follow-up — 2026-08-07

Branch: `lane/cx-bare503`

Steering check: `~/.lanectl/inbox/cx-503.md` did not exist. The lane registry listed `cx-503b`
on this branch/worktree with no notes.

## Item 1 — PP-aware resident-expert sizing

Finding: `build_dev_exps` froze one process-global decision from the first MoE layer. Its GGUF
numerator summed every `_exps.` tensor in the model, then compared that whole bank with the free
VRAM reported by one layer's device. A sharded PP placement therefore charged each card for
expert layers resident on the other card.

Fix:

- replace the global `OnceLock<bool>` with a load-local `ResidentPlan`;
- map each trunk layer through the existing `pp::layer_engine` placement and sum exact GGUF
  expert bytes by CUDA device;
- make one decision per physical device, so distinct PP stages use their own slice and repeated
  device placements combine their slices before deciding;
- retain the non-PP whole-bank path and the existing non-expert/headroom budget semantics;
- keep mixed-layout and explicit disk-tier expert banks on the metadata-aware SLRU paths.

The sizing law was cross-checked against merged SGLang PR #33666: per-stage resources must not
charge one rank for the whole model; shared derived limits additionally need rank-uniform sizing.
Resident slabs are not a cross-rank shared limit, so memra decides independently per owning device.

Verification:

- `cargo check -p memra-engine` — PASS.
- `cargo test -p memra-engine residency_tests --lib` — PASS, 2 passed.
  - split placement `[0,0,1,1]` produced independent 60-byte / 90-byte expert totals;
  - co-located placement `[0,0,0,0]` combined all 100 bytes against the shared device.
- `cargo test -p memra-engine --lib` — PASS, 48 passed, 1 GPU-only test ignored.
