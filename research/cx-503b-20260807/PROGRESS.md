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

## Item 2 — serving worker primary device

Finding: `memra-server` always constructed `Engine::new(0)`. `CUDA_VISIBLE_DEVICES=<physical>`
already makes logical 0 the correct replica-fleet choice, but PP placement is expressed inside
that visible namespace by `MEMRA_PP_DEVICES`; reversed placement (`1,0`) therefore left the
server primary on device 0 while every PP gate/bench made stage 0 the primary on device 1.

Fix:

- select the worker primary from `MEMRA_PP_DEVICES[0]` when the PP placement is present;
- keep logical device 0 as the default, preserving CUDA_VISIBLE_DEVICES semantics without adding
  a duplicate `MEMRA_DEVICE` knob;
- reject an invalid first PP device before CUDA/model initialization;
- log the chosen logical device in the worker-ready line.

GPU-free unit coverage pins default/empty -> 0, reversed placement `1,0` -> 1, whitespace parsing,
and invalid-primary refusal.

Verification:

- `cargo test -p memra-server worker_device` — PASS, 2 passed.
- `cargo check -p memra-engine -p memra-server` — PASS.
- `cargo test -p memra-server` — PASS, 115 passed.

One earlier parallel full-suite invocation failed the unrelated
`a_wedged_gpu_flips_health_even_though_the_worker_thread_is_fine` assertion with exact output
`left: 200`, `right: 503` after its expected GPU-watch critical line. The isolated test passed,
the serial 115-test suite passed, and the exact requested command then passed on rerun; no code
change was made for that transient.
